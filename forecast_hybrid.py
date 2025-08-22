'''
Hybrid forecasting pipeline for {TICKER} (trend + ARIMA residuals)
Fixed MultiIndex flattening & robust Price column creation.

Outputs:
- {TICKER}_engineered_and_forecast.csv
- {TICKER}_future_forecast.csv
'''

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# optional auto-arima
try:
    from pmdarima import auto_arima
    HAVE_PMD = True
except Exception:
    HAVE_PMD = False

# ----------------- user settings -----------------
TICKER = "AAPL"
START_DATE = "2024-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
PRICE_PREFERENCE = ["Adj Close", "Adj_Close", "AdjClose", "Close"]
TEST_RATIO = 0.2
FORECAST_STEPS_FUTURE = 10
AUTO_ARIMA_MAX_PQ = 5
# -------------------------------------------------

# ------------------ helpers -----------------------
def safe_download(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise SystemExit(f"No data downloaded for {ticker} — check ticker or date range.")
    return df

def make_unique_column_names(cols):
    """Ensure column names are unique by appending suffixes for duplicates."""
    seen = {}
    out = []
    for c in cols:
        c_str = str(c)
        if c_str in seen:
            seen[c_str] += 1
            out.append(f"{c_str}__{seen[c_str]}")
        else:
            seen[c_str] = 0
            out.append(c_str)
    return out

def flatten_multiindex_columns(df, price_preferences=None):
    """
    Robust flatten: tries to pick the level containing price-like names (Adj Close / Close).
    If no single level matches, creates readable combined names and ensures uniqueness.
    """
    if price_preferences is None:
        price_preferences = PRICE_PREFERENCE

    if not isinstance(df.columns, pd.MultiIndex):
        return df

    nlevels = df.columns.nlevels
    # 1) try to find a level that contains any preferred price names
    for lvl in range(nlevels):
        level_vals = [str(v) for v in df.columns.get_level_values(lvl)]
        lowercase = [v.lower() for v in level_vals]
        if any(any(pref.lower() in v for pref in price_preferences) for v in lowercase):
            # use this level as column names
            df.columns = df.columns.get_level_values(lvl)
            df.columns = pd.Index(make_unique_column_names(df.columns.tolist()))
            return df

    # 2) If not found, try level 0 (common case where level 0 = 'Adj Close', level1 = ticker)
    try:
        lvl0 = df.columns.get_level_values(0)
        if len(set(lvl0)) == len(lvl0):
            df.columns = lvl0
            df.columns = pd.Index(make_unique_column_names(df.columns.tolist()))
            return df
    except Exception:
        pass

    # 3) fallback: join all levels into one string per column, then make unique
    new_cols = ['_'.join(map(str, tup)).strip().replace(' ', '_') for tup in df.columns]
    df.columns = pd.Index(make_unique_column_names(new_cols))
    return df

def ensure_price_column(df, price_preferences=None):
    """Ensure df has a single numeric 'Price' column selected robustly."""
    if price_preferences is None:
        price_preferences = PRICE_PREFERENCE

    # flatten if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df = flatten_multiindex_columns(df, price_preferences=price_preferences)

    # 1) direct preference matches
    for pref in price_preferences:
        matches = [c for c in df.columns if str(c).lower() == str(pref).lower()]
        if matches:
            # if multiple matches (unlikely after flatten), pick the first by position
            df['Price'] = df.iloc[:, df.columns.get_indexer([matches[0]])[0]]
            return df

    # 2) fuzzy match for "adj" + "close" or any "close"
    matches = [c for c in df.columns if ('adj' in str(c).lower() and 'close' in str(c).lower())]
    if not matches:
        matches = [c for c in df.columns if 'close' in str(c).lower()]
    if matches:
        # select the first matched column by positional index (handles dup names)
        colpos = df.columns.get_loc(matches[0])
        if isinstance(colpos, slice):
            # pick first column if slice
            colpos = colpos.start
        if isinstance(colpos, (list, np.ndarray)):
            colpos = int(colpos[0])
        df['Price'] = df.iloc[:, colpos]
        return df

    # 3) fall back to first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        # pick first numeric column by position
        pos = df.columns.get_loc(numeric_cols[0])
        if isinstance(pos, slice):
            pos = pos.start
        if isinstance(pos, (list, np.ndarray)):
            pos = int(pos[0])
        df['Price'] = df.iloc[:, pos]
        return df

    raise KeyError("No suitable price column found. Available columns: " + ", ".join(map(str, df.columns.tolist())))

def tz_normalize(df, target_tz="America/New_York"):
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(target_tz)
    else:
        df.index = df.index.tz_convert(target_tz)
    return df

def small_defensive_fill(df, cols):
    cols_present = [c for c in cols if c in df.columns]
    df[cols_present] = df[cols_present].ffill(limit=1).bfill(limit=1)
    return df

def run_adf(series, name="series"):
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna(), autolag="AIC")
        print(f"ADF {name}: stat={result[0]:.4f}, p-value={result[1]:.4f}")
        return result
    except Exception:
        return None

def choose_arima_order(train_resid):
    if HAVE_PMD:
        try:
            am = auto_arima(train_resid, seasonal=False, stepwise=True, suppress_warnings=True,
                            error_action="ignore", max_p=AUTO_ARIMA_MAX_PQ, max_q=AUTO_ARIMA_MAX_PQ, d=None)
            order = am.order if hasattr(am, "order") else (1,1,1)
            return order
        except Exception:
            return (1,1,1)
    else:
        return (1,1,1)

def fit_arima_on_resid(train_resid, order=None):
    if order is None:
        order = choose_arima_order(train_resid)
    model = ARIMA(pd.Series(train_resid).values, order=order)
    fit = model.fit()
    return fit, order

def forecast_hybrid(train_series, test_index, arima_order=None):
    steps = len(test_index)
    if steps == 0:
        return pd.Series(dtype=float), {}

    y = train_series.dropna().values
    n = len(y)
    if n < 5:
        last_val = train_series.dropna().iloc[-1]
        fc = pd.Series([last_val]*steps, index=test_index)
        return fc, {'order': None, 'slope': 0.0, 'intercept': float(last_val)}

    x = np.arange(n)
    slope, intercept = np.polyfit(x, y, 1)
    last_x = x[-1]
    x_fore = np.arange(last_x + 1, last_x + 1 + steps)
    trend_proj = intercept + slope * x_fore
    trend_pred = pd.Series(trend_proj, index=test_index)

    in_sample_trend = intercept + slope * x
    resid = y - in_sample_trend

    try:
        arima_fit, chosen_order = fit_arima_on_resid(pd.Series(resid), order=arima_order)
        resid_fc_res = arima_fit.get_forecast(steps=steps)
        resid_pred_vals = np.asarray(resid_fc_res.predicted_mean).ravel()
    except Exception as e:
        print("Warning: ARIMA residual modeling failed, using zero residuals. Error:", e)
        resid_pred_vals = np.zeros(steps)
        chosen_order = None

    resid_pred = pd.Series(resid_pred_vals, index=test_index)
    combined = trend_pred + resid_pred
    info = {'order': chosen_order, 'slope': float(slope), 'intercept': float(intercept)}
    return combined, info

def evaluate_forecast(y_true, y_pred):
    mask = ~pd.isna(y_pred) & ~pd.isna(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = (np.abs((y_true - y_pred) / y_true)).mean() * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE%': mape, 'N': len(y_true)}

# -------------------- main --------------------
if __name__ == "__main__":
    print("Starting hybrid forecasting pipeline for", TICKER)

    # 1) Download
    df_raw = safe_download(TICKER, START_DATE, END_DATE)
    print("Raw columns (yfinance):", df_raw.columns.tolist())

    # 2) Flatten MultiIndex if present
    df_flat = flatten_multiindex_columns(df_raw)
    print("Columns after flatten:", df_flat.columns.tolist())

    # 3) Ensure Price column exists
    df = df_flat.copy()
    df = ensure_price_column(df, price_preferences=PRICE_PREFERENCE)
    print("Using Price column. Sample:")
    print(df['Price'].dropna().head())

    # 4) Normalize timezone for consistency (optional)
    try:
        df = tz_normalize(df, "America/New_York")
    except Exception:
        pass

    # 5) Defensive fill
    df.sort_index(inplace=True)
    df = small_defensive_fill(df, ["Open", "High", "Low", "Close", "Price", "Volume"])

    # 6) Feature engineering
    df["Return"] = df["Price"].pct_change()
    df["LogReturn"] = np.log(df["Price"] / df["Price"].shift(1))
    df["SMA20"] = df["Price"].rolling(window=20, min_periods=1).mean()
    df["SMA50"] = df["Price"].rolling(window=50, min_periods=1).mean()
    df["Vol20"] = df["LogReturn"].rolling(window=20, min_periods=1).std() * np.sqrt(252)
    df.dropna(subset=['Price'], inplace=True)

    # 7) Outlier reporting (non-fatal)
    try:
        zs = np.abs(stats.zscore(df["Return"].dropna()))
        outlier_dates = df["Return"].dropna().index[zs > 3]
        print(f"Detected {len(outlier_dates)} extreme-return dates (|z|>3). Example:", outlier_dates[:5].tolist())
    except Exception:
        pass

    # 8) Stationarity (informational)
    print("\n-- ADF tests (informational) --")
    run_adf(df["Price"], "Price")
    run_adf(df["LogReturn"], "LogReturn")

    # 9) Train/test split (time-ordered)
    n = len(df)
    test_n = int(np.ceil(n * TEST_RATIO))
    if test_n < 1:
        raise SystemExit("Test set size computed as 0 — increase data range or adjust TEST_RATIO.")
    train = df.iloc[:-test_n].copy()
    test = df.iloc[-test_n:].copy()
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # 10) Hybrid forecasting on test horizon
    hybrid_fc, info = forecast_hybrid(train['Price'], test.index, arima_order=None)
    aligned_fc = hybrid_fc.reindex(df.index)
    df['Forecast_on_test'] = aligned_fc
    df['Forecast_hybrid_on_test'] = aligned_fc

    # 11) Evaluate on test horizon (informational)
    metrics = evaluate_forecast(test['Price'], hybrid_fc)
    print("Hybrid evaluation on test set:", metrics)
    print("Hybrid info:", info)

    # 12) Diagnostics (optional)
    try:
        if info.get('order') is not None:
            y = train['Price'].dropna().values
            x = np.arange(len(y))
            slope = info.get('slope', np.polyfit(x, y, 1)[0])
            intercept = info.get('intercept', np.polyfit(x, y, 1)[1])
            in_sample_trend = intercept + slope * x
            resid = pd.Series(y - in_sample_trend, index=train['Price'].dropna().index)
            arima_fit, _ = fit_arima_on_resid(resid, order=info['order'])
            resid_series = pd.Series(arima_fit.resid).dropna()
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lj = acorr_ljungbox(resid_series, lags=[10], return_df=True)
                print("\nLjung-Box (lag=10):\n", lj)
            except Exception:
                pass
    except Exception:
        pass

    # 13) Plot actual vs forecast (optional)
    try:
        plt.figure(figsize=(12,5))
        tail_n = min(200, len(train))
        plt.plot(train.index[-tail_n:], train["Price"].iloc[-tail_n:], label="Historical (train tail)")
        plt.plot(test.index, test["Price"], label="Actual (test)")
        plt.plot(hybrid_fc.index, hybrid_fc.values, label="Hybrid Forecast", linestyle="--", marker="o")
        plt.title(f"{TICKER} Hybrid Trend+ARIMA Forecast")
        plt.xlabel("Date"); plt.ylabel("Price")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.show()
    except Exception:
        pass

    # 14) Save outputs
    out_path = f"{TICKER}_engineered_and_forecast.csv"
    df.to_csv(out_path)
    print(f"Saved engineered data + hybrid forecast to {out_path}")

    # 15) Retrain on full data and forecast next N steps
    try:
        full_series = df['Price'].dropna()
        last_dt = df.index[-1]
        tz = last_dt.tz if getattr(last_dt, "tz", None) is not None else None
        if tz is not None:
            fut_index = pd.date_range(start=last_dt + pd.Timedelta(days=1),
                                      periods=FORECAST_STEPS_FUTURE, freq="B", tz=tz)
        else:
            fut_index = pd.date_range(start=last_dt + pd.Timedelta(days=1),
                                      periods=FORECAST_STEPS_FUTURE, freq="B")
        fut_fc, fut_info = forecast_hybrid(full_series, fut_index, arima_order=info.get('order'))
        fut_df = pd.DataFrame({"forecast": fut_fc.values}, index=fut_index)
        fut_df.to_csv(f"{TICKER}_future_forecast.csv")
        print(f"Saved future forecast to {TICKER}_future_forecast.csv")
    except Exception as e:
        print("Warning: future forecast failed:", e)

    print("Done.")
