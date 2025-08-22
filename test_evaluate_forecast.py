
"""
test_evaluate_forecast.py

Usage examples:
    python test_evaluate_forecast.py --csv AAPL_engineered_and_forecast.csv
    python test_evaluate_forecast.py --ticker AAPL --lookahead 30

Outputs:
 - prints detected events
 - saves events to "<ticker>_support_resistance_events.csv"
 - shows a matplotlib plot with events marked
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Core detection function ----------
def detect_support_resistance_events(df,
                                     price_col='Price',
                                     forecast_col=None,
                                     lookahead=20,
                                     tol=1e-8):
    """
    Detect events where price touches/dips below forecast and then rises again => support,
    or price touches/rises above forecast and then falls again => resistance.

    Returns a DataFrame with columns ['Date', 'Actual Price', 'Forecast', 'Event Type'].
    (Internally also computes and uses positional indices correctly to avoid TypeError.)
    """
    # auto-detect forecast column if not supplied
    if forecast_col is None:
        candidates = ['Forecast_on_test', 'Forecast_hybrid_on_test', 'Forecast', 'forecast', 'forecast_on_test']
        for c in candidates:
            if c in df.columns:
                forecast_col = c
                break
        if forecast_col is None:
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            numeric_cols = [c for c in numeric_cols if c != price_col]
            forecast_col = numeric_cols[0] if numeric_cols else None

    if forecast_col is None or price_col not in df.columns or forecast_col not in df.columns:
        raise KeyError(f"Could not find required columns. price_col='{price_col}', forecast_col='{forecast_col}'")

    # work on a copy
    s_price = df[price_col].astype(float)
    s_fc = df[forecast_col].astype(float)

    diff = s_price - s_fc

    events = []
    n = len(df)

    # We'll use integer positions for lookups, but keep timestamps for storage/display
    for i in range(1, n - 1):
        # skip NaNs
        if pd.isna(diff.iat[i]) or pd.isna(diff.iat[i - 1]):
            continue

        prev = diff.iat[i - 1]
        cur = diff.iat[i]

        # Potential support: previously above (>tol), currently touching/dipping (<= tol)
        if (prev > tol) and (cur <= tol):
            j_end = min(n, i + 1 + lookahead)
            # slice by positional indices
            future_vals = diff.iloc[i + 1:j_end].values
            # boolean array for recovery
            recovered_mask = future_vals > tol
            if recovered_mask.any():
                # position relative to the full series
                rel_pos = int(np.argmax(recovered_mask))
                recovery_pos = i + 1 + rel_pos               # integer positional index of recovery
                recovery_date = df.index[recovery_pos]       # timestamp
                # record event at touch index i
                events.append({
                    'Date': df.index[i],
                    'Actual Price': float(s_price.iat[i]),
                    'Forecast': float(s_fc.iat[i]),
                    'Event Type': 'support',
                    'Touch_Pos': i,
                    'Recovery_Pos': recovery_pos,
                    'Recovery_Date': recovery_date
                })
                # skip ahead to the recovery to avoid overlapping detections
                continue

        # Potential resistance: previously below (< -tol), currently touching/above (>= -tol)
        if (prev < -tol) and (cur >= -tol):
            j_end = min(n, i + 1 + lookahead)
            future_vals = diff.iloc[i + 1:j_end].values
            drop_mask = future_vals < -tol
            if drop_mask.any():
                rel_pos = int(np.argmax(drop_mask))
                drop_pos = i + 1 + rel_pos
                drop_date = df.index[drop_pos]
                events.append({
                    'Date': df.index[i],
                    'Actual Price': float(s_price.iat[i]),
                    'Forecast': float(s_fc.iat[i]),
                    'Event Type': 'resistance',
                    'Touch_Pos': i,
                    'Drop_Pos': drop_pos,
                    'Drop_Date': drop_date
                })
                continue

    # Build final DataFrame (keep only the requested columns)
    if len(events) == 0:
        events_df = pd.DataFrame(columns=['Date', 'Actual Price', 'Forecast', 'Event Type'])
    else:
        events_df = pd.DataFrame(events)
        # Only keep the primary columns the user requested
        events_df = events_df[['Date', 'Actual Price', 'Forecast', 'Event Type']].copy()
        events_df['Date'] = pd.to_datetime(events_df['Date'])

    return events_df


# ---------- Plotting helper ----------
def plot_with_events(df, events_df,
                     price_col='Price',
                     forecast_col=None,
                     title="Actual vs Forecast (support/resistance events)",
                     figsize=(14,6)):
    if forecast_col is None:
        # same logic used earlier for default pick
        candidates = ['Forecast_on_test', 'Forecast_hybrid_on_test', 'Forecast', 'forecast', 'forecast_on_test']
        for c in candidates:
            if c in df.columns:
                forecast_col = c
                break
        if forecast_col is None:
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            numeric_cols = [c for c in numeric_cols if c != price_col]
            forecast_col = numeric_cols[0] if numeric_cols else None

    plt.figure(figsize=figsize)
    plt.plot(df.index, df[price_col], label='Actual Price', linewidth=1.2)
    if forecast_col in df.columns:
        plt.plot(df.index, df[forecast_col], label='Forecast', linestyle='--', linewidth=1.2)

    # mark events
    for _, row in events_df.iterrows():
        dt = pd.to_datetime(row['Date'])
        price = row['Actual Price']
        typ = row['Event Type']
        if typ == 'support':
            plt.scatter(dt, price, marker='v', s=80, label='Support touch' if 'Support touch' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.annotate('support', (dt, price), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=9)
        elif typ == 'resistance':
            plt.scatter(dt, price, marker='^', s=80, label='Resistance touch' if 'Resistance touch' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.annotate('resistance', (dt, price), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- Utility: load CSV (robust) ----------
def load_forecast_csv(path, parse_dates=True, index_col=0):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=parse_dates, index_col=index_col)
    # if index column not parsed or duplicated, try to parse a 'Date' column
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
    return df

# ---------- CLI / main ----------
def main():
    parser = argparse.ArgumentParser(description="Test/evaluate forecast: detect support/resistance touches.")
    parser.add_argument('--csv', type=str, help="Path to engineered forecast CSV (default: <ticker>_engineered_and_forecast.csv)")
    parser.add_argument('--ticker', type=str, default='AAPL', help="Ticker - used to build default CSV filename")
    parser.add_argument('--price-col', type=str, default='Price', help="Column name for actual price (default: Price)")
    parser.add_argument('--forecast-col', type=str, default=None, help="Column name for forecast (auto-detected if not set)")
    parser.add_argument('--lookahead', type=int, default=20, help="Lookahead days to confirm reversal (default 20)")
    parser.add_argument('--save', action='store_true', help="Save detected events to CSV")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        csv_path = f"{args.ticker}_engineered_and_forecast.csv"

    print("Loading CSV:", csv_path)
    df = load_forecast_csv(csv_path)

    # detect events
    events_df = detect_support_resistance_events(df,
                                                price_col=args.price_col,
                                                forecast_col=args.forecast_col,
                                                lookahead=args.lookahead)
    if events_df.empty:
        print("No support/resistance touch events detected with current settings.")
    else:
        print(f"Detected {len(events_df)} events:")
        print(events_df.to_string(index=False))

    # optionally save
    out_name = f"{args.ticker}_support_resistance_events.csv"
    if args.save:
        events_df.to_csv(out_name, index=False)
        print("Saved events CSV to", out_name)

    # plot (always show)
    plot_with_events(df, events_df, price_col=args.price_col, forecast_col=args.forecast_col,
                     title=f"{args.ticker} Actual vs Forecast (support/resistance)")

if __name__ == "__main__":
    main()
