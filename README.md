# Stock-Forecast-Support-Resistance-Detector
Simple repo demonstrating a hybrid trend + ARIMA forecasting pipeline for a single ticker and a test script that detects support / resistance touch events where actual price interacts with the forecast.

## Prerequisites
- Python 3.10+ recommended (works with 3.8+ but 3.10+ tested)
- pip

## Files you need

forecast_hybrid.py — produces the forecast CSV

test_evaluate_forecast.py — detects support/resistance events using that CSV

(optional) requirements.txt — Python packages

## Quick start (copy code → paste in editor → install → run)

Create a folder and put two files inside it:

1. forecast_hybrid.py (paste the forecasting code)

2. test_evaluate_forecast.py (paste the test code)

Open a terminal in that folder (or open the folder in VS Code / PyCharm).

(Optional but recommended) Create and activate a virtual environment:

Windows (PowerShell)

python -m venv .venv
.\.venv\Scripts\Activate.ps1


macOS / Linux

python3 -m venv .venv
source .venv/bin/activate


Install required packages:

pip install pandas numpy yfinance matplotlib statsmodels scipy scikit-learn pmdarima


(If you don’t want pmdarima, omit it — the scripts still work with plain ARIMA.)

Run the forecast script (this creates AAPL_engineered_and_forecast.csv by default):

python forecast_hybrid.py


— If you want another ticker or date range, open forecast_hybrid.py and change TICKER, START_DATE, END_DATE at the top.

Run the test/evaluation script (reads the CSV and shows events/plot):

python test_evaluate_forecast.py


— To save detected events to CSV:

python test_evaluate_forecast.py --save
