# Stock Prediction Models Comparison

An ongoing research project that benchmarks multiple machine learning models for stock price prediction. Models are evaluated side-by-side on the same data using walk-forward validation, so results reflect real-world performance across different market conditions rather than a single lucky test split.

> **Status: Active development. Still in Progress** 

---

## What this project does

Given a stock ticker and date range, the pipeline:

1. Downloads historical OHLCV data from Yahoo Finance
2. Computes technical indicators — RSI, MACD, Bollinger Bands
3. Builds sliding-window sequences (90-day lookback by default)
4. Evaluates models using **walk-forward validation** — the training window expands fold by fold, and the test window slides forward, mimicking how a model would actually be used over time
5. Reports MSE, MAE, R², and MAPE per fold and exports results to CSV

### Models included

| Model | Type | Notes |
|-------|------|-------|
| **XGBoost** | Gradient boosting | Fast; works well on tabular/flattened sequences |
| **LSTM** | Recurrent neural net | Captures long-range temporal dependencies |
| **CNN** | Convolutional neural net | Detects local price patterns |
| **Transformer** | Attention-based | State-of-the-art sequence modelling |
| **ARIMA** | Classical statistical | Baseline; uses auto parameter selection |

---

## Project structure

```
stock-prediction-models-comparison-main/
│
├── config.py                  # All settings live here — change this, not the model files
├── run_pipeline.py            # Non-interactive end-to-end script (recommended entry point)
├── main.py                    # Interactive CLI menu (for training neural nets)
│
├── src/
│   ├── features.py            # RSI, MACD, Bollinger Bands
│   ├── preprocess.py          # Walk-forward splits + sliding-window sequencing
│   ├── evaluate.py            # Loads trained models, computes metrics, exports CSV
│   ├── train.py               # Orchestrates multi-model training with repeat runs
│   ├── save_data.py           # Saves raw and processed data to disk
│   ├── load_data.py           # Loads data from local CSV
│   ├── predict.py             # Next-day price prediction from saved models
│   └── models/
│       ├── lstm_model.py
│       ├── cnn_model.py
│       ├── transformer.py
│       ├── xgboost_model.py
│       └── arima_model.py
│
├── data/
│   ├── raw/                   # Downloaded CSVs from Yahoo Finance
│   └── processed/             # Scaled .npy arrays ready for model training
│
├── models/                    # Saved model files (.keras, .json, .save)
│
└── results/                   # Output CSVs with evaluation metrics
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-fork-url>
cd stock-prediction-models-comparison-main
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install numpy pandas scikit-learn yfinance xgboost joblib
```

For neural net models (LSTM, CNN, Transformer), also install:

```bash
pip install tensorflow
```

**macOS only** — XGBoost requires OpenMP:

```bash
brew install libomp
```

---

## Configuration

All settings are in `config.py`. Edit this file before running anything — no need to touch any other file.

```python
# config.py

TICKER      = "AAPL"        # Stock ticker
START_DATE  = "2018-01-01"  # Data start date
END_DATE    = "2025-01-01"  # Data end date
WINDOW_SIZE = 90            # Lookback window in trading days

N_WFV_SPLITS  = 5           # Number of walk-forward folds
WFV_TEST_RATIO = 0.1        # Each test window = 10% of total data

RSI_PERIOD = 14
MACD_FAST  = 12
MACD_SLOW  = 26
BB_PERIOD  = 20
```

---

## How to run

### Option A — Full pipeline (recommended)

Runs download → features → walk-forward XGBoost → results table → CSV export.
No interaction needed.

```bash
python3 run_pipeline.py
```

Output:

```
Step 1/4  Downloading data …
          1761 trading days  (2018-01-02 → 2024-12-31)

Step 2/4  Computing features (RSI, MACD, Bollinger Bands) …
          1742 rows × 8 features after warm-up drop

Step 3/4  Walk-forward preprocessing (5 folds) …

Step 4/4  Training XGBoost on each fold …

  Fold 1/5  MSE=   260.12  MAE= 13.37  R²=-1.0343  MAPE=  8.08%
  Fold 2/5  MSE=    15.67  MAE=  3.04  R²= 0.8667  MAPE=  2.12%
  ...

  Exported → results/AAPL_wfv_results.csv
```

### Option B — Interactive menu

For training and evaluating neural nets, or making next-day predictions:

```bash
python3 main.py
```

```
[1] Download data        Fetch last 5 years from Yahoo Finance
[2] Train models         Train all or selected models
[3] Evaluate models      Compute MSE/MAE/R²/MAPE, export CSV
[4] Make prediction      Predict next day's closing price
[5] Update actual prices Fill in real prices for saved predictions
[Q] Quit
```

Typical session:

```
Your choice: 1   download data for your ticker
Your choice: 2   train models 
Your choice: 3   evaluate all models and export results/AAPL_evaluation.csv
```

---

## Understanding the results

### Walk-forward results table (`results/AAPL_wfv_results.csv`)

| Column | Meaning |
|--------|---------|
| Fold | Which time period was tested. Fold 1 = earliest market conditions, Fold 5 = most recent |
| Train samples | Number of 90-day sequences the model trained on |
| Test samples | Number of days predicted (~10% of data, typically ~165 trading days) |
| MSE | Mean Squared Error in dollars². Lower is better |
| MAE | Mean Absolute Error in dollars. "On average the prediction was $X off" |
| R² | Variance explained. 1.0 = perfect, 0 = no better than predicting the mean, negative = worse than the mean |
| MAPE | Mean Absolute Percentage Error. The most intuitive metric — "predictions were X% off on average" |

### Multi-model results (`results/AAPL_evaluation.csv`)

Same metrics, one row per model, evaluated on the static test split from the most recently saved processed data.

==
---

## Plan

- [ ] Add sentiment features from news headlines
- [ ] Add more classical baselines (Prophet, exponential smoothing)
- [ ] Hyperparameter tuning via Optuna
- [ ] Multi-step forecasting (predict 5 and 10 days ahead)
- [ ] Dockerfile for reproducible runs
- [ ] Support for crypto tickers (BTC-USD, ETH-USD)

---