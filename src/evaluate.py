import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

TICKER = os.environ.get("TICKER", config.TICKER)
PROCESSED_DIR = os.path.join(config.DATA_DIR, "processed", TICKER)
MODEL_DIR = config.MODEL_DIR
RESULTS_DIR = config.RESULTS_DIR

X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

scaler_path = Path(MODEL_DIR) / f"{TICKER}_scaler.save"
if not scaler_path.exists():
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
scaler = joblib.load(scaler_path)


def inverse_close_only(values: np.ndarray) -> np.ndarray:
    """
    Inverse-transform scaled Close predictions back to dollar values.

    Works regardless of how many features the scaler was fitted with:
    - Single-feature scaler (Close only): reshape to (n, 1) and invert.
    - Multi-feature scaler (Close + indicators): fill a zero matrix,
      place values in column 0, invert, return column 0.
    """
    n_features = scaler.n_features_in_
    if n_features == 1:
        return scaler.inverse_transform(values.reshape(-1, 1)).flatten()
    dummy = np.zeros((len(values), n_features))
    dummy[:, 0] = values.flatten()
    return scaler.inverse_transform(dummy)[:, 0]


y_test_inv = inverse_close_only(y_test)
results = {}

# === LSTM ===
try:
    lstm_model = load_model(os.path.join(MODEL_DIR, f"{TICKER}_lstm_model.keras"))
    y_pred = lstm_model.predict(X_test).flatten()
    y_pred_inv = inverse_close_only(y_pred)
    results["LSTM"] = {
        "MSE_scaled": mean_squared_error(y_test, y_pred),
        "MSE_real": mean_squared_error(y_test_inv, y_pred_inv),
        "MAE": mean_absolute_error(y_test_inv, y_pred_inv),
        "R2": r2_score(y_test_inv, y_pred_inv),
        "MAPE": np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100,
    }
except Exception as e:
    results["LSTM"] = None
    print(f"LSTM error: {e}")

# === CNN ===
try:
    cnn_model = load_model(os.path.join(MODEL_DIR, f"{TICKER}_cnn_model.keras"))
    y_pred = cnn_model.predict(X_test).flatten()
    y_pred_inv = inverse_close_only(y_pred)
    results["CNN"] = {
        "MSE_scaled": mean_squared_error(y_test, y_pred),
        "MSE_real": mean_squared_error(y_test_inv, y_pred_inv),
        "MAE": mean_absolute_error(y_test_inv, y_pred_inv),
        "R2": r2_score(y_test_inv, y_pred_inv),
        "MAPE": np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100,
    }
except Exception as e:
    results["CNN"] = None
    print(f"CNN error: {e}")

# === Transformer ===
try:
    transformer_model = load_model(
        os.path.join(MODEL_DIR, f"{TICKER}_transformer_model.keras")
    )
    y_pred = transformer_model.predict(X_test).flatten()
    y_pred_inv = inverse_close_only(y_pred)
    results["Transformer"] = {
        "MSE_scaled": mean_squared_error(y_test, y_pred),
        "MSE_real": mean_squared_error(y_test_inv, y_pred_inv),
        "MAE": mean_absolute_error(y_test_inv, y_pred_inv),
        "R2": r2_score(y_test_inv, y_pred_inv),
        "MAPE": np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100,
    }
except Exception as e:
    results["Transformer"] = None
    print(f"Transformer error: {e}")

# === XGBoost ===
try:
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(MODEL_DIR, f"{TICKER}_xgboost_model.json"))
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_pred = xgb_model.predict(X_test_flat)
    y_pred_inv = inverse_close_only(y_pred)
    results["XGBoost"] = {
        "MSE_scaled": mean_squared_error(y_test, y_pred),
        "MSE_real": mean_squared_error(y_test_inv, y_pred_inv),
        "MAE": mean_absolute_error(y_test_inv, y_pred_inv),
        "R2": r2_score(y_test_inv, y_pred_inv),
        "MAPE": np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100,
    }
except Exception as e:
    results["XGBoost"] = None
    print(f"XGBoost error: {e}")

# === ARIMA ===
try:
    arima_path = os.path.join(MODEL_DIR, f"{TICKER}_arima_predictions.csv")
    if not os.path.exists(arima_path):
        arima_path = os.path.join(
            MODEL_DIR, "temp", f"{TICKER}_arima_predictions__temp_run_1.csv"
        )

    if os.path.exists(arima_path):
        arima_df = pd.read_csv(arima_path)
        actual = arima_df["actual"].values
        predicted = arima_df["predicted"].values
        results["ARIMA"] = {
            "MSE_real": mean_squared_error(actual, predicted),
            "MAE": mean_absolute_error(actual, predicted),
            "R2": r2_score(actual, predicted),
            "MAPE": np.mean(np.abs((actual - predicted) / actual)) * 100,
        }
    else:
        results["ARIMA"] = None
        print("ARIMA prediction CSV not found.")
except Exception as e:
    results["ARIMA"] = None
    print(f"ARIMA error: {e}")

# === Display results ===
print(f"\n{'='*55}")
print(f"  Model Evaluation Results — {TICKER}")
print(f"{'='*55}")
for model, metrics in results.items():
    if metrics is not None:
        print(f"\n  {model}")
        for key, value in metrics.items():
            print(f"    {key:<14} {value:.6f}")
    else:
        print(f"\n  {model}: no data available")

# === CSV export ===
rows = []
for model, metrics in results.items():
    if metrics is not None:
        rows.append({"Model": model, **metrics})

if rows:
    results_path = Path(RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / f"{TICKER}_evaluation.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Results exported to {csv_path}")
