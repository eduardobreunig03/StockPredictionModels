import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from src.features import build_feature_matrix
from src.preprocess import prepare_data_wfv
from src import save_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inverse_close(scaler, values: np.ndarray) -> np.ndarray:
    """Inverse-transform scaled Close predictions → dollar values."""
    n_features = scaler.n_features_in_
    if n_features == 1:
        return scaler.inverse_transform(values.reshape(-1, 1)).flatten()
    dummy = np.zeros((len(values), n_features))
    dummy[:, 0] = values.flatten()
    return scaler.inverse_transform(dummy)[:, 0]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MSE":  mean_squared_error(y_true, y_pred),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 62

    print(f"\n{sep}")
    print(f"  AAPL Stock Prediction Pipeline")
    print(f"  Ticker : {config.TICKER}")
    print(f"  Period : {config.START_DATE}  →  {config.END_DATE}")
    print(f"  Window : {config.WINDOW_SIZE} days  |  WFV folds: {config.N_WFV_SPLITS}")
    print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Step 1 — Download
    # ------------------------------------------------------------------
    print("Step 1/4  Downloading data …")
    df = yf.download(
        config.TICKER,
        start=config.START_DATE,
        end=config.END_DATE,
        interval=config.INTERVAL,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("Download failed — check ticker or date range.")
    print(
        f"          {len(df)} trading days  "
        f"({df.index[0].date()} → {df.index[-1].date()})"
    )

    # ------------------------------------------------------------------
    # Step 2 — Feature engineering
    # ------------------------------------------------------------------
    print("\nStep 2/4  Computing features (RSI, MACD, Bollinger Bands) …")
    feat_df = build_feature_matrix(df)
    n_rows, n_cols = feat_df.shape
    print(f"          {n_rows} rows × {n_cols} features after warm-up drop")
    print(f"          Columns: {list(feat_df.columns)}")

    # ------------------------------------------------------------------
    # Step 3 — WFV preprocessing
    # ------------------------------------------------------------------
    print(f"\nStep 3/4  Walk-forward preprocessing ({config.N_WFV_SPLITS} folds) …")
    folds, scaler = prepare_data_wfv(
        ticker=config.TICKER,
        feature_matrix=feat_df.values,   # shape (n, 8)
        window_size=config.WINDOW_SIZE,
        n_splits=config.N_WFV_SPLITS,
        test_ratio=config.WFV_TEST_RATIO,
    )

    # Persist last fold so evaluate.py / train.py stay compatible
    X_tr_last, y_tr_last, X_te_last, y_te_last = folds[-1]
    save_data.save_processed_data(
        X_tr_last, y_tr_last, X_te_last, y_te_last, scaler, config.TICKER
    )
    print(
        f"          Last fold saved  "
        f"({len(X_tr_last)} train  /  {len(X_te_last)} test samples)"
    )

    # ------------------------------------------------------------------
    # Step 4 — XGBoost per fold
    # ------------------------------------------------------------------
    print(f"\nStep 4/4  Training XGBoost on each fold …\n")

    fold_rows = []
    for i, (X_tr, y_tr, X_te, y_te) in enumerate(folds, start=1):
        model = xgb.XGBRegressor(
            objective=config.XGB_OBJECTIVE,
            n_estimators=config.XGB_N_ESTIMATORS,
            verbosity=0,
        )
        model.fit(X_tr.reshape(len(X_tr), -1), y_tr)
        y_pred = model.predict(X_te.reshape(len(X_te), -1))

        y_te_inv   = _inverse_close(scaler, y_te)
        y_pred_inv = _inverse_close(scaler, y_pred)
        m = _metrics(y_te_inv, y_pred_inv)

        fold_rows.append(
            {
                "Fold":           i,
                "Train samples":  len(X_tr),
                "Test samples":   len(X_te),
                **m,
            }
        )
        print(
            f"  Fold {i}/{config.N_WFV_SPLITS}  "
            f"MSE={m['MSE']:>9.2f}  MAE={m['MAE']:>6.2f}  "
            f"R²={m['R2']:>7.4f}  MAPE={m['MAPE']:>6.2f}%"
        )

    # ------------------------------------------------------------------
    # Aggregate + display
    # ------------------------------------------------------------------
    metric_cols = ["MSE", "MAE", "R2", "MAPE"]
    mean_row = {
        "Fold":          "MEAN",
        "Train samples": "",
        "Test samples":  "",
        **{c: float(np.mean([r[c] for r in fold_rows])) for c in metric_cols},
    }
    fold_rows.append(mean_row)

    results_df = pd.DataFrame(fold_rows)

    print(f"\n{sep}")
    print(f"  XGBoost Walk-Forward Results — {config.TICKER}")
    print(f"{sep}")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Export CSV
    # ------------------------------------------------------------------
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{config.TICKER}_wfv_results.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\n  Exported → {csv_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
