import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib
from typing import Generator, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# Type alias for a single fold's data
Fold = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Walk-forward validation split
# ---------------------------------------------------------------------------

def walk_forward_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = config.N_WFV_SPLITS,
    test_ratio: float = config.WFV_TEST_RATIO,
) -> Generator[Fold, None, None]:
    """
    Time-series walk-forward validation.

    Yields n_splits folds.  Each fold expands the training window by one
    test-sized step while the test window slides forward by the same step.

    Example with n=1000, n_splits=5, test_ratio=0.1:
        test_size = 100
        fold 0 → train: [0..500),  test: [500..600)
        fold 1 → train: [0..600),  test: [600..700)
        fold 2 → train: [0..700),  test: [700..800)
        fold 3 → train: [0..800),  test: [800..900)
        fold 4 → train: [0..900),  test: [900..1000)
    """
    n = len(X)
    test_size = max(1, int(n * test_ratio))
    # The first train window ends early enough to leave room for all folds
    initial_train_end = n - n_splits * test_size

    if initial_train_end <= 0:
        raise ValueError(
            f"Not enough data for {n_splits} folds with test_ratio={test_ratio}. "
            f"Have {n} samples, need at least {n_splits * test_size + 1}."
        )

    for i in range(n_splits):
        train_end = initial_train_end + i * test_size
        test_end = train_end + test_size
        yield (
            X[:train_end],
            y[:train_end],
            X[train_end:test_end],
            y[train_end:test_end],
        )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _build_sequences(
    scaled_data: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences from scaled data."""
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size : i])
        y.append(scaled_data[i, 0])   # target is always the first column (Close)
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_data(
    ticker: str = config.TICKER,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    interval: str = config.INTERVAL,
    window_size: int = config.WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, object]:
    """
    Download and preprocess historical stock price data.

    Uses a static TRAIN_RATIO split (backward-compatible with the rest of
    the original pipeline).  For walk-forward folds use prepare_data_wfv().

    Returns
    -------
    X_train, y_train, X_test, y_test, scaler, df
    """
    df = yf.download(ticker, start=start, end=end, interval=interval)

    if df.empty:
        raise ValueError("Failed to download data. Check the ticker or dates.")

    close_prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    X, y = _build_sequences(scaled_data, window_size)

    split_index = int(len(X) * config.TRAIN_RATIO)
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    scaler_path = Path(config.MODEL_DIR) / f"{ticker}_scaler.save"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    return X_train, y_train, X_test, y_test, scaler, df


def prepare_data_wfv(
    ticker: str = config.TICKER,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    interval: str = config.INTERVAL,
    window_size: int = config.WINDOW_SIZE,
    n_splits: int = config.N_WFV_SPLITS,
    test_ratio: float = config.WFV_TEST_RATIO,
    feature_matrix: np.ndarray = None,
) -> Tuple[List[Fold], MinMaxScaler]:
    """
    Download data (or accept a pre-built feature matrix), scale, build
    sequences, then return walk-forward folds and the fitted scaler.

    Parameters
    ----------
    feature_matrix : optional np.ndarray of shape (n_timesteps, n_features)
        If supplied, raw download is skipped and this matrix is scaled
        directly.  The first column must be Close price (used as target).

    Returns
    -------
    folds : list of (X_train, y_train, X_test, y_test)
    scaler : the MinMaxScaler fitted on the full feature matrix
    """
    if feature_matrix is None:
        df = yf.download(ticker, start=start, end=end, interval=interval)
        if df.empty:
            raise ValueError("Failed to download data. Check the ticker or dates.")
        feature_matrix = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(feature_matrix)

    X, y = _build_sequences(scaled_data, window_size)

    folds = list(walk_forward_splits(X, y, n_splits=n_splits, test_ratio=test_ratio))

    scaler_path = Path(config.MODEL_DIR) / f"{ticker}_scaler.save"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    return folds, scaler


def prepare_from_series(
    close_prices: np.ndarray,
    window_size: int = config.WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Preprocess an existing close-price series (no download).

    Kept for backward compatibility with main.py.
    Returns X_train, y_train, X_test, y_test, scaler.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))

    X, y = _build_sequences(scaled_data, window_size)

    split_index = int(len(X) * config.TRAIN_RATIO)
    return (
        X[:split_index],
        y[:split_index],
        X[split_index:],
        y[split_index:],
        scaler,
    )
