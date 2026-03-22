import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ---------------------------------------------------------------------------
# Individual indicators
# ---------------------------------------------------------------------------

def add_rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """
    Relative Strength Index using Wilder's exponential smoothing.
    Returns a Series named 'RSI', range [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("RSI")


def add_macd(
    close: pd.Series,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
) -> pd.DataFrame:
    """
    MACD line, signal line, and histogram.
    Returns a DataFrame with columns: MACD, MACD_signal, MACD_hist.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"MACD": macd_line, "MACD_signal": signal_line, "MACD_hist": histogram},
        index=close.index,
    )


def add_bollinger_bands(
    close: pd.Series,
    period: int = config.BB_PERIOD,
    std_mult: float = config.BB_STD,
) -> pd.DataFrame:
    """
    Bollinger Bands: upper, middle (SMA), and lower bands.
    Returns a DataFrame with columns: BB_upper, BB_mid, BB_lower.
    """
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return pd.DataFrame(
        {
            "BB_upper": sma + std_mult * std,
            "BB_mid": sma,
            "BB_lower": sma - std_mult * std,
        },
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Combined feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    rsi_period: int = config.RSI_PERIOD,
    macd_fast: int = config.MACD_FAST,
    macd_slow: int = config.MACD_SLOW,
    macd_signal: int = config.MACD_SIGNAL,
    bb_period: int = config.BB_PERIOD,
    bb_std: float = config.BB_STD,
) -> pd.DataFrame:
    """
    Build a feature matrix from a DataFrame that contains a 'Close' column.

    Output columns (8 total):
        Close, RSI, MACD, MACD_signal, MACD_hist, BB_upper, BB_mid, BB_lower

    Rows with NaN values (indicator warm-up period) are dropped.
    The returned DataFrame index aligns with the original dates.

    Parameters
    ----------
    df : DataFrame with at least a 'Close' column (may be MultiIndex).
    """
    # Flatten MultiIndex columns produced by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze().astype(float)

    rsi = add_rsi(close, period=rsi_period)
    macd = add_macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    bb = add_bollinger_bands(close, period=bb_period, std_mult=bb_std)

    features = pd.concat([close.rename("Close"), rsi, macd, bb], axis=1)
    return features.dropna()
