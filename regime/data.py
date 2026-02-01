"""Data pipeline for market regime detection.

Fetches daily OHLCV data, computes dimensionless features, labels regimes
using forward-looking returns, and produces windowed samples for CNN training.
"""

import numpy as np
import yfinance as yf


REGIME_LABELS = {0: "Bear", 1: "Sideways", 2: "Bull"}
WINDOW_SIZE = 30
NUM_FEATURES = 5
SMA_PERIOD = 20
FORWARD_PERIOD = 20


def fetch_ohlcv(ticker: str = "SPY", period: str = "10y") -> dict:
    """Fetch daily OHLCV data via yfinance.

    Returns dict with keys: open, high, low, close, volume (all 1-D numpy arrays),
    and dates (array of datetime-like objects).
    """
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    # yfinance may return MultiIndex columns for single ticker; flatten
    if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
        df.columns = df.columns.get_level_values(0)
    return {
        "open": df["Open"].values.astype(np.float64),
        "high": df["High"].values.astype(np.float64),
        "low": df["Low"].values.astype(np.float64),
        "close": df["Close"].values.astype(np.float64),
        "volume": df["Volume"].values.astype(np.float64),
        "dates": df.index.values,
    }


def compute_features(ohlcv: dict) -> np.ndarray:
    """Compute 5 dimensionless features per day from OHLCV data.

    Returns array of shape (N, 5) where N <= len(close). First SMA_PERIOD rows
    will be NaN (insufficient history for rolling stats).

    Features:
        0: Log return          ln(close[t] / close[t-1])
        1: Normalized range    (high - low) / close
        2: Volume ratio        volume / SMA(volume, 20)
        3: Price vs SMA        (close - SMA(close, 20)) / SMA(close, 20)
        4: Rolling volatility  std(log_return, 20) * sqrt(252)
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]
    n = len(close)

    features = np.full((n, NUM_FEATURES), np.nan, dtype=np.float64)

    # Feature 0: log return
    features[1:, 0] = np.log(close[1:] / close[:-1])

    # Feature 1: normalized range
    features[:, 1] = (high - low) / close

    # Feature 2: volume ratio (volume / SMA(volume, 20))
    vol_sma = _rolling_mean(volume, SMA_PERIOD)
    # Avoid division by zero for low-volume periods
    safe_vol_sma = np.where(vol_sma > 0, vol_sma, np.nan)
    features[:, 2] = volume / safe_vol_sma

    # Feature 3: price vs SMA(close, 20)
    close_sma = _rolling_mean(close, SMA_PERIOD)
    safe_close_sma = np.where(close_sma > 0, close_sma, np.nan)
    features[:, 3] = (close - safe_close_sma) / safe_close_sma

    # Feature 4: rolling volatility (annualized)
    log_ret = features[:, 0].copy()
    features[:, 4] = _rolling_std(log_ret, SMA_PERIOD) * np.sqrt(252)

    return features


def compute_labels(close: np.ndarray) -> np.ndarray:
    """Label each day's regime using forward-looking 20-day returns.

    Returns int array of shape (N,) with values 0/1/2 (Bear/Sideways/Bull).
    Last FORWARD_PERIOD days will be -1 (unlabelable).
    """
    n = len(close)
    labels = np.full(n, -1, dtype=np.int32)

    # Forward returns
    fwd_ret = np.full(n, np.nan)
    fwd_ret[: n - FORWARD_PERIOD] = (
        close[FORWARD_PERIOD:] / close[: n - FORWARD_PERIOD] - 1.0
    )

    valid = ~np.isnan(fwd_ret)
    p25 = np.percentile(fwd_ret[valid], 25)
    p75 = np.percentile(fwd_ret[valid], 75)

    labels[valid] = 1  # Sideways by default
    labels[valid & (fwd_ret < p25)] = 0  # Bear
    labels[valid & (fwd_ret > p75)] = 2  # Bull

    return labels


def build_dataset(
    ticker: str = "SPY", period: str = "10y", train_frac: float = 0.8
) -> dict:
    """Build windowed dataset for CNN training.

    Returns dict with keys:
        X_train, y_train, X_test, y_test: numpy arrays
        feature_stats: dict with mean/std per feature (from training set)
        dates_train, dates_test: date arrays for each sample's last day
    """
    ohlcv = fetch_ohlcv(ticker, period)
    features = compute_features(ohlcv)
    labels = compute_labels(ohlcv["close"])
    dates = ohlcv["dates"]

    # Build sliding windows, skip samples with any NaN in features or invalid label
    X_windows = []
    y_windows = []
    d_windows = []
    for i in range(WINDOW_SIZE, len(features)):
        window = features[i - WINDOW_SIZE : i]
        label = labels[i - 1]  # label for the last day in the window
        if np.any(np.isnan(window)) or label < 0:
            continue
        X_windows.append(window)
        y_windows.append(label)
        d_windows.append(dates[i - 1])

    X = np.array(X_windows, dtype=np.float32)  # (N, 30, 5)
    y = np.array(y_windows, dtype=np.int32)  # (N,)
    d = np.array(d_windows)

    # Chronological split
    split = int(len(X) * train_frac)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    d_train, d_test = d[:split], d[split:]

    # Normalize features using training set statistics (per-feature)
    mean = X_train.reshape(-1, NUM_FEATURES).mean(axis=0)
    std = X_train.reshape(-1, NUM_FEATURES).std(axis=0)
    std = np.where(std > 0, std, 1.0)  # avoid division by zero

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Add channel dimension for Conv2D: (N, 30, 5) -> (N, 30, 5, 1)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_stats": {"mean": mean, "std": std},
        "dates_train": d_train,
        "dates_test": d_test,
    }


def prepare_inference_input(
    ticker: str, feature_stats: dict, period: str = "3mo"
) -> tuple[np.ndarray, str]:
    """Prepare the most recent window for inference.

    Returns (input_array of shape (1, 30, 5, 1), date_string).
    """
    ohlcv = fetch_ohlcv(ticker, period)
    features = compute_features(ohlcv)
    dates = ohlcv["dates"]

    # Take the last WINDOW_SIZE rows
    if len(features) < WINDOW_SIZE:
        raise ValueError(
            f"Not enough data: got {len(features)} days, need {WINDOW_SIZE}"
        )

    window = features[-WINDOW_SIZE:]
    if np.any(np.isnan(window)):
        # Try stepping back to find a clean window
        for offset in range(1, len(features) - WINDOW_SIZE):
            window = features[-(WINDOW_SIZE + offset) : -offset]
            if not np.any(np.isnan(window)):
                break
        else:
            raise ValueError("Could not find a NaN-free window in recent data")

    date_str = str(dates[-1])[:10]

    mean = feature_stats["mean"]
    std = feature_stats["std"]
    window = (window - mean) / std

    return window.astype(np.float32).reshape(1, WINDOW_SIZE, NUM_FEATURES, 1), date_str


# --- helpers ---


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean. First (window-1) values are NaN."""
    out = np.full_like(arr, np.nan, dtype=np.float64)
    cumsum = np.cumsum(arr)
    out[window - 1 :] = (cumsum[window - 1 :] - np.concatenate([[0], cumsum[:-window]])) / window
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation. First (window-1) values are NaN."""
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(window - 1, len(arr)):
        segment = arr[i - window + 1 : i + 1]
        if np.any(np.isnan(segment)):
            continue
        out[i] = np.std(segment, ddof=1)
    return out


if __name__ == "__main__":
    print("Building dataset for SPY...")
    ds = build_dataset("SPY")
    print(f"Training samples: {len(ds['X_train'])}")
    print(f"Test samples:     {len(ds['X_test'])}")
    print(f"Input shape:      {ds['X_train'].shape}")
    print(f"Label distribution (train):")
    for cls in range(3):
        count = np.sum(ds["y_train"] == cls)
        pct = count / len(ds["y_train"]) * 100
        print(f"  {REGIME_LABELS[cls]:>8s}: {count:5d} ({pct:.1f}%)")
    print(f"Feature stats (mean): {ds['feature_stats']['mean']}")
    print(f"Feature stats (std):  {ds['feature_stats']['std']}")
