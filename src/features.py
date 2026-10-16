"""Features known at the close of day t, for the ML models.

Variances are annualized. Logs keep the heavy right tail of volatility manageable
for both OLS and trees.
"""
import numpy as np
import pandas as pd

FLOOR = 1e-4  # annualized variance floor (1% vol) so zero-return days don't give log(0)


def build_features(df: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    r, sq = df["ret"], df["ret"].pow(2) * trading_days
    vix_var = (df["vix"] / 100) ** 2
    rv5 = sq.rolling(5).mean()
    rv22 = sq.rolling(22).mean()
    f = pd.DataFrame({
        # HAR components: daily, weekly, monthly realized variance
        "log_rv1": np.log(sq + FLOOR),
        "log_rv5": np.log(rv5 + FLOOR),
        "log_rv22": np.log(rv22 + FLOOR),
        "log_rv66": np.log(sq.rolling(66).mean() + FLOOR),
        # Implied volatility: India VIX is the market's 30-day forecast
        "log_vix_var": np.log(vix_var),
        "vix_chg5": np.log(df["vix"]).diff(5),
        "vrp": np.log(vix_var) - np.log(rv22 + FLOOR),  # implied minus realized
        # Leverage effect: falls raise future vol more than rises
        "ret1": r,
        "ret5": r.rolling(5).sum(),
        "neg_ret5": r.clip(upper=0).rolling(5).sum(),
        "range_gk": np.log(_garman_klass(df).rolling(5).mean() * trading_days + FLOOR),
    }, index=df.index)
    return f


def _garman_klass(df: pd.DataFrame) -> pd.Series:
    """Daily Garman-Klass variance from OHLC (uses the intraday range, less noisy than r^2)."""
    hl = np.log(df["High"] / df["Low"])
    co = np.log(df["Close"] / df["Open"])
    return (0.5 * hl**2 - (2 * np.log(2) - 1) * co**2).clip(lower=0)


HAR_COLS = ["log_rv1", "log_rv5", "log_rv22"]
HAR_VIX_COLS = HAR_COLS + ["log_vix_var"]
