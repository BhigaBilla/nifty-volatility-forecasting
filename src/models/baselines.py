"""Model-free baselines. GARCH has to beat these to be worth anything.

Both return annualized variance forecasts for the next h days, using returns up to t.
"""
import pandas as pd


def historical_vol(ret: pd.Series, window: int, trading_days: int = 252) -> pd.Series:
    """Mean of the last `window` squared returns, held flat over the horizon."""
    return (ret.pow(2).rolling(window).mean() * trading_days).rename("hist")


def ewma_vol(ret: pd.Series, lam: float = 0.94, trading_days: int = 252) -> pd.Series:
    """RiskMetrics EWMA: s2_t = lam * s2_{t-1} + (1 - lam) * r_t^2, held flat over the horizon."""
    s2 = ret.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return (s2 * trading_days).rename("ewma")
