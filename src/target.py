"""Forecast target: forward realized volatility over the next h trading days.

At date t (after the close), the target is

    RV_t = (252 / h) * sum_{i=1..h} r_{t+i}^2        (annualized variance)

built from daily close-to-close log returns. Every model forecasts RV_t
using information up to and including day t only.
"""
import pandas as pd


def forward_realized_variance(ret: pd.Series, horizon: int, trading_days: int = 252) -> pd.Series:
    sq = ret.pow(2)
    # rolling(h).sum() at t+h covers r_{t+1..t+h}; shift back by h to align it with t.
    fwd = sq.rolling(horizon).sum().shift(-horizon)
    return (fwd * trading_days / horizon).rename("rv_target")
