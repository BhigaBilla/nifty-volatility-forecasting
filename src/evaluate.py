"""Loss functions and the Diebold-Mariano test."""
import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-12


def qlike(y_var: pd.Series, f_var: pd.Series) -> pd.Series:
    """Patton (2011) QLIKE on variances: y/f - log(y/f) - 1. Zero at a perfect forecast.

    Robust to noise in the realized-variance proxy, unlike MSE on variance."""
    ratio = (y_var + EPS) / (f_var + EPS)
    return ratio - np.log(ratio) - 1


def sq_err_vol(y_var: pd.Series, f_var: pd.Series) -> pd.Series:
    """Squared error on annualized volatility in percentage points."""
    return (100 * np.sqrt(y_var) - 100 * np.sqrt(f_var)) ** 2


def metrics_table(y_var: pd.Series, forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name in forecasts:
        rows[name] = {
            "RMSE_vol_pct": np.sqrt(sq_err_vol(y_var, forecasts[name]).mean()),
            "QLIKE": qlike(y_var, forecasts[name]).mean(),
            "bias_vol_pct": (100 * np.sqrt(forecasts[name]) - 100 * np.sqrt(y_var)).mean(),
        }
    return pd.DataFrame(rows).T.sort_values("QLIKE")


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, horizon: int) -> tuple[float, float]:
    """DM test with Newey-West variance (h-1 lags) and the Harvey-Leybourne-Newbold correction.

    Negative statistic means model A has lower loss. Returns (statistic, two-sided p-value).
    Overlapping h-day targets make the loss differential autocorrelated, hence the HAC variance.
    """
    d = (loss_a - loss_b).dropna().to_numpy()
    T = len(d)
    dc = d - d.mean()
    gamma = [dc @ dc / T] + [dc[k:] @ dc[:-k] / T for k in range(1, horizon)]
    lrv = gamma[0] + 2 * sum(gamma[1:])
    if lrv <= 0:
        lrv = gamma[0]
    dm = d.mean() / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return float(stat), float(p)
