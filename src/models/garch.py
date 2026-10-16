"""GARCH(1,1) with walk-forward (expanding window) estimation.

Parameters are re-estimated every `refit_every` days on all data up to that day.
Between refits the parameters are frozen, but the conditional variance still
updates daily with each new return, so every forecast at day t uses data up to t only.
"""
import warnings

import pandas as pd
from arch import arch_model


def _model(r: pd.Series, dist: str):
    # Returns in percent: the optimizer is much better behaved than on raw decimals.
    return arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist=dist, rescale=False)


def garch_walk_forward(
    ret: pd.Series,
    test_start: str,
    horizon: int,
    refit_every: int = 21,
    dist: str = "normal",
    trading_days: int = 252,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return (annualized h-day variance forecasts indexed by forecast date, parameter log)."""
    r = ret * 100
    first = r.index.searchsorted(pd.Timestamp(test_start))
    if first < 500:
        raise ValueError("need at least ~500 training days before test_start")

    out, params_log = [], []
    for start in range(first, len(r), refit_every):
        end = min(start + refit_every, len(r)) - 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter; checked via params_log
            res = _model(r.iloc[: start + 1], dist).fit(disp="off")
        p = res.params
        params_log.append({
            "refit_date": r.index[start], "converged": res.convergence_flag == 0,
            **p.to_dict(), "persistence": p["alpha[1]"] + p["beta[1]"],
        })
        # Same sample start as the fit, so the variance backcast is identical; the recursion
        # at day t only touches returns up to t, so extending the sample to `end` leaks nothing.
        fixed = _model(r.iloc[: end + 1], dist).fix(p)
        f = fixed.forecast(horizon=horizon, start=r.index[start], reindex=False)
        out.append(f.variance.sum(axis=1))

    var_h = pd.concat(out)  # percent^2 summed over h days
    fc = (var_h / 1e4 * trading_days / horizon).rename("garch")
    return fc, pd.DataFrame(params_log).set_index("refit_date")
