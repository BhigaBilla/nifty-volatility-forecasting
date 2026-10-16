"""Walk-forward HAR (OLS) and XGBoost on log realized variance.

At each refit date s, training rows are dates t <= s - h, because the target at t
needs returns up to t + h and those are only known by day s. Between refits the
model is frozen and predicts each new day from that day's features.

Models predict log variance. exp() of a log forecast estimates the median, which
is biased low for variance, so forecasts are scaled by a smearing factor
(Duan 1983): the mean of exp(residual) on held-out data.
"""
import numpy as np
import pandas as pd

XGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.02, max_depth=3, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, early_stopping_rounds=100,
    random_state=0, n_jobs=-1,
)
VAL_DAYS = 504  # last ~2 years of each training window: early stopping + smearing for XGBoost


def _refit_blocks(index: pd.DatetimeIndex, test_start: str, refit_every: int):
    first = index.searchsorted(pd.Timestamp(test_start))
    for start in range(first, len(index), refit_every):
        yield start, min(start + refit_every, len(index))


def _ols(X: np.ndarray, y: np.ndarray):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return beta, np.mean(np.exp(resid))


def walk_forward(
    X: pd.DataFrame, y_var: pd.Series, model: str, cols: list[str],
    test_start: str, horizon: int, refit_every: int,
) -> tuple[pd.Series, list]:
    """Return (annualized variance forecasts, fitted models or coefficients per refit)."""
    X = X[cols]
    log_y = np.log(y_var)
    valid = X.notna().all(axis=1)
    out, fitted = [], []
    for start, end in _refit_blocks(X.index, test_start, refit_every):
        train_idx = X.index[: max(start - horizon + 1, 0)]  # t <= s - h
        tr = train_idx[valid.loc[train_idx] & log_y.loc[train_idx].notna()]
        Xtr, ytr = X.loc[tr].to_numpy(), log_y.loc[tr].to_numpy()
        pred_idx = X.index[start:end][valid.iloc[start:end].to_numpy()]
        Xp = X.loc[pred_idx].to_numpy()

        if model == "ols":
            beta, smear = _ols(Xtr, ytr)
            pred = beta[0] + Xp @ beta[1:]
            fitted.append(pd.Series(beta, index=["const"] + cols, name=X.index[start]))
        elif model == "xgb":
            from xgboost import XGBRegressor

            # Hold out the last VAL_DAYS, with an h-day gap so validation targets don't overlap training.
            cut = len(Xtr) - VAL_DAYS
            m = XGBRegressor(**XGB_PARAMS)
            m.fit(Xtr[: cut - horizon], ytr[: cut - horizon],
                  eval_set=[(Xtr[cut:], ytr[cut:])], verbose=False)
            smear = np.mean(np.exp(ytr[cut:] - m.predict(Xtr[cut:])))
            pred = m.predict(Xp)
            fitted.append(m)
        else:
            raise ValueError(model)
        out.append(pd.Series(np.exp(pred) * smear, index=pred_idx))
    return pd.concat(out), fitted
