"""Sanity checks that run offline on simulated data."""
import numpy as np
import pandas as pd

from src.models.garch import garch_walk_forward
from src.target import forward_realized_variance


def _sim(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    w, a, b = 2e-6, 0.08, 0.9
    s2, r = w / (1 - a - b), np.empty(n)
    for i in range(n):
        r[i] = np.sqrt(s2) * rng.standard_normal()
        s2 = w + a * r[i] ** 2 + b * s2
    return pd.Series(r, index=pd.bdate_range("2010-01-01", periods=n))


def test_target_alignment():
    r = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0], index=pd.bdate_range("2020-01-01", periods=5))
    y = forward_realized_variance(r, horizon=2, trading_days=2)
    assert y.iloc[0] == 1 + 4      # uses r_1, r_2, not r_0
    assert np.isnan(y.iloc[-1])


def test_garch_forecasts_ignore_future():
    r = _sim()
    test_start = r.index[800]
    fc, _ = garch_walk_forward(r, str(test_start.date()), horizon=5, refit_every=50)
    r2 = r.copy()
    r2.iloc[1000:] *= 5            # change the future after day 999
    fc2, _ = garch_walk_forward(r2, str(test_start.date()), horizon=5, refit_every=50)
    np.testing.assert_allclose(fc.loc[: r.index[999]], fc2.loc[: r.index[999]])
    assert not np.allclose(fc.loc[r.index[1000]:], fc2.loc[r.index[1000]:])


def test_ml_forecasts_ignore_future():
    from src.features import HAR_COLS, build_features
    from src.models.ml import walk_forward

    r = _sim(1500)
    c = 100 * np.exp(r.cumsum())
    df = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                       "vix": 20 + 100 * r.rolling(22).std().bfill(), "ret": r})

    def run(d):
        y = forward_realized_variance(d["ret"], 5)
        X = build_features(d)
        return walk_forward(X, y, "ols", HAR_COLS, str(r.index[800].date()), 5, 63)[0]

    d2 = df.copy()
    d2.iloc[1000:, d2.columns.get_loc("ret")] *= 5
    a, b = run(df), run(d2)
    np.testing.assert_allclose(a.loc[: r.index[999]], b.loc[: r.index[999]])


def test_lstm_forecasts_ignore_future():
    from src.features import build_features
    from src.models.lstm import LSTMConfig, lstm_walk_forward

    r = _sim(1500)
    c = 100 * np.exp(r.cumsum())
    df = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                       "vix": 20 + 100 * r.rolling(22).std().bfill(), "ret": r})
    cfg = LSTMConfig(hidden=8, max_epochs=3, n_seeds=1)

    def run(d):
        y = forward_realized_variance(d["ret"], 5)
        return lstm_walk_forward(build_features(d), y, str(r.index[900].date()), 5, 126, cfg)[0]

    d2 = df.copy()
    d2.iloc[1000:, d2.columns.get_loc("ret")] *= 5
    a, b = run(df), run(d2)
    np.testing.assert_allclose(a.loc[: r.index[999]], b.loc[: r.index[999]], rtol=1e-5)
