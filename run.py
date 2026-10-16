"""One-command pipeline: python run.py

Download data -> realized-vol target -> walk-forward forecasts from every model
-> metrics, Diebold-Mariano tests, plot.
"""
import argparse
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import Config
from src.data import describe, load_prices
from src.explain import summarize, xgb_shap
from src.evaluate import diebold_mariano, metrics_table, qlike, sq_err_vol
from src.features import HAR_COLS, HAR_VIX_COLS, build_features
from src.models.baselines import ewma_vol, historical_vol
from src.models.garch import garch_walk_forward
from src.models.lstm import lstm_walk_forward
from src.models.ml import walk_forward
from src.target import forward_realized_variance

DM_PAIRS = [("garch", "hist"), ("garch", "ewma"), ("har", "garch"), ("har_vix", "har"),
            ("xgb", "garch"), ("xgb", "har_vix"), ("lstm", "garch"), ("lstm", "har_vix"), ("lstm", "xgb")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download data instead of using the cache")
    ap.add_argument("--horizon", type=int, default=Config.horizon)
    ap.add_argument("--test-start", default=Config.test_start)
    ap.add_argument("--garch-dist", default=Config.garch_dist, choices=["normal", "t"])
    args = ap.parse_args()
    cfg = replace(Config(), horizon=args.horizon, test_start=args.test_start, garch_dist=args.garch_dist)
    out = Path(cfg.results_dir)
    out.mkdir(exist_ok=True)

    print("== Data ==")
    df = load_prices(cfg, refresh=args.refresh)
    print(describe(df))
    y = forward_realized_variance(df["ret"], cfg.horizon, cfg.trading_days)
    X = build_features(df, cfg.trading_days)

    print(f"\n== GARCH(1,1) walk-forward (refit every {cfg.refit_every} days, dist={cfg.garch_dist}) ==")
    t0 = time.time()
    garch, params = garch_walk_forward(df["ret"], cfg.test_start, cfg.horizon,
                                       cfg.refit_every, cfg.garch_dist, cfg.trading_days)
    print(f"{len(params)} refits in {time.time() - t0:.1f}s; non-converged: {(~params['converged']).sum()}")
    print(params.drop(columns="converged").describe().loc[["min", "50%", "max"]].round(4).to_string())

    print(f"\n== HAR / XGBoost walk-forward (retrain every {cfg.ml_refit_every} days) ==")
    wf = dict(test_start=cfg.test_start, horizon=cfg.horizon, refit_every=cfg.ml_refit_every)
    har, har_fits = walk_forward(X, y, "ols", HAR_COLS, **wf)
    har_vix, har_vix_fits = walk_forward(X, y, "ols", HAR_VIX_COLS, **wf)
    t0 = time.time()
    xgb, xgb_fits = walk_forward(X, y, "xgb", list(X.columns), **wf)
    print(f"{len(xgb_fits)} XGBoost fits in {time.time() - t0:.1f}s; "
          f"trees used (median): {int(pd.Series([m.best_iteration + 1 for m in xgb_fits]).median())}")
    print("HAR-VIX coefficients on log variance (first / last refit):")
    print(pd.concat([har_vix_fits[0], har_vix_fits[-1]], axis=1).round(3).to_string())
    gain = pd.DataFrame([m.get_booster().get_score(importance_type="gain") for m in xgb_fits]).fillna(0)
    gain.columns = [X.columns[int(c[1:])] if c.startswith("f") and c[1:].isdigit() else c for c in gain.columns]
    print("XGBoost feature importance (gain share, averaged over refits):")
    print((gain.div(gain.sum(axis=1), axis=0).mean().sort_values(ascending=False)).round(3).to_string())

    print(f"\n== LSTM walk-forward (retrain every {cfg.lstm_refit_every} days, 3-seed average) ==")
    t0 = time.time()
    lstm, lstm_log = lstm_walk_forward(X, y, cfg.test_start, cfg.horizon, cfg.lstm_refit_every)
    lstm_log = pd.DataFrame(lstm_log).set_index("refit_date")
    print(f"{len(lstm_log)} refits in {time.time() - t0:.0f}s; validation MSE on log variance "
          f"min {lstm_log['val_mse'].min():.3f} / median {lstm_log['val_mse'].median():.3f} "
          f"/ max {lstm_log['val_mse'].max():.3f}")

    fc = pd.concat([
        historical_vol(df["ret"], cfg.rolling_window, cfg.trading_days),
        ewma_vol(df["ret"], cfg.ewma_lambda, cfg.trading_days),
        garch, har.rename("har"), har_vix.rename("har_vix"), xgb.rename("xgb"),
        lstm.rename("lstm"),
    ], axis=1)
    test = pd.concat([y, fc], axis=1).loc[cfg.test_start:].dropna()
    y_t, fc_t = test["rv_target"], test.drop(columns="rv_target")

    print(f"\n== Test set: {len(test)} days, {test.index.min().date()} to {test.index.max().date()}, "
          f"horizon {cfg.horizon} days ==")
    m = metrics_table(y_t, fc_t)
    print(m.round(4).to_string())

    print("\n== Diebold-Mariano (negative stat = first model better) ==")
    dm_rows = []
    for a, b in DM_PAIRS:
        for lname, lfun in [("QLIKE", qlike), ("SE_vol", sq_err_vol)]:
            stat, p = diebold_mariano(lfun(y_t, fc_t[a]), lfun(y_t, fc_t[b]), cfg.horizon)
            dm_rows.append({"model": a, "vs": b, "loss": lname, "DM_stat": stat, "p_value": p})
    dm = pd.DataFrame(dm_rows)
    print(dm.round(4).to_string(index=False))

    print("\n== SHAP for XGBoost (test period, log-variance units) ==")
    shap = xgb_shap(X, xgb_fits, cfg.test_start, cfg.ml_refit_every)
    rank = summarize(shap, X, out)
    print(rank.round(3).to_string())
    crash = shap.loc["2020-03-09":"2020-03-31"].drop(columns="base").mean().sort_values()
    print("Mean SHAP during the March 2020 crash (2020-03-09 to 2020-03-31), largest 4:")
    print(crash.iloc[::-1].head(4).round(3).to_string())
    shap.to_csv(out / "shap_xgb.csv")

    test.to_csv(out / "forecasts.csv")
    m.to_csv(out / "metrics.csv")
    dm.to_csv(out / "dm_tests.csv", index=False)
    params.to_csv(out / "garch_params.csv")
    lstm_log.to_csv(out / "lstm_log.csv")

    fig, ax = plt.subplots(figsize=(12, 4))
    (100 * y_t ** 0.5).plot(ax=ax, lw=0.6, color="0.6", label=f"realized ({cfg.horizon}d fwd)")
    for name in ["garch", "har_vix", "xgb", "lstm"]:
        (100 * fc_t[name] ** 0.5).plot(ax=ax, lw=0.9, alpha=0.8, label=name)
    ax.set_ylabel("annualized vol, %")
    ax.set_title("Nifty 50: forecast vs realized volatility (test period)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "forecast_vs_realized.png", dpi=120)
    print(f"\nSaved forecasts, metrics, DM tests and plot to {out}/")


if __name__ == "__main__":
    main()
