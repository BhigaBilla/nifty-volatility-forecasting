"""SHAP explanations for the walk-forward XGBoost model.

Uses XGBoost's built-in exact TreeSHAP (predict with pred_contribs=True), so no extra
package is needed. Each test day is explained by the model that actually produced its
forecast. Values are in log-variance units: +0.1 raises that day's variance forecast
by about 10%.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.ml import _refit_blocks


def xgb_shap(X: pd.DataFrame, fits: list, test_start: str, refit_every: int) -> pd.DataFrame:
    """SHAP values (one column per feature, plus 'base') for every test-period forecast."""
    valid = X.notna().all(axis=1).to_numpy()
    out = []
    for (start, end), model in zip(_refit_blocks(X.index, test_start, refit_every), fits):
        idx = X.index[start:end][valid[start:end]]
        contribs = model.get_booster().predict(
            xgb.DMatrix(X.loc[idx].to_numpy()), pred_contribs=True,
            iteration_range=(0, model.best_iteration + 1))
        # SHAP values must add up to the model's own prediction (log variance, before smearing).
        assert np.allclose(contribs.sum(axis=1), model.predict(X.loc[idx].to_numpy()), atol=1e-3)
        out.append(pd.DataFrame(contribs, index=idx, columns=list(X.columns) + ["base"]))
    return pd.concat(out)


def summarize(shap: pd.DataFrame, X: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Print/save the mean |SHAP| ranking and two plots; return the ranking."""
    s = shap.drop(columns="base")
    rank = s.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").to_frame()
    rank["share"] = rank["mean_abs_shap"] / rank["mean_abs_shap"].sum()
    # Direction: correlation between the feature's value and its SHAP contribution.
    rank["corr_value_vs_shap"] = [np.corrcoef(X.loc[s.index, c], s[c])[0, 1] for c in rank.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    rank["mean_abs_shap"][::-1].plot.barh(ax=axes[0], color="#4C72B0")
    axes[0].set_title("XGBoost: mean |SHAP| over test period")
    axes[0].set_xlabel("mean |contribution to log variance|")
    top = rank.index[:4]
    for c in top:
        axes[1].scatter(X.loc[s.index, c].rank(pct=True), s[c], s=3, alpha=0.4, label=c)
    axes[1].axhline(0, color="0.5", lw=0.6)
    axes[1].set_xlabel("feature value (percentile within test period)")
    axes[1].set_ylabel("SHAP (log variance)")
    axes[1].set_title("How the top 4 features push the forecast")
    axes[1].legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(out / "shap_summary.png", dpi=120)

    fig, ax = plt.subplots(figsize=(12, 4))
    s[list(top)].rolling(5).mean().plot(ax=ax, lw=0.8)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_ylabel("SHAP (log variance), 5-day mean")
    ax.set_title("Contribution of top features over time")
    fig.tight_layout()
    fig.savefig(out / "shap_over_time.png", dpi=120)
    plt.close("all")
    return rank
