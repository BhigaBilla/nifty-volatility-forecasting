"""Walk-forward LSTM on log realized variance (PyTorch).

Input: the last `seq_len` days of the same features XGBoost uses, standardized with
training-window statistics. Output: log of the forward h-day realized variance.
Training rows, validation split, purge gap and smearing follow src/models/ml.py.
Each refit trains `n_seeds` networks and averages their log forecasts, because a
single small LSTM's result moves noticeably with the random seed.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.ml import VAL_DAYS, _refit_blocks


@dataclass(frozen=True)
class LSTMConfig:
    seq_len: int = 22
    hidden: int = 32
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 60
    patience: int = 8
    n_seeds: int = 3


class VolLSTM(nn.Module):
    def __init__(self, n_features: int, cfg: LSTMConfig):
        super().__init__()
        self.lstm = nn.LSTM(n_features, cfg.hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def _windows(Z: np.ndarray, ends: np.ndarray, L: int) -> np.ndarray:
    """Stack Z[e-L+1 : e+1] for each end position e -> (n, L, features)."""
    return np.stack([Z[e - L + 1: e + 1] for e in ends]).astype(np.float32)


def _train_one(Xtr, ytr, Xva, yva, n_features, cfg: LSTMConfig, seed: int) -> VolLSTM:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = VolLSTM(n_features, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    best, best_state, bad = np.inf, None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(len(Xtr)), max(1, len(Xtr) // cfg.batch_size)):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = nn.functional.mse_loss(model(Xva_t), yva_t).item()
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _predict(models, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return np.mean([m(torch.from_numpy(X)).numpy() for m in models], axis=0)


def lstm_walk_forward(
    X: pd.DataFrame, y_var: pd.Series, test_start: str, horizon: int,
    refit_every: int, cfg: LSTMConfig = LSTMConfig(),
) -> tuple[pd.Series, list[dict]]:
    """Return (annualized variance forecasts, per-refit training log)."""
    torch.set_num_threads(max(1, torch.get_num_threads()))
    L = cfg.seq_len
    Z_raw = X.to_numpy(dtype=float)
    log_y = np.log(y_var.to_numpy())
    feat_ok = ~np.isnan(Z_raw).any(axis=1)
    # A window ending at e is usable only if all L rows have features.
    win_ok = pd.Series(feat_ok).rolling(L).min().fillna(0).to_numpy().astype(bool)

    out, log = [], []
    for start, end in _refit_blocks(X.index, test_start, refit_every):
        last_train = start - horizon  # target at t needs returns through t + h <= start
        tr_ends = np.flatnonzero(win_ok[: last_train + 1] & ~np.isnan(log_y[: last_train + 1]))
        mu = np.nanmean(Z_raw[tr_ends], axis=0)
        sd = np.nanstd(Z_raw[tr_ends], axis=0) + 1e-8
        Z = np.nan_to_num((Z_raw - mu) / sd)

        cut = len(tr_ends) - VAL_DAYS
        fit_ends, val_ends = tr_ends[: cut - horizon], tr_ends[cut:]
        Xf, yf = _windows(Z, fit_ends, L), log_y[fit_ends].astype(np.float32)
        Xv, yv = _windows(Z, val_ends, L), log_y[val_ends].astype(np.float32)
        models = [_train_one(Xf, yf, Xv, yv, Z.shape[1], cfg, seed) for seed in range(cfg.n_seeds)]

        val_pred = _predict(models, Xv)
        smear = float(np.mean(np.exp(yv - val_pred)))
        pred_ends = np.arange(start, end)[win_ok[start:end]]
        pred = _predict(models, _windows(Z, pred_ends, L))
        out.append(pd.Series(np.exp(pred) * smear, index=X.index[pred_ends]))
        log.append({"refit_date": X.index[start], "train_rows": len(fit_ends),
                    "val_mse": float(np.mean((yv - val_pred) ** 2)), "smear": smear})
    return pd.concat(out), log
