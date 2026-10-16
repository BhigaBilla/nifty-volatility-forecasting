"""Download and cache Nifty 50 and India VIX daily data from Yahoo Finance."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config


def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    # Recent yfinance versions return (field, ticker) MultiIndex columns even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df[["Open", "High", "Low", "Close"]].astype(float)


def load_prices(cfg: Config, refresh: bool = False) -> pd.DataFrame:
    """Return one aligned frame: Nifty OHLC, VIX close, and Nifty daily log return.

    The first download is cached to data/raw.csv so later runs are reproducible
    and offline. Pass refresh=True to re-download.
    """
    cache = Path(cfg.data_dir) / "raw.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col="date", parse_dates=True)

    nifty = _download(cfg.nifty_ticker, cfg.start, cfg.end)
    vix = _download(cfg.vix_ticker, cfg.start, cfg.end)[["Close"]].rename(columns={"Close": "vix"})

    df = nifty.join(vix, how="inner")
    # Drop days Yahoo reports with a zero or missing close, and stale duplicate rows.
    df = df[(df["Close"] > 0) & df["vix"].notna()]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df["ret"] = np.log(df["Close"]).diff()
    df = df.dropna(subset=["ret"])

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def describe(df: pd.DataFrame) -> str:
    r = df["ret"]
    lines = [
        f"rows: {len(df)}  from {df.index.min().date()} to {df.index.max().date()}",
        f"daily log return: mean {r.mean():.5f}  std {r.std():.5f}  "
        f"min {r.min():.4f}  max {r.max():.4f}",
        f"zero-return days: {(r == 0).sum()}  |abs return| > 10%: {(r.abs() > 0.10).sum()}",
        f"India VIX: min {df['vix'].min():.2f}  median {df['vix'].median():.2f}  max {df['vix'].max():.2f}",
    ]
    return "\n".join(lines)
