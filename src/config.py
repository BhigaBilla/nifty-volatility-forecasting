"""Single place for every setting the pipeline uses."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    nifty_ticker: str = "^NSEI"
    vix_ticker: str = "^INDIAVIX"
    start: str = "2008-01-01"
    end: str | None = None          # None = up to latest available
    horizon: int = 5                # forecast realized vol over the next h trading days
    test_start: str = "2015-01-01"  # everything before is the initial training window
    refit_every: int = 21           # re-estimate GARCH parameters every N trading days
    garch_dist: str = "normal"      # "normal" or "t"
    ml_refit_every: int = 63        # re-train HAR / XGBoost every N trading days (quarterly)
    lstm_refit_every: int = 126     # re-train the LSTM every N trading days (half-yearly; it is slower)
    rolling_window: int = 21        # for the historical-vol baseline
    ewma_lambda: float = 0.94       # RiskMetrics
    data_dir: str = "data"
    results_dir: str = "results"
    trading_days: int = 252
