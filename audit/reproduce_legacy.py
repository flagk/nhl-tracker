"""Re-run the legacy walk-forward backtest unchanged and persist every bet.

Purpose: reproduce the headline README numbers (55.8% / 11.69% / 1,805 bets)
with the repo's own code, and keep the per-bet records for the statistical audit.
"""
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtester_realistic import RealisticBacktester  # noqa: E402
from src.main import NHLPredictorPipeline  # noqa: E402

OUT = Path(__file__).parent / "legacy_predictions.csv"

pipeline = NHLPredictorPipeline()
history = pipeline.load_history()
bt = RealisticBacktester(pipeline.trainer)
with contextlib.redirect_stdout(io.StringIO()):  # trainer is very chatty
    res = bt.walk_forward_backtest(history, train_size=500, test_size=100)
preds = pd.DataFrame(res.pop("predictions"))
preds.to_csv(OUT, index=False)
print(res)
print("saved", len(preds), "bets ->", OUT)
