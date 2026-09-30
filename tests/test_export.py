import pandas as pd

from nhlbet.data.store import Store
from nhlbet.report.export import (DAILY_COLS, GAMES_COLS, MODEL_COLS, OOS_COLS, PAPER_COLS, PAPER_SUMMARY_COLS, export_dataset)
from tests.test_report_pipeline import seeded_store

EXPECT = {"games_predictions": GAMES_COLS, "daily_summary": DAILY_COLS, "paper_trading": PAPER_COLS, "paper_summary": PAPER_SUMMARY_COLS,
          "model_versions": MODEL_COLS, "oos_predictions": OOS_COLS}


def run(st, tmp_path):
    export_dataset(st, tmp_path / "x", None, tmp_path / "none.json", tmp_path / "none.csv")
    return tmp_path / "x"


def test_schemas_are_pinned_and_values_right(tmp_path):
    root = run(seeded_store(), tmp_path)
    for name, cols in EXPECT.items():
        assert list(pd.read_csv(root / f"{name}.csv").columns) == cols, name
    g = pd.read_csv(root / "games_predictions.csv").set_index("game_id")
    assert len(g) == 4 and g.loc[1, "decision"] == "BET" and g.loc[3, "decision"] == "NO_BET"
    assert g.loc[1, "profit"] == 10 and g.loc[2, "profit"] == -10 and g.loc[4, "profit"] == 6
    d = pd.read_csv(root / "daily_summary.csv").set_index("date")
    assert d.loc["2024-01-05", "profit"] == 0 and d.loc["2024-01-06", "cum_profit"] == 6


def test_no_bookmaker_names_and_deterministic(tmp_path):
    a = run(seeded_store(), tmp_path)
    first = {p.name: p.read_bytes() for p in a.iterdir()}
    assert not any(b"bookB" in v for v in first.values())
    b = run(seeded_store(), tmp_path)
    assert first == {p.name: p.read_bytes() for p in b.iterdir()}


def test_empty_store_still_writes_headers(tmp_path):
    root = run(Store(":memory:"), tmp_path)
    for name, cols in EXPECT.items():
        assert list(pd.read_csv(root / f"{name}.csv").columns) == cols
