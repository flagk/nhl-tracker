"""Phase 7: run the pandera schemas against the REAL database before they are allowed to halt production. Writes audit/results/p7_schema_probe.json; never raises."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nhlbet.data.store import Store  # noqa: E402
from nhlbet.ops.schemas import TABLES, validate_frame  # noqa: E402

store = Store(str(ROOT / "data/nhl.db"))
out = {}
for name, (schema, sql) in TABLES.items():
    df = store.df(sql)
    out[name] = {"rows": int(len(df)), "failures": validate_frame(schema, df)}
(ROOT / "audit/results").mkdir(parents=True, exist_ok=True)
(ROOT / "audit/results/p7_schema_probe.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
