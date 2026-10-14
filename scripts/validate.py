"""Re-run the honest evaluation from the committed files only:  python scripts/validate.py  -> reports/VALIDATION.json (+ a readable summary on stdout)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlbet.report.validation import run_validation  # noqa: E402

res = run_validation(".")
Path("reports/VALIDATION.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float)[:6000])
