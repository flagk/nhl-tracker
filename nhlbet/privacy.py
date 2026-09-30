"""Public-safe mode: never commit or publish per-bookmaker quotes or bookmaker names.

The Odds API's terms restrict redistributing its data, and this repository may be public. So by default everything that is committed
or published is *derived* (no-vig market probabilities, edges, stakes) and carries no bookmaker prices per book or book names.
Full detail is shown only when the repository is known to be private: the workflows detect that from the GitHub API and set
``PUBLIC_SAFE=0``. If detection fails or the variable is unset, the safe behaviour applies.
"""
from __future__ import annotations

import os


def is_public_safe() -> bool:
    return os.environ.get("PUBLIC_SAFE", "1").strip().lower() not in ("0", "false", "no")
