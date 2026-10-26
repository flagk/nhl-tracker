"""Phase 6(a): diff a blind reimplementation (written from audit/SPEC.md only) against the production code.

    python audit/scripts/p6_diff.py --reimpl /tmp/.../reimpl.py --out audit/results/p6_diff.json

Compares (1) de-vig on random odds, (2) Elo pre-game probabilities over the real score history in data/nhl.db, (3) bet/no-bet decisions and stakes over thousands of random slates,
including edge cases (no usable odds, stale odds, ALERT, early season, unconfirmed goalie, very long shots, large disagreements, prices at or below 1).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nhlbet.data.loaders import load_tables  # noqa: E402
from nhlbet.data.store import Store  # noqa: E402
from nhlbet.features.builder import BuilderConfig, FeatureBuilder  # noqa: E402
from nhlbet.odds.edge import evaluate_game  # noqa: E402
from nhlbet.odds.math import devig  # noqa: E402
from nhlbet.risk.policy import RiskConfig, recommend_slate  # noqa: E402


def load(path):
    spec = importlib.util.spec_from_file_location("reimpl", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def diff_devig(R, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    worst = {}
    for method in ("proportional", "power", "shin"):
        w, bad = 0.0, 0
        for _ in range(n):
            k = int(rng.choice([2, 2, 2, 3]))
            fair = rng.dirichlet(np.ones(k) * rng.uniform(1.5, 8))
            margin = rng.choice([0.0, rng.uniform(0.0, 0.12)])
            odds = list(1.0 / (fair * (1 + margin)) * rng.uniform(0.97, 1.03, k))
            a, b = np.asarray(devig(odds, method)), np.asarray(R.devig(odds, method))
            d = float(np.abs(a - b).max())
            w = max(w, d); bad += d > 1e-7
        worst[method] = {"cases": n, "max_abs_diff": w, "mismatches_over_1e-7": int(bad)}
    return worst


def diff_elo(R):
    store = Store(str(ROOT / "data" / "nhl.db"))
    tables = load_tables(store)
    F = FeatureBuilder(BuilderConfig()).build(tables)
    g = tables["games"].sort_values(["game_date", "game_id"])
    g = g[g.home_score.notna()]
    games = [(int(r.season), str(pd.Timestamp(r.game_date).date()), r.home, r.away, int(r.home_score), int(r.away_score)) for r in g.itertuples()]
    ours = F.loc[g.game_id, "elo_prob"].to_numpy(float)
    theirs = np.asarray(R.elo_probs(games), float)
    d = np.abs(ours - theirs)
    return {"games": int(len(games)), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "games_differing_over_1e-9": int((d > 1e-9).sum()),
            "first_difference": (None if not (d > 1e-9).any() else {"index": int(np.argmax(d > 1e-9)), "game": games[int(np.argmax(d > 1e-9))], "ours": float(ours[int(np.argmax(d > 1e-9))]), "reimplementation": float(theirs[int(np.argmax(d > 1e-9))])})}


def scenario(rng, gid):
    p_home = float(np.clip(rng.beta(6, 6) if rng.uniform() < 0.8 else rng.uniform(0.02, 0.98), 0.02, 0.98))
    nb = int(rng.choice([0, 1, 2, 3, 5, 8], p=[0.03, 0.1, 0.2, 0.3, 0.27, 0.1]))
    mk = float(np.clip(p_home + rng.normal(0, 0.05) if rng.uniform() < 0.9 else rng.uniform(0.05, 0.95), 0.05, 0.95))
    margin = rng.uniform(0.0, 0.08)
    quotes = []
    for _ in range(nb):
        h = 1 / (mk * (1 + margin)) * rng.uniform(0.96, 1.05)
        a = 1 / ((1 - mk) * (1 + margin)) * rng.uniform(0.96, 1.05)
        if rng.uniform() < 0.03:
            h = float(rng.choice([1.0, 0.9, 1.01]))
        if rng.uniform() < 0.01:
            a = 1.0
        quotes.append((float(h), float(a)))
    ctx = {"model_status": str(rng.choice(["OK", "OK", "OK", "WARN", "ALERT"], p=[0.5, 0.2, 0.1, 0.1, 0.1])), "odds_stale": bool(rng.uniform() < 0.05),
           "goalie_confirmed": bool(rng.uniform() < 0.6), "games_played_min": (None if rng.uniform() < 0.1 else int(rng.integers(0, 40)))}
    return {"game_id": gid, "p_home": p_home, "quotes": quotes, "ctx": ctx}


def ours(slate, bankroll):
    games = []
    for g in slate:
        prices = pd.DataFrame({"book": [f"b{i}" for i in range(len(g["quotes"]))], "home": [q[0] for q in g["quotes"]], "away": [q[1] for q in g["quotes"]]})
        q = evaluate_game(g["p_home"], "H", "A", prices) if len(prices) else None
        games.append((g["game_id"], "H", "A", q, dict(g["ctx"])))
    recs = recommend_slate(games, RiskConfig(bankroll=bankroll))
    return [{"action": r.action, "side": r.side, "stake": r.stake, "odds": r.decimal} for r in recs]


def diff_decisions(R, slates=1500, seed=1):
    rng = np.random.default_rng(seed)
    tot = bets = 0
    mism = []
    kinds = {"action": 0, "side": 0, "stake": 0, "odds": 0}
    gid = 0
    for s in range(slates):
        n = int(rng.choice([1, 2, 4, 8, 12], p=[0.1, 0.2, 0.3, 0.3, 0.1]))
        bankroll = float(rng.choice([500.0, 1000.0, 2500.0, 10000.0]))
        slate = []
        for _ in range(n):
            gid += 1
            slate.append(scenario(rng, gid))
        a = ours(slate, bankroll)
        b = R.decide_slate([dict(g) for g in slate], bankroll)
        for g, x, y in zip(slate, a, b):
            tot += 1; bets += x["action"] == "BET"
            bad = []
            if x["action"] != y["action"]:
                bad.append("action")
            elif x["action"] == "BET":
                if x["side"] != y["side"]:
                    bad.append("side")
                if abs(x["stake"] - y["stake"]) > 0.005:
                    bad.append("stake")
                if abs((x["odds"] or 0) - (y["odds"] or 0)) > 1e-9:
                    bad.append("odds")
            for k in bad:
                kinds[k] += 1
            if bad and len(mism) < 8:
                mism.append({"kinds": bad, "bankroll": bankroll, "game": g, "ours": x, "reimplementation": y})
    return {"slates": slates, "games": tot, "bets_by_our_code": bets, "disagreements_by_kind": kinds, "games_with_any_disagreement": sum(kinds.values()), "examples": mism}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reimpl", required=True)
    ap.add_argument("--out", default=str(ROOT / "audit/results/p6_diff.json"))
    a = ap.parse_args()
    R = load(a.reimpl)
    res = {"devig": diff_devig(R), "elo": diff_elo(R), "decisions": diff_decisions(R)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: (v if k != "decisions" else {kk: vv for kk, vv in v.items() if kk != "examples"}) for k, v in res.items()}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
