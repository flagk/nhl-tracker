"""Phase-1 probes of the legacy feature pipeline (no model training needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.feature_engineer import FeatureEngineer  # noqa: E402

H = pd.read_csv("data/history/nhl_history.csv", parse_dates=["Date"])
fe = FeatureEngineer()
rng = np.random.default_rng(0)

print("== 1. Dataset shape / coverage ==")
teams = sorted(set(H.Home) | set(H.Away))
print(f"games={len(H)} teams={len(teams)} range={H.Date.min().date()}..{H.Date.max().date()}")
expected = {"ANA","BOS","BUF","CGY","CAR","CHI","CBJ","COL","DAL","DET","EDM","FLA","LAK","MIN",
            "MTL","NSH","NJD","NYI","NYR","OTT","PHI","PIT","SEA","SJS","STL","TBL","TOR","UTA",
            "VAN","VGK","WPG","WSH"}
print("missing teams vs current 32:", sorted(expected - set(teams)))
print("home win rate (whole file):", round((H.Winner == H.Home).mean(), 4))
seasons = pd.cut(H.Date, pd.to_datetime(["2023-08-01","2024-08-01","2025-08-01","2026-08-01"]),
                 labels=["2023-24","2024-25","2025-26"])
print(H.groupby(seasons, observed=True).size().to_dict(), "(full regular seasons are ~1,312 games each)")

print("\n== 2. Are Points/GoalDiff/GoalsFor/GoalsAgainst home-perspective? ==")
print("GoalsFor==HomeScore:", (H.GoalsFor == H.HomeScore).mean(),
      "| GoalsAgainst==AwayScore:", (H.GoalsAgainst == H.AwayScore).mean(),
      "| Points==2 iff home won:", ((H.Points == 2) == (H.Winner == H.Home)).mean())

print("\n== 3. Consequence: legacy team features on a team's AWAY games use the opponent's numbers ==")
cutoff = "2025-01-01"
rows = []
for t in teams:
    g = H[((H.Home == t) | (H.Away == t)) & (H.Date < cutoff)]
    f = fe.calculate_team_features(t, H, cutoff)
    gf = np.where(g.Home == t, g.HomeScore, g.AwayScore); ga = np.where(g.Home == t, g.AwayScore, g.HomeScore)
    pts = np.where(g.Winner == t, 2, 0)
    rows.append(dict(ppg=(f["ppg"], pts.mean()), goal_diff_pg=(f["goal_diff_pg"], (gf - ga).mean()),
                     gf_pg=(f["gf_pg"], gf.mean()), ga_pg=(f["ga_pg"], ga.mean()), win_pct=(f["win_pct"], (g.Winner == t).mean())))
for k in rows[0]:
    leg = np.array([r[k][0] for r in rows]); cor = np.array([r[k][1] for r in rows])
    print(f"  {k:13s} corr(legacy, correct) across {len(rows)} teams = {np.corrcoef(leg, cor)[0,1]:+.2f}   "
          f"mean|diff|={np.abs(leg-cor).mean():.3f}")

print("\n== 4. Win-streak function ==")
ok = 0; n = 0
for _ in range(300):
    i = int(rng.integers(300, len(H)))
    team = H.Home[i]
    hist = H[H.Date < H.Date[i]]
    tg = hist[(hist.Home == team) | (hist.Away == team)].sort_values("Date")
    s = 0
    for _, r in tg.iloc[::-1].iterrows():
        w = r.Winner == team
        if s == 0: s = 1 if w else -1
        elif (s > 0 and w): s += 1
        elif (s < 0 and not w): s -= 1
        else: break
    n += 1; ok += (fe._calculate_win_streak(tg) == s)
print(f"legacy streak == correct streak in {ok}/{n} sampled team-dates")

print("\n== 5. As-of invariance: corrupt every game on/after the target date; features must not move ==")
bad = 0
for _ in range(200):
    i = int(rng.integers(200, len(H)))
    r = H.iloc[i]
    a = fe.create_training_features(r.Home, r.Away, H[:i], r.Date.strftime("%Y-%m-%d"))
    H2 = H.copy()
    m = H2.Date >= r.Date
    H2.loc[m, ["Points","GoalDiff","GoalsFor","GoalsAgainst"]] = 99
    H2.loc[m, "Winner"] = "ZZZ"
    b = fe.create_training_features(r.Home, r.Away, H2, r.Date.strftime("%Y-%m-%d"))
    bad += not np.allclose(a, b)
print(f"features changed by future corruption: {bad}/200  (0 => date filter is leak-free)")

print("\n== 6. Same-day games / index slicing ==")
print("games sharing a date with an earlier game in file:", int(H.Date.duplicated().sum()),
      "-> same-day results excluded by strict '<' (good, conservative)")
print("history[:idx] equals positional slice when index is RangeIndex:", H.index.equals(pd.RangeIndex(len(H))))

print("\n== 7. Season handling ==")
print("feature windows reset each season? No: 'ppg','gf_pg','win_pct' etc. are career-to-date over",
      "all rows before the game (up to ~2.3 seasons).")
