"""Phase-1 statistical re-analysis of the legacy backtest's saved bets."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
rng = np.random.default_rng(42)
B = 20_000

H = pd.read_csv("data/history/nhl_history.csv", parse_dates=["Date"])
P = pd.read_csv("audit/legacy_predictions.csv", parse_dates=["date"])
win = (P.result == "WIN").to_numpy().astype(float)
n = len(win); w = int(win.sum()); wr = w / n
out: dict = {"n_bets": n, "wins": w, "win_rate": wr}

def roi_even(x): return 2 * x.mean() - 1                      # what the legacy code reports
def roi_at(x, odds):                                          # American odds on every bet
    pay = 100 / abs(odds) if odds < 0 else odds / 100
    return x.mean() * pay - (1 - x.mean())
def breakeven(odds): return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)

print(f"bets={n} wins={w} win_rate={wr:.4f}")
print(f"legacy ROI (even money, +100/-100): {roi_even(win)*100:.2f}%")
lo, hi = stats.binomtest(w, n).proportion_ci(0.95, method="wilson")
print(f"win rate 95% Wilson CI: [{lo:.4f}, {hi:.4f}]")
out["wilson"] = [lo, hi]

# --- bootstrap: iid and cluster-by-week (bets on the same slate are not independent draws)
idx = rng.integers(0, n, (B, n))
boot_wr = win[idx].mean(1)
week = P.date.dt.to_period("W").astype(str).to_numpy()
wk = pd.Series(range(n)).groupby(week).apply(list).tolist()
def cluster_boot():
    pick = rng.integers(0, len(wk), len(wk))
    ii = np.concatenate([wk[k] for k in pick]); return win[ii].mean()
boot_wr_c = np.array([cluster_boot() for _ in range(5000)])
for name, arr in [("iid", boot_wr), ("cluster(week)", boot_wr_c)]:
    l, h = np.percentile(arr, [2.5, 97.5])
    print(f"bootstrap[{name}] win rate CI [{l:.4f},{h:.4f}]  even-money ROI CI [{(2*l-1)*100:.2f}%,{(2*h-1)*100:.2f}%]")
out["boot_iid_wr"] = np.percentile(boot_wr, [2.5, 97.5]).tolist()
out["boot_cluster_wr"] = np.percentile(boot_wr_c, [2.5, 97.5]).tolist()

# --- significance vs break-even baselines (one-sided exact binomial)
for label, p0 in [("even money (legacy assumption) p0=0.5000", 0.5),
                  ("flat -110 p0=0.5238", breakeven(-110)),
                  ("flat -120 p0=0.5455", breakeven(-120)),
                  ("flat -150 p0=0.6000", breakeven(-150))]:
    p = stats.binomtest(w, n, p0, alternative="greater").pvalue
    print(f"H0 win rate <= {label}: one-sided p = {p:.4g}")
    out.setdefault("pvalues", {})[label] = p

# --- ROI under explicit odds assumptions (NO real lines are available in the repo)
print("\nROI if every bet were priced at a flat American price (vig included):")
rows = []
for odds in [+100, -105, -110, -120, -130, -150, -175]:
    lo_, hi_ = np.percentile([roi_at(win[i], odds) for i in idx[:3000]], [2.5, 97.5])
    rows.append((odds, breakeven(odds) * 100, roi_at(win, odds) * 100, lo_ * 100, hi_ * 100))
    print(f"  {odds:+5d}: break-even {breakeven(odds)*100:5.2f}%  ROI {roi_at(win, odds)*100:+6.2f}%  95% CI [{lo_*100:+.1f}%, {hi_*100:+.1f}%]")
out["odds_grid"] = rows

# --- power: is 1,805 bets enough?
sd = np.sqrt(0.5 * 0.5 / n)
for base, lab in [(0.5, "even money"), (breakeven(-110), "-110")]:
    mde = (stats.norm.ppf(0.95) + stats.norm.ppf(0.8)) * np.sqrt(base * (1 - base) / n)
    print(f"minimum detectable win-rate edge at n={n} (one-sided a=.05, 80% power) vs {lab}: +{mde*100:.2f} pts")
    out[f"mde_{lab}"] = mde
for roi in [0.10, 0.05, 0.03, 0.02, 0.01]:
    need = ((stats.norm.ppf(0.95) + stats.norm.ppf(0.8)) / roi) ** 2   # per-bet payoff sd ~ 1 at ~even odds
    print(f"bets needed to detect a true ROI of {roi*100:.0f}% (80% power): ~{need:,.0f}")
out["bets_needed"] = {str(r): ((stats.norm.ppf(0.95) + stats.norm.ppf(0.8)) / r) ** 2 for r in [0.10, 0.05, 0.03, 0.02, 0.01]}

# --- baselines on the SAME games the backtest tested: iloc[500:2900]
T = H.iloc[500:2900].copy()
home_rate = (T.Winner == T.Home).mean()
print(f"\nAlways-home baseline on the 2,400 tested games: {home_rate:.4f}  (model on its bets: {wr:.4f})")
pr = P.assign(is_home_pick=(P.prediction == P.home))
print(f"model picks home {pr.is_home_pick.mean():.1%} of the time; home-pick hit rate {(pr[pr.is_home_pick].result=='WIN').mean():.3f}, away-pick {(pr[~pr.is_home_pick].result=='WIN').mean():.3f}")
home_on_bets = (H.set_index(["Date", "Home", "Away"]).loc[list(zip(P.date, P.home, P.away))].eval("Winner == Home")).to_numpy().astype(float)
diff = win - home_on_bets
d_lo, d_hi = np.percentile([diff[i].mean() for i in idx[:5000]], [2.5, 97.5])
print(f"on the same bets, always-home hit rate = {home_on_bets.mean():.4f}; model - always-home = {diff.mean()*100:+.2f} pts, 95% CI [{d_lo*100:+.2f},{d_hi*100:+.2f}]")
out.update(home_rate_tested=home_rate, home_on_bets=home_on_bets.mean(), model_minus_home=diff.mean(), model_minus_home_ci=[d_lo, d_hi])

# --- calibration of the stated confidence
P["hit"] = win
P["bin"] = pd.cut(P.confidence, [0.55, 0.60, 0.65, 0.70, 0.80, 1.0], right=False)
cal = P.groupby("bin", observed=True).agg(n=("hit", "size"), mean_conf=("confidence", "mean"), hit_rate=("hit", "mean"))
print("\nCalibration of stated confidence on bets placed:\n", cal.round(3).to_string())
print(f"mean stated confidence {P.confidence.mean():.3f} vs realised {wr:.3f}")
eps = 1e-6; pc = P.confidence.clip(eps, 1 - eps)
print(f"Brier(picked side) {np.mean((pc - win)**2):.4f}   logloss {(-(win*np.log(pc)+(1-win)*np.log(1-pc))).mean():.4f}   (coin flip: 0.2500 / 0.6931)")
out["calibration"] = cal.reset_index().astype({"bin": str}).to_dict("records")
out["mean_conf"] = P.confidence.mean()

# --- time stability
P["season"] = np.where(P.date < "2024-08-01", "2023-24", np.where(P.date < "2025-08-01", "2024-25", "2025-26"))
print("\nBy season:\n", P.groupby("season").hit.agg(["size", "mean"]).round(4).to_string())

Path("audit/statistics.json").write_text(json.dumps(out, indent=2, default=float))
