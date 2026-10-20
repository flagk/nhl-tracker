"""The weekly digest: one short message with the system's status, last week's activity, closing-line value, credits and upcoming expiries."""
from __future__ import annotations

from datetime import datetime

from nhlbet.ops.state import open_halts


def build_digest(state: dict, evaluation: dict | None, betting: dict | None, credits_remaining: int | None, expiries: list[str], now: datetime, days_active: int | None = None) -> tuple[str, str]:
    lines = [f"System: {state.get('status', 'UNKNOWN')}" + (" (" + "; ".join(f"{h['check']}: {h['reason'][:80]}" for h in open_halts(state)) + ")" if open_halts(state) else ""),
             f"Betting status: {(betting or {}).get('status', 'RESEARCH ONLY')}" + (" - " + (betting or {}).get("why", "") if (betting or {}).get("status") != "ELIGIBLE" and (betting or {}).get("why") else "")]
    if evaluation:
        h2h = next((r for r in (evaluation.get("markets", {}).get("model") or []) if r.get("market") == "h2h"), None)
        if h2h:
            lines.append(f"Moneyline model picks: {h2h['picks']} settled, ROI {h2h['roi'] * 100:+.1f}%, CLV {h2h['clv_pts']:+.2f} pts over {h2h['clv_n']} ({h2h['verdict_clv']})")
        lines.append(f"Settled model picks, all markets: {evaluation.get('settled_model_picks')}; data through {evaluation.get('data_through')}")
    lines.append(f"Odds credits left: {credits_remaining if credits_remaining is not None else 'unknown'}")
    lines += [f"Upcoming: {e}" for e in expiries] or ["Upcoming expiries: none within 30 days"]
    lines.append("Research and education only; no edge has been demonstrated unless the lines above say so.")
    return f"NHL model weekly digest {now:%Y-%m-%d}", "\n".join(lines)
