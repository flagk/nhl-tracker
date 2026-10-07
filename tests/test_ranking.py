"""The pick ranking: one ordered list across markets, confidence-weighted so experimental markets cannot jump the queue on a big edge alone."""
import pytest

from nhlbet.odds.edge import SideQuote, expected_value
from nhlbet.odds.markets import MarketQuote
from nhlbet.odds.props import PropQuote
from nhlbet.report.ranking import WEIGHTS, rank_picks
from nhlbet.risk.policy import RiskConfig


class Rec:
    action, side = "NO_BET", None


class S:
    def __init__(self, gid, quotes=None, alt=None, props=None, stale=False, status="OK"):
        self.game_id, self.home, self.away, self.start_utc = gid, f"H{gid}", f"A{gid}", None
        self.quotes, self.alt, self.props, self.odds_stale, self.rec = quotes, alt or [], props or [], stale, Rec()
        self.ctx = {"model_status": status, "odds_stale": stale, "goalie_confirmed": True, "games_played_min": 40.0}


def ml(ph, mh, dh, da):
    mk = lambda side, team, p, m, d: SideQuote(side, team, p, m, "bk", d, 1 / d, p - m, expected_value(p, d), 5)
    return {"home": mk("home", "H", ph, mh, dh), "away": mk("away", "A", 1 - ph, 1 - mh, da)}


def mq(market, side, point, pm, pk, dec=1.91):
    return MarketQuote(market, side, f"{side} {point}", point, pm, pk, 0.0, "bk", dec, pm - pk, pm * (dec - 1) - (1 - pm), 4)


def pq(pid, name, p_over, pm=0.5, over=1.91, under=1.91, stat="sog", one_sided=False, n_prev=30):
    return PropQuote(1, pid, name, "H1", "A1", 0.5 if one_sided else 2.5, 2.8, p_over, pm, over, 0.0 if one_sided else under, p_over - pm, p_over * (over - 1) - (1 - p_over),
                     -1.0 if one_sided else (1 - p_over) * (under - 1) - p_over, 3, n_prev, stat=stat, one_sided=one_sided)


def test_ranking_orders_by_confidence_weighted_ev_and_numbers_from_one():
    cfg = RiskConfig(bankroll=1000)
    g = S(1, ml(0.55, 0.52, 1.95, 1.95), [mq("totals", "over", 6.5, 0.50, 0.50), mq("totals", "under", 6.5, 0.50, 0.50)], [pq(7, "Big Edge", 0.75)])
    rows = rank_picks([g], cfg)
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1)) and [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)
    prop = next(r for r in rows if r["kind"] == "player_sog")
    # a 25-point edge at 1.91 is +43% raw EV, but the experimental weight keeps the score far lower
    w = WEIGHTS["player_sog"]
    ev_raw, ev_mkt = 0.75 * 0.91 - 0.25, 0.5 * 0.91 - 0.5
    assert prop["ev_raw"] == pytest.approx(ev_raw) and prop["score"] == pytest.approx(w * ev_raw + (1 - w) * ev_mkt) and prop["experimental"]
    assert prop["pick"] == "Big Edge Over 2.5 shots"
    assert {r["kind"] for r in rows} == {"moneyline", "totals", "player_sog"}


def test_moneyline_is_fully_trusted_and_flags_recommendation():
    cfg = RiskConfig(bankroll=1000)
    g = S(1, ml(0.60, 0.52, 1.95, 1.95))
    g.rec.action, g.rec.side = "BET", "home"
    r = rank_picks([g], cfg)[0]
    assert r["kind"] == "moneyline" and r["weight"] == 1.0 and r["recommended"] and not r["experimental"]


def test_every_pick_is_ranked_even_without_edge_and_stale_or_alert_games_are_skipped():
    cfg = RiskConfig(bankroll=1000)
    fine = S(1, ml(0.50, 0.55, 1.80, 2.10))                          # model sees no edge anywhere: still ranked
    rows = rank_picks([fine, S(2, ml(0.6, 0.5, 2.0, 2.0), stale=True), S(3, ml(0.6, 0.5, 2.0, 2.0), status="ALERT")], cfg)
    assert len(rows) == 1 and rows[0]["game_id"] == 1 and rows[0]["has_edge"] is False


def test_player_props_one_sided_short_history_and_best_side():
    cfg = RiskConfig(bankroll=1000)
    props = [pq(1, "Scorer", 0.30, pm=0.25, over=4.0, stat="goals", one_sided=True), pq(2, "Rookie", 0.9, n_prev=3), pq(3, "Under Guy", 0.40, pm=0.50)]
    rows = rank_picks([S(1, props=props)], cfg)
    picks = {r["pick"] for r in rows}
    assert "Scorer to score (anytime)" in picks and "Under Guy Under 2.5 shots" in picks and not any("Rookie" in p for p in picks)
    assert next(r for r in rows if "Scorer" in r["pick"])["type"] == "Anytime goalscorer"
