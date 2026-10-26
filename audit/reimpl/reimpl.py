import math

def devig(decimals, method):
    q = [1.0 / o for o in decimals]
    s = sum(q)
    if s <= 1 + 1e-12:
        return [x / s for x in q]
    if method == "proportional":
        return [x / s for x in q]
    if method == "power":
        lo, hi = 1.0, 2.0
        while sum(x ** hi for x in q) > 1 and hi < 1e6:
            hi *= 2
        for _ in range(200):
            mid = (lo + hi) / 2
            if sum(x ** mid for x in q) > 1:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        c = (lo + hi) / 2
        p = [x ** c for x in q]
    elif method == "shin":
        def probs(z):
            return [(math.sqrt(z * z + 4 * (1 - z) * x * x / s) - z) / (2 * (1 - z)) for x in q]
        lo, hi = 0.0, 0.999
        while hi - lo > 1e-12:
            mid = (lo + hi) / 2
            if sum(probs(mid)) > 1:
                lo = mid
            else:
                hi = mid
        p = probs((lo + hi) / 2)
    else:
        raise ValueError("unknown method " + str(method))
    t = sum(p)
    return [x / t for x in p]


def _exp(rh, ra, H):
    return 1.0 / (1.0 + 10 ** (-(rh + H - ra) / 400.0))


def elo_probs(games, k=6.0, home_adv=35.0, regress=0.30):
    ratings = {}
    out = [None] * len(games)
    prev_season = None
    i, n = 0, len(games)
    while i < n:
        d = games[i][1]
        j = i
        while j < n and games[j][1] == d:
            j += 1
        season = games[i][0]
        if prev_season is not None and season != prev_season:
            for t in ratings:
                ratings[t] = (1 - regress) * ratings[t] + regress * 1500.0
        prev_season = season
        snap = dict(ratings)
        upd = []
        for g in range(i, j):
            _, _, h, a, hg, ag = games[g]
            rh = snap.get(h, 1500.0)
            ra = snap.get(a, 1500.0)
            e = _exp(rh, ra, home_adv)
            out[g] = e
            margin = abs(hg - ag)
            won = hg > ag
            diff = (rh + home_adv - ra) if won else -(rh + home_adv - ra)
            mult = math.log(margin + 1) * (2.2 / (diff * 0.001 + 2.2))
            delta = k * mult * ((1 if won else 0) - e)
            upd.append((h, a, rh + delta, ra - delta))
        for h, a, nh, na in upd:
            ratings[h] = nh
            ratings[a] = na
        i = j
    return out


DEFAULT_CFG = dict(kelly_fraction=0.25, max_bet_pct=0.02, min_edge=0.03,
                   min_edge_unconfirmed_goalie=0.04, min_ev=0.01, min_prob=0.40,
                   max_decimal=4.0, max_disagreement=0.12, trust_w0=0.5,
                   trust_d0=0.06, min_stake=1.0, min_games_played=10,
                   allow_stale_odds=False, max_game_pct=0.03, max_daily_pct=0.05,
                   max_bets=5)


def _nb():
    return {"action": "NO_BET", "side": None, "stake": 0.0, "odds": None, "ev": None}


def decide(p_home, quotes, ctx, bankroll, cfg=None):
    c = dict(DEFAULT_CFG)
    if cfg:
        c.update(cfg)
    if ctx.get("model_status") == "ALERT":
        return _nb()
    gp = ctx.get("games_played_min")
    if gp is not None and gp < c["min_games_played"]:
        return _nb()
    use = [(h, a) for h, a in quotes
           if h is not None and a is not None and h > 1 and a > 1]
    if not use:
        return _nb()
    if ctx.get("odds_stale") and not c["allow_stale_odds"]:
        return _nb()
    mh = sum(devig([h, a], "shin")[0] for h, a in use) / len(use)
    mkt = {"home": mh, "away": 1 - mh}
    best = {"home": max(h for h, a in use), "away": max(a for h, a in use)}
    need = c["min_edge"] if ctx.get("goalie_confirmed") else c["min_edge_unconfirmed_goalie"]
    cands = []
    for side in ("home", "away"):
        pm = p_home if side == "home" else 1 - p_home
        pk = mkt[side]
        d = best[side]
        edge = pm - pk
        fail = edge < need or pm < c["min_prob"] or d > c["max_decimal"] or abs(edge) > c["max_disagreement"]
        x = pm - pk
        w = c["trust_w0"] / (1 + (abs(x) / c["trust_d0"]) ** 2)
        padj = min(max(pk + w * x, 0.01), 0.99)
        ev = padj * d - 1
        if not fail and ev < c["min_ev"]:
            fail = True
        if not fail:
            cands.append((side, d, padj, ev))
    if not cands:
        return _nb()
    side, d, padj, ev = cands[0]
    for cd in cands[1:]:
        if cd[3] > ev:
            side, d, padj, ev = cd
    f = max(0.0, ((d - 1) * padj - (1 - padj)) / (d - 1))
    pct = min(c["kelly_fraction"] * f, c["max_bet_pct"])
    stake = round(bankroll * pct, 2)
    if stake < c["min_stake"]:
        return _nb()
    return {"action": "BET", "side": side, "stake": stake, "odds": d, "ev": ev}


def _down(x):
    return math.floor(x * 100 + 1e-9) / 100


def decide_slate(games, bankroll, cfg=None):
    c = dict(DEFAULT_CFG)
    if cfg:
        c.update(cfg)
    res = [decide(g["p_home"], g["quotes"], g["ctx"], bankroll, cfg) for g in games]
    bets = [i for i, r in enumerate(res) if r["action"] == "BET"]
    cap = bankroll * c["max_bet_pct"]
    for i in bets:
        if res[i]["stake"] > cap:
            res[i]["stake"] = _down(cap)
    order = sorted(bets, key=lambda i: -res[i]["ev"])  # stable
    keep = order[:c["max_bets"]]
    drop = set(order[c["max_bets"]:])
    gcap = bankroll * c["max_game_pct"]
    if gcap > 0:
        byg = {}
        for i in keep:
            byg.setdefault(games[i]["game_id"], []).append(i)
        for ids in byg.values():
            tot = sum(res[i]["stake"] for i in ids)
            if tot > gcap:
                for i in ids:
                    res[i]["stake"] = _down(res[i]["stake"] * gcap / tot)
    dcap = bankroll * c["max_daily_pct"]
    tot = sum(res[i]["stake"] for i in keep)
    if dcap > 0 and tot > dcap:
        for i in keep:
            res[i]["stake"] = _down(res[i]["stake"] * dcap / tot)
    for i in keep:
        if res[i]["stake"] < c["min_stake"]:
            drop.add(i)
    for i in drop:
        res[i] = _nb()
    return res
