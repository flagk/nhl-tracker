import re
from pathlib import Path

import pandas as pd
import pytest

from nhlbet.report.history import render_history_html, render_history_md
from nhlbet.report.readme import END, START, pages_url, render_block, run_url, update_readme
from tests.test_report_pipeline import mk_slate, seeded_store


def test_history_by_day_and_bets_by_hand():
    st = seeded_store()
    md = render_history_md(st, "reports", site_archive=False, public_safe=True)
    # 2024-01-05: +10 and -10 on $20 staked = $0, 0.0%; 2024-01-06: one $4 bet won +$6 = +150.0%
    assert re.search(r"\| 2024-01-06 \| 2 \| 1 \| \$4\.00 \| \$\+6\.00 \| \+150\.0% \|", md)
    assert re.search(r"\| 2024-01-05 \| 2 \| 2 \| \$20\.00 \| \$\+0\.00 \| \+0\.0% \|", md)
    assert md.index("2024-01-06 |") < md.index("2024-01-05 |")                        # newest first
    assert "Recommended bets" in md and md.count("| won |") == 2 and md.count("| lost |") == 1
    assert "educational" in md and "afford to lose" in md and "../README.md" in md
    assert "bookB" not in md                                                            # public-safe: no bookmaker name
    assert "bookB" in render_history_md(st, "reports", site_archive=False, public_safe=False)


def test_history_empty_store_is_graceful():
    from nhlbet.data.store import Store
    st = Store(":memory:")
    md = render_history_md(st, "reports", public_safe=True)
    assert "No settled results yet" in md and "No recommended bets yet" in md
    assert "No days yet" in render_history_html(st, "reports", public_safe=True)


def test_history_html_links_to_archive_pages_and_escapes():
    html = render_history_html(seeded_store(), "reports", public_safe=True)
    assert "archive/2024-01-05.html" in html and "index.html" in html and "educational" in html
    assert "bookB" not in html and "<script" not in html


def test_readme_block_links_and_table():
    slate = mk_slate()
    env = {"GITHUB_REPOSITORY": "FlagK/nhl-tracker"}
    blk = render_block("2026-10-08", "late", slate, "OK", "2026-10-08 18:00 UTC", 1, env)
    assert blk.startswith(START) and blk.endswith(END)
    for link in ("(reports/latest.md)", "(reports/HISTORY.md)", "(data/logs/bet_log.csv)", "(https://flagk.github.io/nhl-tracker/)",
                 "(https://flagk.github.io/nhl-tracker/history.html)"):
        assert link in blk, link
    assert "TOR @ BOS" in blk and "**BET BOS**" in blk and "1 recommended bet" in blk and "educational" in blk
    local = render_block("2026-10-08", "late", [], "OK", "t", 0, {})                   # not running in GitHub: relative links only
    assert "(site/index.html)" in local and "github.io" not in local and "No NHL games" in local
    nobet = render_block("d", "m", mk_slate(bet=False), "WARN", "t", 0, {})
    assert "No bets today is normal" in nobet and "health **WARN**" in nobet
    assert pages_url({"GITHUB_REPOSITORY": "Bob/Repo"}) == "https://bob.github.io/Repo/" and pages_url({}) is None
    assert run_url({"GITHUB_REPOSITORY": "a/b", "GITHUB_RUN_ID": "7"}) == "https://github.com/a/b/actions/runs/7" and run_url({}) is None


def test_update_readme_replaces_only_the_block_and_is_idempotent(tmp_path):
    p = tmp_path / "README.md"
    p.write_text(f"# Title\n\nintro\n\n{START}\nold stuff\n{END}\n\n## Rest\nkeep me\n")
    b1 = render_block("d1", "late", mk_slate(), "OK", "t1", 1, {})
    assert update_readme(p, b1)
    t1 = p.read_text()
    assert t1.startswith("# Title\n\nintro\n\n") and t1.endswith("## Rest\nkeep me\n") and "old stuff" not in t1 and "d1" in t1
    assert update_readme(p, b1) and p.read_text() == t1                                # idempotent
    assert update_readme(p, render_block("d2", "late", [], "OK", "t2", 0, {})) and "d2" in p.read_text() and "d1" not in p.read_text()
    q = tmp_path / "plain.md"; q.write_text("no markers here\n")
    assert update_readme(q, b1) is False and q.read_text() == "no markers here\n"


def test_repo_readme_has_markers_and_public_safe_note():
    txt = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    assert START in txt and END in txt and txt.index(START) < txt.index("## Honest status")
    assert "public-safe" in txt and "Settings -> Pages" in txt
