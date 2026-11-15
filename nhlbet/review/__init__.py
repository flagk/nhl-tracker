"""Pre-game review: an LLM agent may KEEP, FLAG or VETO picks the model already made; nothing else.

contract.py   candidate picks (stable ids), the review schema and the strict validator
odds_check.py deterministic odds-movement facts and the deterministic veto (no LLM)
apply.py      applies a validated review to produce the final picks, the vetoed list and the append-only logs
evaluate.py   grades kept / flagged / vetoed picks the same way and asks whether the agent adds anything
"""
