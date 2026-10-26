"""Autonomy and reliability: health gate, halt state machine, kill switch, policy guard, hash-chained logs, drift limits, alerts.

PRIME DIRECTIVE: when anything looks wrong the system stops publishing picks and alerts the owner. Nothing in this package changes a threshold, rule or model setting
to make a problem go away; thresholds come from config/policy.yaml, which no code here writes.
"""
