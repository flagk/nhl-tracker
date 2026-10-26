from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CheckResult:
    """One health or kill-switch check. ``level`` is the kind of halt a failure opens: ``soft`` (auto-resumes after N healthy runs) or ``hard`` (only the owner clears it)."""
    name: str
    level: str
    ok: bool
    detail: str = ""
    next_step: str = ""

    def __post_init__(self) -> None:
        if self.level not in ("soft", "hard"):
            raise ValueError(f"check level must be 'soft' or 'hard', not {self.level!r}")


def passed(name: str, level: str = "soft", detail: str = "") -> CheckResult:
    return CheckResult(name, level, True, detail)


def failed(name: str, level: str, detail: str, next_step: str) -> CheckResult:
    return CheckResult(name, level, False, detail, next_step)
