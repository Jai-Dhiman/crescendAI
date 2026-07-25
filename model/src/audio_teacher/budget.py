"""Hard pre-call spend cap for the Gate 0 probe run.

precheck() raises BEFORE the call that would exceed the cap. A
warn-and-continue mode deliberately does not exist ($50 hard stop is a
standing program policy).
"""
from __future__ import annotations


class BudgetExceededError(Exception):
    """The next call's estimated cost would exceed the spend cap."""


class BudgetGuard:
    def __init__(self, max_spend_usd: float, initial_spent_usd: float = 0.0):
        if max_spend_usd <= 0:
            raise ValueError(f"max_spend_usd must be positive, got {max_spend_usd}")
        if initial_spent_usd < 0:
            raise ValueError(
                f"initial_spent_usd must be non-negative, got {initial_spent_usd}"
            )
        self.max_spend_usd = max_spend_usd
        self.spent_usd = initial_spent_usd

    def precheck(self, estimated_next_cost_usd: float) -> None:
        """Raise BudgetExceededError if the next call would exceed the cap.

        Call this BEFORE every sampling call. Refusal charges nothing.
        """
        projected = self.spent_usd + estimated_next_cost_usd
        if projected > self.max_spend_usd:
            raise BudgetExceededError(
                f"next call estimated at ${estimated_next_cost_usd:.4f} would take "
                f"spend to ${projected:.4f}, over the ${self.max_spend_usd:.2f} cap "
                f"(spent so far ${self.spent_usd:.4f}). Responses so far are saved; "
                f"a re-run resumes from the manifest."
            )

    def record(self, actual_cost_usd: float) -> None:
        self.spent_usd += actual_cost_usd
