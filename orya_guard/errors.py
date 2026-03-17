from __future__ import annotations


class OryaGuardError(Exception):
    """User-facing error with a suggested next step."""

    def __init__(self, message: str, next_step: str) -> None:
        super().__init__(message)
        self.message = message
        self.next_step = next_step
