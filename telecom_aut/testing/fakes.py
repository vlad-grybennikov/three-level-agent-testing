"""Deterministic stand-ins for the LLM client (protocol: complete_json)."""

from __future__ import annotations


class ScriptedLLM:
    """Returns queued dict responses in order and records every prompt.

    The unit-test analogue of "frozen inputs": each stage call consumes
    exactly one scripted response, so tests are deterministic and offline.
    """

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []  # (system, user) per call

    def complete_json(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError(
                f"ScriptedLLM exhausted after {len(self.calls)} calls;"
                f" last prompt:\n{user[:400]}"
            )
        return self.responses.pop(0)


class LoopingLLM:
    """Cycles through a fixed response list forever (budget tests)."""

    def __init__(self, cycle: list[dict]) -> None:
        self.cycle = list(cycle)
        self.n = 0

    def complete_json(self, system: str, user: str) -> dict:
        response = self.cycle[self.n % len(self.cycle)]
        self.n += 1
        return response
