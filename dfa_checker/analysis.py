from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .automata import Automaton, DFA, NFA


@dataclass(frozen=True)
class TestCase:
    tokens: Tuple[str, ...]
    expected: bool
    label: str = ""

    @staticmethod
    def from_raw(raw_tokens: Iterable[str] | str, expected: bool, label: str = "") -> "TestCase":
        if isinstance(raw_tokens, str):
            tokens = tuple(raw_tokens)
        else:
            tokens = tuple(raw_tokens)
        return TestCase(tokens=tokens, expected=expected, label=label)


@dataclass(frozen=True)
class TestResult:
    case: TestCase
    actual: bool

    @property
    def passed(self) -> bool:
        return self.actual == self.case.expected


def run_test_cases(automaton: Automaton, test_cases: Sequence[TestCase]) -> List[TestResult]:
    results: List[TestResult] = []
    for case in test_cases:
        actual = automaton.accepts(case.tokens)
        results.append(TestResult(case=case, actual=actual))
    return results


def summarize_results(results: Sequence[TestResult]) -> dict[str, int]:
    summary = {"total": len(results), "passed": 0, "failed": 0}
    for result in results:
        if result.passed:
            summary["passed"] += 1
        else:
            summary["failed"] += 1
    return summary


def ensure_dfa(automaton: Automaton) -> DFA:
    if isinstance(automaton, DFA):
        return automaton
    if isinstance(automaton, NFA):
        return automaton.to_dfa()
    raise TypeError("Unsupported automaton type for DFA conversion.")
