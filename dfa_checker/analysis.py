from __future__ import annotations

from collections import deque

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

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


def analyze_graph(automaton: Automaton) -> Dict[str, object]:
    states = list(automaton.states)
    state_set = set(states)
    transitions = automaton.transitions
    epsilon_symbol = getattr(automaton, "epsilon_symbol", None)

    reachable: Set[str] = set()
    queue: deque[str] = deque([automaton.start_state])
    while queue:
        state = queue.popleft()
        if state in reachable:
            continue
        reachable.add(state)
        for destinations in transitions.get(state, {}).values():
            for dest in destinations:
                if dest not in reachable:
                    queue.append(dest)

    missing: List[Tuple[str, str]] = []
    nondeterministic: Set[str] = set()
    alphabet = list(automaton.alphabet)
    for state in states:
        mapping = transitions.get(state, {})
        for symbol in alphabet:
            destinations = mapping.get(symbol, frozenset())
            if not destinations:
                missing.append((state, symbol))
            if len(destinations) > 1:
                nondeterministic.add(state)
        for symbol, destinations in mapping.items():
            if symbol not in alphabet and len(destinations) > 1:
                nondeterministic.add(state)

    transition_count = sum(len(destinations) for mapping in transitions.values() for destinations in mapping.values())

    reverse: Dict[str, Set[str]] = {state: set() for state in states}
    for state, mapping in transitions.items():
        for destinations in mapping.values():
            for dest in destinations:
                reverse.setdefault(dest, set()).add(state)

    accepting = set(automaton.accept_states)
    alive: Set[str] = set()
    queue.clear()
    for state in accepting:
        if state in state_set:
            queue.append(state)
    while queue:
        state = queue.popleft()
        if state in alive:
            continue
        alive.add(state)
        for src in reverse.get(state, set()):
            if src not in alive:
                queue.append(src)

    dead_states = sorted(state_set - alive)
    unreachable = sorted(state_set - reachable)

    has_epsilon = False
    if epsilon_symbol:
        for mapping in transitions.values():
            if epsilon_symbol in mapping and mapping[epsilon_symbol]:
                has_epsilon = True
                break

    report: Dict[str, object] = {
        "state_count": len(states),
        "reachable_count": len(reachable),
        "unreachable": unreachable,
        "dead_states": dead_states,
        "missing_symbols": missing,
        "nondeterministic_states": sorted(nondeterministic),
        "transition_count": transition_count,
        "is_total": not missing,
        "has_epsilon": has_epsilon,
    }
    return report
