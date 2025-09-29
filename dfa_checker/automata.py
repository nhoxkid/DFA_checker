from __future__ import annotations

from collections import deque
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set


class AutomatonError(Exception):
    """Base error for automata operations."""


class AutomatonValidationError(AutomatonError):
    """Raised when an automaton definition is invalid."""


TransitionMap = Dict[str, Dict[str, FrozenSet[str]]]


class Automaton:
    """Common functionality shared by DFA and NFA structures."""

    def __init__(
        self,
        states: Sequence[str],
        alphabet: Sequence[str],
        transitions: Mapping[str, Mapping[str, Iterable[str]]],
        start_state: str,
        accept_states: Iterable[str],
    ) -> None:
        self._states = tuple(self._normalize_state(s) for s in states)
        if not self._states:
            raise AutomatonValidationError("Automaton must declare at least one state.")
        if len(set(self._states)) != len(self._states):
            raise AutomatonValidationError("State names must be unique.")

        self._alphabet = tuple(self._normalize_symbol(sym) for sym in alphabet)
        if len(set(self._alphabet)) != len(self._alphabet):
            raise AutomatonValidationError("Alphabet symbols must be unique.")

        self._start_state = self._normalize_state(start_state)
        self._accept_states = frozenset(self._normalize_state(s) for s in accept_states)
        self._transitions = self._build_transition_map(transitions)

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_state(state: str) -> str:
        if not isinstance(state, str) or not state:
            raise AutomatonValidationError("States must be non-empty strings.")
        return state.strip()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if not isinstance(symbol, str) or not symbol:
            raise AutomatonValidationError("Alphabet symbols must be non-empty strings.")
        return symbol.strip()

    def _build_transition_map(
        self, transitions: Mapping[str, Mapping[str, Iterable[str]]]
    ) -> TransitionMap:
        result: TransitionMap = {state: {} for state in self._states}
        for state, mapping in transitions.items():
            norm_state = self._normalize_state(state)
            state_transitions = result.setdefault(norm_state, {})
            for symbol, destinations in mapping.items():
                norm_symbol = self._normalize_symbol(symbol)
                if destinations is None:
                    dests: FrozenSet[str] = frozenset()
                else:
                    dests = frozenset(self._normalize_state(dst) for dst in destinations)
                state_transitions[norm_symbol] = dests
        return result

    # ------------------------------------------------------------------
    @property
    def states(self) -> Sequence[str]:
        return self._states

    @property
    def alphabet(self) -> Sequence[str]:
        return self._alphabet

    @property
    def alphabet_set(self) -> Set[str]:
        return set(self._alphabet)

    @property
    def start_state(self) -> str:
        return self._start_state

    @property
    def accept_states(self) -> FrozenSet[str]:
        return self._accept_states

    @property
    def transitions(self) -> TransitionMap:
        return self._transitions

    # ------------------------------------------------------------------
    def validate(
        self,
        *,
        allow_partial: bool,
        allowed_special_symbols: Optional[Set[str]] = None,
    ) -> bool:
        allowed_symbols = set(self._alphabet)
        if allowed_special_symbols:
            allowed_symbols.update(allowed_special_symbols)

        if self._start_state not in self._states:
            raise AutomatonValidationError("Start state must be part of the state set.")
        if not self._accept_states.issubset(set(self._states)):
            raise AutomatonValidationError("Accept states must be a subset of the state set.")

        for state, mapping in self._transitions.items():
            if state not in self._states:
                raise AutomatonValidationError(f"State '{state}' used in transitions but not declared.")
            for symbol, destinations in mapping.items():
                if symbol not in allowed_symbols:
                    raise AutomatonValidationError(
                        f"Symbol '{symbol}' is not part of the alphabet or permitted specials."
                    )
                if not allow_partial and symbol in self._alphabet and not destinations:
                    raise AutomatonValidationError(
                        f"Missing transition target for state '{state}' on symbol '{symbol}'."
                    )
                for dest in destinations:
                    if dest not in self._states:
                        raise AutomatonValidationError(
                            f"Destination state '{dest}' referenced but not declared."
                        )
        return True

    def transition_from(self, state: str, symbol: str) -> FrozenSet[str]:
        state = self._normalize_state(state)
        symbol = self._normalize_symbol(symbol)
        return self._transitions.get(state, {}).get(symbol, frozenset())

    def _normalize_input(self, input_symbols: Iterable[str]) -> List[str]:
        if isinstance(input_symbols, str):
            if not self._alphabet:
                raise AutomatonError("Cannot parse input without a defined alphabet.")
            if any(len(sym) != 1 for sym in self._alphabet):
                raise AutomatonError(
                    "Provide the input as an iterable of symbols when alphabet tokens are multi-character."
                )
            return list(input_symbols)
        return [self._normalize_symbol(sym) for sym in input_symbols]


class DFA(Automaton):
    """Deterministic Finite Automaton."""

    def __init__(
        self,
        states: Sequence[str],
        alphabet: Sequence[str],
        transitions: Mapping[str, Mapping[str, str]],
        start_state: str,
        accept_states: Iterable[str],
    ) -> None:
        normalized: Dict[str, Dict[str, Iterable[str]]] = {}
        for state, mapping in transitions.items():
            normalized[state] = {}
            for symbol, destination in mapping.items():
                if isinstance(destination, str):
                    destinations = [destination]
                else:
                    raise AutomatonValidationError("DFA transitions must map to a single destination state.")
                normalized[state][symbol] = destinations
        super().__init__(states, alphabet, normalized, start_state, accept_states)
        self.validate()

    def validate(self) -> bool:  # type: ignore[override]
        super().validate(allow_partial=False)
        alphabet = set(self.alphabet)
        for state in self.states:
            mapping = self.transitions.get(state, {})
            missing = alphabet - set(mapping.keys())
            if missing:
                missing_symbols = ", ".join(sorted(missing))
                raise AutomatonValidationError(
                    f"State '{state}' lacks transitions for symbols: {missing_symbols}."
                )
            for symbol, destinations in mapping.items():
                if len(destinations) != 1:
                    raise AutomatonValidationError(
                        f"State '{state}' has non-deterministic transition on symbol '{symbol}'."
                    )
        return True

    def accepts(self, input_symbols: Iterable[str]) -> bool:
        current_state = self.start_state
        for symbol in self._normalize_input(input_symbols):
            destinations = self.transition_from(current_state, symbol)
            if not destinations:
                return False
            current_state = next(iter(destinations))
        return current_state in self.accept_states

    def as_nfa(self) -> "NFA":
        transitions: Dict[str, Dict[str, Iterable[str]]] = {}
        for state, mapping in self.transitions.items():
            transitions[state] = {symbol: list(destinations) for symbol, destinations in mapping.items()}
        return NFA(self.states, self.alphabet, transitions, self.start_state, self.accept_states)


class NFA(Automaton):
    """Non-deterministic Finite Automaton (supports epsilon transitions)."""

    def __init__(
        self,
        states: Sequence[str],
        alphabet: Sequence[str],
        transitions: Mapping[str, Mapping[str, Iterable[str]]],
        start_state: str,
        accept_states: Iterable[str],
        epsilon_symbol: str = "epsilon",
    ) -> None:
        self._epsilon_symbol = self._normalize_symbol(epsilon_symbol)
        if self._epsilon_symbol in alphabet:
            raise AutomatonValidationError("Epsilon symbol must not be part of the explicit alphabet.")
        super().__init__(states, alphabet, transitions, start_state, accept_states)
        self.validate()

    @property
    def epsilon_symbol(self) -> str:
        return self._epsilon_symbol

    def validate(self) -> bool:  # type: ignore[override]
        super().validate(allow_partial=True, allowed_special_symbols={self._epsilon_symbol})
        return True

    # ------------------------------------------------------------------
    def epsilon_closure(self, states: Iterable[str]) -> FrozenSet[str]:
        closure: Set[str] = set(self._normalize_state(state) for state in states)
        stack = list(closure)
        while stack:
            state = stack.pop()
            for next_state in self.transition_from(state, self._epsilon_symbol):
                if next_state not in closure:
                    closure.add(next_state)
                    stack.append(next_state)
        return frozenset(closure)

    def accepts(self, input_symbols: Iterable[str]) -> bool:
        current_states = self.epsilon_closure([self.start_state])
        for symbol in self._normalize_input(input_symbols):
            next_states: Set[str] = set()
            for state in current_states:
                transitions = self.transition_from(state, symbol)
                for destination in transitions:
                    next_states.update(self.epsilon_closure([destination]))
            current_states = frozenset(next_states)
            if not current_states:
                return False
        return any(state in self.accept_states for state in current_states)

    def to_dfa(self) -> DFA:
        dfa_alphabet = list(self.alphabet)
        empty_subset: FrozenSet[str] = frozenset()

        def subset_name(subset: FrozenSet[str]) -> str:
            if not subset:
                return "EMPTY"
            return "{" + ",".join(sorted(subset)) + "}"

        start_subset = self.epsilon_closure([self.start_state])
        state_queue = deque([start_subset])
        subset_to_name: Dict[FrozenSet[str], str] = {start_subset: subset_name(start_subset)}
        transitions: Dict[str, Dict[str, str]] = {}
        accept_states: Set[str] = set()
        used_empty = False
        accept_lookup = set(self.accept_states)

        while state_queue:
            subset = state_queue.popleft()
            state_name = subset_to_name[subset]
            transitions.setdefault(state_name, {})

            if subset & accept_lookup:
                accept_states.add(state_name)

            for symbol in dfa_alphabet:
                next_states: Set[str] = set()
                for state in subset:
                    for destination in self.transition_from(state, symbol):
                        next_states.update(self.epsilon_closure([destination]))
                if next_states:
                    frozen_next = frozenset(next_states)
                    if frozen_next not in subset_to_name:
                        subset_to_name[frozen_next] = subset_name(frozen_next)
                        state_queue.append(frozen_next)
                    transitions[state_name][symbol] = subset_to_name[frozen_next]
                else:
                    transitions[state_name][symbol] = subset_name(empty_subset)
                    used_empty = True

        if used_empty:
            empty_name = subset_name(empty_subset)
            subset_to_name.setdefault(empty_subset, empty_name)
            if empty_name not in transitions:
                transitions[empty_name] = {}
            transitions[empty_name] = {symbol: empty_name for symbol in dfa_alphabet}

        dfa_states = list(subset_to_name.values())
        start_state = subset_to_name[start_subset]
        return DFA(dfa_states, dfa_alphabet, transitions, start_state, accept_states)
