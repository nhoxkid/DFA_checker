from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from . import _accelerator
except ImportError:
    _accelerator = None


class AutomatonError(Exception):
    """Base meltdown for automata drama."""


class AutomatonValidationError(AutomatonError):
    """Input is cursed or whatever."""


TransitionMap = Dict[str, Dict[str, FrozenSet[str]]]


def _iter_bits(mask: int) -> Iterable[int]:
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _single_bit_index(bitset: int) -> int:
    if bitset & (bitset - 1):
        raise AutomatonValidationError("DFA transition went full spider-web (non-deterministic).")
    return bitset.bit_length() - 1 if bitset else -1


class Automaton:
    __slots__ = (
        "_states",
        "_alphabet",
        "_start_state",
        "_accept_states",
        "_transitions",
        "_state_to_idx",
        "_symbol_to_idx",
        "_transition_matrix",
        "_extra_matrices",
        "_blank_row",
        "_start_idx",
        "_accept_mask",
        "_bitset_cache",
    )

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
            raise AutomatonValidationError("Need at least one state, shocker.")
        if len(set(self._states)) != len(self._states):
            raise AutomatonValidationError("State names gotta be unique, man.")

        self._alphabet = tuple(self._normalize_symbol(sym) for sym in alphabet)
        if len(set(self._alphabet)) != len(self._alphabet):
            raise AutomatonValidationError("Duplicate alphabet symbols? nope.")

        self._start_state = self._normalize_state(start_state)
        self._accept_states = frozenset(self._normalize_state(s) for s in accept_states)
        self._transitions = self._build_transition_map(transitions)

        self._state_to_idx = {state: idx for idx, state in enumerate(self._states)}
        self._symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self._alphabet)}

        if self._start_state not in self._state_to_idx:
            raise AutomatonValidationError("Start state is playing hide-and-seek.")
        missing_accepts = [s for s in self._accept_states if s not in self._state_to_idx]
        if missing_accepts:
            raise AutomatonValidationError(
                f"Accept states not declared: {', '.join(sorted(missing_accepts))}."
            )

        self._start_idx = self._state_to_idx[self._start_state]
        self._accept_mask = 0
        for state in self._accept_states:
            self._accept_mask |= 1 << self._state_to_idx[state]

        self._transition_matrix, self._extra_matrices = self._build_fast_matrices()
        self._blank_row = tuple(0 for _ in self._states)
        self._bitset_cache: Dict[int, FrozenSet[str]] = {0: frozenset()}

    # ---------------------------------------------------------------
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
            if not isinstance(mapping, Mapping):
                raise AutomatonValidationError("Transitions per state must be mapping-like.")
            state_transitions = result.setdefault(norm_state, {})
            for symbol, destinations in mapping.items():
                norm_symbol = self._normalize_symbol(symbol)
                if destinations is None:
                    norm_dests: FrozenSet[str] = frozenset()
                else:
                    norm_dests = frozenset(
                        self._normalize_state(dst) for dst in destinations
                    )
                state_transitions[norm_symbol] = norm_dests
        return result

    def _build_fast_matrices(self) -> Tuple[Tuple[Tuple[int, ...], ...], Dict[str, Tuple[int, ...]]]:
        state_count = len(self._states)
        extra: Dict[str, List[int]] = {}
        matrix: List[Tuple[int, ...]] = []
        for idx, state in enumerate(self._states):
            mapping = self._transitions.get(state, {})
            row: List[int] = []
            for symbol in self._alphabet:
                row.append(self._names_to_bitset(mapping.get(symbol, frozenset())))
            matrix.append(tuple(row))
            for symbol, destinations in mapping.items():
                if symbol in self._symbol_to_idx:
                    continue
                extra.setdefault(symbol, [0] * state_count)[idx] = self._names_to_bitset(destinations)
        extra_final: Dict[str, Tuple[int, ...]] = {
            symbol: tuple(values) for symbol, values in extra.items()
        }
        return tuple(matrix), extra_final

    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    def validate(
        self,
        *,
        allow_partial: bool,
        allowed_special_symbols: Optional[Set[str]] = None,
    ) -> bool:
        self._validate_structure(allow_partial=allow_partial, allowed_special_symbols=allowed_special_symbols)
        return True

    def _validate_structure(
        self,
        *,
        allow_partial: bool,
        allowed_special_symbols: Optional[Set[str]] = None,
    ) -> None:
        allowed = set(self._alphabet)
        if allowed_special_symbols:
            allowed.update(allowed_special_symbols)

        for state in self._transitions.keys():
            if state not in self._state_to_idx:
                raise AutomatonValidationError(
                    f"State '{state}' shows up in transitions but not in the state list."
                )

        for state in self._states:
            mapping = self._transitions.get(state, {})
            if not allow_partial:
                for symbol in self._alphabet:
                    if not mapping.get(symbol):
                        raise AutomatonValidationError(
                            f"State '{state}' forgot to handle symbol '{symbol}'."
                        )
            for symbol, destinations in mapping.items():
                if symbol not in allowed:
                    raise AutomatonValidationError(
                        f"Symbol '{symbol}' is illegal in this alphabet party."
                    )
                if symbol in self._symbol_to_idx or (allowed_special_symbols and symbol in allowed_special_symbols):
                    for dest in destinations:
                        if dest not in self._state_to_idx:
                            raise AutomatonValidationError(
                                f"Destination '{dest}' was never declared."
                            )

    def transition_from(self, state: str, symbol: str) -> FrozenSet[str]:
        state = self._normalize_state(state)
        symbol = self._normalize_symbol(symbol)
        idx = self._state_to_idx.get(state)
        if idx is None:
            return frozenset()
        if symbol in self._symbol_to_idx:
            mask = self._transition_matrix[idx][self._symbol_to_idx[symbol]]
        else:
            mask = self._extra_matrices.get(symbol, self._blank_row)[idx]
        return self._bitset_to_names(mask)

    def _normalize_input(self, input_symbols: Iterable[str]) -> List[str]:
        if isinstance(input_symbols, str):
            if not self._alphabet:
                raise AutomatonError("Empty alphabet means no clue how to split strings.")
            if any(len(sym) != 1 for sym in self._alphabet):
                raise AutomatonError(
                    "Give me an iterable of tokens when alphabet symbols are not single characters."
                )
            return list(input_symbols)
        return [self._normalize_symbol(sym) for sym in input_symbols]

    def _input_to_symbol_ids(self, input_symbols: Iterable[str]) -> List[int]:
        tokens = self._normalize_input(input_symbols)
        ids: List[int] = []
        for token in tokens:
            ids.append(self._symbol_to_idx.get(token, -1))
        return ids

    def _names_to_bitset(self, names: Iterable[str]) -> int:
        mask = 0
        for name in names:
            try:
                idx = self._state_to_idx[name]
            except KeyError as exc:
                raise AutomatonValidationError(f"State '{name}' does not exist.") from exc
            mask |= 1 << idx
        return mask

    def _bitset_to_names(self, bitset: int) -> FrozenSet[str]:
        cached = self._bitset_cache.get(bitset)
        if cached is None:
            cached = frozenset(self._states[idx] for idx in _iter_bits(bitset))
            self._bitset_cache[bitset] = cached
        return cached


class DFA(Automaton):
    __slots__ = ("_delta", "_accelerator_ready")

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
            if not isinstance(mapping, Mapping):
                raise AutomatonValidationError("Each DFA row needs to be a mapping, bro.")
            normalized[state] = {}
            for symbol, destination in mapping.items():
                if isinstance(destination, str):
                    normalized[state][symbol] = [destination]
                else:
                    raise AutomatonValidationError(
                        "DFA transitions must point to exactly one state."
                    )
        super().__init__(states, alphabet, normalized, start_state, accept_states)
        self._delta = self._build_delta()
        self._accelerator_ready = _accelerator is not None
        self.validate()

    def _build_delta(self) -> Tuple[Tuple[int, ...], ...]:
        table: List[Tuple[int, ...]] = []
        for row in self._transition_matrix:
            table.append(tuple(_single_bit_index(bitset) for bitset in row))
        return tuple(table)

    def validate(self) -> bool:  # type: ignore[override]
        super().validate(allow_partial=False)
        for state_idx, row in enumerate(self._delta):
            for symbol_idx, destination in enumerate(row):
                if destination < 0:
                    raise AutomatonValidationError(
                        f"State '{self._states[state_idx]}' forgot symbol '{self._alphabet[symbol_idx]}'."
                    )
        return True

    def accepts(self, input_symbols: Iterable[str]) -> bool:
        symbol_ids = self._input_to_symbol_ids(input_symbols)
        if _accelerator is not None and self._accelerator_ready:
            try:
                return bool(
                    _accelerator.dfa_accepts(self._delta, self._start_idx, self._accept_mask, symbol_ids)
                )
            except (ValueError, TypeError):
                pass
        return self._accepts_python(symbol_ids)

    def _accepts_python(self, symbol_ids: Sequence[int]) -> bool:
        current = self._start_idx
        for symbol_id in symbol_ids:
            if symbol_id < 0 or symbol_id >= len(self._delta[current]):
                return False
            destination = self._delta[current][symbol_id]
            if destination < 0:
                return False
            current = destination
        return bool(self._accept_mask & (1 << current))

    def transition_path(self, tokens: Sequence[str]) -> Tuple[List[Tuple[str, str, str]], bool]:
        symbol_ids = self._input_to_symbol_ids(tokens)
        if any(symbol_id < 0 for symbol_id in symbol_ids):
            return [], False
        if _accelerator is not None and self._accelerator_ready:
            try:
                trace = _accelerator.dfa_trace(self._delta, self._start_idx, symbol_ids)
            except (ValueError, TypeError):
                trace = None
            if trace is not None:
                states_idx = list(trace)
                path: List[Tuple[str, str, str]] = []
                for idx, dest_idx in enumerate(states_idx[1:]):
                    src_idx = states_idx[idx]
                    path.append((self._states[src_idx], tokens[idx], self._states[dest_idx]))
                accepted = bool(self._accept_mask & (1 << states_idx[-1]))
                return path, accepted
        return self._trace_python(symbol_ids, tokens)

    def _trace_python(self, symbol_ids: Sequence[int], tokens: Sequence[str]) -> Tuple[List[Tuple[str, str, str]], bool]:
        path: List[Tuple[str, str, str]] = []
        current = self._start_idx
        for idx, symbol_id in enumerate(symbol_ids):
            if symbol_id < 0 or symbol_id >= len(self._delta[current]):
                return path, False
            destination = self._delta[current][symbol_id]
            if destination < 0:
                return path, False
            path.append((self._states[current], tokens[idx], self._states[destination]))
            current = destination
        accepted = bool(self._accept_mask & (1 << current))
        return path, accepted

    def as_nfa(self) -> "NFA":
        transitions: Dict[str, Dict[str, List[str]]] = {}
        for state_idx, state in enumerate(self._states):
            row: Dict[str, List[str]] = {}
            for symbol_idx, symbol in enumerate(self._alphabet):
                destination = self._delta[state_idx][symbol_idx]
                if destination >= 0:
                    row[symbol] = [self._states[destination]]
                else:
                    row[symbol] = []
            transitions[state] = row
        return NFA(self._states, self._alphabet, transitions, self._start_state, self._accept_states)


class NFA(Automaton):
    __slots__ = (
        "_epsilon_symbol",
        "_epsilon_matrix",
        "_epsilon_closure_masks",
        "_symbol_closure_matrix",
        "_accelerator_ready",
    )

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
            raise AutomatonValidationError("Epsilon symbol sneaked into the alphabet.")
        super().__init__(states, alphabet, transitions, start_state, accept_states)
        self._epsilon_matrix = self._extra_matrices.get(
            self._epsilon_symbol,
            self._blank_row,
        )
        self._accelerator_ready = False
        if _accelerator is not None:
            try:
                closures_tuple = tuple(int(v) for v in _accelerator.compute_epsilon_closures(self._epsilon_matrix))
                symbol_matrix = _accelerator.build_symbol_matrix(self._transition_matrix, closures_tuple)
                self._epsilon_closure_masks = closures_tuple
                self._symbol_closure_matrix = tuple(
                    tuple(int(v) for v in row) for row in symbol_matrix
                )
                self._accelerator_ready = True
            except (ValueError, TypeError):
                self._epsilon_closure_masks = self._build_epsilon_closures()
                self._symbol_closure_matrix = self._build_symbol_matrix()
        else:
            self._epsilon_closure_masks = self._build_epsilon_closures()
            self._symbol_closure_matrix = self._build_symbol_matrix()
        self.validate()

    @property
    def epsilon_symbol(self) -> str:
        return self._epsilon_symbol

    def validate(self) -> bool:  # type: ignore[override]
        super().validate(allow_partial=True, allowed_special_symbols={self._epsilon_symbol})
        return True

    def _build_epsilon_closures(self) -> Tuple[int, ...]:
        state_count = len(self._states)
        if not state_count:
            return tuple()
        closures = [0] * state_count

        def compute(state_idx: int) -> Tuple[int, int]:
            stack = [state_idx]
            visited = 1 << state_idx
            while stack:
                here = stack.pop()
                mask = self._epsilon_matrix[here] if here < len(self._epsilon_matrix) else 0
                while mask:
                    bit = mask & -mask
                    mask ^= bit
                    nxt = bit.bit_length() - 1
                    if not (visited & (1 << nxt)):
                        visited |= 1 << nxt
                        stack.append(nxt)
            return state_idx, visited

        max_workers = min(32, state_count) or 1
        if state_count == 1:
            idx, mask = compute(0)
            closures[idx] = mask
            return tuple(closures)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(compute, idx): idx for idx in range(state_count)}
            for future in as_completed(futures):
                idx, mask = future.result()
                closures[idx] = mask
        return tuple(closures)

    def _build_symbol_matrix(self) -> Tuple[Tuple[int, ...], ...]:
        table: List[Tuple[int, ...]] = []
        for state_idx in range(len(self._states)):
            row: List[int] = []
            for symbol_idx in range(len(self._alphabet)):
                mask = self._transition_matrix[state_idx][symbol_idx]
                closed = 0
                temp = mask
                while temp:
                    bit = temp & -temp
                    temp ^= bit
                    nxt = bit.bit_length() - 1
                    closed |= self._epsilon_closure_masks[nxt]
                row.append(closed)
            table.append(tuple(row))
        return tuple(table)

    def epsilon_closure(self, states: Iterable[str]) -> FrozenSet[str]:
        mask = 0
        for state in states:
            idx = self._state_to_idx.get(self._normalize_state(state))
            if idx is not None:
                mask |= self._epsilon_closure_masks[idx]
        return self._bitset_to_names(mask)

    def accepts(self, input_symbols: Iterable[str]) -> bool:
        symbol_ids = self._input_to_symbol_ids(input_symbols)
        if _accelerator is not None:
            try:
                return bool(
                    _accelerator.nfa_accepts(
                        self._symbol_closure_matrix,
                        self._epsilon_closure_masks[self._start_idx],
                        self._accept_mask,
                        symbol_ids,
                    )
                )
            except (ValueError, TypeError):
                pass
        return self._accepts_python(symbol_ids)

    def _accepts_python(self, symbol_ids: Sequence[int]) -> bool:
        current = self._epsilon_closure_masks[self._start_idx]
        for symbol_id in symbol_ids:
            if symbol_id < 0:
                return False
            nxt_mask = self._subset_step(current, symbol_id)
            if not nxt_mask:
                return False
            current = nxt_mask
        return bool(current & self._accept_mask)

    def _subset_step_python(self, subset_mask: int, symbol_idx: int) -> int:
        mask = 0
        temp = subset_mask
        while temp:
            bit = temp & -temp
            temp ^= bit
            state_idx = bit.bit_length() - 1
            mask |= self._symbol_closure_matrix[state_idx][symbol_idx]
        return mask

    def _subset_step(self, subset_mask: int, symbol_idx: int) -> int:
        if self._accelerator_ready:
            try:
                return int(
                    _accelerator.subset_step(self._symbol_closure_matrix, subset_mask, symbol_idx)
                )
            except (ValueError, TypeError):
                pass
        return self._subset_step_python(subset_mask, symbol_idx)

    def to_dfa(self) -> DFA:
        alphabet = list(self._alphabet)
        start_mask = self._epsilon_closure_masks[self._start_idx]
        subset_queue = deque([start_mask])
        mask_to_name: Dict[int, str] = {}

        def subset_name(mask: int) -> str:
            if not mask:
                return "EMPTY"
            names = [self._states[idx] for idx in _iter_bits(mask)]
            return "{" + ",".join(sorted(names)) + "}"

        mask_to_name[start_mask] = subset_name(start_mask)
        transitions: Dict[str, Dict[str, str]] = {}
        accept_states: Set[str] = set()
        used_empty = False
        empty_mask = 0

        executor: Optional[ThreadPoolExecutor] = None
        if len(alphabet) > 1:
            executor = ThreadPoolExecutor(max_workers=min(32, len(alphabet)))

        try:
            while subset_queue:
                mask = subset_queue.popleft()
                name = mask_to_name[mask]
                transitions.setdefault(name, {})
                if mask & self._accept_mask:
                    accept_states.add(name)

                symbol_results: Dict[int, int] = {}
                if executor:
                    futures = {
                        executor.submit(self._subset_step, mask, idx): idx
                        for idx in range(len(alphabet))
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        symbol_results[idx] = future.result()
                else:
                    for idx in range(len(alphabet)):
                        symbol_results[idx] = self._subset_step(mask, idx)

                for symbol_idx, next_mask in symbol_results.items():
                    symbol = alphabet[symbol_idx]
                    if next_mask:
                        if next_mask not in mask_to_name:
                            mask_to_name[next_mask] = subset_name(next_mask)
                            subset_queue.append(next_mask)
                        transitions[name][symbol] = mask_to_name[next_mask]
                    else:
                        used_empty = True
                        empty_name = subset_name(empty_mask)
                        mask_to_name.setdefault(empty_mask, empty_name)
                        transitions[name][symbol] = empty_name
        finally:
            if executor:
                executor.shutdown(wait=True)

        if used_empty:
            empty_name = subset_name(empty_mask)
            transitions.setdefault(empty_name, {})
            for symbol in alphabet:
                transitions[empty_name][symbol] = empty_name

        dfa_states = list(mask_to_name.values())
        start_name = mask_to_name[start_mask]
        return DFA(dfa_states, alphabet, transitions, start_name, accept_states)
