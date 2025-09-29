from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from .analysis import TestCase, ensure_dfa, run_test_cases, summarize_results
from .automata import Automaton, AutomatonError, AutomatonValidationError, DFA, NFA
from .graphviz import write_dot

TOKEN_SPLIT_RE = re.compile(r"[\s,]+")
EMPTY_INPUT_LABEL = "<empty>"


@dataclass
class Session:
    automaton: Automaton
    automaton_type: str
    test_cases: List[TestCase] = field(default_factory=list)
    epsilon_symbol: Optional[str] = None
    interactive: bool = False
    _cached_dfa: Optional[DFA] = field(default=None, init=False, repr=False)

    def dfa(self) -> DFA:
        if self._cached_dfa is None:
            self._cached_dfa = ensure_dfa(self.automaton)
        return self._cached_dfa




def build_session_from_payload(payload: Mapping[str, Any]) -> Session:
    if not isinstance(payload, Mapping):
        raise ValueError('Config payload must be a mapping.')
    data = dict(payload)

    automaton_type = str(data.get('type', '')).lower()
    if automaton_type not in {'dfa', 'nfa'}:
        raise ValueError("Config field 'type' must be either 'dfa' or 'nfa'.")

    states = _require_string_sequence(data, 'states')
    alphabet = _require_string_sequence(data, 'alphabet')
    start_state = _require_string(data, 'start_state')
    accept_states = _require_string_sequence(data, 'accept_states')

    transitions_obj = data.get('transitions')
    if not isinstance(transitions_obj, dict):
        raise ValueError("Config field 'transitions' must be an object.")

    if automaton_type == 'dfa':
        transitions = _normalize_dfa_transitions_from_config(transitions_obj)
        automaton: Automaton = DFA(states, alphabet, transitions, start_state, accept_states)
        epsilon_symbol: Optional[str] = None
    else:
        epsilon_symbol = data.get('epsilon_symbol', 'epsilon')
        if not isinstance(epsilon_symbol, str) or not epsilon_symbol:
            raise ValueError("Config field 'epsilon_symbol' must be a non-empty string.")
        transitions = _normalize_nfa_transitions_from_config(transitions_obj)
        automaton = NFA(
            states,
            alphabet,
            transitions,
            start_state,
            accept_states,
            epsilon_symbol=epsilon_symbol,
        )

    test_cases = _load_test_cases_from_payload(data.get('test_cases'))
    session = Session(
        automaton=automaton,
        automaton_type=automaton_type,
        test_cases=test_cases,
        epsilon_symbol=epsilon_symbol if automaton_type == 'nfa' else None,
        interactive=False,
    )
    return session


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive DFA/NFA checker and graph generator."
    )
    parser.add_argument("--config", help="Path to a JSON file that defines the automaton.")
    parser.add_argument(
        "--tests",
        help="Optional JSON file containing additional test cases to execute.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where DOT graph files will be written.",
    )
    parser.add_argument(
        "--base-name",
        default="automaton",
        help="Base filename used for generated DOT files.",
    )
    parser.add_argument(
        "--epsilon-symbol",
        help="Override the epsilon symbol when building an NFA interactively.",
    )
    parser.add_argument(
        "--skip-interactive-tests",
        action="store_true",
        help="Skip the interactive test-case prompt.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        session = _build_session(args)
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        return 130
    except (
        AutomatonValidationError,
        AutomatonError,
        ValueError,
        FileNotFoundError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _display_summary(session)
    _run_tests(session)
    highlight_path = determine_highlight_path(session)
    paths = write_graphs_for_session(session, args.output_dir, args.base_name, highlight_path)
    if paths:
        print("\nDOT files written:")
        for path in paths:
            print(f"  {path}")
    return 0


def _build_session(args: argparse.Namespace) -> Session:
    if args.config:
        session = _build_from_config(Path(args.config))
        session.interactive = False
    else:
        session = _build_interactively(
            epsilon_override=args.epsilon_symbol,
            skip_tests=args.skip_interactive_tests,
        )
        session.interactive = True

    if args.tests:
        extra_cases = _load_test_cases_from_file(Path(args.tests))
        session.test_cases.extend(extra_cases)
    return session


def _build_from_config(path: Path) -> Session:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config file must define a JSON object.")
    return build_session_from_payload(payload)


def _build_interactively(*, epsilon_override: Optional[str], skip_tests: bool) -> Session:
    print("Interactive automaton builder")
    automaton_type = _prompt_type()
    alphabet = _prompt_symbol_list(
        "Enter alphabet symbols (separate with spaces or commas): ", allow_empty=True
    )
    states = _prompt_symbol_list(
        "Enter state names (separate with spaces or commas): ", allow_empty=False
    )
    start_state = _prompt_choice("Enter the start state: ", states)
    accept_states = _prompt_subset("Enter accepting states (press Enter for none): ", states)

    if automaton_type == "nfa":
        epsilon_symbol = epsilon_override or _prompt_optional(
            "Enter epsilon symbol [epsilon]: ", default="epsilon"
        )
        transitions = _collect_nfa_transitions(states, alphabet, epsilon_symbol)
        automaton = NFA(
            states,
            alphabet,
            transitions,
            start_state,
            accept_states,
            epsilon_symbol=epsilon_symbol,
        )
    else:
        epsilon_symbol = None
        transitions = _collect_dfa_transitions(states, alphabet)
        automaton = DFA(states, alphabet, transitions, start_state, accept_states)

    session = Session(
        automaton=automaton,
        automaton_type=automaton_type,
        epsilon_symbol=epsilon_symbol,
        interactive=True,
    )
    if not skip_tests:
        session.test_cases.extend(_prompt_test_cases(automaton))
    return session


def _display_summary(session: Session) -> None:
    automaton = session.automaton
    print("\nAutomaton Summary")
    print(f"  Type: {session.automaton_type.upper()}")
    print(f"  States: {', '.join(automaton.states)}")
    alphabet_text = ", ".join(automaton.alphabet) if automaton.alphabet else "<empty>"
    print(f"  Alphabet: {alphabet_text}")
    print(f"  Start state: {automaton.start_state}")
    accept_text = ", ".join(automaton.accept_states) if automaton.accept_states else "<none>"
    print(f"  Accept states: {accept_text}")
    print("  Transition function:")
    for state in automaton.states:
        mapping = automaton.transitions.get(state, {})
        if mapping:
            parts: List[str] = []
            for symbol in automaton.alphabet:
                if symbol in mapping:
                    destinations = mapping[symbol]
                    dest_text = "|".join(sorted(destinations)) if destinations else "-"
                    parts.append(f"{symbol}->{dest_text}")
            extra_symbols = [sym for sym in mapping.keys() if sym not in automaton.alphabet]
            for symbol in sorted(extra_symbols):
                destinations = mapping[symbol]
                dest_text = "|".join(sorted(destinations)) if destinations else "-"
                parts.append(f"{symbol}->{dest_text}")
            transition_text = ", ".join(parts) if parts else "<none>"
        else:
            transition_text = "<none>"
        print(f"    {state}: {transition_text}")


def _run_tests(session: Session):
    if not session.test_cases:
        print("\nNo test cases were provided.")
        return []
    print("\nRunning test cases...")
    results = run_test_cases(session.automaton, session.test_cases)
    summary = summarize_results(results)
    print(f"  Passed {summary['passed']} of {summary['total']} test cases.")
    for result in results:
        tokens_text = " ".join(result.case.tokens) if result.case.tokens else EMPTY_INPUT_LABEL
        expected_text = "accept" if result.case.expected else "reject"
        actual_text = "accept" if result.actual else "reject"
        status = "PASS" if result.passed else "FAIL"
        label_prefix = f"{result.case.label}: " if result.case.label else ""
        print(
            f"    [{status}] {label_prefix}{tokens_text} -> expected {expected_text}, got {actual_text}"
        )
    return results


def determine_highlight_path(session: Session) -> List[Tuple[str, str]]:
    if not session.test_cases:
        return []
    try:
        dfa = session.dfa()
    except AutomatonError:
        return []
    path: List[Tuple[str, str]] = []
    current_state = dfa.start_state
    for symbol in session.test_cases[0].tokens:
        destinations = dfa.transition_from(current_state, symbol)
        if not destinations:
            break
        next_state = next(iter(destinations))
        path.append((current_state, next_state))
        current_state = next_state
    return path


def write_graphs_for_session(
    session: Session,
    output_dir: Path | str,
    base_name: Optional[str],
    highlight_path: Sequence[Tuple[str, str]],
) -> List[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = (base_name or 'automaton').strip() or 'automaton'
    if session.interactive:
        name = _prompt_optional(
            f"Base filename for DOT outputs [{name}]: ", default=name
        )
        name = (name or 'automaton').strip() or 'automaton'

    written_paths: List[Path] = []
    if session.automaton_type == 'dfa':
        dfa_path = out_dir / f"{name}_dfa.dot"
        write_dot(session.automaton, str(dfa_path), highlight_path=highlight_path)
        written_paths.append(dfa_path)
        nfa_path = out_dir / f"{name}_dfa_as_nfa.dot"
        write_dot(session.automaton.as_nfa(), str(nfa_path))
        written_paths.append(nfa_path)
    else:
        nfa_path = out_dir / f"{name}_nfa.dot"
        write_dot(session.automaton, str(nfa_path))
        written_paths.append(nfa_path)
        dfa_path = out_dir / f"{name}_dfa.dot"
        write_dot(session.dfa(), str(dfa_path), highlight_path=highlight_path)
        written_paths.append(dfa_path)
    return [path.resolve() for path in written_paths]


def _collect_dfa_transitions(
    states: Sequence[str],
    alphabet: Sequence[str],
) -> dict[str, dict[str, str]]:
    transitions: dict[str, dict[str, str]] = {}
    if not alphabet:
        print("\nAlphabet is empty; DFA will only evaluate the empty input.")
    else:
        print("\nEnter DFA transitions (one destination per symbol).")
    state_set = set(states)
    for state in states:
        transitions[state] = {}
        for symbol in alphabet:
            prompt = f"  d({state}, {symbol}) = "
            while True:
                raw = _safe_input(prompt).strip()
                if not raw:
                    print("    Destination state is required.")
                    continue
                destinations = _split_tokens(raw)
                if len(destinations) != 1:
                    print("    Please provide exactly one destination state.")
                    continue
                destination = destinations[0]
                if destination not in state_set:
                    print(f"    Unknown state '{destination}'.")
                    continue
                transitions[state][symbol] = destination
                break
    return transitions


def _collect_nfa_transitions(
    states: Sequence[str],
    alphabet: Sequence[str],
    epsilon_symbol: str,
) -> dict[str, dict[str, List[str]]]:
    print(
        "\nEnter NFA transitions (separate multiple destinations with spaces or commas; leave blank for none)."
    )
    transitions: dict[str, dict[str, List[str]]] = {}
    for state in states:
        transitions[state] = {}
        for symbol in alphabet:
            destinations = _prompt_destination_list(state, symbol, states, allow_empty=True)
            transitions[state][symbol] = destinations
        epsilon_destinations = _prompt_destination_list(
            state,
            epsilon_symbol,
            states,
            allow_empty=True,
            label_override="epsilon",
        )
        if epsilon_destinations:
            transitions[state][epsilon_symbol] = epsilon_destinations
    return transitions


def _prompt_destination_list(
    state: str,
    symbol: str,
    states: Sequence[str],
    *,
    allow_empty: bool,
    label_override: Optional[str] = None,
) -> List[str]:
    state_set = set(states)
    label = label_override or symbol
    prompt = f"  d({state}, {label}) = "
    while True:
        raw = _safe_input(prompt).strip()
        if not raw:
            if allow_empty:
                return []
            print("    Destination state is required.")
            continue
        if raw.lower() in {"none", "-"}:
            return []
        tokens = _split_tokens(raw)
        if not tokens:
            if allow_empty:
                return []
            print("    Provide at least one destination state.")
            continue
        invalid = [token for token in tokens if token not in state_set]
        if invalid:
            print(f"    Unknown states: {', '.join(invalid)}.")
            continue
        return tokens


def _prompt_test_cases(automaton: Automaton) -> List[TestCase]:
    print("\nAdd test cases to validate the automaton.")
    while True:
        raw = _safe_input("How many test cases would you like to enter? (0 to skip): ").strip()
        if raw.isdigit():
            count = int(raw)
            break
        print("  Please enter a non-negative integer.")
    cases: List[TestCase] = []
    for index in range(1, count + 1):
        tokens = _prompt_input_tokens(automaton, index)
        expected = _prompt_expected(f"Expected result for case {index} ([A]ccept/[R]eject): ")
        label = _safe_input(f"Optional label for case {index} (press Enter to skip): ").strip()
        cases.append(TestCase(tokens=tuple(tokens), expected=expected, label=label or f"case {index}"))
    return cases


def _prompt_input_tokens(automaton: Automaton, index: int) -> List[str]:
    while True:
        raw = _safe_input(f"Input symbols for case {index}: ").strip()
        try:
            return _parse_input_tokens(raw, automaton.alphabet)
        except ValueError as exc:
            print(f"  {exc}")


def _parse_input_tokens(raw: str, alphabet: Sequence[str]) -> List[str]:
    if not raw:
        return []
    alphabet_list = list(alphabet)
    if not alphabet_list:
        raise ValueError("Alphabet is empty; only the empty input is valid.")
    if TOKEN_SPLIT_RE.search(raw):
        tokens = [token for token in TOKEN_SPLIT_RE.split(raw) if token]
    else:
        single_chars = all(len(sym) == 1 for sym in alphabet_list)
        tokens = list(raw) if single_chars else [raw]
    alphabet_set = set(alphabet_list)
    invalid = [token for token in tokens if token not in alphabet_set]
    if invalid:
        raise ValueError(f"Symbols {', '.join(invalid)} are not part of the alphabet.")
    return tokens


def _prompt_type() -> str:
    while True:
        raw = _safe_input("Choose automaton type ([D]FA/[N]FA): ").strip().lower()
        if raw in {"dfa", "d"}:
            return "dfa"
        if raw in {"nfa", "n"}:
            return "nfa"
        print("  Please enter 'DFA' or 'NFA'.")


def _prompt_symbol_list(prompt_text: str, allow_empty: bool) -> List[str]:
    while True:
        raw = _safe_input(prompt_text).strip()
        tokens = _split_tokens(raw)
        if tokens or allow_empty:
            return tokens
        print("  Please provide at least one value.")


def _split_tokens(text: str) -> List[str]:
    if not text:
        return []
    return [token for token in TOKEN_SPLIT_RE.split(text) if token]


def _prompt_choice(prompt_text: str, options: Sequence[str]) -> str:
    options_set = set(options)
    while True:
        raw = _safe_input(prompt_text).strip()
        if raw in options_set:
            return raw
        print(f"  Value must be one of: {', '.join(options)}.")


def _prompt_subset(prompt_text: str, options: Sequence[str]) -> List[str]:
    options_set = set(options)
    while True:
        raw = _safe_input(prompt_text).strip()
        if not raw:
            return []
        values = _split_tokens(raw)
        invalid = [value for value in values if value not in options_set]
        if invalid:
            print(f"  Unknown states: {', '.join(invalid)}.")
            continue
        return values


def _prompt_expected(prompt_text: str) -> bool:
    while True:
        raw = _safe_input(prompt_text).strip().lower()
        if raw in {"accept", "a", "yes", "y", "true", "t", "1"}:
            return True
        if raw in {"reject", "r", "no", "n", "false", "f", "0"}:
            return False
        print("  Please respond with accept or reject.")


def _prompt_optional(prompt_text: str, *, default: str) -> str:
    raw = _safe_input(prompt_text).strip()
    return raw or default


def _safe_input(prompt_text: str) -> str:
    try:
        return input(prompt_text)
    except EOFError as exc:
        raise KeyboardInterrupt from exc


def _require_string_sequence(payload: dict[str, Any], key: str) -> List[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Config field '{key}' must be a list of strings.")
    return list(value)


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config field '{key}' must be a non-empty string.")
    return value


def _normalize_dfa_transitions_from_config(
    transitions: dict[str, Any]
) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for state, mapping in transitions.items():
        if not isinstance(mapping, dict):
            raise ValueError("DFA transition entries must be objects.")
        state_map: dict[str, str] = {}
        for symbol, destination in mapping.items():
            if isinstance(destination, str):
                state_map[str(symbol)] = destination
            elif (
                isinstance(destination, list)
                and len(destination) == 1
                and isinstance(destination[0], str)
            ):
                state_map[str(symbol)] = destination[0]
            else:
                raise ValueError(
                    f"DFA transition for state '{state}' and symbol '{symbol}' must be a single destination string."
                )
        normalized[str(state)] = state_map
    return normalized


def _normalize_nfa_transitions_from_config(
    transitions: dict[str, Any]
) -> dict[str, dict[str, List[str]]]:
    normalized: dict[str, dict[str, List[str]]] = {}
    for state, mapping in transitions.items():
        if not isinstance(mapping, dict):
            raise ValueError("NFA transition entries must be objects.")
        state_map: dict[str, List[str]] = {}
        for symbol, destinations in mapping.items():
            if destinations is None:
                state_map[str(symbol)] = []
            elif isinstance(destinations, str):
                state_map[str(symbol)] = [destinations]
            elif isinstance(destinations, list):
                if not all(isinstance(dest, str) for dest in destinations):
                    raise ValueError("NFA transitions must list destination state names.")
                state_map[str(symbol)] = list(destinations)
            else:
                raise ValueError("NFA transition destinations must be strings or lists of strings.")
        normalized[str(state)] = state_map
    return normalized


def _load_test_cases_from_file(path: Path) -> List[TestCase]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _load_test_cases_from_payload(payload)


def _load_test_cases_from_payload(data: Any) -> List[TestCase]:
    if data is None:
        return []
    if isinstance(data, dict):
        entries = data.get("cases", [])
    else:
        entries = data
    if not isinstance(entries, list):
        raise ValueError("Test cases must be provided as a list.")
    cases: List[TestCase] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError("Each test case must be an object with 'input' and 'expected'.")
        raw_tokens = entry.get("input", [])
        expected = bool(entry.get("expected", False))
        label = entry.get("label") or f"case {index}"
        tokens = _normalize_test_case_tokens(raw_tokens)
        cases.append(TestCase(tokens=tokens, expected=expected, label=label))
    return cases


def _normalize_test_case_tokens(raw_tokens: Any) -> Tuple[str, ...]:
    if isinstance(raw_tokens, str):
        raw = raw_tokens.strip()
        if not raw:
            return ()
        if TOKEN_SPLIT_RE.search(raw):
            tokens = [token for token in TOKEN_SPLIT_RE.split(raw) if token]
        else:
            tokens = [raw]
        return tuple(tokens)
    if isinstance(raw_tokens, list):
        if not all(isinstance(token, str) for token in raw_tokens):
            raise ValueError("Test case symbols must be strings.")
        return tuple(token.strip() for token in raw_tokens)
    raise ValueError("Test case 'input' must be a string or a list of strings.")
