from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .automata import Automaton


def automaton_to_dot(
    automaton: Automaton,
    *,
    graph_name: str = "Automaton",
    rankdir: str = "LR",
    highlight_path: Sequence[Tuple[str, str]] = (),
) -> str:
    """Return a Graphviz DOT representation for the provided automaton."""
    highlight_edges = set(highlight_path)

    lines: List[str] = [f'digraph "{graph_name}" {{']
    lines.append(f"  rankdir={rankdir};")
    lines.append("  node [shape=circle];")
    lines.append("  __start__ [shape=point];")
    lines.append(f'  __start__ -> "{automaton.start_state}";')

    for state in automaton.states:
        shape = "doublecircle" if state in automaton.accept_states else "circle"
        lines.append(f'  "{state}" [shape={shape}];')

    for source, destination, labels in _collect_edges(automaton):
        label = ", ".join(labels)
        attributes = [f'label="{label}"']
        if (source, destination) in highlight_edges:
            attributes.append('color="red"')
            attributes.append('fontcolor="red"')
        attr_text = ", ".join(attributes)
        lines.append(f'  "{source}" -> "{destination}" [{attr_text}];')

    lines.append("}")
    return "\n".join(lines)


def _collect_edges(automaton: Automaton) -> Iterable[Tuple[str, str, List[str]]]:
    grouped: dict[Tuple[str, str], List[str]] = {}
    for state, mapping in automaton.transitions.items():
        for symbol, destinations in mapping.items():
            if not destinations:
                continue
            for destination in destinations:
                grouped.setdefault((state, destination), []).append(symbol)
    for (source, destination), labels in sorted(grouped.items()):
        labels.sort()
        yield source, destination, labels


def write_dot(automaton: Automaton, path: str, **kwargs) -> str:
    """Generate a DOT file at `path` and return the absolute path."""
    dot = automaton_to_dot(automaton, **kwargs)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(dot + "\n")
    return path
