from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox

from .analysis import analyze_graph, run_test_cases, summarize_results
from .automata import AutomatonError
from .cli import (
    EMPTY_INPUT_LABEL,
    Session,
    TOKEN_SPLIT_RE,
    build_session_from_payload,
    determine_highlight_path,
    load_payload_from_text,
    write_graphs_for_session,
)

THEMES: Dict[str, Dict[str, str]] = {
    "light": {
        "bg": "#f6f8fa",
        "surface": "#ffffff",
        "text": "#24292e",
        "muted": "#57606a",
        "border": "#d0d7de",
        "input_bg": "#ffffff",
        "input_fg": "#24292e",
        "accent": "#0d6efd",
        "accent_fg": "#ffffff",
        "accent_hover": "#0b5ed7",
        "accent_hover_fg": "#ffffff",
        "accent_disabled": "#9ec5fe",
        "accent_disabled_fg": "#1f2933",
    },
    "dark": {
        "bg": "#0d1117",
        "surface": "#161b22",
        "text": "#c9d1d9",
        "muted": "#8b949e",
        "border": "#30363d",
        "input_bg": "#0d1117",
        "input_fg": "#c9d1d9",
        "accent": "#2f81f7",
        "accent_fg": "#0d1117",
        "accent_hover": "#508bff",
        "accent_hover_fg": "#0d1117",
        "accent_disabled": "#1f3b70",
        "accent_disabled_fg": "#8b949e",
    },
}

DEFAULT_TEMPLATE = """#states
s0
s1
s2
s3
#initial
s0
#accepting
s1
#alphabet
a
b
c
#transitions
s0:a>s1
s0:b>s3
s0:c>s0
s1:a>s2
s1:b>s2
s1:c>s3
"""

ANIMATION_INTERVAL_MS = 500


class AutomatonStudio(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Automaton Studio")
        self.geometry("1180x780")
        self.minsize(960, 640)

        self.session: Optional[Session] = None
        self.current_theme = "light"
        self.status_var = tk.StringVar(value="Paste a config and hit Analyze.")
        self.output_dir_var = tk.StringVar(value="artifacts")
        self.base_name_var = tk.StringVar(value="automaton")
        self.validation_result_var = tk.StringVar(value="No string tested.")
        self.simulation_status_var = tk.StringVar(value="Simulation idle.")

        self._busy = False
        self._worker: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        self._analysis_lock = threading.Lock()
        self._analysis_lock_acquired = False

        self._simulation_path: List[Tuple[str, str, str]] = []
        self._simulation_accepts = False
        self._simulation_index = 0
        self._simulation_running = False
        self._simulation_after: Optional[str] = None

        self._last_tokens: Sequence[str] = ()
        self._last_highlight: List[Tuple[str, str]] = []
        self._last_dot_paths: List[str] = []
        self._graph_report: Dict[str, object] = {}
        self._button_bindings: Dict[tk.Button, bool] = {}
        self._current_palette: Dict[str, str] = THEMES[self.current_theme].copy()

        self._build_ui()
        self.apply_theme()
        self._reset_simulation(clear_path=True)
        self._update_interaction_states()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    # ---------------------------------------------------------------
    def _build_ui(self) -> None:
        self.base_font = ("Segoe UI", 10)
        self.semibold_font = ("Segoe UI Semibold", 12)
        self.mono_font = ("JetBrains Mono", 10)

        self.header = tk.Frame(self, bd=0)
        self.header.pack(fill="x", padx=16, pady=(16, 8))

        self.title_label = tk.Label(
            self.header,
            text="Automaton Studio",
            font=("Segoe UI Semibold", 16),
        )
        self.title_label.pack(side="left")

        self.theme_button = tk.Button(
            self.header,
            text="Dark mode",
            command=self.toggle_theme,
            relief="flat",
            padx=16,
            pady=6,
        )
        self.theme_button.pack(side="right")

        self.body = tk.Frame(self, bd=0)
        self.body.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.body.columnconfigure(0, weight=3)
        self.body.columnconfigure(1, weight=2)
        self.body.rowconfigure(0, weight=1)

        # left pane --------------------------------------------------
        self.editor_frame = tk.Frame(self.body, bd=0)
        self.editor_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.editor_frame.rowconfigure(1, weight=1)

        self.config_label = tk.Label(
            self.editor_frame,
            text="Automaton Config",
            font=self.semibold_font,
        )
        self.config_label.grid(row=0, column=0, sticky="w")

        self.config_container = tk.Frame(self.editor_frame, bd=1, relief="solid")
        self.config_container.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.config_container.rowconfigure(0, weight=1)
        self.config_container.columnconfigure(0, weight=1)

        self.config_text = tk.Text(
            self.config_container,
            wrap="word",
            font=self.mono_font,
            relief="flat",
            undo=True,
            height=22,
        )
        self.config_text.grid(row=0, column=0, sticky="nsew")
        self.config_text.insert("1.0", DEFAULT_TEMPLATE)

        self.config_scroll = tk.Scrollbar(
            self.config_container,
            orient="vertical",
            command=self.config_text.yview,
        )
        self.config_scroll.grid(row=0, column=1, sticky="ns")
        self.config_text.configure(yscrollcommand=self.config_scroll.set)

        self.options_frame = tk.Frame(self.editor_frame, bd=0)
        self.options_frame.grid(row=2, column=0, sticky="ew", pady=12)
        self.options_frame.columnconfigure(1, weight=1)

        tk.Label(self.options_frame, text="Output dir", font=self.base_font).grid(
            row=0, column=0, sticky="w"
        )
        self.output_dir_entry = tk.Entry(
            self.options_frame,
            textvariable=self.output_dir_var,
            font=self.base_font,
        )
        self.output_dir_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.output_browse = tk.Button(
            self.options_frame,
            text="Browse",
            width=10,
            command=self._browse_output_dir,
        )
        self.output_browse.grid(row=0, column=2)

        tk.Label(self.options_frame, text="Base name", font=self.base_font).grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.base_name_entry = tk.Entry(
            self.options_frame,
            textvariable=self.base_name_var,
            font=self.base_font,
        )
        self.base_name_entry.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))

        self.button_row = tk.Frame(self.editor_frame, bd=0)
        self.button_row.grid(row=3, column=0, sticky="ew")
        self.button_row.columnconfigure(0, weight=1)
        self.button_row.columnconfigure(1, weight=1)
        self.button_row.columnconfigure(2, weight=1)

        self.analyze_button = tk.Button(
            self.button_row,
            text="Analyze",
            command=self.analyze,
            padx=16,
            pady=8,
        )
        self.analyze_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.clear_button = tk.Button(
            self.button_row,
            text="Clear",
            command=self.clear,
            padx=16,
            pady=8,
        )
        self.clear_button.grid(row=0, column=1, sticky="ew", padx=6)

        self.graph_button = tk.Button(
            self.button_row,
            text="Generate Graph",
            command=self._generate_graphs,
            padx=16,
            pady=8,
        )
        self.graph_button.grid(row=0, column=2, sticky="ew", padx=(6, 0))
        # right pane -------------------------------------------------
        self.output_frame = tk.Frame(self.body, bd=0)
        self.output_frame.grid(row=0, column=1, sticky="nsew")
        self.output_frame.rowconfigure(5, weight=1)

        self.output_label = tk.Label(
            self.output_frame,
            text="Results",
            font=self.semibold_font,
        )
        self.output_label.grid(row=0, column=0, sticky="w")

        self.validation_frame = tk.Frame(self.output_frame, bd=0)
        self.validation_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.validation_frame.columnconfigure(1, weight=1)

        tk.Label(self.validation_frame, text="Test string", font=self.base_font).grid(
            row=0, column=0, sticky="w"
        )
        self.test_entry = tk.Entry(
            self.validation_frame,
            font=self.base_font,
        )
        self.test_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.validate_button = tk.Button(
            self.validation_frame,
            text="Check",
            command=self._validate_string,
            padx=12,
            pady=6,
            width=10,
        )
        self.validate_button.grid(row=0, column=2, sticky="ew")

        self.validation_result_label = tk.Label(
            self.validation_frame,
            textvariable=self.validation_result_var,
            font=self.base_font,
        )
        self.validation_result_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.control_frame = tk.Frame(self.validation_frame, bd=0)
        self.control_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.columnconfigure(1, weight=1)
        self.control_frame.columnconfigure(2, weight=1)

        self.start_button = tk.Button(
            self.control_frame,
            text="Start",
            command=self._start_simulation,
            padx=12,
            pady=6,
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.pause_button = tk.Button(
            self.control_frame,
            text="Pause",
            command=self._pause_simulation,
            padx=12,
            pady=6,
        )
        self.pause_button.grid(row=0, column=1, sticky="ew", padx=4)

        self.reset_button = tk.Button(
            self.control_frame,
            text="Reset",
            command=self._reset_simulation,
            padx=12,
            pady=6,
        )
        self.reset_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        self.simulation_label = tk.Label(
            self.output_frame,
            text="Step Playback",
            font=self.semibold_font,
        )
        self.simulation_label.grid(row=2, column=0, sticky="w", pady=(12, 0))

        self.simulation_container = tk.Frame(self.output_frame, bd=1, relief="solid")
        self.simulation_container.grid(row=3, column=0, sticky="nsew", pady=(6, 0))
        self.simulation_container.rowconfigure(0, weight=1)
        self.simulation_container.columnconfigure(0, weight=1)

        self.simulation_text = tk.Text(
            self.simulation_container,
            wrap="word",
            font=self.base_font,
            height=6,
            relief="flat",
            state="disabled",
        )
        self.simulation_text.grid(row=0, column=0, sticky="nsew")
        self.simulation_scroll = tk.Scrollbar(
            self.simulation_container,
            orient="vertical",
            command=self.simulation_text.yview,
        )
        self.simulation_scroll.grid(row=0, column=1, sticky="ns")
        self.simulation_text.configure(yscrollcommand=self.simulation_scroll.set)

        self.simulation_status_label = tk.Label(
            self.simulation_container,
            textvariable=self.simulation_status_var,
            anchor="w",
            font=self.base_font,
            padx=6,
            pady=6,
        )
        self.simulation_status_label.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.graph_container = tk.Frame(self.output_frame, bd=1, relief="solid")
        self.graph_container.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        self.graph_container.columnconfigure(0, weight=1)

        self.graph_text = tk.Text(
            self.graph_container,
            wrap="word",
            font=self.base_font,
            height=6,
            relief="flat",
            state="disabled",
        )
        self.graph_text.grid(row=0, column=0, sticky="nsew")

        self.output_container = tk.Frame(self.output_frame, bd=1, relief="solid")
        self.output_container.grid(row=5, column=0, sticky="nsew", pady=(12, 0))
        self.output_container.rowconfigure(0, weight=1)
        self.output_container.columnconfigure(0, weight=1)

        self.output_text = tk.Text(
            self.output_container,
            wrap="word",
            font=self.base_font,
            state="disabled",
            relief="flat",
        )
        self.output_text.grid(row=0, column=0, sticky="nsew")
        self.output_scroll = tk.Scrollbar(
            self.output_container,
            orient="vertical",
            command=self.output_text.yview,
        )
        self.output_scroll.grid(row=0, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=self.output_scroll.set)

        self.status_bar = tk.Label(
            self,
            textvariable=self.status_var,
            anchor="w",
            font=self.base_font,
            padx=16,
            pady=8,
        )
        self.status_bar.pack(fill="x", side="bottom")

        self._surface_frames = [
            self,
            self.header,
            self.body,
            self.editor_frame,
            self.config_container,
            self.options_frame,
            self.button_row,
            self.output_frame,
            self.validation_frame,
            self.control_frame,
            self.simulation_container,
            self.graph_container,
            self.output_container,
        ]
        self._labels = [
            self.title_label,
            self.config_label,
            self.output_label,
            self.validation_result_label,
            self.simulation_label,
            self.simulation_status_label,
            self.status_bar,
        ]
        self._buttons = [
            self.theme_button,
            self.analyze_button,
            self.clear_button,
            self.graph_button,
            self.output_browse,
            self.validate_button,
            self.start_button,
            self.pause_button,
            self.reset_button,
        ]
        self._text_widgets = [self.config_text, self.simulation_text, self.graph_text, self.output_text]
        self._entries = [self.output_dir_entry, self.base_name_entry, self.test_entry]
    # ---------------------------------------------------------------
    def apply_theme(self) -> None:
        palette = THEMES[self.current_theme]
        self._current_palette = palette
        self.configure(bg=palette["bg"])
        for frame in self._surface_frames:
            frame_bg = palette["bg"] if frame is self else palette["surface"]
            frame.configure(bg=frame_bg)
        self.config_container.configure(bg=palette["border"])
        self.simulation_container.configure(bg=palette["border"])
        if hasattr(self, "graph_container"):
            self.graph_container.configure(bg=palette["border"])
        self.output_container.configure(bg=palette["border"])
        for label in self._labels:
            label.configure(bg=palette["surface"], fg=palette["text"])
        for entry in self._entries:
            entry.configure(
                bg=palette["input_bg"],
                fg=palette["input_fg"],
                insertbackground=palette["text"],
                highlightthickness=1,
                highlightcolor=palette["border"],
                highlightbackground=palette["border"],
                relief="flat",
                borderwidth=1,
            )
        for text_widget in self._text_widgets:
            state = text_widget.cget("state")
            if state == "disabled":
                text_widget.configure(state="normal")
            text_widget.configure(
                bg=palette["input_bg"],
                fg=palette["input_fg"],
                insertbackground=palette["text"],
            )
            if state == "disabled":
                text_widget.configure(state="disabled")
        for button in self._buttons:
            button.configure(relief="flat", borderwidth=0, highlightthickness=0)
        self._style_buttons(palette)
        self.status_bar.configure(bg=palette["surface"], fg=palette["muted"])
        self.theme_button.configure(
            text="Dark mode" if self.current_theme == "light" else "Light mode"
        )

    def toggle_theme(self) -> None:
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self.apply_theme()

    def _style_buttons(self, palette: Dict[str, str]) -> None:
        self._current_palette = palette
        for button in self._buttons:
            button.configure(
                activebackground=palette["accent_hover"],
                activeforeground=palette["accent_hover_fg"],
                disabledforeground=palette["accent_disabled_fg"],
            )
            if button not in self._button_bindings:
                button.bind("<Enter>", lambda event, b=button: self._on_button_hover(b, True))
                button.bind("<Leave>", lambda event, b=button: self._on_button_hover(b, False))
                self._button_bindings[button] = True
        self._refresh_button_colors()

    def _refresh_button_colors(self) -> None:
        palette = self._current_palette
        for button in self._buttons:
            if button["state"] == "disabled":
                button.configure(bg=palette["accent_disabled"], fg=palette["accent_disabled_fg"])
            else:
                button.configure(bg=palette["accent"], fg=palette["accent_fg"])

    def _on_button_hover(self, button: tk.Button, entering: bool) -> None:
        if button["state"] == "disabled":
            return
        palette = self._current_palette
        if entering:
            button.configure(bg=palette["accent_hover"], fg=palette["accent_hover_fg"])
        else:
            button.configure(bg=palette["accent"], fg=palette["accent_fg"])
    # ---------------------------------------------------------------
    def analyze(self) -> None:
        if self._busy:
            return
        payload = self.config_text.get("1.0", "end").strip()
        if not payload:
            messagebox.showinfo("Automaton Studio", "Drop a config in the editor first.", parent=self)
            return
        output_dir = self.output_dir_var.get().strip() or "artifacts"
        base_name = self.base_name_var.get().strip() or "automaton"
        if not self._analysis_lock.acquire(blocking=False):
            messagebox.showinfo("Automaton Studio", "Already chewing on a config.", parent=self)
            return
        self._analysis_lock_acquired = True
        self._worker_stop.clear()
        self._set_busy(True, "Crunching automata magic...")
        self._reset_simulation(clear_path=True)
        worker = threading.Thread(
            target=self._run_analysis_worker,
            args=(payload, output_dir, base_name),
            daemon=True,
        )
        self._worker = worker
        worker.start()

    def _run_analysis_worker(self, payload: str, output_dir: str, base_name: str) -> None:
        def cancelled() -> bool:
            if self._worker_stop.is_set():
                self.after(0, self._handle_worker_cancelled)
                return True
            return False

        try:
            config_payload = load_payload_from_text(payload)
        except ValueError as exc:
            if cancelled():
                return
            self.after(0, lambda: self._on_analysis_error(exc))
            return
        if cancelled():
            return
        try:
            session = build_session_from_payload(config_payload)
        except (AutomatonError, ValueError) as exc:
            if cancelled():
                return
            self.after(0, lambda: self._on_analysis_error(exc))
            return
        if cancelled():
            return

        summary_lines = self._session_summary_lines(session)
        if cancelled():
            return

        if session.test_cases:
            results = run_test_cases(session.automaton, session.test_cases)
            summary = summarize_results(results)
            test_lines = self._format_test_lines(results, summary)
        else:
            test_lines = ["No test cases defined."]
        if cancelled():
            return

        graph_report = analyze_graph(session.automaton)
        highlight = determine_highlight_path(session)
        if cancelled():
            return

        dot_paths: List[str] = []
        graph_error: Optional[Exception] = None
        try:
            paths = write_graphs_for_session(session, output_dir, base_name, highlight)
            dot_paths = [str(path) for path in paths]
        except Exception as exc:  # pylint: disable=broad-except
            graph_error = exc
        if cancelled():
            return

        self.after(
            0,
            lambda: self._on_analysis_success(
                session,
                summary_lines,
                test_lines,
                dot_paths,
                highlight,
                graph_report,
                graph_error,
            ),
        )

    def _handle_worker_cancelled(self) -> None:
        self._worker = None
        self._set_busy(False)
        self._release_analysis_lock()
    def _on_analysis_success(
        self,
        session: Session,
        summary_lines: Sequence[str],
        test_lines: Sequence[str],
        dot_paths: Sequence[str],
        highlight: Sequence[Tuple[str, str]],
        graph_report: Dict[str, object],
        graph_error: Optional[Exception] = None,
    ) -> None:
        self.session = session
        self._last_highlight = list(highlight)
        self._last_dot_paths = list(dot_paths)
        self._graph_report = dict(graph_report)
        self._worker = None
        self._set_busy(False)
        self._release_analysis_lock()

        lines: List[str] = ["Automaton Summary", "-----------------"]
        lines.extend(summary_lines)
        lines.append("")
        lines.extend(self._format_graph_report(graph_report))
        lines.append("")
        lines.append("Test Report")
        lines.append("-----------")
        lines.extend(test_lines)
        if dot_paths:
            lines.append("")
            lines.append("DOT files")
            lines.append("---------")
            lines.extend(dot_paths)
        self._update_output(lines)
        self._update_graph_panel()

        if graph_error is not None:
            messagebox.showwarning(
                "Automaton Studio",
                f"Graphs were not updated: {graph_error}",
                parent=self,
            )
        else:
            self.status_var.set(f"Automaton ready. Generated {len(dot_paths)} DOT file(s).")

        self.validation_result_var.set("No string tested.")
        self.test_entry.delete(0, "end")
        self._reset_simulation(clear_path=True)
        self._update_interaction_states()

    def _on_analysis_error(self, error: Exception | str) -> None:
        self._worker = None
        self._set_busy(False)
        self._release_analysis_lock()
        message = str(error)
        messagebox.showerror("Automaton Studio", message, parent=self)
        self.status_var.set(f"Error: {message}")

    def _update_output(self, lines: Sequence[str]) -> None:
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", "\n".join(lines))
        self.output_text.configure(state="disabled")
        self.output_text.see("1.0")

    def _session_summary_lines(self, session: Session) -> List[str]:
        automaton = session.automaton
        lines = [f"Type: {session.automaton_type.upper()}"]
        lines.append(f"States: {', '.join(automaton.states) if automaton.states else '<none>'}")
        alphabet_text = ", ".join(automaton.alphabet) if automaton.alphabet else "<empty>"
        lines.append(f"Alphabet: {alphabet_text}")
        lines.append(f"Start state: {automaton.start_state}")
        accept_text = ", ".join(automaton.accept_states) if automaton.accept_states else "<none>"
        lines.append(f"Accept states: {accept_text}")
        lines.append("Transition function:")
        for state in automaton.states:
            mapping = automaton.transitions.get(state, {})
            parts: List[str] = []
            for symbol in automaton.alphabet:
                if symbol in mapping:
                    destinations = mapping[symbol]
                    dest_text = "|".join(sorted(destinations)) if destinations else "-"
                    parts.append(f"{symbol}->{dest_text}")
            extras = [sym for sym in mapping.keys() if sym not in automaton.alphabet]
            for symbol in sorted(extras):
                destinations = mapping[symbol]
                dest_text = "|".join(sorted(destinations)) if destinations else "-"
                parts.append(f"{symbol}->{dest_text}")
            lines.append(f"  {state}: {', '.join(parts) if parts else '<none>'}")
        return lines

    def _update_graph_panel(self) -> None:
        self.graph_text.configure(state="normal")
        self.graph_text.delete("1.0", "end")
        if not self._graph_report:
            self.graph_text.insert("end", "No graph diagnostics available.\n")
        else:
            lines = self._format_graph_report(self._graph_report)
            self.graph_text.insert("end", "\n".join(lines) + "\n")
        self.graph_text.configure(state="disabled")

    def _format_graph_report(self, report: Dict[str, object]) -> List[str]:
        lines = ["Graph Check", "-----------"]
        reachable = report.get("reachable_count")
        total = report.get("state_count")
        if reachable is not None and total is not None:
            lines.append(f"Reachable states: {reachable} / {total}")
        unreachable = report.get("unreachable") or []
        if unreachable:
            lines.append("Unreachable: " + ", ".join(unreachable))
        dead_states = report.get("dead_states") or []
        if dead_states:
            lines.append("Dead states: " + ", ".join(dead_states))
        missing = report.get("missing_symbols") or []
        if missing:
            formatted = ", ".join(f"{state}:{symbol}" for state, symbol in missing)
            lines.append("Missing transitions: " + formatted)
        nondet = report.get("nondeterministic_states") or []
        if nondet:
            lines.append("Non-deterministic states: " + ", ".join(nondet))
        transition_count = report.get("transition_count")
        if transition_count is not None:
            lines.append("Total transitions: " + str(transition_count))
        if report.get("is_total"):
            lines.append("Transition function is total.")
        if report.get("has_epsilon"):
            lines.append("Includes epsilon transitions.")
        return lines

    def _format_test_lines(self, results, summary: Dict[str, int]) -> List[str]:
        lines = [f"Passed {summary['passed']} of {summary['total']} cases."]
        for result in results:
            tokens = " ".join(result.case.tokens) if result.case.tokens else EMPTY_INPUT_LABEL
            expected = "accept" if result.case.expected else "reject"
            actual = "accept" if result.actual else "reject"
            status = "PASS" if result.passed else "FAIL"
            label = f"{result.case.label}: " if result.case.label else ""
            lines.append(f"  [{status}] {label}{tokens} -> expected {expected}, got {actual}")
        return lines
    # ---------------------------------------------------------------
    def _browse_output_dir(self) -> None:
        if self._busy:
            return
        directory = filedialog.askdirectory(parent=self, mustexist=False)
        if directory:
            self.output_dir_var.set(directory)

    def clear(self) -> None:
        if self._busy:
            return
        self.config_text.delete("1.0", "end")
        self.config_text.insert("1.0", DEFAULT_TEMPLATE)
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.configure(state="disabled")
        if hasattr(self, "graph_text"):
            self.graph_text.configure(state="normal")
            self.graph_text.delete("1.0", "end")
            self.graph_text.configure(state="disabled")
        self.status_var.set("Cleared.")
        self.validation_result_var.set("No string tested.")
        self.test_entry.delete(0, "end")
        self.session = None
        self._last_tokens = ()
        self._last_highlight.clear()
        self._last_dot_paths.clear()
        self._graph_report.clear()
        self._update_graph_panel()
        self._reset_simulation(clear_path=True)
        self._update_interaction_states()
        self._refresh_button_colors()

    def _set_busy(self, busy: bool, status: Optional[str] = None) -> None:
        self._busy = busy
        if status:
            self.status_var.set(status)
        self._update_interaction_states()
        self._refresh_button_colors()

    def _update_interaction_states(self) -> None:
        if self._busy:
            for button in (
                self.analyze_button,
                self.clear_button,
                self.graph_button,
                self.output_browse,
                self.validate_button,
                self.start_button,
                self.pause_button,
                self.reset_button,
            ):
                button.configure(state="disabled")
            self._refresh_button_colors()
            return

        self.analyze_button.configure(state="normal")
        self.clear_button.configure(state="normal")
        self.output_browse.configure(state="normal")
        self.graph_button.configure(state="normal" if self.session else "disabled")
        has_session = self.session is not None
        self.validate_button.configure(state="normal" if has_session else "disabled")

        if self._simulation_running:
            self.start_button.configure(state="disabled")
            self.pause_button.configure(state="normal")
            self.reset_button.configure(state="disabled")
        else:
            has_path = bool(self._simulation_path)
            self.start_button.configure(state="normal" if has_session and has_path else "disabled")
            self.pause_button.configure(state="disabled")
            self.reset_button.configure(state="normal" if has_path else "disabled")

    def _release_analysis_lock(self) -> None:
        if self._analysis_lock_acquired:
            self._analysis_lock.release()
            self._analysis_lock_acquired = False
    # ---------------------------------------------------------------
    def _validate_string(self) -> None:
        if not self.session:
            messagebox.showinfo("Automaton Studio", "Analyze an automaton first.", parent=self)
            return
        raw = self.test_entry.get().strip()
        try:
            tokens = self._tokenize_input(raw)
        except ValueError as exc:
            messagebox.showerror("Automaton Studio", str(exc), parent=self)
            return
        try:
            accepted = self.session.automaton.accepts(tokens)
        except AutomatonError as exc:
            messagebox.showerror("Automaton Studio", str(exc), parent=self)
            return

        verdict = "Valid" if accepted else "Invalid"
        self.validation_result_var.set(f"{verdict} string ({'accept' if accepted else 'reject'}).")
        self.status_var.set(f"String check: {verdict.lower()}.")
        self._last_tokens = tokens

        path, dfa_accepts = self._compute_dfa_path(tokens)
        self._prepare_simulation(path, dfa_accepts and accepted)
        if path:
            self._last_highlight = [(src, dst) for src, _, dst in path]

    def _tokenize_input(self, raw: str) -> List[str]:
        if not self.session:
            return []
        automaton = self.session.automaton
        if not raw:
            return []
        if TOKEN_SPLIT_RE.search(raw):
            tokens = [token for token in TOKEN_SPLIT_RE.split(raw) if token]
        else:
            alphabet = list(automaton.alphabet)
            if alphabet and all(len(sym) == 1 for sym in alphabet):
                tokens = list(raw)
            else:
                tokens = [raw]
        invalid = [token for token in tokens if token not in automaton.alphabet_set]
        if invalid:
            raise ValueError(f"Symbols {', '.join(invalid)} are not in the alphabet.")
        return tokens

    def _compute_dfa_path(self, tokens: Sequence[str]) -> Tuple[List[Tuple[str, str, str]], bool]:
        if not self.session:
            return [], False
        try:
            path, accepted = self.session.dfa().transition_path(tokens)
        except AutomatonError:
            return [], False
        return path, accepted

    def _prepare_simulation(self, path: Sequence[Tuple[str, str, str]], accepts: bool) -> None:
        self._simulation_path = list(path)
        self._simulation_accepts = accepts
        self._simulation_index = 0
        self._simulation_running = False
        if self._simulation_after is not None:
            self.after_cancel(self._simulation_after)
        self._simulation_after = None
        self.simulation_text.configure(state="normal")
        self.simulation_text.delete("1.0", "end")
        if self._simulation_path:
            self.simulation_text.insert("end", "Ready. Press Start to animate.\n")
            verdict = "accept" if self._simulation_accepts else "reject"
            self.simulation_status_var.set(f"Simulation idle (final verdict {verdict}).")
        else:
            self.simulation_text.insert("end", "No deterministic path for this string.\n")
            self.simulation_status_var.set("Simulation unavailable for this string.")
        self.simulation_text.configure(state="disabled")
        self._update_interaction_states()
    def _start_simulation(self) -> None:
        if self._simulation_running or not self._simulation_path:
            return
        self._simulation_index = 0
        self._simulation_running = True
        self.simulation_text.configure(state="normal")
        self.simulation_text.delete("1.0", "end")
        self.simulation_text.configure(state="disabled")
        self._advance_simulation_step()
        self._update_interaction_states()

    def _advance_simulation_step(self) -> None:
        if not self._simulation_running:
            return
        if self._simulation_index >= len(self._simulation_path):
            verdict = "ACCEPT" if self._simulation_accepts else "REJECT"
            self.simulation_status_var.set(f"Finished: {verdict}.")
            self._simulation_running = False
            self._simulation_after = None
            self._update_interaction_states()
            return
        src, symbol, dest = self._simulation_path[self._simulation_index]
        self.simulation_text.configure(state="normal")
        self.simulation_text.insert(
            "end",
            f"{self._simulation_index + 1}. δ({src}, {symbol}) → {dest}\n",
        )
        self.simulation_text.configure(state="disabled")
        self.simulation_text.see("end")
        self.simulation_status_var.set(
            f"Step {self._simulation_index + 1}/{len(self._simulation_path)}: {src} --{symbol}--> {dest}"
        )
        self._simulation_index += 1
        self._simulation_after = self.after(ANIMATION_INTERVAL_MS, self._advance_simulation_step)
        self._update_interaction_states()

    def _pause_simulation(self) -> None:
        if not self._simulation_running:
            return
        if self._simulation_after is not None:
            self.after_cancel(self._simulation_after)
        self._simulation_after = None
        self._simulation_running = False
        self.simulation_status_var.set(
            f"Paused at step {self._simulation_index}/{len(self._simulation_path)}."
        )
        self._update_interaction_states()

    def _reset_simulation(self, clear_path: bool = False) -> None:
        if self._simulation_after is not None:
            self.after_cancel(self._simulation_after)
        self._simulation_after = None
        self._simulation_running = False
        if clear_path:
            self._simulation_path = []
            self._simulation_accepts = False
        self._simulation_index = 0
        self.simulation_text.configure(state="normal")
        self.simulation_text.delete("1.0", "end")
        if self._simulation_path:
            verdict = "accept" if self._simulation_accepts else "reject"
            self.simulation_text.insert("end", "Ready. Press Start to animate.\n")
            self.simulation_status_var.set(f"Simulation idle (final verdict {verdict}).")
        else:
            self.simulation_text.insert("end", "No simulation loaded.\n")
            self.simulation_status_var.set("Simulation idle.")
        self.simulation_text.configure(state="disabled")
        self._update_interaction_states()

    def _generate_graphs(self) -> None:
        if not self.session:
            messagebox.showinfo("Automaton Studio", "Nothing to draw yet.", parent=self)
            return
        highlight = self._last_highlight or determine_highlight_path(self.session)
        output_dir = self.output_dir_var.get().strip() or "artifacts"
        base_name = self.base_name_var.get().strip() or "automaton"
        try:
            paths = write_graphs_for_session(self.session, output_dir, base_name, highlight)
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Automaton Studio", str(exc), parent=self)
            return
        self._last_dot_paths = [str(path) for path in paths]
        self.status_var.set(f"DOT updated ({len(paths)} file(s)).")
        self._update_interaction_states()
    # ---------------------------------------------------------------
    def _on_close(self) -> None:
        self._worker_stop.set()
        worker = self._worker
        if worker and worker.is_alive():
            worker.join(timeout=0.5)
        self.destroy()


def run_gui() -> int:
    app = AutomatonStudio()
    app.mainloop()
    return 0
