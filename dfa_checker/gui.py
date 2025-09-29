from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox

from .analysis import run_test_cases, summarize_results
from .cli import (
    EMPTY_INPUT_LABEL,
    Session,
    build_session_from_payload,
    determine_highlight_path,
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
        "accent": "#0969da",
        "accent_fg": "#ffffff",
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
    },
}

DEFAULT_TEMPLATE = """{
  "type": "dfa",
  "alphabet": ["0", "1"],
  "states": ["q0", "q1"],
  "start_state": "q0",
  "accept_states": ["q1"],
  "transitions": {
    "q0": {"0": "q0", "1": "q1"},
    "q1": {"0": "q0", "1": "q1"}
  },
  "test_cases": [
    {"label": "odd ones", "input": "1", "expected": true},
    {"label": "stay even", "input": "10", "expected": false}
  ]
}"""


class AutomatonStudio(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Automaton Studio")
        self.geometry("1120x720")
        self.minsize(900, 600)

        self.session: Optional[Session] = None
        self.current_theme = "light"
        self.status_var = tk.StringVar(value="Paste a config and smash Analyze.")
        self.output_dir_var = tk.StringVar(value="artifacts")
        self.base_name_var = tk.StringVar(value="automaton")
        self._busy = False
        self._worker: Optional[threading.Thread] = None

        self._build_ui()
        self.apply_theme()

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
            font=("Segoe UI Semibold", 15),
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

        # left pane
        self.editor_frame = tk.Frame(self.body, bd=0)
        self.editor_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.editor_frame.rowconfigure(1, weight=1)

        self.config_label = tk.Label(
            self.editor_frame,
            text="Automaton Config (JSON)",
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
            height=20,
        )
        self.config_text.grid(row=0, column=0, sticky="nsew")
        self.config_text.insert("1.0", DEFAULT_TEMPLATE)

        self.config_scroll = tk.Scrollbar(
            self.config_container, orient="vertical", command=self.config_text.yview
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
        self.clear_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # right pane
        self.output_frame = tk.Frame(self.body, bd=0)
        self.output_frame.grid(row=0, column=1, sticky="nsew")
        self.output_frame.rowconfigure(1, weight=1)

        self.output_label = tk.Label(
            self.output_frame,
            text="Results",
            font=self.semibold_font,
        )
        self.output_label.grid(row=0, column=0, sticky="w")

        self.output_container = tk.Frame(self.output_frame, bd=1, relief="solid")
        self.output_container.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
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
            self.output_container, orient="vertical", command=self.output_text.yview
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
            self.output_container,
        ]
        self._labels = [
            self.title_label,
            self.config_label,
            self.output_label,
            self.status_bar,
        ]
        self._buttons = [
            self.theme_button,
            self.analyze_button,
            self.clear_button,
            self.output_browse,
        ]
        self._text_widgets = [self.config_text, self.output_text]
        self._entries = [self.output_dir_entry, self.base_name_entry]

    # ---------------------------------------------------------------
    def apply_theme(self) -> None:
        palette = THEMES[self.current_theme]
        self.configure(bg=palette["bg"])
        for frame in self._surface_frames:
            frame.configure(bg=palette["surface"] if frame is not self else palette["bg"])
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
            button.configure(
                bg=palette["accent"],
                fg=palette["accent_fg"],
                activebackground=palette["accent"],
                activeforeground=palette["accent_fg"],
                relief="flat",
                borderwidth=0,
            )
        self.config_container.configure(bg=palette["border"])
        self.output_container.configure(bg=palette["border"])
        self.status_bar.configure(bg=palette["surface"], fg=palette["muted"])
        self.theme_button.configure(
            text="Dark mode" if self.current_theme == "light" else "Light mode"
        )

    def toggle_theme(self) -> None:
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self.apply_theme()

    # ---------------------------------------------------------------
    def analyze(self) -> None:
        if self._busy:
            return
        payload = self.config_text.get("1.0", "end").strip()
        if not payload:
            messagebox.showinfo("Automaton Studio", "Drop a JSON config first.", parent=self)
            return
        output_dir = self.output_dir_var.get().strip() or "artifacts"
        base_name = self.base_name_var.get().strip() or "automaton"
        self._set_busy(True, "Crunching automata magic...")
        worker = threading.Thread(
            target=self._run_analysis_worker,
            args=(payload, output_dir, base_name),
            daemon=True,
        )
        self._worker = worker
        worker.start()

    def _run_analysis_worker(self, payload: str, output_dir: str, base_name: str) -> None:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            self.after(0, lambda: self._on_analysis_error(f"JSON oof: {exc}"))
            return
        try:
            session = build_session_from_payload(data)
            summary_lines = self._session_summary_lines(session)
            if session.test_cases:
                results = run_test_cases(session.automaton, session.test_cases)
                summary = summarize_results(results)
                test_lines = self._format_test_lines(results, summary)
            else:
                test_lines = ["No test cases defined."]
            highlight = determine_highlight_path(session)
            dot_paths = [
                str(path)
                for path in write_graphs_for_session(session, output_dir, base_name, highlight)
            ]
        except Exception as exc:  # pylint: disable=broad-except
            self.after(0, lambda: self._on_analysis_error(exc))
            return
        self.after(
            0,
            lambda: self._on_analysis_success(
                session,
                summary_lines,
                test_lines,
                dot_paths,
                highlight,
            ),
        )

    def _on_analysis_success(
        self,
        session: Session,
        summary_lines: Sequence[str],
        test_lines: Sequence[str],
        dot_paths: Sequence[str],
        highlight: Sequence[Tuple[str, str]],
    ) -> None:
        self.session = session
        lines: List[str] = ["Automaton Summary", "-----------------"]
        lines.extend(summary_lines)
        lines.append("")
        lines.append("Test Report")
        lines.append("-----------")
        lines.extend(test_lines)
        if highlight:
            display_path = [highlight[0][0]] + [dst for _, dst in highlight]
            lines.append("")
            lines.append("First test walk: " + " -> ".join(display_path))
        if dot_paths:
            lines.append("")
            lines.append("DOT files")
            lines.append("---------")
            lines.extend(dot_paths)
        self._update_output(lines)
        self._set_busy(False)
        if dot_paths:
            self.status_var.set(f"Generated {len(dot_paths)} DOT file(s).")
        else:
            self.status_var.set("Automaton processed.")

    def _on_analysis_error(self, error: Exception | str) -> None:
        self._set_busy(False)
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

    def clear(self) -> None:
        if self._busy:
            return
        self.config_text.delete("1.0", "end")
        self.config_text.insert("1.0", DEFAULT_TEMPLATE)
        self._update_output(["Ready for another automaton."])
        self.status_var.set("Cleared.")
        self.session = None

    def _browse_output_dir(self) -> None:
        if self._busy:
            return
        directory = filedialog.askdirectory(parent=self, mustexist=False)
        if directory:
            self.output_dir_var.set(directory)

    def _set_busy(self, busy: bool, status: Optional[str] = None) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for button in (self.analyze_button, self.clear_button, self.output_browse):
            button.configure(state=state)
        if status:
            self.status_var.set(status)


def run_gui() -> int:
    app = AutomatonStudio()
    app.mainloop()
    return 0
