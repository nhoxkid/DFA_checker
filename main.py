from __future__ import annotations

import argparse
import sys
from typing import Sequence

from dfa_checker.cli import run as run_cli
from dfa_checker.gui import run_gui


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gui", action="store_true", help=argparse.SUPPRESS)
    args, remaining = parser.parse_known_args(argv)
    if args.gui:
        return run_gui()
    return run_cli(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
