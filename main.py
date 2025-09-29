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
    parser.add_argument("--cli", action="store_true", help=argparse.SUPPRESS)
    args, remaining = parser.parse_known_args(argv)
    if args.cli or remaining:
        return run_cli(remaining)
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
