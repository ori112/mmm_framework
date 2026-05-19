"""Entry point: python -m agent_mmm <command> [args]"""
from __future__ import annotations

import sys


_HELP = """\
Usage: python -m agent_mmm <command>

Commands (Phase 0):
  intake-quick       5-question intake -> mmm-workspace/spec.yaml
  intake             Full intake -> mmm-workspace/spec.yaml

Coming in later phases:
  analyze-data       Audit data quality (Phase 1)
  recommend-controls Suggest IL control variables (Phase 1)
  recommend-priors   Recommend channel priors (Phase 1)
  build              Construct MMM model (Phase 1)
  fit                Fit model via NUTS (Phase 1)
  diagnose           Convergence and fit diagnostics (Phase 1)
  attribute          Channel contributions and ROAS (Phase 2)
  report             Generate stakeholder reports (Phase 2)
  optimize           Budget optimization (Phase 3)
  improve            Iterative improvement tournament (Phase 3)
"""


def main() -> None:
    from agent_mmm.cli.commands import COMMANDS

    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(_HELP)
        return

    cmd = args[0]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd!r}\n")
        print(_HELP)
        sys.exit(1)

    COMMANDS[cmd](args[1:])  # type: ignore[operator]


if __name__ == "__main__":
    main()
