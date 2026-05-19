"""Thin CLI shims: one function per /mmm-* skill, callable from agents and __main__."""
from __future__ import annotations


def cmd_intake_quick(args: list[str]) -> None:
    from agent_mmm.spec.intake import build_spec_from_answers
    build_spec_from_answers(quick=True)


def cmd_intake(args: list[str]) -> None:
    from agent_mmm.spec.intake import build_spec_from_answers
    build_spec_from_answers(quick=False)


def cmd_analyze_data(args: list[str]) -> None:
    raise NotImplementedError("Phase 1 — not yet implemented.")


def cmd_recommend_controls(args: list[str]) -> None:
    raise NotImplementedError("Phase 1 — not yet implemented.")


def cmd_recommend_priors(args: list[str]) -> None:
    raise NotImplementedError("Phase 1 — not yet implemented.")


def cmd_build(args: list[str]) -> None:
    raise NotImplementedError("Phase 1 — not yet implemented.")


def cmd_fit(args: list[str]) -> None:
    raise NotImplementedError("Phase 1 — not yet implemented.")


def cmd_diagnose(args: list[str]) -> None:
    raise NotImplementedError("Phase 1 — not yet implemented.")


def cmd_attribute(args: list[str]) -> None:
    raise NotImplementedError("Phase 2 — not yet implemented.")


def cmd_optimize(args: list[str]) -> None:
    raise NotImplementedError("Phase 3 — not yet implemented.")


def cmd_improve(args: list[str]) -> None:
    raise NotImplementedError("Phase 3 — not yet implemented.")


def cmd_report(args: list[str]) -> None:
    raise NotImplementedError("Phase 2 — not yet implemented.")


COMMANDS: dict[str, object] = {
    "intake-quick": cmd_intake_quick,
    "intake": cmd_intake,
    "analyze-data": cmd_analyze_data,
    "recommend-controls": cmd_recommend_controls,
    "recommend-priors": cmd_recommend_priors,
    "build": cmd_build,
    "fit": cmd_fit,
    "diagnose": cmd_diagnose,
    "attribute": cmd_attribute,
    "optimize": cmd_optimize,
    "improve": cmd_improve,
    "report": cmd_report,
}
