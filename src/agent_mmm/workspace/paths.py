from pathlib import Path

WORKSPACE = Path("mmm-workspace")


def ensure_workspace() -> Path:
    WORKSPACE.mkdir(exist_ok=True)
    (WORKSPACE / "reports").mkdir(exist_ok=True)
    return WORKSPACE


def spec_path() -> Path:
    return WORKSPACE / "spec.yaml"


def spec_used_path() -> Path:
    return WORKSPACE / "spec_used.yaml"


def idata_path() -> Path:
    return WORKSPACE / "idata.nc"


def diagnostics_path() -> Path:
    return WORKSPACE / "diagnostics.json"


def leaderboard_path() -> Path:
    return WORKSPACE / "leaderboard.json"


def reports_dir() -> Path:
    return WORKSPACE / "reports"


def report_path(role: str) -> Path:
    return reports_dir() / f"{role}.md"
