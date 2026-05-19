"""Read/write/sort the mmm-workspace/leaderboard.json tournament leaderboard."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent_mmm.workspace.paths import leaderboard_path


def load_leaderboard() -> dict:
    path = leaderboard_path()
    if not path.exists():
        return {"runs": [], "best_run_id": None, "best_score": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_leaderboard(data: dict) -> None:
    path = leaderboard_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_run(
    run_id: str,
    round_num: int,
    variant_name: str,
    score: float,
    cv_r2: float,
    in_sample_r2: float,
    overfit_gap: float,
    converged: bool,
    n_divergences: int,
) -> None:
    data = load_leaderboard()
    entry = {
        "run_id": run_id,
        "round": round_num,
        "variant": variant_name,
        "score": score,
        "cv_r2": cv_r2,
        "in_sample_r2": in_sample_r2,
        "overfit_gap": overfit_gap,
        "converged": converged,
        "n_divergences": n_divergences,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    data["runs"].append(entry)

    # Update best
    if data["best_score"] is None or score > data["best_score"]:
        data["best_run_id"] = run_id
        data["best_score"] = score

    # Sort runs descending by score
    data["runs"] = sorted(data["runs"], key=lambda r: r["score"], reverse=True)
    save_leaderboard(data)


def best_run() -> dict | None:
    data = load_leaderboard()
    if not data["runs"]:
        return None
    return data["runs"][0]
