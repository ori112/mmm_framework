from __future__ import annotations

from pathlib import Path

import yaml

from .schema import Spec


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Spec.model_validate(raw)


def save_spec(spec: Spec, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(spec.model_dump(exclude_none=True), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def freeze_spec(spec: Spec, dest: str | Path) -> None:
    """Write an immutable snapshot of spec alongside idata.nc for reproducibility."""
    save_spec(spec, dest)
