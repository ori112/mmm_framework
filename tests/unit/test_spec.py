"""Unit tests for spec schema, loader, and workspace paths."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from agent_mmm.spec.schema import Channel, DataCfg, Spec, SamplerCfg, TargetUnit
from agent_mmm.spec.loader import load_spec, save_spec


def _minimal_spec() -> Spec:
    return Spec(
        target=TargetUnit(column="revenue"),
        channels=[Channel(name="google", spend_col="google_spend")],
    )


def test_spec_defaults():
    spec = _minimal_spec()
    assert spec.region == "IL"
    assert spec.currency == "ILS"
    assert spec.industry == "other"
    assert spec.fourier_order == 2
    assert spec.sampler.draws == 1000
    assert spec.sampler.chains == 4
    assert spec.brownfield is None


def test_spec_channel_defaults():
    ch = Channel(name="tv", spend_col="tv_spend")
    assert ch.channel_type == "digital"
    assert ch.adstock == "geometric"
    assert ch.saturation == "logistic"
    assert ch.l_max == 8
    assert ch.beta_sigma is None


def test_spec_rejects_unknown_industry():
    with pytest.raises(Exception):
        Spec(
            target=TargetUnit(column="revenue"),
            channels=[],
            industry="unknown_industry",  # type: ignore[arg-type]
        )


def test_save_and_load_roundtrip():
    spec = _minimal_spec()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spec.yaml"
        save_spec(spec, path)
        assert path.exists()
        loaded = load_spec(path)
    assert loaded.region == "IL"
    assert loaded.currency == "ILS"
    assert loaded.channels[0].name == "google"
    assert loaded.channels[0].spend_col == "google_spend"


def test_spec_yaml_structure():
    spec = _minimal_spec()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spec.yaml"
        save_spec(spec, path)
        raw = yaml.safe_load(path.read_text())
    assert "target" in raw
    assert "channels" in raw
    assert raw["region"] == "IL"
    assert raw["currency"] == "ILS"


def test_target_unit_value_per_unit():
    target = TargetUnit(column="installs", type="acquisitions", value_per_unit=50.0)
    assert target.value_per_unit == 50.0
    assert target.type == "acquisitions"
