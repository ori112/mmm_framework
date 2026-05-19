"""Interactive intake questionnaire → spec.yaml."""
from __future__ import annotations

from pathlib import Path

from .loader import save_spec
from .schema import Channel, DataCfg, Spec, SamplerCfg, TargetUnit

_INDUSTRIES = ["retail", "automotive", "insurance", "telco", "saas", "other"]
_TARGET_TYPES = ["revenue", "acquisitions", "volume"]


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or default


def _ask_choice(prompt: str, choices: list[str], default: str = "") -> str:
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice}")
    while True:
        raw = _ask(prompt, default)
        if raw in choices:
            return raw
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            pass
        print(f"  Please choose one of: {', '.join(choices)}")


def build_spec_from_answers(
    quick: bool = False,
    output: Path | None = None,
) -> Spec:
    """Run interactive intake and write spec.yaml. Returns the validated Spec."""
    from agent_mmm.workspace.paths import spec_path

    print("\n=== MMM Intake" + (" — Quick Mode (5 steps)" if quick else " — Full Mode") + " ===\n")

    # Step 1 – data file
    data_path = _ask("Path to your data file (CSV)", "data/sales.csv")
    date_col = _ask("Date column name", "date")

    # Step 2 – target KPI
    target_col = _ask("Target KPI column name", "revenue")
    print("Target type:")
    target_type = _ask_choice("Select", _TARGET_TYPES, "revenue")
    value_per_unit = 1.0
    if target_type in ("acquisitions", "volume"):
        vpu = _ask("Revenue value per unit (for ROAS equivalence)", "1.0")
        value_per_unit = float(vpu)

    # Step 3 – channels
    raw_channels = _ask("Channel names (comma-separated)", "google,facebook,tv")
    channel_names = [c.strip() for c in raw_channels.split(",") if c.strip()]
    channels: list[Channel] = []
    for ch in channel_names:
        spend_col = _ask(f"  Spend column for '{ch}'", f"{ch}_spend")
        channels.append(Channel(name=ch, spend_col=spend_col))

    # Step 4 – company context
    company = _ask("Company name (optional)", "Unknown")

    if quick:
        industry = "other"
        sampler = SamplerCfg()
        fourier_order = 2
    else:
        # Step 5 – industry (for IL seasonality presets)
        print("Industry (used for Israeli seasonality presets):")
        industry = _ask_choice("Select", _INDUSTRIES, "other")

        # Step 6 – adstock / saturation per channel
        print("\nChannel configuration (press Enter to accept defaults):")
        updated: list[Channel] = []
        for ch in channels:
            print(f"\n  Channel: {ch.name}")
            ch_type = _ask_choice("    Type", ["digital", "offline"], ch.channel_type)
            ads = _ask_choice("    Adstock", ["geometric", "delayed"], ch.adstock)
            sat = _ask_choice("    Saturation", ["logistic", "hill"], ch.saturation)
            l_max = int(_ask("    Max lag (l_max)", str(ch.l_max)))
            updated.append(ch.model_copy(update=dict(
                channel_type=ch_type, adstock=ads, saturation=sat, l_max=l_max
            )))
        channels = updated

        # Step 7 – sampler
        print("\nSampler configuration:")
        draws = int(_ask("  MCMC draws", "1000"))
        tune = int(_ask("  MCMC tune", "1000"))
        chains = int(_ask("  MCMC chains", "4"))
        seed_raw = _ask("  Random seed (blank = none)", "")
        sampler = SamplerCfg(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=int(seed_raw) if seed_raw else None,
        )
        fourier_order = int(_ask("Fourier seasonality order", "2"))

    spec = Spec(
        company=company,
        industry=industry,  # type: ignore[arg-type]
        data=DataCfg(source="csv", path=data_path, date_col=date_col),
        target=TargetUnit(
            column=target_col,
            type=target_type,  # type: ignore[arg-type]
            value_per_unit=value_per_unit,
        ),
        channels=channels,
        sampler=sampler,
        fourier_order=fourier_order,
    )

    out = output or spec_path()
    save_spec(spec, out)
    print(f"\n  Spec written to {out}")
    return spec
