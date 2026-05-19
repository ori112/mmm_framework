"""Generate model variants for tournament rounds."""
from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass
class Variant:
    l_max: int
    fourier_order: int
    prior_width_factor: float

    @property
    def name(self) -> str:
        return f"l_max={self.l_max}_fourier={self.fourier_order}_width={self.prior_width_factor}"


def generate_variants(
    l_max_options: list[int] | None = None,
    fourier_options: list[int] | None = None,
    width_options: list[float] | None = None,
    n: int | None = None,
) -> list[Variant]:
    """Generate Cartesian product of hyperparameter options.

    n: if set, limit to at most n variants (sampled uniformly from the grid).
    """
    l_max_options = l_max_options or [4, 8, 13]
    fourier_options = fourier_options or [2, 4]
    width_options = width_options or [0.7, 1.0, 1.5]

    all_variants = [
        Variant(l, f, w)
        for l, f, w in itertools.product(l_max_options, fourier_options, width_options)
    ]

    if n is not None and n < len(all_variants):
        import random
        random.seed(0)
        all_variants = random.sample(all_variants, n)

    return all_variants


def apply_variant_to_spec(spec, variant: Variant):
    """Return a copy of spec with the variant's hyperparameters applied."""
    import copy
    s = copy.deepcopy(spec)
    s.fourier_order = variant.fourier_order
    for ch in s.channels:
        ch.l_max = variant.l_max
    return s
