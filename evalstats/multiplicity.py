"""Many variants at once, and the cost of reporting whichever won.

Prompt engineering is a search over variants, and the eval set is usually asked
to do two jobs at once: pick the winner, and say how good the winner is. It
cannot do both. The variant that scores highest on a given eval set is the one
whose errors happened to fall most favourably on that set, so its margin over
the baseline is biased upward even when nothing really improved.

Two corrections live here. Benjamini-Hochberg controls the false discovery rate
across the variants tested, which stops the best of twenty coin flips being
reported as a finding. selection_optimism() measures the inflation directly, by
running the search on variants that are all truly identical to the baseline and
seeing how large a win it produces anyway.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.stats.multitest import multipletests

from .compare import paired


@dataclass(frozen=True)
class VariantResult:
    name: str
    diff: float
    ci: tuple[float, float]
    p_raw: float
    p_adjusted: float
    survives: bool


def compare_variants(baseline, variants: dict, alpha: float = 0.05,
                     n_boot: int = 10_000, seed: int = 0) -> list[VariantResult]:
    """Each variant against the baseline, with Benjamini-Hochberg control.

    Returned worst-adjusted-p last, so the list reads as a ranking. `survives`
    is the only column that should be quoted as a finding.
    """
    if not variants:
        raise ValueError("no variants to compare")
    names = list(variants)
    results = [paired(baseline, variants[k], n_boot=n_boot, seed=seed)
               for k in names]
    raw = [r.p_value for r in results]
    keep, adj, _, _ = multipletests(raw, alpha=alpha, method="fdr_bh")
    out = [VariantResult(n, r.diff, r.ci, r.p_value, float(a), bool(k))
           for n, r, a, k in zip(names, results, adj, keep)]
    return sorted(out, key=lambda v: v.p_adjusted)


def selection_optimism(n_items: int, n_variants: int, base_accuracy: float,
                       discordance: float, trials: int = 2_000,
                       seed: int = 0) -> dict:
    """How big a win the best of k identical variants shows, by simulation.

    Every variant here is drawn with the *same* true accuracy as the baseline,
    so the honest answer is always zero. Whatever the best-looking variant
    scores above baseline is pure selection: the number a prompt-engineering
    write-up would report after trying k things and keeping the winner.
    """
    if n_variants < 1:
        raise ValueError("need at least one variant")
    rng = np.random.default_rng(seed)
    half = discordance / 2        # equal disagreement both ways => no true edge
    best = np.empty(trials)
    for t in range(trials):
        # cell 0: variant right, baseline wrong. cell 1: the reverse.
        cells = rng.choice(3, size=(n_variants, n_items),
                           p=[half, half, 1 - discordance])
        diffs = (cells == 0).mean(axis=1) - (cells == 1).mean(axis=1)
        best[t] = diffs.max()
    return {
        "n_items": n_items,
        "n_variants": n_variants,
        "mean_apparent_gain": float(best.mean()),
        "p90_apparent_gain": float(np.percentile(best, 90)),
        "share_positive": float((best > 0).mean()),
    }
