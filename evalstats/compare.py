"""Comparing two systems on one eval set, paired and unpaired.

The distinction this module exists for: almost every eval runs both systems on
the *same* items, and almost every write-up then compares them as if the two
accuracy numbers came from independent samples. They did not. Treating paired
data as unpaired throws away the pairing, and the pairing is where most of the
information is, because the two systems get the same easy items right and the
same hard items wrong. What is left after that cancellation - the items they
disagree on - is the entire signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Comparison:
    """One system-versus-system result, with the uncertainty attached."""
    diff: float               # mean(b) - mean(a), in score units
    ci: tuple[float, float]   # 95% interval on that difference
    p_value: float
    n_items: int
    method: str
    n_discordant: int | None = None   # paired binary only

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def _check(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(
            f"paired comparison needs one score per system per item, "
            f"got {a.shape} and {b.shape}")
    if a.ndim != 1:
        raise ValueError(f"expected one score per item, got shape {a.shape}")
    if len(a) == 0:
        raise ValueError("no items to compare")
    return a, b


def _is_binary(*arrays) -> bool:
    return all(np.isin(x, (0.0, 1.0)).all() for x in arrays)


def paired(a, b, n_boot: int = 10_000, seed: int = 0) -> Comparison:
    """Compare two systems scored on the same items, keeping the pairing.

    Binary scores use McNemar's exact test, which conditions on the discordant
    items and asks whether b-right-a-wrong outnumbers a-right-b-wrong. Items
    both systems get right, or both get wrong, carry no information about which
    is better and are correctly ignored.

    Continuous scores use a bootstrap over the per-item differences.
    """
    a, b = _check(a, b)
    rng = np.random.default_rng(seed)
    d = b - a
    # All resamples at once. The loop version costs ~0.2s per call, which is
    # nothing on its own and forty minutes inside a coverage study.
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    if _is_binary(a, b):
        b_wins = int(np.sum((b == 1) & (a == 0)))
        a_wins = int(np.sum((a == 1) & (b == 0)))
        n_disc = b_wins + a_wins
        # Exact binomial on the discordant pairs. With none, the systems agree
        # on every item and there is no evidence either way.
        p = 1.0 if n_disc == 0 else float(
            stats.binomtest(b_wins, n_disc, 0.5).pvalue)
        return Comparison(float(d.mean()), ci, p, len(a),
                          "McNemar exact (paired)", n_disc)

    t = stats.ttest_rel(b, a)
    return Comparison(float(d.mean()), ci, float(t.pvalue), len(a),
                      "paired t-test")


def unpaired(a, b) -> Comparison:
    """Compare the same two systems as if the scores were independent samples.

    This is the wrong analysis for a shared eval set, and it is the one most
    write-ups report. It is here to be measured against `paired`, not used.
    """
    a, b = _check(a, b)
    if _is_binary(a, b):
        # Two-proportion z-test, the usual "85% vs 80%" comparison.
        n = len(a)
        p1, p2 = a.mean(), b.mean()
        pool = (a.sum() + b.sum()) / (2 * n)
        se_pool = np.sqrt(pool * (1 - pool) * 2 / n)
        z = 0.0 if se_pool == 0 else (p2 - p1) / se_pool
        p = float(2 * stats.norm.sf(abs(z)))
        se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
        method = "two-proportion z (unpaired)"
    else:
        t = stats.ttest_ind(b, a, equal_var=False)
        p = float(t.pvalue)
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        method = "Welch t-test (unpaired)"

    diff = float(b.mean() - a.mean())
    half = 1.96 * se
    return Comparison(diff, (diff - half, diff + half), p, len(a), method)
