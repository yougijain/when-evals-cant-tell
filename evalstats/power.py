"""How many items an eval needs before it can see a difference that size.

The question this module answers is the one that should be asked before an eval
is run and almost never is: given two systems that really do differ by some
amount, how often would an eval of n items notice?

Paired power is parameterised by the *discordance rate*, the share of items on
which the two systems disagree. That is the quantity that governs how much a
paired eval can tell you, and unlike a correlation it is something you can read
straight off a results file. Two systems with the same accuracy gap are far
easier to separate when they disagree often than when they disagree rarely.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1 - alpha / 2))


def cohens_h(p1: float, p2: float) -> float:
    """Arcsine-transformed effect size for two proportions."""
    return float(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))


def power_unpaired(n: int, p1: float, p2: float, alpha: float = 0.05) -> float:
    """Power of the two-proportion test, n items scored by each system.

    This is the power of the analysis most eval write-ups actually run.
    """
    h = abs(cohens_h(p1, p2))
    return float(stats.norm.cdf(h * np.sqrt(n / 2) - _z(alpha)))


def discordance_split(delta: float, discordance: float) -> tuple[float, float]:
    """Split a discordance rate into the two one-sided disagreement rates.

    `discordance` is P(the systems disagree on an item); `delta` is the true
    accuracy difference. The two are not independent: a difference of delta
    cannot arise unless the systems disagree at least that often, which is the
    constraint this enforces.
    """
    if discordance < abs(delta):
        raise ValueError(
            f"a true difference of {delta:.3f} needs the systems to disagree "
            f"on at least {abs(delta):.1%} of items, not {discordance:.1%}")
    return (discordance + delta) / 2, (discordance - delta) / 2


def power_paired(n: int, delta: float, discordance: float,
                 alpha: float = 0.05) -> float:
    """Power of McNemar's test on n paired items (Connor 1987 approximation).

    Only the discordant items carry information, so power is driven by
    n * discordance rather than by n alone.
    """
    if delta == 0:
        return alpha
    p10, p01 = discordance_split(delta, discordance)
    s = p10 + p01
    num = abs(delta) * np.sqrt(n) - _z(alpha) * np.sqrt(s)
    den = np.sqrt(s - delta ** 2)
    return float(stats.norm.cdf(num / den))


def _search_n(power_fn, target: float, hi: int = 2_000_000) -> int:
    """Smallest n reaching `target` power, by bisection on a monotone curve."""
    if power_fn(hi) < target:
        raise ValueError(f"even n={hi:,} does not reach {target:.0%} power")
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if power_fn(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def required_items_unpaired(p1: float, p2: float, target_power: float = 0.80,
                            alpha: float = 0.05) -> int:
    return _search_n(lambda n: power_unpaired(n, p1, p2, alpha), target_power)


def required_items_paired(delta: float, discordance: float,
                          target_power: float = 0.80,
                          alpha: float = 0.05) -> int:
    return _search_n(lambda n: power_paired(n, delta, discordance, alpha),
                     target_power)


def mde_paired(n: int, discordance: float, target_power: float = 0.80,
               alpha: float = 0.05, tol: float = 1e-5) -> float:
    """Smallest true difference an n-item paired eval detects at `target_power`.

    The honest headline for an eval that found nothing: not "no difference",
    but "no difference larger than this would reliably have shown up".
    """
    lo, hi = 0.0, discordance
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if power_paired(n, mid, discordance, alpha) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi
