import numpy as np
import pytest

from evalstats import compare


def _paired_binary(n=400, p_both=0.70, p_b_only=0.10, p_a_only=0.05, seed=0):
    """An eval where both systems see the same items.

    p_both      both right        (carries no information about which is better)
    p_b_only    only b is right   (evidence for b)
    p_a_only    only a is right   (evidence for a)
    remainder   both wrong        (also carries no information)
    """
    rng = np.random.default_rng(seed)
    cell = rng.choice(4, n, p=[p_both, p_b_only, p_a_only,
                               1 - p_both - p_b_only - p_a_only])
    a = np.isin(cell, [0, 2]).astype(float)
    b = np.isin(cell, [0, 1]).astype(float)
    return a, b


def test_paired_recovers_the_difference():
    a, b = _paired_binary()
    r = compare.paired(a, b)
    assert r.diff == pytest.approx(b.mean() - a.mean())
    assert r.ci[0] < r.diff < r.ci[1]
    assert r.n_items == len(a)


def test_mcnemar_ignores_the_items_both_systems_agree_on():
    """The concordant items are not evidence, and must not move the p-value.

    This is the property that makes the paired test more powerful. If adding
    items both systems get right changed the answer, the test would be
    measuring difficulty rather than difference.
    """
    a, b = _paired_binary()
    p_before = compare.paired(a, b).p_value
    pad = np.ones(500)
    p_after = compare.paired(np.r_[a, pad], np.r_[b, pad]).p_value
    assert p_after == pytest.approx(p_before)


def test_paired_beats_unpaired_on_the_same_data():
    """Same items, same scores, two analyses. The pairing is worth keeping."""
    a, b = _paired_binary()
    assert compare.paired(a, b).p_value < compare.unpaired(a, b).p_value


def test_no_discordant_items_is_no_evidence():
    x = np.array([1.0, 1, 0, 0, 1])
    r = compare.paired(x, x.copy())
    assert r.n_discordant == 0
    assert r.p_value == 1.0
    assert r.diff == 0.0


def test_continuous_scores_use_a_paired_t_test():
    rng = np.random.default_rng(1)
    a = rng.normal(3.0, 1.0, 300)
    b = a + rng.normal(0.4, 0.5, 300)
    r = compare.paired(a, b)
    assert "paired t" in r.method
    assert r.diff == pytest.approx(0.4, abs=0.1)
    assert r.p_value < 0.001


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="one score per system per item"):
        compare.paired([1, 0, 1], [1, 0])


def test_bootstrap_interval_covers_the_truth_at_the_nominal_rate():
    """The interval is only worth printing if it means what it says.

    400 evals with a known 5-point true difference; the 95% interval should
    contain it about 95% of the time. This is the check that makes every CI
    elsewhere in the repo trustworthy, and it is why the bootstrap is
    vectorised - the loop version makes this study take forty minutes.
    """
    rng = np.random.default_rng(7)
    delta, disc, n = 0.05, 0.25, 400
    p10, p01 = (disc + delta) / 2, (disc - delta) / 2
    covered = 0
    for _ in range(400):
        cell = rng.choice(3, n, p=[p10, p01, 1 - p10 - p01])
        a = (cell == 1).astype(float)
        b = (cell == 0).astype(float)
        lo, hi = compare.paired(a, b, n_boot=2000, seed=int(rng.integers(1e6))).ci
        covered += lo <= delta <= hi
    assert 0.92 <= covered / 400 <= 0.98, covered / 400
