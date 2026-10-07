import numpy as np
import pytest

from evalstats import multiplicity


def _variant(baseline, flip_to_right, flip_to_wrong, rng):
    """A variant that fixes some baseline errors and breaks some successes."""
    v = baseline.copy()
    wrong = np.flatnonzero(baseline == 0)
    right = np.flatnonzero(baseline == 1)
    v[rng.choice(wrong, flip_to_right, replace=False)] = 1
    v[rng.choice(right, flip_to_wrong, replace=False)] = 0
    return v


def _null_variant(baseline, rng, rate=0.40):
    """A variant with the same true accuracy as the baseline, differing by noise.

    Flipping a fixed count each way would make every variant's difference
    exactly zero and no search could ever select on it. The counts are drawn
    with equal expectation instead, so the true difference is zero while the
    realised one varies - which is the thing a search over variants picks from.
    """
    v = baseline.copy()
    wrong = np.flatnonzero(baseline == 0)
    right = np.flatnonzero(baseline == 1)
    expected = rate * min(len(wrong), len(right))
    v[rng.choice(wrong, rng.binomial(len(wrong), expected / len(wrong)),
                 replace=False)] = 1
    v[rng.choice(right, rng.binomial(len(right), expected / len(right)),
                 replace=False)] = 0
    return v


def test_a_real_improvement_survives_correction():
    rng = np.random.default_rng(0)
    base = (rng.random(2000) < 0.70).astype(float)
    variants = {"real": _variant(base, 220, 20, rng)}
    res = multiplicity.compare_variants(base, variants)
    assert res[0].survives and res[0].diff > 0


def test_correction_removes_the_false_positives_a_search_generates():
    """The reason the correction is here, measured rather than demonstrated once.

    Twenty variants per run, none of them a real improvement. Asserting against
    a single run would mean picking a seed where an uncorrected false positive
    happened to appear, which is the selection this whole project is about. So
    the rate is measured across runs instead: uncorrected, a few per cent of
    variants clear p < 0.05 and would be written up; after Benjamini-Hochberg,
    essentially none do.

    n_boot=1 because only the p-values are under test here and the bootstrap
    interval would dominate the runtime.
    """
    raw_hits = survivors = total = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        base = (rng.random(600) < 0.70).astype(float)
        variants = {f"v{i}": _null_variant(base, rng) for i in range(20)}
        res = multiplicity.compare_variants(base, variants, n_boot=1)
        raw_hits += sum(v.p_raw < 0.05 for v in res)
        survivors += sum(v.survives for v in res)
        total += len(res)

    # Uncorrected, the search does produce findings. McNemar's exact test is
    # conservative, so the rate sits at or below the nominal 5%.
    assert 0 < raw_hits / total <= 0.08, raw_hits / total
    # Corrected, it does not.
    assert survivors / total < 0.01, survivors / total
    assert survivors < raw_hits


def test_results_are_ordered_by_adjusted_p():
    rng = np.random.default_rng(2)
    base = (rng.random(800) < 0.70).astype(float)
    variants = {"a": _variant(base, 90, 10, rng), "b": _variant(base, 40, 40, rng)}
    res = multiplicity.compare_variants(base, variants)
    assert [v.p_adjusted for v in res] == sorted(v.p_adjusted for v in res)


def test_searching_more_variants_inflates_the_winner():
    """Selection optimism grows with the size of the search, not the data."""
    one = multiplicity.selection_optimism(200, 1, 0.70, 0.25, trials=400)
    ten = multiplicity.selection_optimism(200, 10, 0.70, 0.25, trials=400)
    assert ten["mean_apparent_gain"] > one["mean_apparent_gain"] > -0.01
    assert ten["share_positive"] > 0.95


def test_a_single_variant_is_unbiased():
    """With no search there is nothing to select on, so the gain centres on zero."""
    r = multiplicity.selection_optimism(500, 1, 0.70, 0.25, trials=800)
    assert r["mean_apparent_gain"] == pytest.approx(0.0, abs=0.005)


def test_no_variants_is_an_error():
    with pytest.raises(ValueError, match="no variants"):
        multiplicity.compare_variants(np.ones(10), {})
