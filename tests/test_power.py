import numpy as np
import pytest

from evalstats import compare, power


def _simulate_power(n, delta, discordance, alpha=0.05, trials=3000, seed=0):
    """Empirical power of McNemar, to check the closed form against reality."""
    rng = np.random.default_rng(seed)
    p10, p01 = power.discordance_split(delta, discordance)
    hits = 0
    for _ in range(trials):
        cell = rng.choice(3, n, p=[p10, p01, 1 - p10 - p01])
        a = (cell == 1).astype(float)
        b = (cell == 0).astype(float)
        hits += compare.paired(a, b, n_boot=1).p_value < alpha
    return hits / trials


def test_closed_form_power_matches_simulation():
    """The approximation is only worth having if it is close to the truth."""
    for n, delta, disc in [(200, 0.05, 0.25), (500, 0.03, 0.20), (1000, 0.02, 0.30)]:
        assert power.power_paired(n, delta, disc) == pytest.approx(
            _simulate_power(n, delta, disc), abs=0.05), (n, delta, disc)


def test_power_rises_with_items_and_with_disagreement():
    assert power.power_paired(100, 0.05, 0.2) < power.power_paired(1000, 0.05, 0.2)
    # Same accuracy gap, but systems that disagree more often are easier to
    # separate: the informative items are the discordant ones.
    assert power.power_paired(300, 0.05, 0.40) < power.power_paired(300, 0.05, 0.10)


def test_a_difference_cannot_exceed_the_disagreement_rate():
    with pytest.raises(ValueError, match="disagree on at least"):
        power.discordance_split(delta=0.10, discordance=0.05)


def test_no_true_difference_leaves_only_the_false_positive_rate():
    assert power.power_paired(500, 0.0, 0.2) == pytest.approx(0.05)


def test_required_items_is_the_smallest_n_that_works():
    n = power.required_items_paired(0.05, 0.25, target_power=0.80)
    assert power.power_paired(n, 0.05, 0.25) >= 0.80
    assert power.power_paired(n - 1, 0.05, 0.25) < 0.80


def test_mde_is_the_smallest_detectable_difference():
    n, disc = 200, 0.25
    d = power.mde_paired(n, disc)
    assert power.power_paired(n, d, disc) == pytest.approx(0.80, abs=0.01)
    assert power.power_paired(n, d * 0.8, disc) < 0.80


def test_pairing_needs_fewer_items_than_not_pairing():
    """The whole argument, as a sample-size statement.

    Two systems at 80% and 85% that disagree on a quarter of items. Keeping the
    pairing should need materially fewer items than throwing it away.
    """
    paired_n = power.required_items_paired(0.05, 0.25)
    unpaired_n = power.required_items_unpaired(0.80, 0.85)
    assert paired_n < unpaired_n
