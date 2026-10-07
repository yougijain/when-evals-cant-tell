import numpy as np
import pandas as pd
import pytest

from evalstats.__main__ import main
from evalstats.io import load_results


def _write(tmp_path, rows):
    path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _tidy(n=200, systems=("base", "variant"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        for s in systems:
            rows.append({"item_id": f"q{i}", "system": s,
                         "score": float(rng.random() < 0.7)})
    return rows


def test_loads_a_tidy_file_into_items_by_systems(tmp_path):
    wide = load_results(_write(tmp_path, _tidy()))
    assert wide.shape == (200, 2)
    assert list(wide.columns) == ["base", "variant"]


def test_missing_columns_are_named(tmp_path):
    path = _write(tmp_path, [{"item_id": "q1", "model": "a", "score": 1}])
    with pytest.raises(ValueError, match=r"missing columns \['system'\]"):
        load_results(path)


def test_a_system_scored_twice_on_one_item_is_rejected(tmp_path):
    rows = _tidy(n=5) + [{"item_id": "q0", "system": "base", "score": 1.0}]
    with pytest.raises(ValueError, match="duplicate"):
        load_results(_write(tmp_path, rows))


def test_systems_scored_on_different_items_are_rejected(tmp_path):
    """The quiet failure this loader exists to catch.

    Two averages over two different item sets are not comparable, and nothing
    downstream can tell once the scores have been averaged.
    """
    rows = [r for r in _tidy(n=20) if not (r["item_id"] == "q3"
                                           and r["system"] == "variant")]
    with pytest.raises(ValueError, match="not scored on every item"):
        load_results(_write(tmp_path, rows))


def test_cli_reports_detectability_and_a_verdict(tmp_path, capsys):
    assert main([str(_write(tmp_path, _tidy())), "--baseline", "base"]) == 0
    out = capsys.readouterr().out
    assert "200 items" in out
    assert "it detects a difference of" in out
    assert "Benjamini-Hochberg" in out


def test_cli_rejects_an_unknown_baseline(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main([str(_write(tmp_path, _tidy())), "--baseline", "nope"])
