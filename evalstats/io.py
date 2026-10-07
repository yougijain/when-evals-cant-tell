"""Reading a tidy eval-results file.

One row per (item, system, score). That shape is deliberate: it makes the
pairing explicit and it makes a missing score impossible to overlook, where a
wide matrix quietly lets one system be scored on a different set of items than
another and still look comparable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("item_id", "system", "score")


def load_results(path: str | Path) -> pd.DataFrame:
    """Load and validate a tidy results file into a wide items-by-systems frame.

    Rejects anything that would make a paired comparison meaningless: missing
    columns, a system scored twice on one item, or systems scored on different
    item sets. The last is the quiet one. It leaves two perfectly plausible
    accuracy numbers that are not comparable, and nothing downstream can detect
    it once the scores have been averaged.
    """
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}; "
                         f"expected one row per {REQUIRED}")

    dupes = df.duplicated(["item_id", "system"]).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicate (item_id, system) rows in "
                         f"{path.name}; each system scores each item once")

    wide = df.pivot(index="item_id", columns="system", values="score")
    if wide.isna().any().any():
        incomplete = wide.columns[wide.isna().any()].tolist()
        n_missing = int(wide.isna().sum().sum())
        raise ValueError(
            f"{n_missing} missing scores: {incomplete} were not scored on every "
            f"item. Paired comparison needs one common item set.")

    if not np.isfinite(wide.values).all():
        raise ValueError("scores must be finite numbers")
    return wide
