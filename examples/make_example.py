#!/usr/bin/env python
"""Write examples/eval_results.csv: a worked case for the CLI.

Three prompt variants against a baseline on 300 items. One variant is a real
improvement, two change nothing. Those are the proportions a prompt-engineering
session usually produces, and the point of the example is that the eval is too
small to tell them apart with any confidence.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED, N_ITEMS, BASE_ACC = 11, 300, 0.70
OUT = Path(__file__).resolve().parent / "eval_results.csv"


def _variant(base, rng, up_rate, down_rate):
    v = base.copy()
    wrong, right = np.flatnonzero(base == 0), np.flatnonzero(base == 1)
    v[rng.choice(wrong, rng.binomial(len(wrong), up_rate), replace=False)] = 1
    v[rng.choice(right, rng.binomial(len(right), down_rate), replace=False)] = 0
    return v


def main() -> None:
    rng = np.random.default_rng(SEED)
    base = (rng.random(N_ITEMS) < BASE_ACC).astype(float)
    systems = {
        "baseline": base,
        # Genuinely better: fixes far more than it breaks.
        "few_shot": _variant(base, rng, up_rate=0.40, down_rate=0.05),
        # No real effect, equal churn in both directions.
        "chain_of_thought": _variant(base, rng, up_rate=0.20, down_rate=0.09),
        "reworded": _variant(base, rng, up_rate=0.18, down_rate=0.08),
    }
    rows = [
        {"item_id": f"q{i:03d}", "system": name, "score": scores[i]}
        for name, scores in systems.items()
        for i in range(N_ITEMS)
    ]
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"  wrote {OUT.name}: {N_ITEMS} items x {len(systems)} systems")
    for name, s in systems.items():
        print(f"    {name:<18} {s.mean():.1%}")


if __name__ == "__main__":
    main()
