"""Point the toolkit at a results file: python -m evalstats results.csv

Prints what the eval supports, in the order the questions should be asked:
what it could have detected at all, then what it found, then what survives
having looked at several variants.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import compare, multiplicity, power
from .io import load_results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evalstats", description=__doc__.splitlines()[0])
    ap.add_argument("results", help="tidy CSV: item_id, system, score")
    ap.add_argument("--baseline", help="system to compare against "
                                       "(default: the first column)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80,
                    help="target power for the detectability line")
    args = ap.parse_args(argv)

    wide = load_results(args.results)
    systems = list(wide.columns)
    base_name = args.baseline or systems[0]
    if base_name not in systems:
        ap.error(f"--baseline {base_name!r} is not one of {systems}")
    variants = {s: wide[s].values for s in systems if s != base_name}
    if not variants:
        ap.error("need at least two systems to compare")

    base = wide[base_name].values
    n = len(base)
    print(f"{n:,} items, {len(systems)} systems, baseline = {base_name}\n")

    if set(np.unique(wide.values)) <= {0.0, 1.0}:
        disc = float(np.mean([(base != v).mean() for v in variants.values()]))
        mde = power.mde_paired(n, max(disc, 1e-6), args.power, args.alpha)
        print(f"What this eval could see: at {args.power:.0%} power it detects a "
              f"difference of {mde * 100:.1f} points or larger.")
        print(f"(systems disagree on {disc:.1%} of items on average)\n")

    results = multiplicity.compare_variants(base, variants, alpha=args.alpha)
    width = max(len(r.name) for r in results)
    print(f"{'system':<{width}}  {'diff':>8}  {'95% CI':>18}  {'p':>8}  "
          f"{'p (BH)':>8}  verdict")
    for r in results:
        ci = f"[{r.ci[0] * 100:+.1f}, {r.ci[1] * 100:+.1f}]"
        verdict = "supported" if r.survives else "not supported"
        print(f"{r.name:<{width}}  {r.diff * 100:>+7.2f}pp  {ci:>18}  "
              f"{r.p_raw:>8.3f}  {r.p_adjusted:>8.3f}  {verdict}")

    n_sig = sum(r.survives for r in results)
    print(f"\n{n_sig} of {len(results)} differences survive Benjamini-Hochberg "
          f"at alpha={args.alpha}.")
    if len(results) > 1 and n_sig == 0:
        print("Reporting the best-looking variant here would be reporting noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
