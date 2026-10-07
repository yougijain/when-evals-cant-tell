#!/usr/bin/env python
"""What an eval of a given size can tell you. Writes RESULTS.md and figures/.

Everything below is computed here. Nothing is typed in.

Run:
    python study.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evalstats import multiplicity, power

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RESULTS = ROOT / "RESULTS.md"

# Two systems five points apart is the comparison a write-up usually reports.
BASE_ACC, GAP = 0.80, 0.05
# How often the two systems disagree. Near-identical systems sit low; systems
# from different families sit high.
DISC_LOW, DISC_HIGH = 0.10, 0.25
ITEM_GRID = [50, 100, 200, 500, 1000, 2000, 5000]

PAPER, INK, INK2, INK3, RULE = "#fdfdfb", "#1b1b1a", "#4a4a46", "#6f6f68", "#d8d6cf"
BLUE, RUST = "#1a5e94", "#b4561f"
DASH = (0, (4, 3))


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9.5, "text.color": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "normal",
        "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.titlepad": 10,
        "axes.labelsize": 9, "axes.labelcolor": INK2,
        "axes.edgecolor": RULE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.facecolor": PAPER,
        "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
        "xtick.color": RULE, "ytick.color": RULE,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "grid.color": RULE, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
        "lines.linewidth": 1.6, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def blind_below(discordance: float) -> int:
    """Eval size below which no difference at all is detectable.

    The largest difference a discordance rate admits is the rate itself, so the
    smallest eval that can detect *that* is the smallest eval that can rule out
    anything. Below it `mde_paired` has no answer to give.
    """
    return power.required_items_paired(discordance, discordance)


def power_table() -> list[dict]:
    rows = []
    for n in ITEM_GRID:
        rows.append({
            "n": n,
            "unpaired": power.power_unpaired(n, BASE_ACC, BASE_ACC + GAP),
            "paired_high": power.power_paired(n, GAP, DISC_HIGH),
            "paired_low": power.power_paired(n, GAP, DISC_LOW),
        })
    return rows


def plot_power(rows) -> None:
    ns = np.array([r["n"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.grid(axis="y")
    ax.axhline(0.80, color=INK3, lw=0.9, ls=DASH)
    ax.text(ns[0], 0.815, "80% power", fontsize=8, color=INK3, va="bottom")
    ax.plot(ns, [r["paired_low"] for r in rows], color=BLUE,
            label=f"paired, systems disagree on {DISC_LOW:.0%}")
    ax.plot(ns, [r["paired_high"] for r in rows], color=RUST,
            label=f"paired, disagree on {DISC_HIGH:.0%}")
    ax.plot(ns, [r["unpaired"] for r in rows], color=INK,
            label="unpaired (the usual analysis)")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n:,}" for n in ns])
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Items in the eval")
    ax.set_ylabel("Chance of detecting the difference")
    ax.set_title(f"Detecting a {GAP:.0%}-point gap "
                 f"({BASE_ACC:.0%} vs {BASE_ACC + GAP:.0%})")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "01_power.png")
    plt.close(fig)


def mde_curve():
    ns = np.unique(np.round(np.logspace(np.log10(30), np.log10(5000), 60)).astype(int))
    out = {d: np.array([power.mde_paired(int(n), d) for n in ns])
           for d in (DISC_LOW, DISC_HIGH)}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.grid(axis="y")
    ax.plot(ns, out[DISC_LOW] * 100, color=BLUE,
            label=f"systems disagree on {DISC_LOW:.0%}")
    ax.plot(ns, out[DISC_HIGH] * 100, color=RUST,
            label=f"disagree on {DISC_HIGH:.0%}")

    # A curve stops where the eval stops being able to rule anything out: below
    # that size not even the largest difference the discordance rate admits
    # would reliably show up. Shade it rather than leaving an unexplained gap.
    defined = ~np.isnan(out[DISC_LOW])
    if not defined.all():
        first = blind_below(DISC_LOW)
        ax.annotate(
            f"below {first} items, a {DISC_LOW:.0%}-disagreement\n"
            f"eval rules out nothing at all",
            xy=(ns[defined][0], out[DISC_LOW][defined][0] * 100),
            xytext=(ns[0] * 1.1, 4.0), fontsize=8, color=INK3,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK3, lw=0.7,
                            shrinkA=4, shrinkB=3))

    ax.set_xscale("log")
    ax.set_xticks(ITEM_GRID)
    ax.set_xticklabels([f"{n:,}" for n in ITEM_GRID])
    ax.set_xlim(ns[0], ns[-1])
    ax.set_ylim(0, 26)
    ax.set_xlabel("Items in the eval")
    ax.set_ylabel("Smallest difference detectable (points)")
    ax.set_title("What an eval of this size could have found")
    ax.legend(loc="upper right")
    fig.savefig(FIG / "02_detectable.png")
    plt.close(fig)
    return ns, out


def selection_curve(n_items=300, ks=(1, 2, 3, 5, 10, 20, 50), trials=1500):
    rows = [multiplicity.selection_optimism(n_items, k, BASE_ACC, DISC_HIGH,
                                            trials=trials, seed=3) for k in ks]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.grid(axis="y")
    ax.axhline(0, color=INK3, lw=0.9, ls=DASH)
    ax.text(ks[-1], 0.0012, "the truth: no variant is better",
            fontsize=8, color=INK3, ha="right", va="bottom")
    mean = [r["mean_apparent_gain"] * 100 for r in rows]
    p90 = [r["p90_apparent_gain"] * 100 for r in rows]
    ax.fill_between(ks, mean, p90, color=RUST, alpha=0.13, linewidth=0)
    ax.plot(ks, p90, color=RUST, lw=1.0, ls=DASH, label="90th percentile")
    ax.plot(ks, mean, color=RUST, label="typical reported gain")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Prompt variants tried")
    ax.set_ylabel("Apparent gain of the best one (points)")
    ax.set_title(f"Searching {n_items} items for a variant that does nothing")
    ax.legend(loc="upper left")
    fig.savefig(FIG / "03_selection.png")
    plt.close(fig)
    return list(zip(ks, rows))


def write_results(rows, mde, selection) -> None:
    ns, mde_vals = mde
    pick = lambda n: int(np.argmin(np.abs(ns - n)))
    L = []
    L.append("# Results — When Evals Can't Tell\n")
    L.append("_Auto-generated by `study.py`. Every number is computed; none is "
             "typed in._\n")

    at100 = next(r for r in rows if r["n"] == 100)
    need_u = power.required_items_unpaired(BASE_ACC, BASE_ACC + GAP)
    need_h = power.required_items_paired(GAP, DISC_HIGH)
    need_l = power.required_items_paired(GAP, DISC_LOW)

    L.append("## Headline figures")
    L.append("| Figure | Value |")
    L.append("|---|---|")
    L.append(f"| Comparison | **{BASE_ACC:.0%}** vs **{BASE_ACC + GAP:.0%}**, "
             f"a {GAP * 100:.0f}-point gap |")
    L.append(f"| Power at 100 items, unpaired | **{at100['unpaired']:.0%}** |")
    L.append(f"| Power at 100 items, paired | "
             f"**{at100['paired_high']:.0%}**–**{at100['paired_low']:.0%}** |")
    L.append(f"| Items needed for 80% power, unpaired | **{need_u:,}** |")
    L.append(f"| Items needed for 80% power, paired | "
             f"**{need_l:,}**–**{need_h:,}** |")
    L.append(f"| Smallest difference 100 items can detect | "
             f"**{mde_vals[DISC_HIGH][pick(100)] * 100:.1f}** points |")
    L.append("")

    L.append("## 1. Most evals cannot see the difference they report\n")
    L.append(f"| Items | Unpaired | Paired, disagree {DISC_HIGH:.0%} | "
             f"Paired, disagree {DISC_LOW:.0%} |")
    L.append("|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['n']:,} | {r['unpaired']:.0%} | "
                 f"{r['paired_high']:.0%} | {r['paired_low']:.0%} |")
    L.append(f"\nAt 100 items the usual analysis notices a {GAP * 100:.0f}-point "
             f"gap {at100['unpaired']:.0%} of the time. Reaching 80% power takes "
             f"**{need_u:,}** items unpaired, or **{need_l:,}** paired when the "
             f"systems disagree on only {DISC_LOW:.0%}.\n")
    L.append("![power](figures/01_power.png)\n")

    L.append("## 2. Pairing helps most when the systems are most alike\n")
    L.append(f"Keeping the pairing is worth {need_u - need_h:,} items when the "
             f"systems disagree on {DISC_HIGH:.0%}, and {need_u - need_l:,} when "
             f"they disagree on {DISC_LOW:.0%}. The information lives in the "
             f"items the systems disagree on, so two near-identical systems are "
             f"the easy case, not the hard one — the opposite of the intuition "
             f"that similar systems are harder to tell apart.\n")

    L.append("## 3. What an eval that found nothing has ruled out\n")
    cell = lambda v: "nothing" if np.isnan(v) else f"{v * 100:.1f} points"
    L.append(f"| Items | Disagree {DISC_HIGH:.0%} | Disagree {DISC_LOW:.0%} |")
    L.append("|---|---|---|")
    for n in (50, 100, 200, 500, 1000, 2000):
        L.append(f"| {n:,} | {cell(mde_vals[DISC_HIGH][pick(n)])} | "
                 f"{cell(mde_vals[DISC_LOW][pick(n)])} |")
    floor = blind_below(DISC_LOW)
    L.append("\n\"No significant difference\" on 100 items rules out differences "
             f"larger than about {mde_vals[DISC_HIGH][pick(100)] * 100:.0f} "
             "points. It does not rule out a difference.\n")
    L.append(f"The {DISC_LOW:.0%} column stops short because a difference "
             f"cannot exceed the rate at which the systems disagree. Under "
             f"{DISC_LOW:.0%} disagreement an eval of fewer than {floor} items "
             f"cannot rule out any difference at all, however the test comes "
             f"out.\n")
    L.append("![detectable](figures/02_detectable.png)\n")

    L.append("## 4. The cost of trying several prompts\n")
    L.append("| Variants tried | Typical gain of the best | 90th percentile | "
             "Share showing a gain |")
    L.append("|---|---|---|---|")
    for k, r in selection:
        L.append(f"| {k} | {r['mean_apparent_gain'] * 100:+.2f}pp | "
                 f"{r['p90_apparent_gain'] * 100:+.2f}pp | "
                 f"{r['share_positive']:.0%} |")
    best_k, best_r = selection[-1]
    L.append(f"\nNone of these variants is better than the baseline. Every gain "
             f"in the table is selection. Try {best_k} and the best one shows "
             f"**{best_r['mean_apparent_gain'] * 100:+.2f}pp** on average and "
             f"clears the baseline in {best_r['share_positive']:.0%} of runs, "
             f"which is the number a write-up reports after keeping the winner.\n")
    L.append("![selection](figures/03_selection.png)\n")

    L.append("## Figures")
    for f in sorted(FIG.glob("*.png")):
        L.append(f"- `figures/{f.name}`")
    RESULTS.write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {RESULTS.name}")


def main() -> None:
    _style()
    FIG.mkdir(exist_ok=True)
    print("Simulating what an eval of a given size can tell you...")
    rows = power_table()
    plot_power(rows)
    mde = mde_curve()
    selection = selection_curve()
    write_results(rows, mde, selection)
    print(f"  figures : {FIG}")


if __name__ == "__main__":
    main()
