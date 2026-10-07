# When Evals Can't Tell
### How large an LLM eval has to be before its result means anything

[![tests](https://github.com/yougijain/when-evals-cant-tell/actions/workflows/tests.yml/badge.svg)](https://github.com/yougijain/when-evals-cant-tell/actions/workflows/tests.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[Generated results](RESULTS.md) · [The study](study.py) · [The toolkit](evalstats/)

**The question:** you run a 100-item eval. The new prompt scores 85%, the old
one 80%. Do you ship it?

The honest answer is that you cannot tell, and the eval does not say so. This
repo works out how often a result like that is real, what it would take to find
out, and how much of a reported gain is left once you account for having tried
several prompts before reporting one.

---

## TL;DR — the finding

> **A 100-item eval detects a real 5-point gap 15% of the time.** A genuine
> improvement of that size leaves no mark in 85 runs out of 100. Reaching 80%
> power takes **903 items** under the comparison most write-ups run.

| | Items for 80% power | Power at 100 items |
|---|---|---|
| Unpaired, two proportions (what gets reported) | **903** | **15%** |
| Paired, systems disagree on 25% of items | **783** | **17%** |
| Paired, systems disagree on 10% of items | **312** | **35%** |

Two things fall out of that table.

**Pairing is free and almost nobody takes it.** Both systems were run on the
same items, so the comparison is paired by construction, and treating it as two
independent samples throws the pairing away. McNemar's test on the same data
costs nothing to run.

**It helps most when the systems are most alike.** Disagreement on 10% of items
needs 312 items where 25% needs 783. The information lives entirely in the items
the two systems answer differently, so near-identical systems are the easy case,
not the hard one. That is backwards from the usual intuition, and it means the
discordance rate, not the accuracy gap, is what to look at when sizing an eval.

![power](figures/01_power.png)

### What a null result has ruled out

"No significant difference" on 100 items rules out differences larger than about
14 points. Below 77 items, a pair of systems that disagree on 10% of items
cannot be separated at all: no difference in the admissible range would show up,
whichever way the test comes out. Reporting a null from an eval that size says
something about the eval, not about the systems.

### The cost of trying several prompts

Fifty prompt variants, none of them better than the baseline, scored on 300
items. Keep the winner and report its score:

| Variants tried | Apparent gain of the best | Share showing a gain |
|---|---|---|
| 1 | -0.03pp | 46% |
| 5 | +3.30pp | 95% |
| 50 | **+6.50pp** | **100%** |

Every one of those gains is selection. The search is real work and the number it
produces is not, which is what `evalstats.multiplicity` measures and corrects.

![selection](figures/03_selection.png)

---

## Using it on your own eval

One row per (item, system, score), scores binary or continuous:

```csv
item_id,system,score
q001,baseline,1
q001,few_shot,1
q002,baseline,0
q002,few_shot,1
```

```console
$ python -m evalstats examples/eval_results.csv

300 items, 4 systems, baseline = baseline

What this eval could see: at 80% power it detects a difference of 5.6 points or larger.
(systems disagree on 12.0% of items on average)

system                diff              95% CI         p    p (BH)  verdict
few_shot            +6.00pp       [+2.0, +10.0]     0.005     0.015  supported
chain_of_thought    +1.00pp        [-2.7, +4.7]     0.711     0.755  not supported
reworded            +1.00pp        [-3.0, +5.3]     0.755     0.755  not supported

1 of 3 differences survive Benjamini-Hochberg at alpha=0.05.
```

Detectability comes first on purpose. A p-value is unreadable without it: the
same 0.711 means "no effect" from a well-powered eval and "no idea" from an
underpowered one, and only the first line of that output tells you which.

As a library:

```python
from evalstats import compare, power

power.required_items_paired(delta=0.05, discordance=0.25)   # 783
power.mde_paired(n=100, discordance=0.25)                   # 0.138
compare.paired(baseline_scores, variant_scores)             # diff, CI, p
```

## What is in here

| Module | What it answers |
|---|---|
| `evalstats/power.py` | How many items to detect a difference this size, and what this many items could have found |
| `evalstats/compare.py` | Two systems on one eval set: McNemar exact for binary scores, paired *t* for continuous, bootstrap intervals |
| `evalstats/multiplicity.py` | Several variants at once: Benjamini-Hochberg across them, and how much the search alone inflates the winner |
| `evalstats/io.py` | Reads a tidy results file and refuses the shapes that quietly break a paired analysis |
| `study.py` | Runs the above over a grid of eval sizes and writes `RESULTS.md` and `figures/` |

Nothing here calls a model. The input is a results file you already have.

## Reproducing

```console
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest tests/ -q        # 27 tests
python study.py         # rewrites RESULTS.md and figures/, ~5s
```

Every number in `RESULTS.md` is computed by `study.py`; none is typed in. CI
re-runs the study on each push and fails if the file comes out different, so the
claims cannot drift away from the code behind them.

## Scope

The power results are analytic (Connor's 1987 McNemar approximation, Cohen's *h*
for the unpaired case) and assume scores that are independent across items. Evals
built from templated variations of a few seed questions break that assumption and
have less information than their item count suggests, so treat the sample sizes
here as a floor.

The simulations in sections 2 and 4 of `RESULTS.md` use synthetic scores with a
known ground truth, which is the point: the true answer has to be known to show
that the usual analysis misses it.

## Licence

MIT, see [LICENSE](LICENSE).
