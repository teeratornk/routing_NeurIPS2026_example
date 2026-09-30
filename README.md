# Reproducing "When to Call a Time-Series Foundation Model"

A standalone example that re-derives the paper's headline numbers from the
released per-request records. No forecaster is run, no gate is refitted and no
GPU is needed: it reads the absolute percentage error of both forecasters at every
request, the deployed gates' stored scores and the thresholds frozen on 2024, all
from `full_archive/`.

This root directory is the example. `full_archive/` is the complete evaluation
archive behind every table in the paper (per-request error records of every arm,
the eight deployed gates, result files, table sources, analysis scripts, manifests
and licences), described in `full_archive/README.md`. Clone the repository to get
both (about 85 MB).

**Python 3.9 to 3.12.** numpy 1.26.4 ships no wheel for 3.13, so a newer Python
tries to compile it from source and fails without a C toolchain. If you see pip
building numpy, you are on the wrong Python.

```bash
python3 -m venv .venv && . .venv/bin/activate   # on Debian/Ubuntu the system
pip install -r requirements.txt                 # Python refuses a bare install
python run_example.py
```

The run finishes in a few seconds. On a many-core server set `OMP_NUM_THREADS=4`.
It ends with:

```
All checks reproduced the published values.
```

and exit code 0. Every number it prints is checked against the published value
in `expected/`, so a mismatch exits non-zero rather than requiring you to spot it.

## What it reproduces

**1. The routing result.** For each of the eight (model, horizon) cells it takes
the 2025 requests, applies the deployed gate's stored score against the threshold
frozen on 2024 (`full_archive/gates/thresholds.csv`), and compares the served error
with the strictest *deployable* fixed baseline: one model per (country, horizon),
chosen on the 2024 errors, which the paper calls **c-static**. The c-static rule is
recomputed here from the 2024 records, and both sets of decisions are checked
against the frozen served record. The eight gains reproduce the published values
to four decimals, and their median is the paper's 0.19 pp headline.

The gate is applied, not refitted. Refitting needs its 22 request-time features,
two of which (`anchor_load`, `recent_load_range`) are load values and are not
redistributed, for the reason given under "What is not here". The eight fitted
gates are in `full_archive/gates/`; the refit's sensitivity to the random seed is
reported in the paper's appendix.

**2. Metric dependence.** The same decisions on the same requests, with the
threshold held where the pooled mean put it, re-scored by the median over
countries of within-country median APE. The gain roughly halves: 0.19 pp becomes
0.10 pp, and one of the eight cells goes slightly negative. The point is that
routing value depends on the metric, not only on the per-request loss.

**3. The long-horizon comparison.** At one and two years, Chronos-2 at its frozen
output-window cap trails the structural forecaster by 3.86 and 5.17 pp. This
compares **fixed choices**: no gate is fitted at those leads, so it says nothing
about whether a selective policy could exploit isolated Chronos-2 wins.

## Layout

    run_example.py     the whole example
    src/routing.py     the metrics and the escalation rule
    expected/          the published values each check is compared against
    full_archive/      the evaluation archive the example reads:
      records/structural/per_origin_structural_{2025,dev,2025_year,2025_h17520}.parquet
                       the structural forecaster's APE per request
      records/fm/tsl*_q_*.parquet, rollout_ablation_h*_cap*.parquet
                       the foundation models' APE per request
      records/served/served_per_request.parquet
                       served errors of every policy, with the deployed gate's score
      gates/           the eight deployed gates and the 2024-frozen thresholds

Requests are assigned to fit, selection and test windows by **target** timestamp,
so no split can hold a request whose target it has not yet reached.

## What is not here

- **Raw load values, forecasts and the two load-valued gate features.** The load
  provider's redistribution terms are not stated on its data page, and a forecast
  beside its error reveals the load, so neither is released. Anyone with access to
  the source data can regenerate the load values and the foundation-model
  forecasts; the runner scripts are in `full_archive/scripts/`.
- **The structural forecaster's discovery and fitting library.** Its per-request
  errors are here, and they are what the accuracy results depend on. It is
  available on reasonable request from the corresponding author.

The two foundation models are public and pinned by revision in the paper's
appendix.

The example is released under the MIT License (see `LICENSE`); the archive's own
licences are in `full_archive/LICENSE-CODE` and `full_archive/LICENSE-DATA`.
