"""Assemble the anonymised reproducibility supplement for the TS-LIMITS paper.

The bundle was hand-assembled once and drifted within a week: `tab_estimands`
was regenerated after a cross-table interval fix, `tab_countrybreak` was added
to the paper and never copied in, and the README promised a development-window
record file the archive did not contain. The checklist tells reviewers every
number is reproducible from this archive, so a stale copy is a false claim.

Hence a script. The file list lives here beside the README text, so adding a
table to the paper and forgetting the archive is a --check failure rather than
something a reviewer discovers.

    .venv/bin/python scripts/build_tslimits_supplement.py           # build
    .venv/bin/python scripts/build_tslimits_supplement.py --check   # verify only

--check exits 0 when the archive matches the paper and the declared inputs, and
1 otherwise. Build writes reports/tslimits/supplement/ and the .tar.gz beside it,
computing MANIFEST last so it cannot describe a tree that changed after it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "reports" / "tslimits"
OUT = SRC / "supplement"
TARBALL = SRC / "supplement_anonymous.tar.gz"
PAPER = REPO.parent / "NeurIPS_ts"

# Frozen per-request predictions. Named individually: a glob would silently
# absorb whatever a future run drops into reports/tslimits.
RECORDS = (
    "per_origin_structural_2025.parquet",
    "per_origin_structural_2025_year.parquet",
    "per_origin_structural_2025_h17520.parquet",
    "per_origin_structural_2025_h26280.parquet",
    "per_origin_structural_dev.parquet",
    # sufficient statistics for the cluster bootstrap, so interval variants
    # can be recomputed without refitting anything
    "cluster_matrices.npz",
    "served_per_request.parquet",
)

# Record families that are one file per country or per rollout cap. Each carries
# the count it must have, so a missing country fails the build.
RECORD_GLOBS = (
    ("fm/tsl_chronos_q_*.parquet", 19),
    ("fm/tsl_chronos_year_q_*.parquet", 19),
    ("fm/tsl_chronos_h17520_q_*.parquet", 19),
    ("fm/tsl_chronos_h26280_q_*.parquet", 19),
    ("fm/tsldev_chronos_q_*.parquet", 19),
    # TimesFM writes one file per (country, horizon) rather than one per country.
    ("fm/tsl_timesfm_q_*.parquet", 76),
    ("fm/tsldev_timesfm_q_*.parquet", 76),
    ("rollout_ablation_h720.parquet", 1),
    ("rollout_ablation_h8760_cap*.parquet", 4),
    ("rollout_ablation_h17520_cap*.parquet", 1),
    ("rollout_sel2024_h8760_cap*.parquet", 5),
    ("rollout_sel2024_h17520_cap*.parquet", 5),
)

# Scored outputs and the generated .tex the paper includes.
TABLES = (
    "baselines.csv",
    "bootstrap_ci.csv",
    "bootstrap_ci_panel_median.csv",
    "bootstrap_ci_pooled_mean.csv",
    "by_year.csv",
    "ceiling_blocked.csv",
    "ceiling_blocked_null.csv",
    "ceiling_classes.csv",
    "ceiling_conditional.csv",
    "ceiling_privileged.csv",
    "country_breakdown.csv",
    "country_static.csv",
    "country_static_ridge.csv",
    "country_static_shallow.csv",
    "interval_variants.csv",
    "context_sensitivity.csv",
    "context_sensitivity_timesfm.csv",
    "latency_endtoend.csv",
    "dateblock_ci.csv",
    "dateblock_ci_28d.csv",
    "headroom.csv",
    "aggregation_2x2.csv",
    "longlead_diagnostics.csv",
    "cluster_leverage.csv",
    "frontier.csv",
    "gate_frontier.csv",
    "gate_policies.csv",
    "headroom_2025.csv",
    "noise_floor.csv",
    "paired_aggregate.csv",
    "paired_estimand.csv",
    "per_origin_structural_gates.csv",
    "per_origin_structural_gates_dev.csv",
    "per_origin_structural_gates_2025.csv",
    "per_origin_structural_gates_2025_year.csv",
    "per_origin_structural_gates_2025_h17520.csv",
    "per_origin_structural_gates_2025_h26280.csv",
    "seed_sensitivity.csv",
    "tail_robustness.csv",
    "why_concentration.csv",
    "why_diagnostics.csv",
    "year_scale.csv",
)

# Measurement records that carry a hostname and must be scrubbed.
TABLES_JSON = (
    "energy_probe.json",
    "energy_inference_chronos.json",
    "energy_inference_timesfm.json",
    "router_cost.json",
)

# The .tex the paper \inputs. Every one is checked byte-for-byte against the
# paper directory, which is what catches drift.
TABLES_TEX = (
    "tab_byyear.tex",
    "tab_capselect.tex",
    "tab_controls.tex",
    "tab_countrybreak.tex",
    "tab_energy.tex",
    "tab_intervals.tex",
    "tab_context.tex",
    "tab_latency.tex",
    "tab_noweather.tex",
    "tab_context_timesfm.tex",
    "tab_agg2x2.tex",
    "tab_longlead.tex",
    "tab_estimands.tex",
    "tab_main.tex",
    "tab_paired.tex",
    "tab_threshold.tex",
    "tab_why.tex",
    "tab_year.tex",
    "tab_dst.tex",
    "tab_longlead_cstatic.tex",
    "tab_gate_pareto.tex",
    "tab_nogdp.tex",
    "tab_replication2026.tex",
    "tab_thrstab.tex",
    "tab_selection.tex",
    "tab_covariates.tex",
    "tab_policy2x2.tex",
    "tab_native_family.tex",
    "tab_capselect8192.tex",
)

# Tables the paper writes inline rather than \input. Shipped for completeness;
# no paper file to compare them against.
TABLES_TEX_STANDALONE = ("tab_gap.tex", "tab_policies.tex")

SCRIPTS = (
    "tslimits_bootstrap.py",
    "tslimits_by_year.py",
    "tslimits_country_static.py",
    "tslimits_fig_estimand.py",
    "tslimits_interval_variants.py",
    "tslimits_context.py",
    "tslimits_latency.py",
    "tslimits_dateblock.py",
    "tslimits_headroom.py",
    "tslimits_seed_sensitivity.py",
    "tslimits_fig_ladder.py",
    "tslimits_paired_estimand.py",
    "tslimits_rollout_ablation.py",
    "tslimits_router_cost.py",
    "tslimits_tables.py",
    "tslimits_tail_robustness.py",
    "tslimits_year_scale.py",
    "tslimits_threshold_stability.py",
    "tslimits_selection_bootstrap.py",
)

# Keys in the measurement JSON that identify a machine or a scheduler run.
SCRUB_KEYS = ("host", "hostname", "node", "nodelist", "job_id", "jobid", "slurm_job_id", "user")
REDACTED = "<anonymised>"

README = """# Anonymised reproducibility supplement

Everything needed to reproduce every accuracy number, table and figure in the
paper from frozen records, without re-running any model and without the
structural forecaster's discovery and fitting library, which is not part of this
release.

## What is here

    records/    frozen per-request predictions
      per_origin_structural_2025.parquet          h in {1,24,168,720}, 27,735 requests
      per_origin_structural_2025_year.parquet     h=8760,  6,935 requests
      per_origin_structural_2025_h17520.parquet   h=17520, 6,935 requests
      per_origin_structural_2025_h26280.parquet   h=26280
      per_origin_structural_dev.parquet           2018-2024 development window
      cluster_matrices.npz                        per-country bootstrap matrices
      fm/tsl_chronos_q_*.parquet                  Chronos-2, paired horizons
      fm/tsl_timesfm_q_*_h*.parquet               TimesFM-2.5, one file per horizon
      fm/tsl_chronos_year_q_*.parquet             Chronos-2, h=8760, library default
      fm/tsl_chronos_h17520_q_*.parquet           Chronos-2, h=17520, library default
      fm/tsl_chronos_h26280_q_*.parquet           Chronos-2, h=26280, library default
      fm/tsldev_chronos_q_*.parquet               Chronos-2, development window
      fm/tsldev_timesfm_q_*_h*.parquet            TimesFM-2.5, development window
      rollout_ablation_h720.parquet               the rollout-partition ablation
      rollout_ablation_h{8760,17520}_cap*.parquet 2025 arms at each cap
      rollout_sel2024_h{8760,17520}_cap*.parquet  the 2024 cap-selection sweep

    tables/     every scored CSV and generated .tex the paper includes,
                plus the energy and router-cost measurement records

    scripts/    the generators that turn records/ into tables/

## Reproducing the paper

Each table is regenerated by a named script:

    tslimits_tables.py          Tables 1-6 and the appendix tables
    tslimits_year_scale.py      the horizon ladder and its intervals
    tslimits_bootstrap.py       country-cluster intervals for the routing cells
    tslimits_interval_variants.py BCa, Bonferroni and per-country leverage
    tslimits_seed_sensitivity.py the five-seed range
    tslimits_paired_estimand.py the paired estimand-difference test
    tslimits_country_static.py  the country-control policy levels and the
                                per-country breakdown
    tslimits_by_year.py         the rolling per-test-year replication
    tslimits_tail_robustness.py trimming, winsorising, load weighting
    tslimits_rollout_ablation.py the rollout-partition arms (needs a GPU)
    tslimits_fig_estimand.py    Figure 1
    tslimits_fig_ladder.py      the horizon-ladder figure
    tslimits_router_cost.py     the gate's size and serving latency

Only `tslimits_rollout_ablation.py` needs a GPU, and only to regenerate records
that are already in `records/`. Everything else is CPU and reads frozen records.

## What is not here

The discovery and fitting library for the structural forecaster. The records in
`records/` are its frozen outputs, which is what the accuracy results in this
paper depend on. The two foundation models are public and pinned by revision in
the paper's appendix.

## Anonymisation

Cluster hostnames and scheduler job identifiers have been replaced with
`<anonymised>` in the measurement JSON. Hardware identifiers the paper reports
(CPU and GPU model) are kept, since the latency and energy numbers are
meaningless without them. No author names, affiliations, email addresses or
absolute filesystem paths appear anywhere in this archive.
"""


def _scrub(obj):
    """Replace machine and scheduler identifiers anywhere in a JSON document."""
    if isinstance(obj, dict):
        return {
            k: (REDACTED if k.lower() in SCRUB_KEYS and isinstance(v, str) else _scrub(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _resolve_globs() -> tuple[list[Path], list[str]]:
    """Expand the record families and report any whose count is wrong."""
    files: list[Path] = []
    problems: list[str] = []
    for pattern, expected in RECORD_GLOBS:
        hits = sorted(SRC.glob(pattern))
        if len(hits) != expected:
            problems.append(f"{pattern}: found {len(hits)}, expected {expected}")
        files.extend(hits)
    return files, problems


def check() -> int:
    """Verify the built archive still matches the paper and the declared inputs."""
    problems: list[str] = []

    if not OUT.is_dir():
        print(f"[supp] no archive at {OUT}; run without --check to build it")
        return 1

    for name in TABLES_TEX:
        paper, bundled = PAPER / name, OUT / "tables" / name
        if not paper.is_file():
            problems.append(f"{name}: not in the paper directory")
        elif not bundled.is_file():
            problems.append(f"{name}: in the paper, missing from the archive")
        elif paper.read_bytes() != bundled.read_bytes():
            problems.append(f"{name}: archive copy differs from the paper's")

    # A table added to the paper and never declared here is the drift that
    # started this script, so look for it from the paper's side too.
    declared = set(TABLES_TEX) | set(TABLES_TEX_STANDALONE)
    for paper_tex in sorted(PAPER.glob("tab_*.tex")):
        if paper_tex.name not in declared:
            problems.append(f"{paper_tex.name}: in the paper, not declared in this script")

    for name in RECORDS:
        if not (OUT / "records" / name).is_file():
            problems.append(f"records/{name}: missing from the archive")
    _, glob_problems = _resolve_globs()
    problems.extend(glob_problems)

    for name in TABLES_JSON:
        bundled = OUT / "tables" / name
        if not bundled.is_file():
            problems.append(f"tables/{name}: missing from the archive")
            continue
        raw = bundled.read_text()
        for key in SCRUB_KEYS:
            if f'"{key}"' in raw and REDACTED not in raw:
                problems.append(f"tables/{name}: carries a {key} field that was not scrubbed")

    if problems:
        print(f"[supp] FAIL: {len(problems)} problem(s)")
        for p in problems:
            print(f"  {p}")
        print("[supp] rebuild with: .venv/bin/python scripts/build_tslimits_supplement.py")
        return 1

    print(f"[supp] OK: archive matches the paper ({len(TABLES_TEX)} tables verified byte-for-byte)")
    return 0


# Raw load values and forecasts are not redistributed (a forecast beside an APE
# reveals the load), so the record copies keep the evaluation quantities only,
# the same rule as build_tslimits_archive.py.
LOAD_REVEALING = ("y_true", "y_pred", "actual", "anchor_load", "recent_load_range")


def _strip(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    if "ape" not in d.columns and {"q50", "actual"} <= set(d.columns):
        d["ape"] = 100.0 * (d["q50"] - d["actual"]).abs() / d["actual"].abs()
    drop = [c for c in d.columns if c in LOAD_REVEALING or re.fullmatch(r"q\d+", c)]
    return d.drop(columns=drop)


def _copy_record(src: Path, dest: Path) -> None:
    """Parquet records are copied without load-revealing columns; other files verbatim."""
    if src.suffix == ".parquet":
        _strip(src).to_parquet(dest, index=False)
    else:
        shutil.copy2(src, dest)


def build() -> int:
    record_globs, problems = _resolve_globs()
    if problems:
        print("[supp] FAIL: record families are incomplete")
        for p in problems:
            print(f"  {p}")
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("records/fm", "tables", "scripts"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    n = 0
    for name in RECORDS:
        _copy_record(SRC / name, OUT / "records" / name)
        n += 1
    for path in record_globs:
        dest = OUT / "records" / ("fm" if path.parent.name == "fm" else "") / path.name
        _copy_record(path, dest)
        n += 1

    for name in TABLES:
        shutil.copy2(SRC / name, OUT / "tables" / name)
        n += 1
    for name in TABLES_JSON:
        doc = json.loads((SRC / name).read_text())
        (OUT / "tables" / name).write_text(json.dumps(_scrub(doc), indent=2) + "\n")
        n += 1
    # The .tex are taken from the paper, so the archive cannot disagree with it.
    for name in TABLES_TEX:
        shutil.copy2(PAPER / name, OUT / "tables" / name)
        n += 1
    for name in TABLES_TEX_STANDALONE:
        src = SRC / name
        if src.is_file():
            shutil.copy2(src, OUT / "tables" / name)
            n += 1

    for name in SCRIPTS:
        shutil.copy2(REPO / "scripts" / name, OUT / "scripts" / name)
        n += 1

    (OUT / "README.md").write_text(README)

    # Manifests last: written earlier they would describe a tree still being
    # copied into, which is how the previous bundle came to disagree with itself.
    listing = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
    (OUT / "MANIFEST.txt").write_text("MANIFEST.txt\n" + "\n".join(listing) + "\n")
    digests = []
    for rel in sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file()):
        h = hashlib.sha256((OUT / rel).read_bytes()).hexdigest()
        digests.append(f"{h}  {rel}")
    (OUT / "MANIFEST.sha256").write_text("\n".join(digests) + "\n")

    with tarfile.open(TARBALL, "w:gz") as tar:
        tar.add(OUT, arcname="supplement")

    size = TARBALL.stat().st_size / 1e6
    print(f"[supp] built {OUT} with {n} data files")
    print(f"[supp] wrote {TARBALL.name}, {size:.1f} MB")
    return check()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify only, do not rebuild")
    args = ap.parse_args()
    if not SRC.is_dir():
        print(f"[supp] no source directory: {SRC}")
        return 1
    if not PAPER.is_dir():
        print(f"[supp] no paper directory: {PAPER}")
        return 1
    return check() if args.check else build()


if __name__ == "__main__":
    sys.exit(main())
