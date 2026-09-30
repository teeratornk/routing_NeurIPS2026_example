"""Anonymous reproduction archive for the TS-LIMITS submission.

What goes in, and why only that:
  records/   per-request evaluation quantities for every arm the paper reads:
             (country, horizon, anchor_t, target_t, ape) and, for the served
             files, the per-request served losses and the gate score. Raw load
             values, targets and point/quantile predictions are NOT included:
             a prediction beside an APE reveals the target, and the load
             provider's redistribution terms are not stated on its data page.
             Anyone with source access can regenerate them.
  gates/     the eight deployed margin regressors, refit here from the
             development records (2018-2023, seed 0, deterministic) and
             pickled, with the feature list and the frozen 2024 thresholds
             under both metrics.
  results/   every CSV and JSON under reports/tslimits that a table reads.
  tables/    the tab_*.tex files exactly as the paper inputs them.
  scripts/   the tslimits_*.py, revision_2026_*.py and sbatch files.
  README.md  versions, hosts, and which script regenerates which table.

Out:  reports/tslimits/archive/ and reports/tslimits/tslimits_archive.tar.gz
Run:  PYTHONPATH=src .venv/bin/python scripts/build_tslimits_archive.py
"""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import pickle
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "reports" / "tslimits"
OUT = SRC / "archive"
PAPER = REPO.parent / "NeurIPS_ts"
FM_PREFIXES = (
    "tsl_chronos_q",
    "tsl_timesfm_q",
    "tsldev_chronos_q",
    "tsldev_timesfm_q",
    "tsl_chronos_ctx8192_h1h24_q",
    "tsl_chronos_ctx8192_q",
    "tsl_chronos_ctx8192_h8760_q",
    "tsl_chronos_ctx8192_h17520_q",
    "tsldev_chronos_ctx8192_h1h24_q",
    "tsldev_chronos_ctx8192_q",
    "tsl_timesfm_ctx16256_q",
    "tsl_timesfm_ctxnative_q",
    "tsldev_timesfm_ctx16256_q",
    "tsldev_timesfm_ctxnative_q",
    "tsl_chronos_year_q",
    "tsl_chronos_h17520_q",
    "tsl26_chronos_q",
    "tsl26_chronos_ctx8192_q",
    "tsl26_timesfm_q",
    "tsl26_timesfm_ctx16256_q",
    "tsl26_timesfm_ctxnative_q",
    "tslcov_cal_q",
    "tslcov_calclim_q",
    "tslcov_cal_h8760_q",
    "tslcov_cal_h17520_q",
    "tsldense_chronos_h24_q",
)
ROLLOUT_GLOBS = ("rollout_*.parquet", "ctx8192_rollout_*.parquet")
STRUCTURAL_GLOBS = ("per_origin_structural_*.parquet", "per_origin_seasonal_naive_2025.parquet")
SERVED_GLOBS = ("served_per_request*.parquet",)
KEEP_FM = ["cc", "test_year", "horizon", "origin_ts", "target_ts"]
KEEP_ST = ["country", "test_year", "horizon", "anchor_t", "target_t", "ape"]


def _mod(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---- anonymisation. The scripts are copied as text and every trace of the
# authors' machine is rewritten; the build fails if any survives.
REPO_STR = str(REPO)
# hex-encoded so the patterns themselves never appear in the shipped copy of this file
IDENTIFYING = tuple(
    bytes.fromhex(h).decode()
    for h in ("6d6b616465657468756d", "2f686f6d652f", "407369656d656e73", "6865616431")
)
SCRUB_KEYS = ("host", "hostname", "node", "nodelist", "job_id", "jobid", "slurm_job_id", "user")


def _scrub_text(text: str) -> str:
    text = text.replace(REPO_STR, "$REPO")
    text = re.sub(r"#SBATCH --output=reports/", "#SBATCH --output=reports/", text)
    text = text.replace('-o "%i %j"', '-o "%i %j"')
    text = re.sub(r"$HOME/... "$HOME/...", text)
    return text


def _scrub_json(doc: Any) -> Any:
    if isinstance(doc, dict):
        return {
            k: ("<anonymised>" if k.lower() in SCRUB_KEYS else _scrub_json(v))
            for k, v in doc.items()
        }
    if isinstance(doc, list):
        return [_scrub_json(v) for v in doc]
    return doc


def _audit(root: Path) -> None:
    """Fail the build if any identifying string survives in a text file."""
    hits = []
    for f in sorted(x for x in root.rglob("*") if x.is_file()):
        if f.suffix in (".parquet", ".pkl", ".pdf", ".npz"):
            continue
        try:
            body = f.read_text()
        except UnicodeDecodeError:
            continue
        for needle in IDENTIFYING:
            if needle in body:
                hits.append(f"{f.relative_to(root)}: {needle}")
    assert not hits, "identifying strings in the archive:\n  " + "\n  ".join(hits)
    print(f"[archive] anonymity audit passed ({len(IDENTIFYING)} patterns)", flush=True)


LICENSE_DATA = """Creative Commons Attribution 4.0 International (CC BY 4.0)

Applies to records/, results/ and tables/ of this archive: the per-request
evaluation quantities, the result files and the table sources. You may share
and adapt them for any purpose with attribution to the paper. Full text:
https://creativecommons.org/licenses/by/4.0/legalcode
"""

LICENSE_CODE = """MIT License

Applies to scripts/ and gates/ of this archive.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def _fm_record(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    if "ape" not in d.columns:
        d["ape"] = 100.0 * (d["q50"] - d["actual"]).abs() / d["actual"].abs()
    keep = [c for c in (*KEEP_FM, "year", "cap", "passes", "ape") if c in d.columns]
    return d[keep]


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in (
        "records/fm",
        "records/structural",
        "records/served",
        "gates",
        "results",
        "tables",
        "scripts",
    ):
        (OUT / sub).mkdir(parents=True)
    n = 0
    for prefix in FM_PREFIXES:
        for f in sorted(glob.glob(str(SRC / "fm" / f"{prefix}_*.parquet"))):
            _fm_record(Path(f)).to_parquet(OUT / "records" / "fm" / Path(f).name, index=False)
            n += 1
    for pat in ROLLOUT_GLOBS:
        for f in sorted(glob.glob(str(SRC / pat))):
            _fm_record(Path(f)).to_parquet(OUT / "records" / "fm" / Path(f).name, index=False)
            n += 1
    for pat in STRUCTURAL_GLOBS:
        for f in sorted(glob.glob(str(SRC / pat))):
            d = pd.read_parquet(f)
            keep = [c for c in KEEP_ST if c in d.columns]
            # the gate features are request-time quantities and carry no load value
            feats = [
                c
                for c in d.columns
                if c not in keep
                and c not in ("y_true", "y_pred", "anchor_load", "recent_load_range")
            ]
            d[keep + feats].to_parquet(OUT / "records" / "structural" / Path(f).name, index=False)
            n += 1
    for pat in SERVED_GLOBS:
        for f in sorted(glob.glob(str(SRC / pat))):
            shutil.copy2(f, OUT / "records" / "served" / Path(f).name)
            n += 1
    print(f"[archive] {n} record files", flush=True)
    # ---- the deployed gates, refit deterministically, with their frozen thresholds
    bs = _mod("tslimits_bootstrap")
    cols = list(bs.ENRICHED)
    cs = pd.read_csv(SRC / "country_static.csv").set_index(["fm", "horizon"])
    bcm = pd.read_csv(SRC / "bootstrap_ci_panel_median.csv").set_index(["fm", "horizon"])
    thresholds = []
    for fm_label, (_t, dev_pat) in bs.FM_PATTERNS.items():
        dev = bs._paired("dev", dev_pat)
        fit = dev[dev["test_year"].isin(bs.FIT_YEARS)]
        for h in sorted(fit["horizon"].unique()):
            f = fit[fit["horizon"] == h].reset_index(drop=True)
            reg = bs._reg(cols).fit(f[cols], f["ape_st"] - f["ape_fm"])
            name = f"gate_{fm_label.split('-Uni')[0].lower().replace('.', '')}_h{int(h)}.pkl"
            with (OUT / "gates" / name).open("wb") as fh:
                pickle.dump(reg, fh)
            key = (fm_label, int(h))
            thresholds.append(
                {
                    "fm": fm_label,
                    "horizon": int(h),
                    "gate_file": name,
                    "budget_pooled_mean": float(cs.loc[key, "router_budget"]),
                    "threshold_pooled_mean": float(cs.loc[key, "router_threshold"]),
                    "budget_panel_median": float(bcm.loc[key, "frozen_enr_budget"]),
                    "threshold_panel_median": float(bcm.loc[key, "frozen_enr_threshold"]),
                }
            )
            print(f"[archive] gate {name}: {reg.n_iter_} iterations", flush=True)
    pd.DataFrame(thresholds).to_csv(OUT / "gates" / "thresholds.csv", index=False)
    (OUT / "gates" / "features.json").write_text(
        json.dumps(
            {
                "features": cols,
                "fit_years": list(bs.FIT_YEARS),
                "selection_year": list(bs.VAL_YEARS),
                "seed": bs.SEED,
                "learner": "HistGradientBoostingRegressor, sklearn 1.5.2 defaults, 300 iterations",
            },
            indent=2,
        )
    )
    # ---- results, tables, scripts
    for f in sorted(glob.glob(str(SRC / "*.csv"))):
        shutil.copy2(f, OUT / "results" / Path(f).name)
    for f in sorted(glob.glob(str(SRC / "*.json"))):
        doc = json.loads(Path(f).read_text())
        (OUT / "results" / Path(f).name).write_text(json.dumps(_scrub_json(doc), indent=2) + "\n")
    for f in sorted(PAPER.glob("tab_*.tex")):
        shutil.copy2(f, OUT / "tables" / f.name)
    for pat in (
        "tslimits_*.py",
        "tslimits_*.sbatch",
        "revision_2026_*.py",
        "revision_2025_frozen_eval.py",
        "revision_year_frozen2017.py",
        "d216_gpu_chronos_quantiles.py",
        "revision_timesfm_zeroshot_quantiles.py",
        "build_tslimits_archive.py",
        "build_tslimits_supplement.py",
    ):
        for f in sorted(glob.glob(str(REPO / "scripts" / pat))):
            (OUT / "scripts" / Path(f).name).write_text(_scrub_text(Path(f).read_text()))
    readme = REPO / "scripts" / "tslimits_archive_README.md"
    assert readme.exists(), "the archive README is missing"
    shutil.copy2(readme, OUT / "README.md")
    (OUT / "LICENSE-DATA").write_text(LICENSE_DATA)
    (OUT / "LICENSE-CODE").write_text(LICENSE_CODE)
    manifest_md = REPO / "scripts" / "tslimits_archive_MANIFEST_tables.md"
    shutil.copy2(manifest_md, OUT / "MANIFEST-tables.md")
    _audit(OUT)
    # the file manifest is written last so it cannot describe a tree that changed after it
    listing, digests = [], []
    for f in sorted(x for x in OUT.rglob("*") if x.is_file()):
        rel = f.relative_to(OUT).as_posix()
        listing.append(f"{f.stat().st_size:>12d}  {rel}")
        digests.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel}")
    (OUT / "MANIFEST.txt").write_text("\n".join(listing) + "\n")
    (OUT / "MANIFEST.sha256").write_text("\n".join(digests) + "\n")
    tar = SRC / "tslimits_archive.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(OUT, arcname="tslimits_archive")
    print(f"[archive] wrote {tar} ({tar.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
