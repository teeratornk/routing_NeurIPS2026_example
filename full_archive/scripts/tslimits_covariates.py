"""Covariate-aware Chronos-2 against the univariate configuration, same grid.

Reads the tslcov_* arms written by tslimits_chronos_covariates.py and the
univariate records they pair with (tsl_chronos_q_* at h<=720; the frozen-cap
rollout arms at the long leads), joins one-to-one on (country, horizon,
anchor_t), and reports panel-median MdAPE per arm with a paired country-
cluster interval on covariate minus univariate. The structural model's own
median is printed beside them for scale.

Out:  reports/tslimits/covariates.csv, tab_covariates.tex
Run:  .venv/bin/python scripts/tslimits_covariates.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("reports/tslimits")
SEED, B = 0, 2000
FROZEN_CAP = {8760: 48, 17520: 56}
ARMS = (("cal", "calendar + holidays"), ("calclim", "calendar + holidays + climatology"))


def _load(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d = d.rename(columns={"cc": "country", "origin_ts": "anchor_t"})
    d["anchor_t"] = pd.to_datetime(d["anchor_t"])
    d["ape"] = 100.0 * (d["q50"] - d["actual"]).abs() / d["actual"].abs()
    return d[["country", "horizon", "anchor_t", "actual", "ape"]]


def main() -> int:
    uni = _load("reports/tslimits/fm/tsl_chronos_q_*.parquet")
    for h, cap in FROZEN_CAP.items():
        f = OUT / f"rollout_ablation_h{h}_cap{cap}.parquet"
        if f.exists():
            uni = pd.concat([uni, _load(str(f))], ignore_index=True)
    ys = pd.read_csv(OUT / "year_scale.csv").set_index("horizon")
    rows = []
    for arm, label in ARMS:
        cov = pd.concat(
            [
                _load(f"reports/tslimits/fm/tslcov_{arm}_q_*.parquet"),
                _load(f"reports/tslimits/fm/tslcov_{arm}_h8760_q_*.parquet"),
                _load(f"reports/tslimits/fm/tslcov_{arm}_h17520_q_*.parquet"),
            ],
            ignore_index=True,
        )
        if cov.empty:
            continue
        for h, g in cov.groupby("horizon"):
            u = uni[uni["horizon"] == h]
            j = g.merge(
                u,
                on=["country", "horizon", "anchor_t"],
                suffixes=("_cov", "_uni"),
                validate="one_to_one",
            )
            if j["country"].nunique() < 19 or len(j) != len(u):
                print(
                    f"[cov] {arm} h={int(h)}: {j['country'].nunique()} countries, "
                    f"{len(j)}/{len(u)} paired; skipped (incomplete)",
                    flush=True,
                )
                continue
            assert float((j["actual_cov"] - j["actual_uni"]).abs().max()) == 0.0
            med = j.groupby("country")[["ape_cov", "ape_uni"]].median()
            pm_cov, pm_uni = float(med["ape_cov"].median()), float(med["ape_uni"].median())
            rng = np.random.default_rng(SEED)
            arr = med.to_numpy()
            d = np.empty(B)
            for bi in range(B):
                q = np.median(arr[rng.integers(0, len(arr), len(arr))], axis=0)
                d[bi] = q[0] - q[1]
            wins = float((j["ape_cov"] < j["ape_uni"]).mean())
            rows.append(
                {
                    "arm": arm,
                    "label": label,
                    "horizon": int(h),
                    "n": len(j),
                    "univariate": round(pm_uni, 4),
                    "covariate": round(pm_cov, 4),
                    "cov_minus_uni_pp": round(pm_cov - pm_uni, 4),
                    "lo": round(float(np.percentile(d, 2.5)), 4),
                    "hi": round(float(np.percentile(d, 97.5)), 4),
                    "cov_wins_share": round(wins, 4),
                    "structural": round(float(ys.loc[int(h), "structural"]), 4)
                    if int(h) in ys.index
                    else np.nan,
                }
            )
            print(
                f"[cov] {arm:8s} h={int(h):<5} univariate {pm_uni:.3f} covariate {pm_cov:.3f} "
                f"diff {pm_cov - pm_uni:+.3f} "
                f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] wins {wins:.2f}",
                flush=True,
            )
    d = pd.DataFrame(rows).sort_values(["arm", "horizon"])
    d.to_csv(OUT / "covariates.csv", index=False)
    lab = {1: "1\\,h", 24: "1\\,d", 168: "1\\,w", 720: "1\\,mo", 8760: "1\\,yr", 17520: "2\\,yr"}
    tex = [
        "\\begin{tabular}{llrrlrr}",
        "\\toprule",
        "covariates & $h$ & univariate & with covariates & difference [95\\% CI] & wins & "
        "structural \\\\",
        "\\midrule",
    ]
    for r in d.itertuples():
        tex.append(
            f"{r.label} & {lab[r.horizon]} & {r.univariate:.2f} & {r.covariate:.2f} & "
            f"${r.cov_minus_uni_pp:+.2f}$ [{r.lo:+.2f}, {r.hi:+.2f}] & "
            f"{100 * r.cov_wins_share:.0f}\\% & {r.structural:.2f} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_covariates.tex").write_text("\n".join(tex) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
