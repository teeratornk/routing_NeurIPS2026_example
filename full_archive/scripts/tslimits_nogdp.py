"""The no-GDP structural arm on the 2025 grid, against the deployed model.

The deployed year layer carries lagged GDP that the foundation models never
see. This arm drops that term (drift only) and re-scores every horizon on the
same origin grid, so the long-lead lead can be read with and without the one
exogenous input the FM lacks. Both artifacts come from
tslimits_per_origin_structural.py; the deployed ones are the gated files.

Out:  reports/tslimits/nogdp.csv, tab_nogdp.tex
Run:  .venv/bin/python scripts/tslimits_nogdp.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("reports/tslimits")
SEED, B = 0, 2000
FILES = {  # horizon -> (deployed, no-GDP)
    "paired": ("per_origin_structural_2025.parquet", "per_origin_structural_2025_nogdp.parquet"),
    8760: (
        "per_origin_structural_2025_year.parquet",
        "per_origin_structural_2025_year_nogdp.parquet",
    ),
    17520: (
        "per_origin_structural_2025_h17520.parquet",
        "per_origin_structural_2025_h17520_nogdp.parquet",
    ),
}


def main() -> int:
    ys = pd.read_csv(OUT / "year_scale.csv").set_index("horizon")
    rows = []
    for key, (f_dep, f_no) in FILES.items():
        dep = pd.read_parquet(OUT / f_dep)[["country", "horizon", "anchor_t", "ape"]]
        no = pd.read_parquet(OUT / f_no)[["country", "horizon", "anchor_t", "ape"]]
        j = dep.merge(
            no,
            on=["country", "horizon", "anchor_t"],
            suffixes=("_dep", "_no"),
            validate="one_to_one",
        )
        assert len(j) == len(dep) == len(no), f"{key}: {len(j)} of {len(dep)}/{len(no)}"
        for h, g in j.groupby("horizon"):
            med = g.groupby("country")[["ape_dep", "ape_no"]].median()
            pm_dep, pm_no = float(med["ape_dep"].median()), float(med["ape_no"].median())
            rng = np.random.default_rng(SEED)
            arr = med.to_numpy()
            d = np.empty(B)
            for bi in range(B):
                q = np.median(arr[rng.integers(0, len(arr), len(arr))], axis=0)
                d[bi] = q[1] - q[0]
            rows.append(
                {
                    "horizon": int(h),
                    "deployed": round(pm_dep, 4),
                    "no_gdp": round(pm_no, 4),
                    "no_gdp_minus_deployed_pp": round(pm_no - pm_dep, 4),
                    "lo": round(float(np.percentile(d, 2.5)), 4),
                    "hi": round(float(np.percentile(d, 97.5)), 4),
                    "chronos2_tuned": round(float(ys.loc[int(h), "chronos2_tuned"]), 4)
                    if int(h) in ys.index
                    else np.nan,
                    "seasonal_naive": round(float(ys.loc[int(h), "seasonal_naive"]), 4)
                    if int(h) in ys.index
                    else np.nan,
                }
            )
            print(
                f"[nogdp] h={int(h):<5} deployed {pm_dep:.3f} no-GDP {pm_no:.3f} diff "
                f"{pm_no - pm_dep:+.3f} "
                f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]",
                flush=True,
            )
    d = pd.DataFrame(rows).sort_values("horizon")
    d.to_csv(OUT / "nogdp.csv", index=False)
    lab = {1: "1\\,h", 24: "1\\,d", 168: "1\\,w", 720: "1\\,mo", 8760: "1\\,yr", 17520: "2\\,yr"}
    tex = [
        "\\begin{tabular}{lrrlrr}",
        "\\toprule",
        "$h$ & deployed & no GDP & no GDP $-$ deployed [95\\% CI] & Chronos-2 & seas.\\ naive \\\\",
        "\\midrule",
    ]
    for r in d.itertuples():
        tex.append(
            f"{lab[r.horizon]} & {r.deployed:.2f} & {r.no_gdp:.2f} & "
            f"${r.no_gdp_minus_deployed_pp:+.2f}$ [{r.lo:+.2f}, {r.hi:+.2f}] & "
            f"{r.chronos2_tuned:.2f} & {r.seasonal_naive:.2f} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_nogdp.tex").write_text("\n".join(tex) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
