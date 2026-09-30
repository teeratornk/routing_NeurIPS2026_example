"""DST sensitivity: re-score the frozen policies without the transition weeks.

The gate's calendar features are UTC with no daylight-saving adjustment, so
"hour" means a different local hour across countries and seasons. Every policy
column in served_per_request.parquet is the realised per-request loss under
the frozen 2024 threshold, so dropping the requests whose TARGET falls within
seven days of the two 2025 EU transitions (2025-03-30, 2025-10-26) and
re-aggregating is exact post-processing: no gate refit, no re-selection. The
country-cluster interval is rebuilt from the filtered frame the same way
tslimits_country_static.py builds it (per-country sums and counts, B=2000,
seed 0).

Out:  reports/tslimits/dst_sensitivity.csv, tab_dst.tex
Run:  .venv/bin/python scripts/tslimits_dst_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("reports/tslimits")
SEED, B = 0, 2000
TRANSITIONS = (pd.Timestamp("2025-03-30"), pd.Timestamp("2025-10-26"))
WINDOW = pd.Timedelta(days=7)
SHORT = {"Chronos-2-Uni-ZS": "Chronos-2", "TimesFM-2.5-Uni-ZS": "TimesFM-2.5"}


def _gain(frame: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float, float]:
    """Router gain over c-static, pooled mean, with a country-cluster interval."""
    g = frame.groupby("country")
    sums = g[["cc_static", "router_cc"]].sum().to_numpy()
    cnt = g["cc_static"].size().to_numpy().astype(float)
    point = (sums[:, 0].sum() - sums[:, 1].sum()) / cnt.sum()
    draws = np.empty(B)
    for bi in range(B):
        ix = rng.integers(0, len(cnt), len(cnt))
        draws[bi] = (sums[ix, 0].sum() - sums[ix, 1].sum()) / cnt[ix].sum()
    return float(point), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    pr = pd.read_parquet(OUT / "served_per_request.parquet")
    pr["target_t"] = pr["anchor_t"] + pd.to_timedelta(pr["horizon"], unit="h")
    near = np.zeros(len(pr), dtype=bool)
    for t0 in TRANSITIONS:
        near |= (pr["target_t"] - t0).abs() <= WINDOW
    print(f"[dst] dropping {int(near.sum())} of {len(pr)} requests ({near.mean():.2%})", flush=True)
    rows = []
    for (model, h), cell in pr.groupby(["model", "horizon"], sort=True):
        rng = np.random.default_rng(SEED)
        p_all, lo_all, hi_all = _gain(cell, rng)
        rng = np.random.default_rng(SEED)
        keep = cell[~near[cell.index]]
        p_dst, lo_dst, hi_dst = _gain(keep, rng)
        rows.append(
            {
                "fm": model,
                "horizon": int(h),
                "n_all": len(cell),
                "n_kept": len(keep),
                "gain_all_pp": round(p_all, 4),
                "lo_all": round(lo_all, 4),
                "hi_all": round(hi_all, 4),
                "gain_dst_pp": round(p_dst, 4),
                "lo_dst": round(lo_dst, 4),
                "hi_dst": round(hi_dst, 4),
                "excludes_zero_all": bool(lo_all > 0),
                "excludes_zero_dst": bool(lo_dst > 0),
            }
        )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "dst_sensitivity.csv", index=False)
    cs = pd.read_csv(OUT / "country_static.csv")
    chk = d.merge(cs[["fm", "horizon", "router_over_cc_pp"]], on=["fm", "horizon"])
    assert np.allclose(chk["gain_all_pp"], chk["router_over_cc_pp"], atol=2e-4), (
        "full-window gain does not reproduce country_static.csv"
    )
    tex = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "FM & $h$ & all requests (pp) & without transition weeks (pp) & excludes 0 \\\\",
        "\\midrule",
    ]
    for r in d.itertuples():
        tex.append(
            f"{SHORT[r.fm]} & {r.horizon} & ${r.gain_all_pp:+.3f}$ [{r.lo_all:.3f}, "
            f"{r.hi_all:.3f}] & "
            f"${r.gain_dst_pp:+.3f}$ [{r.lo_dst:.3f}, {r.hi_dst:.3f}] & "
            f"{'yes' if r.excludes_zero_dst else 'no'} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_dst.tex").write_text("\n".join(tex) + "\n")
    print(
        d[["fm", "horizon", "gain_all_pp", "gain_dst_pp", "excludes_zero_dst"]].to_string(
            index=False
        )
    )
    print(
        f"[dst] median gain all {d.gain_all_pp.median():.4f} -> without transitions "
        f"{d.gain_dst_pp.median():.4f}; "
        f"excludes zero {int(d.excludes_zero_dst.sum())}/8",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
