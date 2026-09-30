"""TS-LIMITS: a moving-block bootstrap over dates, beside the country bootstrap.

Every interval in the paper resamples the 19 countries. That captures
cross-country heterogeneity and nothing else. It does not capture a common
weather or calendar shock hitting several European grids on the same day, and
those are exactly the days where the margin between a structural model and a
foundation model is largest.

This resamples the other margin: contiguous blocks of DATES, with all countries
moving together. The two designs are crossed rather than nested, so neither
dominates, and the honest reading is that a claim surviving both is more robust
than one surviving either.

Three things the design has to respect:

  * blocks, not single dates. Requests a day apart share a weather regime and a
    day type, so an i.i.d. date bootstrap would understate the variance. The
    default block is 7 days, which also absorbs the weekly cycle.
  * the unit count jumps from 19 to about 365, so intervals narrow. That is a
    property of the design, not evidence of a stronger result; report both.
  * at long horizons the block is cut on the anchor while the outcome lands h
    hours later, so blocks overlap in outcome space. Reported, not corrected.

Reads the per-request served losses persisted by tslimits_country_static.py, so
nothing is refitted here.

Out: reports/tslimits/dateblock_ci.csv
Run: .venv/bin/python scripts/tslimits_dateblock.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("reports/tslimits")
SERVED = OUT / "served_per_request.parquet"

B = 2000
SEED = 0
# 7 absorbs the weekly cycle; 28 is the sensitivity a reviewer asked for at h=720,
# where a 7-day block is shorter than the horizon it is cut on.
BLOCK_DAYS = int(__import__("os").environ.get("TSLIMITS_BLOCK_DAYS", "7"))
CONTRAST = ("cc_static", "router_cc")


def main() -> int:
    if not SERVED.is_file():
        print(f"[db] missing {SERVED}; run scripts/tslimits_country_static.py first", flush=True)
        return 1
    pr = pd.read_parquet(SERVED)
    pr["date"] = pr["anchor_t"].dt.normalize()
    rng = np.random.default_rng(SEED)
    rows = []

    for (model, h), g in pr.groupby(["model", "horizon"], sort=True):
        dates = np.sort(g["date"].unique())
        nd = len(dates)
        pos = {d: i for i, d in enumerate(dates)}
        di = g["date"].map(pos).to_numpy()
        a = g[CONTRAST[0]].to_numpy()
        b = g[CONTRAST[1]].to_numpy()
        point = float(a.mean() - b.mean())

        # rows grouped by date index, so a resampled block gathers whole days
        order = np.argsort(di, kind="stable")
        di_s, a_s, b_s = di[order], a[order], b[order]
        starts = np.searchsorted(di_s, np.arange(nd), side="left")
        ends = np.searchsorted(di_s, np.arange(nd), side="right")

        nblocks = int(np.ceil(nd / BLOCK_DAYS))
        draws = np.empty(B)
        for k in range(B):
            heads = rng.integers(0, nd, nblocks)
            idx = []
            for s0 in heads:
                for j in range(BLOCK_DAYS):
                    dd = (s0 + j) % nd  # circular, so every date is equally likely
                    idx.append(np.arange(starts[dd], ends[dd]))
            sel = np.concatenate(idx)
            draws[k] = a_s[sel].mean() - b_s[sel].mean()

        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append(
            {
                "model": model,
                "horizon": int(h),
                "point_pp": round(point, 4),
                "dateblock_lo": round(float(lo), 4),
                "dateblock_hi": round(float(hi), 4),
                "dateblock_sig": bool(lo > 0),
                "n_dates": nd,
                "block_days": BLOCK_DAYS,
                "outcome_overlaps_block": bool(h > BLOCK_DAYS * 24),
            }
        )

    d = pd.DataFrame(rows)
    suffix = "" if BLOCK_DAYS == 7 else f"_{BLOCK_DAYS}d"
    d.to_csv(OUT / f"dateblock_ci{suffix}.csv", index=False)

    cc = pd.read_csv(OUT / "country_static.csv")[
        ["fm", "horizon", "router_over_cc_pp", "router_over_cc_lo", "router_over_cc_hi"]
    ].rename(columns={"fm": "model"})
    m = d.merge(cc, on=["model", "horizon"], validate="one_to_one")
    print(f"[db] block {BLOCK_DAYS} days, B={B}, all countries move together", flush=True)
    for r in m.itertuples():
        w_c = r.router_over_cc_hi - r.router_over_cc_lo
        w_d = r.dateblock_hi - r.dateblock_lo
        flag = "  outcome outside block" if r.outcome_overlaps_block else ""
        print(
            f"[db] {r.model:<20} h={r.horizon:<6} {r.point_pp:+.4f}  "
            f"country [{r.router_over_cc_lo:+.4f},{r.router_over_cc_hi:+.4f}] w={w_c:.4f}  "
            f"date [{r.dateblock_lo:+.4f},{r.dateblock_hi:+.4f}] w={w_d:.4f}  "
            f"{'sig' if r.dateblock_sig else 'NS '}{flag}",
            flush=True,
        )
    print(
        f"[db] excludes zero: country design {int(cc.router_over_cc_lo.gt(0).sum())}/8, "
        f"date-block {int(d.dateblock_sig.sum())}/8",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
