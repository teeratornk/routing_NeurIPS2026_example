"""TS-LIMITS: end-to-end serving latency, with the gate charged on every request.

The paper reports that routing cuts foundation-model calls at short horizons. It
does not follow that routing is faster. The gate runs on every request including
the ones it declines, so the expected serial latency is

    router = gate + p * FM + (1 - p) * structural

and against always calling the FM this is an improvement only when

    p  <  1 - gate / (FM - structural).

At a day and below the escalated fraction is close to 0.9, and on an H200 the
foundation-model call is tens of milliseconds, so the 2.04 ms gate is a large
share of the difference. Whether the router wins on latency there is an
arithmetic question the paper never asked, and the answer is not uniform.

Two things this cannot settle. The gate and the structural model were measured on
different hosts, and the structural figure is carried over rather than
re-measured. And the gate's 2.04 ms is a predict() call on a pre-built feature
row: constructing the 22 features, including a 168-hour rolling statistic and the
weather and holiday joins, is not in it. Both push the same way, so the numbers
below flatter the router.

Out: reports/tslimits/latency_endtoend.csv
     reports/tslimits/tab_latency.tex
Run: .venv/bin/python scripts/tslimits_latency.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path("reports/tslimits")
COST = Path("reports/revision/cost")

SHORT = {"Chronos-2-Uni-ZS": "Chronos-2", "TimesFM-2.5-Uni-ZS": "TimesFM-2.5"}


def _timesfm_latency_ms() -> dict[int, float]:
    """Per-horizon single-request latency, batch 1, from the energy probe."""
    d = json.loads((OUT / "energy_inference_timesfm.json").read_text())
    out: dict[int, float] = {}
    for rec in d["runs"]:
        if int(rec["batch_size"]) != 1:
            continue
        h = int(rec["horizon"])
        # several repeats per (horizon, batch); take the first, as the p50 is
        # already a within-run median over calls
        out.setdefault(h, float(rec["latency_p50_s"]) * 1000.0)
    return out


def main() -> int:
    gate_ms = (
        json.loads((OUT / "router_cost.json").read_text())["single_request_latency"]["p50_us"]
        / 1000.0
    )
    struct = {
        int(k): v / 1000.0
        for k, v in json.loads((COST / "cost_benchmark.json").read_text())["composition"][
            "latency_single_us"
        ].items()
    }
    chronos = {
        int(k): float(v)
        for k, v in json.loads((COST / "chronos_cost_gpu.json").read_text())[
            "latency_single_ms"
        ].items()
    }
    timesfm = _timesfm_latency_ms()
    fm_ms = {"Chronos-2-Uni-ZS": chronos, "TimesFM-2.5-Uni-ZS": timesfm}

    cs = pd.read_csv(OUT / "country_static.csv")
    rows = []
    for r in cs.itertuples():
        h = int(r.horizon)
        lat = fm_ms.get(str(r.fm), {}).get(h)
        if lat is None:
            continue
        p = float(r.router_realized_frac)
        st = struct[h]
        router = gate_ms + p * lat + (1.0 - p) * st
        # router beats always-FM iff p < 1 - gate/(FM - structural)
        pstar = 1.0 - gate_ms / (lat - st)
        rows.append(
            {
                "fm": str(r.fm),
                "horizon": h,
                "escalated_frac": round(p, 4),
                "gate_ms": round(gate_ms, 4),
                "structural_ms": round(st, 4),
                "fm_ms": round(lat, 4),
                "router_ms": round(router, 4),
                "always_fm_ms": round(lat, 4),
                "delta_vs_fm_ms": round(router - lat, 4),
                "breakeven_frac": round(pstar, 4),
                "router_faster": bool(router < lat),
            }
        )

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "latency_endtoend.csv", index=False)

    tex = [
        "\\begin{tabular}{llrrrrc}",
        "\\toprule",
        "FM & $h$ & escalated & router & always-FM & $\\Delta$ & $p^{\\star}$ \\\\",
        "\\midrule",
    ]
    for r in d.itertuples():
        tex.append(
            f"{SHORT.get(r.fm, r.fm)} & {r.horizon} & {r.escalated_frac:.2f} & "
            f"{r.router_ms:.2f} & {r.always_fm_ms:.2f} & "
            f"${r.delta_vs_fm_ms:+.2f}$ & {r.breakeven_frac:.2f} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    (OUT / "tab_latency.tex").write_text("\n".join(tex))

    slower = d[~d.router_faster]
    print(f"[lat] gate {gate_ms:.3f} ms charged on every request", flush=True)
    for r in d.itertuples():
        mark = "faster" if r.router_faster else "SLOWER"
        print(
            f"[lat] {SHORT.get(r.fm, r.fm):<12} h={r.horizon:<6} p={r.escalated_frac:.4f} "
            f"router {r.router_ms:7.3f} ms vs FM {r.always_fm_ms:7.3f} "
            f"({r.delta_vs_fm_ms:+7.3f})  breakeven {r.breakeven_frac:.4f}  {mark}",
            flush=True,
        )
    print(
        f"[lat] router is slower end to end in {len(slower)}/{len(d)} cells: "
        + ", ".join(f"{SHORT.get(r.fm, r.fm)} h={r.horizon}" for r in slower.itertuples()),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
