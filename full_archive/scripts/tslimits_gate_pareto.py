"""Gate Pareto: accuracy gain against artifact size and latency, per learner.

The deployed gate (HGB, 738 kB, ~2 ms) is compared with the shallow tree and
the ridge gate on the same frozen pipeline. Gain and positive cells come from
country_static{_sfx}.csv, size and predict() latency from router_cost{_sfx}.json,
and the end-to-end latency uses the accounting of tslimits_latency.py: gate on
every request, the FM on the escalated fraction, the structural model on the
rest, with the FM and structural latencies carried over from their own
measurements (three hosts; the paper says so).

Out:  reports/tslimits/gate_pareto.csv, tab_gate_pareto.tex
Run:  .venv/bin/python scripts/tslimits_gate_pareto.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("reports/tslimits")
COST = Path("reports/revision/cost")
LEARNERS = (
    ("hgb", "", "HGB (deployed)"),
    ("shallow", "_shallow", "shallow tree"),
    ("ridge", "_ridge", "ridge"),
)


def _timesfm_latency_ms() -> dict[int, float]:
    d = json.loads((OUT / "energy_inference_timesfm.json").read_text())
    out: dict[int, float] = {}
    for rec in d["runs"]:
        if int(rec["batch_size"]) == 1:
            out.setdefault(int(rec["horizon"]), float(rec["latency_p50_s"]) * 1000.0)
    return out


def main() -> int:
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
    fm_ms = {"Chronos-2-Uni-ZS": chronos, "TimesFM-2.5-Uni-ZS": _timesfm_latency_ms()}
    rows = []
    for _key, sfx, label in LEARNERS:
        cs = pd.read_csv(OUT / f"country_static{sfx}.csv")
        rc = json.loads((OUT / f"router_cost{sfx}.json").read_text())
        gate_ms = rc["single_request_latency"]["p50_us"] / 1000.0
        e2e, faster = [], 0
        for r in cs.itertuples():
            lat = fm_ms[str(r.fm)].get(int(r.horizon))
            if lat is None:
                continue
            p = float(r.router_realized_frac)
            router = gate_ms + p * lat + (1.0 - p) * struct[int(r.horizon)]
            e2e.append(router)
            faster += int(router < lat)
        rows.append(
            {
                "learner": label,
                "median_gain_pp": round(float(cs["router_over_cc_pp"].median()), 4),
                "positive_cells": int((cs["router_over_cc_pp"] > 0).sum()),
                "excl_zero_cells": int((cs["router_over_cc_lo"] > 0).sum()),
                "size_kb": round(rc["pickled_bytes"] / 1000.0, 1),  # decimal kB, as in the body
                "gate_ms": round(gate_ms, 3),
                "e2e_median_ms": round(float(pd.Series(e2e).median()), 2),
                "faster_than_always_fm": f"{faster} of {len(e2e)}",
            }
        )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "gate_pareto.csv", index=False)
    tex = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "gate & median gain (pp) & positive & excl.\\ 0 & size (kB) & gate (ms) & model+gate (ms) "
        "\\\\",
        "\\midrule",
    ]
    for r in d.itertuples():
        tex.append(
            f"{r.learner} & ${r.median_gain_pp:+.3f}$ & {r.positive_cells} of 8 & "
            f"{r.excl_zero_cells} of 8 & "
            f"{r.size_kb:.1f} & {r.gate_ms:.2f} & {r.e2e_median_ms:.1f} "
            f"({r.faster_than_always_fm} faster) \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_gate_pareto.tex").write_text("\n".join(tex) + "\n")
    print(d.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
