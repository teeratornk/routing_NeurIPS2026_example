"""TS-LIMITS: what the router itself costs, measured rather than asserted.

Routing is only worth it if the gate is cheap against the call it avoids. The
paper originally asserted the gate answers "in tens of microseconds"; measuring
it showed 2.3 ms for a single request, about twenty times the structural model
it gates, which is a materially different claim. So it is measured here and the
numbers come from an artifact.

Two latencies matter and they differ by two orders of magnitude:

    single    one request scored on its own, which is what an online router
              that is handed one request at a time actually pays. Dominated by
              per-call overhead, not by the trees.
    batched   the same gate scoring 1,000 requests, amortised per request. This
              is what a router fed a daily batch pays.

Reporting only the second would flatter the gate; reporting only the first would
misdescribe a batched deployment.

Size is the pickled estimator, which is what has to be shipped and loaded, and
is reported in kibibytes alongside the structural artifact's 13,032 bytes so the
ratio is between like units.

Out: reports/tslimits/router_cost.json
Run: PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python scripts/tslimits_router_cost.py
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import pathlib
import pickle
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

OUT = Path("reports/tslimits")
STRUCTURAL_BYTES = 13_032  # the concurrent submission's deployed coefficient blob
WARMUP = 50
N_SINGLE = 500
N_BATCH = 50
BATCH = 1000
HORIZON = 24


def _mod(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_BS = _mod("tslimits_bootstrap")


def _cpu_model() -> str:
    """The CPU model name, since platform.processor() gives only the arch here."""
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _percentiles(x: np.ndarray) -> dict[str, float]:
    return {
        "p50_us": round(float(np.percentile(x, 50)), 2),
        "p95_us": round(float(np.percentile(x, 95)), 2),
        "min_us": round(float(x.min()), 2),
    }


def main() -> int:
    cols = list(_BS.ENRICHED)
    dev = _BS._paired("dev", "reports/tslimits/fm/tsldev_chronos_q_*.parquet")
    f = dev[(dev["test_year"].isin(_BS.FIT_YEARS)) & (dev["horizon"] == HORIZON)]
    f = f.reset_index(drop=True)
    reg = _BS._reg(cols).fit(f[cols], f["ape_st"] - f["ape_fm"])

    buf = io.BytesIO()
    pickle.dump(reg, buf)
    size = buf.getbuffer().nbytes

    x1 = f[cols].iloc[:1]
    for _ in range(WARMUP):
        reg.predict(x1)
    single = np.empty(N_SINGLE)
    for i in range(N_SINGLE):
        t0 = time.perf_counter()
        reg.predict(x1)
        single[i] = (time.perf_counter() - t0) * 1e6

    xb = f[cols].iloc[:BATCH]
    for _ in range(5):
        reg.predict(xb)
    batched = np.empty(N_BATCH)
    for i in range(N_BATCH):
        t0 = time.perf_counter()
        reg.predict(xb)
        batched[i] = (time.perf_counter() - t0) * 1e6 / BATCH

    out = {
        "fit_rows": len(f),
        "n_features": len(cols),
        "horizon": HORIZON,
        "max_iter": getattr(reg, "max_iter", None),
        "n_iter_fitted": int(getattr(reg, "n_iter_", 0)),
        "early_stopping_active": bool(len(f) > 10_000),
        "pickled_bytes": int(size),
        "pickled_kib": round(size / 1024, 1),
        "structural_bytes": STRUCTURAL_BYTES,
        "size_ratio_vs_structural": round(size / STRUCTURAL_BYTES, 1),
        # Carried over from the concurrent submission's cost analysis
        # (reports/revision/RESULTS_SUMMARY.md), not measured here. The ratios
        # below inherit that provenance and the paper says so.
        "structural_latency_us": 111.0,
        "structural_latency_source": "concurrent submission, not measured here",
        "single_request_latency": _percentiles(single),
        "batched_1000_latency_per_request": _percentiles(batched),
        "single_vs_structural_ratio": round(float(np.percentile(single, 50)) / 111.0, 1),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            # platform.processor() returns the bare arch on Linux, which is
            # useless for reproducing a latency. Read the real model name.
            "cpu_model": _cpu_model(),
            "processor": platform.processor() or platform.machine(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            "warmup_predicts": WARMUP,
            "n_single": N_SINGLE,
            "n_batch": N_BATCH,
            "cpu_pinning": "none; single-threaded by OMP_NUM_THREADS only",
            "note": "single-threaded; run with OMP_NUM_THREADS=1",
        },
    }
    # one record per gate learner; the deployed HGB keeps the unsuffixed name
    _sfx = "" if _BS.GATE_LEARNER == "hgb" else f"_{_BS.GATE_LEARNER}"
    out["gate_learner"] = _BS.GATE_LEARNER
    (OUT / f"router_cost{_sfx}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    print(
        f"\n[router] {out['pickled_kib']} KiB "
        f"({out['size_ratio_vs_structural']}x the structural artifact); "
        f"single {out['single_request_latency']['p50_us']} us "
        f"({out['single_vs_structural_ratio']}x structural), "
        f"batched {out['batched_1000_latency_per_request']['p50_us']} us/request",
        flush=True,
    )
    print(f"[router] wrote {OUT}/router_cost.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
