"""TS-LIMITS Phase 0: is serving energy directly measurable on a compute node?

Feasibility probe, not a measurement. The paper's headline experiment is joules
per forecast, and on the login node the CPU side is blocked: intel-rapl is
present but energy_uj is root-only (CVE-2020-8694). This answers three
questions on an actual GPU node before Phase 1 is designed around either path.

  1. Is /sys/class/powercap/*/energy_uj readable here? If yes, CPU-side joules
     are directly measurable and the structural model can be measured in the
     same units as the foundation models.
  2. What sampling resolution does NVML give, and how noisy is idle draw? The
     idle baseline has to be subtracted, so its dispersion bounds the smallest
     workload we can resolve.
  3. Does integrating sampled power over a known GPU workload give a stable
     joule figure across repeats?

Writes reports/tslimits/energy_probe.json. Reports what it finds; it does not
assume either path works.

Run: sbatch scripts/tslimits_energy_probe.sbatch
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

OUT = Path("reports/tslimits/energy_probe.json")
RAPL_ROOT = Path("/sys/class/powercap")
IDLE_SECONDS = 20.0
LOAD_SECONDS = 20.0
WARMUP_SECONDS = 8.0
POLL_SECONDS = 0.1
REPEATS = 3


def _rapl_domains() -> list[dict[str, Any]]:
    """Every RAPL domain, with whether its energy counter is actually readable."""
    found: list[dict[str, Any]] = []
    if not RAPL_ROOT.is_dir():
        return found
    for d in sorted(RAPL_ROOT.glob("intel-rapl:*")):
        counter = d / "energy_uj"
        entry: dict[str, Any] = {"domain": d.name, "exists": counter.exists()}
        try:
            entry["value_uj"] = int(counter.read_text().strip())
            entry["readable"] = True
        except (OSError, ValueError) as exc:
            entry["readable"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        name = d / "name"
        if name.exists():
            with contextlib.suppress(OSError):
                entry["name"] = name.read_text().strip()
        found.append(entry)
    return found


def _target_gpu() -> str:
    """Physical index of the GPU we were actually allocated.

    nvidia-smi indices are NOT remapped by CUDA_VISIBLE_DEVICES, so a
    hardcoded "-i 0" measures physical GPU 0 regardless of what SLURM handed
    us. On a shared node that silently reports a different job's power draw and
    still looks like a plausible measurement. torch's cuda:0 is the FIRST entry
    of CUDA_VISIBLE_DEVICES, so that is the device to sample.
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not vis:
        return "0"
    return vis.split(",")[0].strip()


def _read_gpu_power() -> float | None:
    """One instantaneous GPU power reading (W), or None if unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                _target_gpu(),
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for line in res.stdout.strip().splitlines():
            line = line.strip()
            if line and line.replace(".", "", 1).isdigit():
                return float(line)
    except (subprocess.SubprocessError, ValueError):
        return None
    return None


def _sample_gpu_power(duration_s: float, poll_s: float) -> list[float]:
    """Poll GPU power for a fixed duration (used for the idle baseline)."""
    watts: list[float] = []
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        w = _read_gpu_power()
        if w is not None:
            watts.append(w)
        time.sleep(poll_s)
    return watts


class _PowerSampler(threading.Thread):
    """Sample GPU power CONCURRENTLY with the workload.

    The first version of this probe ran the load to completion and only then
    started sampling, so it integrated post-load decay power over a window that
    also contained the load phase. That is not a measurement of anything: it
    reported 3103 J for a cold repeat against 1403 J for a warm one, an 85%
    spread that was an artifact of the design rather than of the hardware.
    """

    def __init__(self, poll_s: float) -> None:
        super().__init__(daemon=True)
        self.poll_s = poll_s
        self.watts: list[float] = []
        # NOT _stop: threading.Thread defines a private _stop() that join()
        # calls internally, so shadowing it with an Event makes join() raise
        # "TypeError: 'Event' object is not callable".
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            w = _read_gpu_power()
            if w is not None:
                self.watts.append(w)
            self._halt.wait(self.poll_s)

    def request_stop(self) -> None:
        self._halt.set()


def _gpu_load(seconds: float) -> str:
    """Keep one GPU busy with dense matmuls; returns a status string."""
    try:
        import torch
    except ImportError:
        return "torch unavailable: no synthetic load applied"
    if not torch.cuda.is_available():
        return "torch present but CUDA unavailable"
    dev = torch.device("cuda:0")
    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)
    t_end = time.monotonic() + seconds
    n = 0
    while time.monotonic() < t_end:
        a = (a @ b).relu() * 1e-3
        n += 1
    torch.cuda.synchronize()
    return f"applied {n} matmuls of 4096x4096 on {torch.cuda.get_device_name(0)}"


def _stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n if n > 1 else 0.0
    return {
        "n": n,
        "mean_w": round(mean, 2),
        "sd_w": round(var**0.5, 3),
        "min_w": round(s[0], 2),
        "median_w": round(s[n // 2], 2),
        "max_w": round(s[-1], 2),
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "host": socket.gethostname(),
        "poll_seconds": POLL_SECONDS,
        "rapl": _rapl_domains(),
    }
    report["rapl_readable"] = any(d.get("readable") for d in report["rapl"])
    print(f"[probe] host={report['host']}  RAPL readable={report['rapl_readable']}", flush=True)

    idle = _sample_gpu_power(IDLE_SECONDS, POLL_SECONDS)
    report["gpu_idle"] = _stats(idle)
    print(f"[probe] idle: {report['gpu_idle']}", flush=True)

    # Absorb CUDA context creation and first-call compilation before timing.
    # Without this the first repeat ran 47 s against 21 s for the others.
    warm = _gpu_load(WARMUP_SECONDS)
    report["warmup"] = warm
    print(f"[probe] warm-up: {warm}", flush=True)

    idle_w = float(report["gpu_idle"].get("median_w", 0.0))
    runs = []
    for i in range(REPEATS):
        sampler = _PowerSampler(POLL_SECONDS)
        sampler.start()
        t0 = time.monotonic()
        status = _gpu_load(LOAD_SECONDS)
        elapsed = time.monotonic() - t0
        sampler.request_stop()
        sampler.join(timeout=10)
        st = _stats(sampler.watts)
        joules = (st.get("mean_w", 0.0) - idle_w) * elapsed if st.get("n") else None
        runs.append(
            {
                "repeat": i,
                "load_status": status,
                "elapsed_s": round(elapsed, 2),
                "power": st,
                "idle_adjusted_joules": round(joules, 1) if joules is not None else None,
            }
        )
        print(f"[probe] repeat {i}: {runs[-1]}", flush=True)
    report["gpu_load_runs"] = runs

    js = [r["idle_adjusted_joules"] for r in runs if r["idle_adjusted_joules"] is not None]
    report["joule_repeat_spread_pct"] = (
        round(100.0 * (max(js) - min(js)) / max(abs(sum(js) / len(js)), 1e-9), 1)
        if len(js) > 1
        else None
    )
    idle_sd = float(report["gpu_idle"].get("sd_w", 0.0))
    report["verdict"] = {
        "cpu_joules_directly_measurable": report["rapl_readable"],
        "gpu_power_sampling_available": report["gpu_idle"].get("n", 0) > 0,
        "idle_baseline_sd_w": idle_sd,
        "idle_baseline_clean": idle_sd < 1.0,
        "note": (
            "idle sd above ~1 W means the node was not quiet while the baseline "
            "was taken; the baseline is subtracted from every workload figure, so "
            "Phase 1 must measure it on an unshared node"
        ),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[probe] wrote {OUT}", flush=True)
    print(json.dumps(report["verdict"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
