"""TS-LIMITS Phase 3: joules per forecast for real foundation-model inference.

Phase 0 validated the method on a synthetic matmul (idle-adjusted joules
repeatable to 1.5%) but measured nothing about the models. This measures actual
Chronos-2 and TimesFM-2.5 serving on the same request shapes the paper scores:
context 2048, horizons 24 and 720, batch 1 / 32 / 256.

Two energy numbers are reported per configuration, and the distinction is not
cosmetic:

    marginal    (mean power - idle) * elapsed / forecasts. What is actually
                saved by declining one call on a GPU that stays up regardless.
    provisioned  mean power * elapsed / forecasts. What the call costs if the
                accelerator is provisioned for this workload and could be
                scaled down.

The H200 idles near 76 W, so on short batches the two differ by an order of
magnitude. Reporting only one would let the paper pick whichever flatters the
argument, so both go in the table.

Cold and warm are separated: model load, per-horizon compile and the first call
are timed individually, then steady state is measured over a window long enough
for the integration to be stable. Measured outcome: TimesFM's compile() returns
below our timing resolution, so no separate cost is attributed to it, and cold
start is confined to the first call of a process (19.9x the warm latency for
Chronos-2, 6.9x for TimesFM-2.5) rather than recurring per horizon.

Requires an unshared node. The idle baseline is subtracted from every figure
and Phase 0 measured it at sd 12.17 W with co-resident jobs against 0.056 W
quiet, so a contaminated baseline is recorded and flagged rather than hidden.

Out: reports/tslimits/energy_inference_{model}.json
Run: sbatch scripts/tslimits_energy_inference.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PANEL = Path(
    "data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet"
)
CHRONOS_CKPT = "amazon/chronos-2"
CONTEXT = 2048
HORIZONS = (1, 24, 168, 720)
BATCHES = (1, 32)
REPEATS = 5
IDLE_SECONDS = 20.0
# A settled loaded-resident baseline: the power an idle service draws with the
# model already resident. The old "between windows" sample was taken seconds
# after inference stopped, so it was still on the decay curve and over-stated
# resident idle. Wait for the card to settle, then sample for as long as the
# quiet baseline.
RESIDENT_SETTLE_SECONDS = 90.0
RESIDENT_SAMPLE_SECONDS = 30.0
MIN_WARM_SECONDS = 20.0
POLL_SECONDS = 0.1


def _assert_single_gpu() -> None:
    """Refuse to measure when more than one GPU is visible.

    --exclusive grants every GPU on the node regardless of --gres, so without
    an explicit CUDA_VISIBLE_DEVICES the pipelines' device_map="auto" shards
    the model across cards. That failed loudly for Chronos, but the dangerous
    case is the one that does not: a sharded model whose power is sampled on a
    single card yields a fraction of the true joules and looks plausible.
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n = len([x for x in vis.split(",") if x.strip()])
    if n != 1:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={vis!r} exposes {n} GPUs; this measurement "
            "requires exactly one. Set CUDA_VISIBLE_DEVICES=0 in the sbatch."
        )


def _timesfm_ids() -> tuple[str, str]:
    """Checkpoint and MODEL revision, read from the canonical runner.

    Duplicating these caused the bug this replaces: the first version hardcoded
    the timesfm *package* git revision (the one pinned in the sbatch's uv line)
    as the *model* revision. That hash is not in the local HF cache, so on an
    offline node from_pretrained tried the Hub and raised
    LocalEntryNotFoundError, after the Chronos half had already run.

    Parsed rather than imported: executing that module pulls its own imports,
    and this must not be able to fail for a reason unrelated to two strings.
    """
    src = (Path(__file__).parent / "revision_timesfm_zeroshot_quantiles.py").read_text(
        encoding="utf-8"
    )
    ck = re.search(r'^CHECKPOINT = "([^"]+)"', src, re.M)
    rv = re.search(r'^REVISION = "([^"]+)"', src, re.M)
    if not (ck and rv):
        raise RuntimeError("could not read CHECKPOINT/REVISION from the timesfm runner")
    return ck.group(1), rv.group(1)


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


class _PowerSampler(threading.Thread):
    """Sample GPU power concurrently with the workload.

    NOT _stop: threading.Thread defines a private _stop() that join() calls,
    so shadowing it with an Event makes join() raise "'Event' object is not
    callable".
    """

    def __init__(self, poll_s: float) -> None:
        super().__init__(daemon=True)
        self.poll_s = poll_s
        self.watts: list[float] = []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            w = _read_gpu_power()
            if w is not None:
                self.watts.append(w)
            self._halt.wait(self.poll_s)

    def request_stop(self) -> None:
        self._halt.set()


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
        "median_w": round(s[n // 2], 2),
        "max_w": round(s[-1], 2),
    }


@dataclass(frozen=True)
class Sweep:
    """What to measure. Explicit rather than module constants, because the grid
    a reviewer asks for differs from the one first reported."""

    horizons: tuple[int, ...]
    batches: tuple[int, ...]
    repeats: int
    warm_seconds: float


def _dispersion(xs: list[float]) -> dict[str, Any]:
    """Median and IQR across repeats. One run is a number, not a measurement."""
    if not xs:
        return {"n_rep": 0}
    a = np.asarray(xs, dtype=float)
    med = float(np.median(a))
    q25, q75 = (float(np.percentile(a, q)) for q in (25, 75))
    return {
        "n_rep": int(a.size),
        "median": round(med, 4),
        "q25": round(q25, 4),
        "q75": round(q75, 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "rel_iqr_pct": round(100.0 * (q75 - q25) / med, 2) if med > 0 else None,
    }


def _environment() -> dict[str, Any]:
    """Which machine and stack produced these joules. Absent, the number is
    not reproducible even in principle."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        env["gpu_name_torch"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception as exc:
        env["torch_error"] = f"{type(exc).__name__}: {exc}"[:120]
    fields = (
        "name,driver_version,power.limit,power.default_limit,"
        "clocks.max.sm,memory.total,persistence_mode"
    )
    try:
        res = subprocess.run(
            ["nvidia-smi", "-i", _target_gpu(), f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if res.returncode == 0:
            vals = [v.strip() for v in res.stdout.split(",")]
            env["gpu"] = dict(zip(fields.split(","), vals, strict=False))
    except Exception as exc:
        env["gpu_error"] = f"{type(exc).__name__}: {exc}"[:120]
    return env


def _contexts(panel: pd.DataFrame, n_series: int) -> list[np.ndarray]:
    """`n_series` distinct context windows of length CONTEXT, cycling countries."""
    out: list[np.ndarray] = []
    ccs = sorted(panel["country"].unique())
    per_cc = {
        cc: panel[panel["country"] == cc].sort_values("t")["load"].to_numpy(dtype=np.float32)
        for cc in ccs
    }
    i = 0
    while len(out) < n_series:
        cc = ccs[i % len(ccs)]
        v = per_cc[cc]
        # stride the window start so repeated series are not identical
        start = len(v) - CONTEXT - 1 - (i // len(ccs)) * 24
        if start < 0:
            start = 0
        out.append(v[start : start + CONTEXT])
        i += 1
    return out


def _run_chronos(panel: pd.DataFrame, report: dict[str, Any], sweep: Sweep) -> None:
    try:
        from chronos import Chronos2Pipeline
    except ImportError:
        from chronos.chronos2.pipeline import Chronos2Pipeline

    t0 = time.monotonic()
    pipe = Chronos2Pipeline.from_pretrained(CHRONOS_CKPT, device_map="auto")
    report["model_load_s"] = round(time.monotonic() - t0, 2)
    print(f"[energy] chronos loaded in {report['model_load_s']}s", flush=True)
    report["gpu_idle_resident"] = _resident_baseline()

    def call(ctxs: list[np.ndarray], h: int) -> None:
        frames = [
            pd.DataFrame(
                {
                    "series_id": f"s{j}",
                    "t": pd.date_range("2020-01-01", periods=len(c), freq="h"),
                    "target": c.astype(float),
                }
            )
            for j, c in enumerate(ctxs)
        ]
        pipe.predict_df(
            pd.concat(frames, ignore_index=True),
            prediction_length=h,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="series_id",
            timestamp_column="t",
            target="target",
            context_length=CONTEXT,
        )

    _measure(
        panel,
        call,
        report,
        None,
        sweep,
        report["gpu_idle_preload"]["median_w"],
        report["gpu_idle_resident"]["median_w"],
    )


def _run_timesfm(panel: pd.DataFrame, report: dict[str, Any], sweep: Sweep) -> None:
    import timesfm

    t0 = time.monotonic()
    ckpt, revision = _timesfm_ids()
    print(f"[energy] timesfm {ckpt}@{revision[:8]}", flush=True)
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(ckpt, revision=revision)
    report["model_load_s"] = round(time.monotonic() - t0, 2)
    print(f"[energy] timesfm loaded in {report['model_load_s']}s", flush=True)
    report["gpu_idle_resident"] = _resident_baseline()

    state: dict[str, Any] = {"model": model}

    def compile_fn(h: int, bs: int) -> float:
        t = time.monotonic()
        model.compile(
            timesfm.ForecastConfig(
                max_context=CONTEXT,
                max_horizon=h,
                normalize_inputs=True,
                per_core_batch_size=bs,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        return time.monotonic() - t

    def call(ctxs: list[np.ndarray], h: int) -> None:
        state["model"].forecast(horizon=h, inputs=list(ctxs))

    _measure(
        panel,
        call,
        report,
        compile_fn,
        sweep,
        report["gpu_idle_preload"]["median_w"],
        report["gpu_idle_resident"]["median_w"],
    )


def _idle_baseline() -> float:
    """Median idle watts over IDLE_SECONDS. Raises if the meter is dead."""
    idle: list[float] = []
    t_end = time.monotonic() + IDLE_SECONDS
    while time.monotonic() < t_end:
        w = _read_gpu_power()
        if w is not None:
            idle.append(w)
        time.sleep(POLL_SECONDS)
    st = _stats(idle)
    if st["n"] == 0:
        raise RuntimeError("no GPU power samples for the idle baseline")
    return float(st["median_w"])


def _resident_baseline() -> dict[str, float]:
    """Idle power with the model loaded and no inference running.

    Sampled after RESIDENT_SETTLE_SECONDS of quiet so the card is off the
    post-inference decay curve. This is the baseline that answers "what does
    declining a call save on a service that stays up", which the between-window
    sample was standing in for and could not support.
    """
    print(
        f"[energy] settling {RESIDENT_SETTLE_SECONDS:.0f}s for the loaded-resident baseline",
        flush=True,
    )
    t_end = time.monotonic() + RESIDENT_SETTLE_SECONDS
    while time.monotonic() < t_end:
        time.sleep(1.0)
    watts: list[float] = []
    t_end = time.monotonic() + RESIDENT_SAMPLE_SECONDS
    while time.monotonic() < t_end:
        w = _read_gpu_power()
        if w is not None:
            watts.append(w)
        time.sleep(POLL_SECONDS)
    st = _stats(watts)
    if st["n"] == 0:
        raise RuntimeError("no GPU power samples for the loaded-resident baseline")
    print(
        f"[energy] loaded-resident idle {st['median_w']:.2f} W "
        f"(sd {st['sd_w']:.2f}, n={st['n']})",
        flush=True,
    )
    return st


def _measure(
    panel: pd.DataFrame,
    call: Any,
    report: dict[str, Any],
    compile_fn: Any,
    sweep: Sweep,
    idle_preload_w: float,
    idle_resident_w: float,
) -> None:
    """Sweep (horizon, batch), repeating each configuration `sweep.repeats` times.

    TWO idle baselines are recorded, because the choice between them moves the
    marginal figure by up to a factor of five and both are defensible.

      preload   sampled once before the model is loaded, on a quiet card. This
                is the convention the first submission used and the one the
                marginal figure is reported against: it answers "what does this
                call cost above an idle accelerator".
      between   sampled between measurement windows, inside the sweep. This
                captures thermal and co-tenancy drift, but on this card it reads
                126-130 W against a preload 82 W, because the GPU has not
                returned to idle between workloads. Subtracting it removes most
                of the signal.

    Reporting both is the point. Provisioned joules, which subtract nothing,
    reproduce across independent runs to about 1%; marginal joules do not, and a
    paper that quotes one without the baseline convention is quoting an
    arbitrary choice.

    Repeats share one model load and one GPU thermal state, so what is reported
    is within-process run-to-run variation. Re-running the job would be the
    stronger claim and the paper should not conflate the two.
    """
    runs: list[dict[str, Any]] = []
    for h in sweep.horizons:
        for bs in sweep.batches:
            for rep in range(sweep.repeats):
                entry: dict[str, Any] = {"horizon": h, "batch_size": bs, "repeat": rep}
                try:
                    idle_between = _idle_baseline()
                    entry["idle_preload_w"] = round(idle_preload_w, 3)
                    entry["idle_between_w"] = round(idle_between, 3)
                    ctxs = _contexts(panel, bs)
                    if compile_fn is not None and rep == 0:
                        entry["compile_s"] = round(compile_fn(h, bs), 2)
                    t = time.monotonic()
                    call(ctxs, h)
                    entry["first_call_s"] = round(time.monotonic() - t, 3)

                    sampler = _PowerSampler(POLL_SECONDS)
                    sampler.start()
                    t0 = time.monotonic()
                    lat: list[float] = []
                    while time.monotonic() - t0 < sweep.warm_seconds:
                        t1 = time.monotonic()
                        call(ctxs, h)
                        lat.append(time.monotonic() - t1)
                    elapsed = time.monotonic() - t0
                    sampler.request_stop()
                    sampler.join(timeout=10)

                    power = _stats(sampler.watts)
                    # Index, never .get(..., 0.0). An empty power window used to
                    # fall through to 0 W and report 0 J per forecast -- the most
                    # flattering possible number for the expensive model -- inside
                    # a success-shaped JSON flagged as a clean baseline.
                    if power["n"] == 0:
                        raise RuntimeError("no GPU power samples in the measurement window")
                    n_fc = len(lat) * bs
                    mean_w = float(power["mean_w"])
                    entry.update(
                        {
                            "calls": len(lat),
                            "forecasts": n_fc,
                            "elapsed_s": round(elapsed, 2),
                            "power": power,
                            # the sampler forks nvidia-smi per reading, so the
                            # realised interval exceeds POLL_SECONDS; report what
                            # actually happened rather than the nominal constant
                            "effective_poll_s": round(elapsed / power["n"], 4),
                            "latency_p50_s": round(float(np.percentile(lat, 50)), 4),
                            "latency_p95_s": round(float(np.percentile(lat, 95)), 4),
                            # headline: quiet-card baseline, the stated convention
                            "marginal_j_per_forecast": round(
                                (mean_w - idle_preload_w) * elapsed / n_fc, 4
                            ),
                            # a settled idle-with-model-resident baseline: what
                            # declining a call saves on a service that stays up
                            "marginal_j_resident_baseline": round(
                                (mean_w - idle_resident_w) * elapsed / n_fc, 4
                            ),
                            # the between-window sample, kept only as a
                            # sensitivity: it is taken seconds after inference
                            # stops, so it sits on the decay curve and is not a
                            # steady resident idle
                            "marginal_j_between_baseline": round(
                                (mean_w - idle_between) * elapsed / n_fc, 4
                            ),
                            "provisioned_j_per_forecast": round(mean_w * elapsed / n_fc, 4),
                        }
                    )
                    print(
                        f"[energy] h={h} bs={bs} rep={rep} "
                        f"marginal={entry['marginal_j_per_forecast']} J",
                        flush=True,
                    )
                except Exception as exc:  # OOM at a large batch is a result, not a crash
                    entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
                    print(f"[energy] {h}/{bs}/rep{rep} FAILED: {entry['error']}", flush=True)
                runs.append(entry)
    report["runs"] = runs

    agg: list[dict[str, Any]] = []
    for h in sweep.horizons:
        for bs in sweep.batches:
            ok = [
                r for r in runs
                if r["horizon"] == h and r["batch_size"] == bs and "error" not in r
            ]
            if not ok:
                continue
            agg.append(
                {
                    "horizon": h,
                    "batch_size": bs,
                    "marginal_j": _dispersion([r["marginal_j_per_forecast"] for r in ok]),
                    "provisioned_j": _dispersion([r["provisioned_j_per_forecast"] for r in ok]),
                    "latency_p50_s": _dispersion([r["latency_p50_s"] for r in ok]),
                    "marginal_j_resident": _dispersion(
                        [r["marginal_j_resident_baseline"] for r in ok]
                    ),
                    "marginal_j_between": _dispersion(
                        [r["marginal_j_between_baseline"] for r in ok]
                    ),
                    "idle_between_w": _dispersion([r["idle_between_w"] for r in ok]),
                }
            )
    report["aggregate"] = agg



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("chronos", "timesfm"), required=True)
    ap.add_argument("--panel", type=Path, default=PANEL)
    ap.add_argument("--out", type=Path, default=Path("reports/tslimits"))
    ap.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    ap.add_argument("--batches", nargs="+", type=int, default=list(BATCHES))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--warm-seconds", type=float, default=MIN_WARM_SECONDS)
    args = ap.parse_args()
    sweep = Sweep(
        horizons=tuple(args.horizons),
        batches=tuple(args.batches),
        repeats=int(args.repeats),
        warm_seconds=float(args.warm_seconds),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "model": args.model,
        "host": socket.gethostname(),
        "context": CONTEXT,
        "poll_seconds_nominal": POLL_SECONDS,
        "idle_seconds": IDLE_SECONDS,
        "warm_seconds": sweep.warm_seconds,
        "repeats": sweep.repeats,
        "horizons": list(sweep.horizons),
        "batches": list(sweep.batches),
        "environment": _environment(),
    }
    print(f"[energy] host={report['host']} model={args.model}", flush=True)

    # The baseline that is actually subtracted is re-taken per repeat inside
    # _measure, so drift shows up as dispersion. This first reading is only a
    # gate: if the node is not quiet, say so before spending an hour on it.
    # Sampled BEFORE the model is loaded, on a quiet card. This is the baseline
    # the headline marginal figure subtracts, and it must be taken here rather
    # than between workloads: the card reads ~82 W idle and ~128 W between
    # windows, and subtracting the latter removes most of the measured signal.
    idle: list[float] = []
    t_end = time.monotonic() + IDLE_SECONDS
    while time.monotonic() < t_end:
        w = _read_gpu_power()
        if w is not None:
            idle.append(w)
        time.sleep(POLL_SECONDS)
    pre = _stats(idle)
    if pre["n"] == 0:
        raise RuntimeError("no GPU power samples for the pre-load idle baseline")
    report["gpu_idle_preload"] = pre
    report["idle_baseline_clean"] = bool(pre["sd_w"] < 1.0)
    print(f"[energy] preload idle {pre} clean={report['idle_baseline_clean']}", flush=True)

    _assert_single_gpu()
    panel = pd.read_parquet(args.panel, columns=["country", "t", "load"])
    if args.model == "chronos":
        _run_chronos(panel, report, sweep)
    else:
        _run_timesfm(panel, report, sweep)

    if not report["idle_baseline_clean"]:
        report["warning"] = (
            "idle sd above 1 W: the node was not quiet, and the baseline is "
            "subtracted from every marginal figure; re-run on an unshared node"
        )
    out = args.out / f"energy_inference_{args.model}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[energy] wrote {out}", flush=True)

    # A JSON of nothing but errors is not a measurement. Exit non-zero so the
    # job is marked FAILED rather than leaving a success-shaped artifact.
    ok = [r for r in report["runs"] if "error" not in r]
    print(f"[energy] {len(ok)}/{len(report['runs'])} configurations measured", flush=True)
    if not ok:
        print("[energy] every configuration failed; not a usable measurement", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
