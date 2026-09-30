"""Revision (reviewer comment 6) — TimesFM-2.5 zero-shot on the common protocol.

Second foundation model on the SAME apples-to-apples grid as the paper's
Chronos-2-Uni-ZS column, to test whether the FM coverage-decay finding
generalizes beyond Chronos-2.

Direct analogue of scripts/d216_gpu_chronos_quantiles.py, but swaps the FM
primitive for TimesFM-2.5 (``timesfm.TimesFM_2p5_200M_torch``), captured through
the SAME context-construction convention used by fm_tier1._predict_timesfm
(context = values[origin-context : origin], ends AT the anchor) and the SAME
common-protocol origin grid (load_forecast.eval.common_protocol._origin_positions:
representative rotating stride-24, frozen pre-2018, h in {1,24,168,720}).

Persists BOTH the point forecast (for MdAPE, matching how the Chronos point
column was scored) AND the decile quantiles (for cov80), so the FM's coverage
can be scored under the same protocol as the four-scale HBQ and the Chronos-2
column.

Output: <out>/timesfm_q_{cc}.parquet with columns
  cc, test_year, horizon, origin_ts, target_ts, actual,
  point, q10, q50, q90  (q10/q50/q90 from the 0.1/0.5/0.9 quantile heads).

Usage (isolated uv timesfm env on a GPU node):
  python scripts/revision_timesfm_zeroshot_quantiles.py \
      --countries AT BE BG CH CZ DE ES FR GR HR HU IT LU MK NL PL PT RO SI SK \
      --horizons 1 24 168 720 --context 2048 --out reports/d215/uq_timesfm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from load_forecast.eval.common_protocol import DEFAULT_TEST_YEARS, _origin_positions

PANEL = Path("data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode.parquet")
# HF-cached checkpoint + frozen revision (matches fm_tier1.TIMESFM_REVISION).
CHECKPOINT = "google/timesfm-2.5-200m-pytorch"
REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _quantile_index_map(q_arr: np.ndarray) -> dict[str, int]:
    """Map q10/q50/q90 -> last-axis index of the TimesFM quantile output.

    TimesFM-2.5 exposes the 9 deciles [0.1..0.9]. Depending on the release the
    forecast may return either 9 channels (deciles only) or 10 channels
    (channel 0 = mean/point, channels 1..9 = deciles). Resolve robustly.
    """
    n = q_arr.shape[-1]
    if n == 10:  # [mean, 0.1, 0.2, ..., 0.9]
        return {"q10": 1, "q50": 5, "q90": 9}
    if n == 9:  # [0.1, 0.2, ..., 0.9]
        return {"q10": 0, "q50": 4, "q90": 8}
    raise ValueError(f"unexpected TimesFM quantile channel count: {n}")


def run(
    countries,
    horizons,
    context,
    batch_size,
    out_dir: Path,
    years=DEFAULT_TEST_YEARS,
    prefix: str = "timesfm_q",
    panel_path: Path = PANEL,
) -> None:
    import timesfm

    print(f"[gpu-tfm] loading TimesFM-2.5 from {CHECKPOINT}@{REVISION[:8]}…", flush=True)
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CHECKPOINT, revision=REVISION)

    panel = pd.read_parquet(panel_path, columns=["country", "t", "load", "year", "hod"])
    out_dir.mkdir(parents=True, exist_ok=True)
    qmap = None

    for h in horizons:
        # TimesFM compile is horizon-specific -> (re)compile once per horizon.
        print(f"[gpu-tfm] compiling for horizon={h} (context={context})…", flush=True)
        model.compile(
            timesfm.ForecastConfig(
                max_context=context,
                max_horizon=h,
                normalize_inputs=True,
                per_core_batch_size=batch_size,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        for cc in countries:
            g = panel[panel["country"] == cc].sort_values("t").reset_index(drop=True)
            values = g["load"].to_numpy(dtype=np.float32)
            load = g["load"].to_numpy(dtype=float)
            ts = g["t"].to_numpy()
            n = len(g)
            cc_rows = []
            for ty in years:
                anchor = _origin_positions(g, ty, h)
                if not anchor.size:
                    continue
                origins = (anchor + 1).astype(
                    int
                )  # context ends AT anchor (iloc[:origin] exclusive)
                pt_all, q10_all, q50_all, q90_all = [], [], [], []
                for batch in _chunks(list(origins), batch_size):
                    contexts = [values[max(0, o - context) : o] for o in batch]
                    point_fc, quant_fc = model.forecast(horizon=h, inputs=contexts)
                    point_fc = np.asarray(point_fc, dtype=float)  # (b, h)
                    quant_fc = np.asarray(quant_fc, dtype=float)  # (b, h, Q)
                    if qmap is None:
                        qmap = _quantile_index_map(quant_fc)
                        print(
                            f"[gpu-tfm] point{point_fc.shape} quant{quant_fc.shape} -> {qmap}",
                            flush=True,
                        )
                    pt_all.extend(point_fc[:, h - 1])
                    q10_all.extend(quant_fc[:, h - 1, qmap["q10"]])
                    q50_all.extend(quant_fc[:, h - 1, qmap["q50"]])
                    q90_all.extend(quant_fc[:, h - 1, qmap["q90"]])
                tgt = anchor + h
                ok = tgt < n
                cc_rows.append(
                    pd.DataFrame(
                        {
                            "cc": cc,
                            "test_year": ty,
                            "horizon": h,
                            "origin_ts": ts[anchor][ok],
                            "target_ts": ts[tgt[ok]],
                            "actual": load[tgt[ok]],
                            "point": np.asarray(pt_all, dtype=float)[ok],
                            "q10": np.asarray(q10_all, dtype=float)[ok],
                            "q50": np.asarray(q50_all, dtype=float)[ok],
                            "q90": np.asarray(q90_all, dtype=float)[ok],
                        }
                    )
                )
                print(f"  {cc} h={h} ty={ty}: {anchor.size} origins", flush=True)
            if cc_rows:
                out_path = out_dir / f"{prefix}_{cc}_h{h}.parquet"
                df = pd.concat(cc_rows, ignore_index=True)
                df.to_parquet(out_path, index=False)
                ape = 100 * np.abs(df["point"] - df["actual"]) / np.abs(df["actual"])
                c80 = ((df["q10"] <= df["actual"]) & (df["actual"] <= df["q90"])).mean()
                print(
                    f"[gpu-tfm] wrote {out_path.name}: median point-APE={ape.median():.3f}%, "
                    f"pooled cov80={c80:.3f}",
                    flush=True,
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", required=True)
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 24, 168, 720])
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("reports/d215/uq_timesfm"))
    ap.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_TEST_YEARS))
    ap.add_argument("--prefix", default="timesfm_q")
    ap.add_argument(
        "--panel",
        type=Path,
        default=PANEL,
        help="hourly panel parquet; pass the ext2025 panel to score the 2025 window",
    )
    args = ap.parse_args()
    run(
        args.countries,
        args.horizons,
        args.context,
        args.batch_size,
        args.out,
        years=tuple(args.years),
        prefix=args.prefix,
        panel_path=args.panel,
    )


if __name__ == "__main__":
    main()
