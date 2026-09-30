"""D216 GPU — Chronos-2 QUANTILE predictions on the common-protocol grid.

Companion to d216_common_protocol_uq.py. Persists Chronos-2 zero-shot quantile
forecasts (0.025/0.10/0.50/0.90/0.975) at the endpoint of each common-protocol
origin (representative stride-24, frozen pre-2018, h in {1,24,168,720}) so the
FM's coverage/CRPS can be scored under the SAME protocol as the four-scale HBQ.

Mirrors fm_tier1._predict_chronos2_with_pipeline's context construction (context
= frame.iloc[start:anchor+1], i.e. ends AT the anchor) but reads the quantile
columns at the horizon-th step instead of only the point column.

Output: reports/d215/uq/chronos_q_{cc}.parquet with columns
  cc, test_year, horizon, origin_ts, target_ts, actual, q025,q10,q50,q90,q975.

Usage (offline venv on GPU node):
  python scripts/d216_gpu_chronos_quantiles.py --countries DE --horizons 1 24 168 720
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from load_forecast.eval.common_protocol import DEFAULT_TEST_YEARS, _origin_positions

PANEL = Path("data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode.parquet")
QLEVELS = [0.025, 0.10, 0.50, 0.90, 0.975]
QNAME = {0.025: "q025", 0.10: "q10", 0.50: "q50", 0.90: "q90", 0.975: "q975"}


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _qcol(pred_df: pd.DataFrame, q: float):
    for cand in (q, str(q), f"{q:.3f}", f"{q:g}", f"{q:.2f}"):
        if cand in pred_df.columns:
            return cand
    return None


def run(
    countries,
    horizons,
    context,
    checkpoint,
    batch_size,
    out_dir: Path,
    years=DEFAULT_TEST_YEARS,
    prefix: str = "chronos_q",
    panel_path: Path = PANEL,
    stride: int = 24,
) -> None:
    try:
        from chronos import Chronos2Pipeline
    except ImportError:
        from chronos.chronos2.pipeline import Chronos2Pipeline
    print(f"[gpu-q] loading Chronos-2 from {checkpoint}…", flush=True)
    pipeline = Chronos2Pipeline.from_pretrained(checkpoint, device_map="auto")

    panel = pd.read_parquet(panel_path, columns=["country", "t", "load", "year", "hod"])
    out_dir.mkdir(parents=True, exist_ok=True)
    qcols_resolved = None

    for cc in countries:
        g = panel[panel["country"] == cc].sort_values("t").reset_index(drop=True)
        frame = pd.DataFrame({"t": g["t"].to_numpy(), "target": g["load"].to_numpy(dtype=float)})
        load = g["load"].to_numpy(dtype=float)
        ts = g["t"].to_numpy()
        rows = []
        for h in horizons:
            for ty in years:
                anchor = _origin_positions(g, ty, h, stride=stride)
                if not anchor.size:
                    continue
                origins = (anchor + 1).tolist()
                qpred = {q: [] for q in QLEVELS}
                for batch in _chunks(origins, batch_size):
                    ctx_rows, ids = [], []
                    for offset, origin in enumerate(batch):
                        sid = f"s{offset}"
                        ids.append(sid)
                        start = max(0, origin - context)
                        c = frame.iloc[start:origin]
                        ctx_rows.append(
                            pd.DataFrame(
                                {
                                    "series_id": sid,
                                    "t": c["t"].to_numpy(),
                                    "target": c["target"].to_numpy(dtype=float),
                                }
                            )
                        )
                    context_df = pd.concat(ctx_rows, ignore_index=True)
                    pred_df = pipeline.predict_df(
                        context_df,
                        prediction_length=h,
                        quantile_levels=QLEVELS,
                        id_column="series_id",
                        timestamp_column="t",
                        target="target",
                        context_length=context,
                    )
                    if qcols_resolved is None:
                        qcols_resolved = {q: _qcol(pred_df, q) for q in QLEVELS}
                        print(f"[gpu-q] quantile columns -> {qcols_resolved}", flush=True)
                    for sid in ids:
                        gpred = pred_df[pred_df["series_id"].astype(str).eq(sid)]
                        if "t" in gpred:
                            gpred = gpred.sort_values("t")
                        endpoint = gpred.iloc[h - 1]
                        for q in QLEVELS:
                            col = qcols_resolved[q]
                            qpred[q].append(float(endpoint[col]) if col else np.nan)
                tgt = anchor + h
                d = {
                    "cc": cc,
                    "test_year": ty,
                    "horizon": h,
                    "origin_ts": ts[anchor],
                    "target_ts": ts[tgt],
                    "actual": load[tgt],
                }
                for q in QLEVELS:
                    d[QNAME[q]] = np.asarray(qpred[q], dtype=float)
                rows.append(pd.DataFrame(d))
                print(f"  {cc} h={h} ty={ty}: {anchor.size} origins", flush=True)
        if rows:
            df = pd.concat(rows, ignore_index=True)
            df.to_parquet(out_dir / f"{prefix}_{cc}.parquet", index=False)
            # quick coverage sanity
            c80 = ((df["q10"] <= df["actual"]) & (df["actual"] <= df["q90"])).mean()
            print(f"[gpu-q] {cc} empirical cov80 (pooled) = {c80:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", required=True)
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 24, 168, 720])
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--checkpoint", default="amazon/chronos-2")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("reports/d215/uq"))
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(DEFAULT_TEST_YEARS),
        help="target years to score (default test years; pass 2013..2017 "
        "to build the conformal-calibration set on the frozen window)",
    )
    ap.add_argument(
        "--prefix",
        default="chronos_q",
        help="output filename prefix (chronos_q for test, chronos_qcal "
        "for the <=2017 calibration window)",
    )
    ap.add_argument(
        "--panel",
        type=Path,
        default=PANEL,
        help="hourly panel parquet; pass the ext2025 panel to score the 2025 window",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=24,
        help="origin stride in hours; 24 is the published rotating grid, 1 scores every hour "
        "(a dense-grid sensitivity, one cell, never the published protocol)",
    )
    args = ap.parse_args()
    run(
        args.countries,
        args.horizons,
        args.context,
        args.checkpoint,
        args.batch_size,
        args.out,
        years=tuple(args.years),
        prefix=args.prefix,
        panel_path=args.panel,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
