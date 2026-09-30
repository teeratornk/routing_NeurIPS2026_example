"""TS-LIMITS: does reaching a horizon by rollout cost accuracy, at a fixed horizon?

The paper observes that Chronos-2 trails the structural composition at one and
two years and that it reaches those targets through nine and eighteen chained
autoregressive passes. It does NOT claim the passes cause the gap, because the
two move together: every comparison across horizons varies the horizon and the
pass count at once. This script separates them.

The trick is that the pass count is a free parameter. Chronos-2 emits
`num_output_patches` patches of 16 steps per forward call, capped by
`max_output_patches`, and `predict` accepts that cap as a keyword
(pipeline.py:574). The number of chained passes is therefore

    ceil(prediction_length / (max_output_patches * 16))

so holding the horizon at h=720 and lowering the cap forces the SAME model to
reach the SAME target in 1, 2, 4, 6, 9 or 15 passes. Crucially the rollout is
the library's own: it unrolls a set of quantile trajectories with probability
mass weights rather than feeding back the median, which the source notes would
"lead to uncertainty collapse" (pipeline.py:393). Hand-rolling on q50 would
measure a strawman the library was built to avoid.

h=720 is chosen because it is a paper horizon AND is inside the native window:
ceil(720/16) = 45 patches, so the 45-cap arm is a genuine DIRECT forecast in one
forward pass. That is the arm the causal claim needs and that no cross-horizon
comparison can supply.

The 9-pass arm is the one to watch. Nine is what the model actually spends to
reach h=8760. If nine passes at h=720 cost little, then pass count alone does
not explain the year-scale gap and the paper's hypothesis is wrong.

Out: reports/tslimits/rollout_ablation.parquet (per-origin, one row per arm)
Run: see scripts/tslimits_rollout_ablation.sbatch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))

PANEL = Path(
    "data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet"
)
EU19 = (
    "AT", "BE", "BG", "CH", "CZ", "DE", "ES", "FR", "GR", "HR",
    "HU", "IT", "LU", "NL", "PL", "PT", "RO", "SI", "SK",
)
QLEVELS = [0.1, 0.5, 0.9]
QNAME = {0.1: "q10", 0.5: "q50", 0.9: "q90"}
PATCH = 16  # chronos_config.output_patch_size, verified at the checkpoint
# caps chosen so the induced pass counts at h=720 are 1, 2, 4, 6, 9, 15
CAPS = (45, 23, 12, 8, 5, 3)


def _passes(h: int, cap: int) -> int:
    """Replicates get_num_output_patches' loop in pipeline.py:665."""
    remaining, n = h, 0
    while remaining > 0:
        remaining -= min(int(np.ceil(remaining / PATCH)), cap) * PATCH
        n += 1
    return n


def _origin_positions(g: pd.DataFrame, test_year: int, h: int) -> np.ndarray:
    from load_forecast.eval.common_protocol import _origin_positions as f

    return f(g, test_year, h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=720)
    # The selection window. Choosing a cap by looking at 2025 would be test-set
    # selection, which is exactly the discipline the rest of the study keeps, so
    # the cap is picked on 2024 and then frozen.
    ap.add_argument("--year", type=int, default=2025, choices=(2024, 2025))
    ap.add_argument("--countries", nargs="+", default=list(EU19))
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--checkpoint", default="amazon/chronos-2")
    ap.add_argument("--caps", nargs="+", type=int, default=list(CAPS))
    ap.add_argument("--out", type=Path, default=Path("reports/tslimits"))
    ap.add_argument("--tag", default="rollout_ablation")
    args = ap.parse_args()
    h = args.horizon

    plan = {cap: _passes(h, cap) for cap in args.caps}
    print(f"[abl] h={h}: cap -> passes {plan}", flush=True)
    assert len(set(plan.values())) == len(plan), f"caps give duplicate pass counts: {plan}"

    try:
        from chronos import Chronos2Pipeline
    except ImportError:
        from chronos.chronos2.pipeline import Chronos2Pipeline
    print(f"[abl] loading {args.checkpoint} ...", flush=True)
    pipe = Chronos2Pipeline.from_pretrained(args.checkpoint, device_map="auto")
    cfg = pipe.model.chronos_config
    print(f"[abl] native single-shot length {cfg.max_output_patches * cfg.output_patch_size}",
          flush=True)
    assert pipe.model.chronos_config.output_patch_size == PATCH, "patch size drifted"

    panel = pd.read_parquet(PANEL, columns=["country", "t", "load", "year", "hod"])
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for cc in args.countries:
        g = panel[panel["country"] == cc].sort_values("t").reset_index(drop=True)
        frame = pd.DataFrame({"t": g["t"].to_numpy(), "target": g["load"].to_numpy(dtype=float)})
        load = g["load"].to_numpy(dtype=float)
        ts = g["t"].to_numpy()
        anchor = _origin_positions(g, args.year, h)
        if not anchor.size:
            continue
        origins = (anchor + 1).tolist()

        for cap in args.caps:
            qpred = {q: [] for q in QLEVELS}
            for i in range(0, len(origins), args.batch_size):
                batch = origins[i : i + args.batch_size]
                ctx_rows, ids = [], []
                for off, origin in enumerate(batch):
                    sid = f"s{off}"
                    ids.append(sid)
                    c = frame.iloc[max(0, origin - args.context) : origin]
                    ctx_rows.append(
                        pd.DataFrame(
                            {"series_id": sid, "t": c["t"].to_numpy(),
                             "target": c["target"].to_numpy(dtype=float)}
                        )
                    )
                pred = pipe.predict_df(
                    pd.concat(ctx_rows, ignore_index=True),
                    prediction_length=h,
                    quantile_levels=QLEVELS,
                    id_column="series_id",
                    timestamp_column="t",
                    target="target",
                    context_length=args.context,
                    max_output_patches=cap,   # the whole experiment is this line
                )
                for sid in ids:
                    gp = pred[pred["series_id"].astype(str).eq(sid)]
                    if "t" in gp:
                        gp = gp.sort_values("t")
                    end = gp.iloc[h - 1]
                    for q in QLEVELS:
                        col = next((c for c in (q, str(q), f"{q:.1f}") if c in pred.columns), None)
                        qpred[q].append(float(end[col]) if col else np.nan)
            tgt = anchor + h
            d = {
                "cc": cc, "horizon": h, "cap": cap, "passes": plan[cap], "year": args.year,
                "origin_ts": ts[anchor], "target_ts": ts[tgt], "actual": load[tgt],
            }
            for q in QLEVELS:
                d[QNAME[q]] = np.asarray(qpred[q], dtype=float)
            df = pd.DataFrame(d)
            df["ape"] = 100.0 * (df["q50"] - df["actual"]).abs() / df["actual"].abs()
            rows.append(df)
            print(f"  {cc} cap={cap:>2} passes={plan[cap]:>2}: "
                  f"n={len(df)} median APE {df['ape'].median():.3f}", flush=True)
        # Write after every country. The first version of this script only wrote
        # at the end, so a failure on the last country would have thrown away
        # every GPU-hour before it.
        pd.concat(rows, ignore_index=True).to_parquet(
            args.out / f"{args.tag}.parquet", index=False
        )

    out = pd.concat(rows, ignore_index=True)
    n_nan = int(out[["q10", "q50", "q90"]].isna().sum().sum())
    assert not n_nan, f"{n_nan} unresolved quantile cells; check the column naming"
    # Each country must contribute the SAME origins to every arm, which is what
    # makes the arms paired. Countries need not match each other: Slovenia is one
    # origin short at h=720 because an anchor hour is missing, and an earlier
    # version of this assertion compared counts across countries and so would
    # have failed on legitimate data at the very end of a run.
    per_cc = out.groupby("cc")["cap"].value_counts().groupby(level=0).nunique()
    assert bool(per_cc.eq(1).all()), f"arms not paired within {list(per_cc[per_cc.ne(1)].index)}"
    out.to_parquet(args.out / f"{args.tag}.parquet", index=False)
    print("\n[abl] panel-median MdAPE by pass count, horizon held at "
          f"{h}:", flush=True)
    for cap in args.caps:
        s = out[out["cap"] == cap]
        pm = s.groupby("cc")["ape"].median().median()
        print(f"  passes={plan[cap]:>2} (cap {cap:>2}): {pm:.4f}", flush=True)
    print(f"[abl] wrote {args.out}/{args.tag}.parquet", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
