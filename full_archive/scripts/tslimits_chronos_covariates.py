"""Chronos-2 with known-future covariates on the tslimits origin grid.

The deployed comparison is univariate: the structural model reads the civil
calendar, the national holiday list and a <=2017 temperature climatology,
and the foundation model reads the load history alone. Chronos-2 accepts
known-future covariates natively (predict_df's future_df), so this arm hands
it the same calendar functions the structural model has, and nothing else:
no realised weather, no GDP, no load-derived feature.

Arms (all known into the future, all deterministic functions of the calendar
and the frozen holiday table):
  cal      hour, day-of-week and day-of-year as sin/cos pairs; national
           holiday and bridge-day flags (the holiday layer's own rule)
  calclim  cal plus the <=2017 (country, month, hour) temperature climatology,
           recovered from the panel as level minus anomaly (a calendar function)

Everything else is d216_gpu_chronos_quantiles.py: same grid, same context
construction, same endpoint extraction, same quantile levels. Output schema is
identical, so the routing scripts can read these files through FM_PATTERNS.

Usage (GPU node):
  python scripts/tslimits_chronos_covariates.py --arm cal --countries CH --horizons 24 \
      --years 2025 --panel <ext2025> --out reports/tslimits/fm --prefix tslcov_cal_q
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from load_forecast.eval.common_protocol import DEFAULT_TEST_YEARS, _origin_positions

PANEL = Path(
    "data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet"
)
HOLIDAYS = Path("data/external/holidays/clean20_public_holidays.parquet")
QLEVELS = [0.025, 0.10, 0.50, 0.90, 0.975]
QNAME = {0.025: "q025", 0.10: "q10", 0.50: "q50", 0.90: "q90", 0.975: "q975"}
CAL = ("hod_sin", "hod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_hol", "is_bridge")
ARMS = {"cal": CAL, "calclim": (*CAL, "clim_T")}


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _qcol(pred_df: pd.DataFrame, q: float):
    for cand in (q, str(q), f"{q:.3f}", f"{q:g}", f"{q:.2f}"):
        if cand in pred_df.columns:
            return cand
    return None


def _holiday_keys(cc: str) -> tuple[set, set]:
    """National holidays and bridge days, the rule of exp_holiday_layer."""
    hol = pd.read_parquet(HOLIDAYS)
    hol = hol[(hol["country"] == cc) & hol["global_holiday"]]
    dates = set(pd.to_datetime(hol["date"]).dt.normalize())
    day = pd.Timedelta(days=1)
    bridge = set()
    for d in dates:
        if d.weekday() == 3:
            bridge.add(d + day)
        if d.weekday() == 1:
            bridge.add(d - day)
    bridge = {d for d in bridge if d not in dates and d.weekday() < 5}
    return dates, bridge


def covariates(g: pd.DataFrame, cc: str) -> pd.DataFrame:
    """Per-hour known-future covariates for one country's full series."""
    t = pd.DatetimeIndex(g["t"])
    hol, bridge = _holiday_keys(cc)
    day = t.normalize()
    out = pd.DataFrame(
        {
            "t": g["t"].to_numpy(),
            "hod_sin": np.sin(2 * np.pi * t.hour / 24.0),
            "hod_cos": np.cos(2 * np.pi * t.hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * t.dayofweek / 7.0),
            "dow_cos": np.cos(2 * np.pi * t.dayofweek / 7.0),
            "doy_sin": np.sin(2 * np.pi * (t.dayofyear - 1) / 365.25),
            "doy_cos": np.cos(2 * np.pi * (t.dayofyear - 1) / 365.25),
            "is_hol": np.asarray([d in hol for d in day], dtype=float),
            "is_bridge": np.asarray([d in bridge for d in day], dtype=float),
            # level minus anomaly is the frozen <=2017 climatology (gated
            # year-invariant in the panel appends), so it is known ahead
            "clim_T": (g["T_air"] - g["d_T_air_hourly"]).to_numpy(dtype=float),
        }
    )
    assert not out.isna().any().any(), f"{cc}: NaN covariate"
    return out


def run(
    countries,
    horizons,
    context,
    checkpoint,
    batch_size,
    out_dir: Path,
    arm: str,
    years=DEFAULT_TEST_YEARS,
    prefix: str = "tslcov_q",
    panel_path: Path = PANEL,
    max_output_patches: int | None = None,
) -> None:
    try:
        from chronos import Chronos2Pipeline
    except ImportError:
        from chronos.chronos2.pipeline import Chronos2Pipeline
    cov_cols = list(ARMS[arm])
    print(f"[cov] arm={arm} covariates={cov_cols}", flush=True)
    pipeline = Chronos2Pipeline.from_pretrained(checkpoint, device_map="auto")
    panel = pd.read_parquet(
        panel_path, columns=["country", "t", "load", "year", "hod", "T_air", "d_T_air_hourly"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    qcols_resolved = None
    for cc in countries:
        g = panel[panel["country"] == cc].sort_values("t").reset_index(drop=True)
        cov = covariates(g, cc)
        frame = pd.DataFrame({"t": g["t"].to_numpy(), "target": g["load"].to_numpy(dtype=float)})
        for c in cov_cols:
            frame[c] = cov[c].to_numpy()
        load = g["load"].to_numpy(dtype=float)
        ts = g["t"].to_numpy()
        rows = []
        for h in horizons:
            for ty in years:
                anchor = _origin_positions(g, ty, h)
                if not anchor.size:
                    continue
                origins = (anchor + 1).tolist()
                qpred = {q: [] for q in QLEVELS}
                for batch in _chunks(origins, batch_size):
                    ctx_rows, fut_rows, ids = [], [], []
                    for offset, origin in enumerate(batch):
                        sid = f"s{offset}"
                        ids.append(sid)
                        start = max(0, origin - context)
                        c = frame.iloc[start:origin]
                        f = frame.iloc[origin : origin + h]
                        assert len(f) == h, f"{cc}: future window truncated at origin {origin}"
                        ctx_rows.append(
                            pd.DataFrame(
                                {
                                    "series_id": sid,
                                    "t": c["t"].to_numpy(),
                                    "target": c["target"].to_numpy(dtype=float),
                                    **{k: c[k].to_numpy(dtype=float) for k in cov_cols},
                                }
                            )
                        )
                        fut_rows.append(
                            pd.DataFrame(
                                {
                                    "series_id": sid,
                                    "t": f["t"].to_numpy(),
                                    **{k: f[k].to_numpy(dtype=float) for k in cov_cols},
                                }
                            )
                        )
                    context_df = pd.concat(ctx_rows, ignore_index=True)
                    future_df = pd.concat(fut_rows, ignore_index=True)
                    pred_df = pipeline.predict_df(
                        context_df,
                        future_df=future_df,
                        prediction_length=h,
                        quantile_levels=QLEVELS,
                        id_column="series_id",
                        timestamp_column="t",
                        target="target",
                        context_length=context,
                        # the frozen long-lead partition (48 at h=8760, 56 at
                        # h=17520); None is the library default, as for h<=720
                        **(
                            {"max_output_patches": max_output_patches} if max_output_patches else {}
                        ),
                    )
                    if qcols_resolved is None:
                        qcols_resolved = {q: _qcol(pred_df, q) for q in QLEVELS}
                        print(f"[cov] quantile columns -> {qcols_resolved}", flush=True)
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
            ape = 100.0 * (df["q50"] - df["actual"]).abs() / df["actual"].abs()
            print(f"[cov] {cc} median point-APE = {ape.median():.3f}%", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--countries", nargs="+", required=True)
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 24, 168, 720])
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--checkpoint", default="amazon/chronos-2")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("reports/tslimits/fm"))
    ap.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_TEST_YEARS))
    ap.add_argument(
        "--prefix", required=True, help="output prefix; one per arm, never the univariate one"
    )
    ap.add_argument("--panel", type=Path, default=PANEL)
    ap.add_argument(
        "--max-output-patches",
        type=int,
        default=None,
        help="rollout cap for the long leads (48 at 8760, 56 at 17520 are the frozen ones)",
    )
    args = ap.parse_args()
    run(
        args.countries,
        args.horizons,
        args.context,
        args.checkpoint,
        args.batch_size,
        args.out,
        args.arm,
        years=tuple(args.years),
        prefix=args.prefix,
        panel_path=args.panel,
        max_output_patches=args.max_output_patches,
    )


if __name__ == "__main__":
    main()
