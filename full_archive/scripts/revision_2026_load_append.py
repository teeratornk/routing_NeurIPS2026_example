"""Append the Jan-Mar 2026 ENTSO-E Power Statistics rows to the per-country load files.

Why not etl_entsoe_multiyear.py: it rebuilds every file from all raw sources and
its cross-source dedupe (sort by t, keep last) is not deterministic where the
2015 and 2019 sources overlap, so a rebuild moved thousands of MW in those two
years. The deposited <=2025 rows are therefore kept verbatim and only the new
file's rows are parsed, through the same _parse_long_format (same Cov_ratio
rescale) the deposit used.

Source: https://www.entsoe.eu/publications/data/power-stats/2026/monthly_hourly_load_values_2026.csv
        retrieved 2026-08-27, partial year (CreateDate 2026-06-09), covering
        2026-01-01 .. 2026-03-31 23:00 UTC.
Out:    data/processed/multiyear/hourly_<cc>.parquet (rows appended)
        reports/revision/load2026_gates.csv
Run:    .venv/bin/python scripts/revision_2026_load_append.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from load_forecast.discovery.jepa.data import _parse_long_format

RAW = Path("data/ENTSO-E/monthly_hourly_load_values_2026.csv")
HOURLY_DIR = Path("data/processed/multiyear")
BACKUP = HOURLY_DIR / "_backup_pre2026"
GATES = Path("reports/revision/load2026_gates.csv")
END = pd.Timestamp("2026-03-31 23:00")
COUNTRIES = (
    "AT",
    "BE",
    "BG",
    "CH",
    "CZ",
    "DE",
    "ES",
    "FR",
    "GR",
    "HR",
    "HU",
    "IT",
    "LU",
    "MK",
    "NL",
    "PL",
    "PT",
    "RO",
    "SI",
    "SK",
)


def main() -> int:
    sha = hashlib.sha256(RAW.read_bytes()).hexdigest()
    with RAW.open("r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    sep = "\t" if first.count("\t") >= first.count(";") else ";"
    raw = pd.read_csv(RAW, sep=sep)
    raw.columns = [c.lstrip("﻿") for c in raw.columns]
    long = _parse_long_format(raw)
    long["country"] = long["country"].str.upper()
    long = long[(long["t"] >= "2026-01-01") & (long["t"] <= END)]
    long = long.sort_values("t").drop_duplicates(subset=["country", "t"], keep="last")
    gates = []
    for cc in COUNTRIES:
        path = HOURLY_DIR / f"hourly_{cc.lower()}.parquet"
        old = pd.read_parquet(path)
        ref = pd.read_parquet(BACKUP / path.name)
        untouched = (
            old.shape == ref.shape
            and (old["t"].to_numpy() == ref["t"].to_numpy()).all()
            and np.allclose(
                old["load"].to_numpy(), ref["load"].to_numpy(), equal_nan=True, rtol=0, atol=0
            )
        )
        if old["t"].max() >= pd.Timestamp("2026-01-01"):
            gates.append(
                {
                    "country": cc,
                    "status": "already extended",
                    "rows2026": int((old.t >= "2026-01-01").sum()),
                    "pass": True,
                }
            )
            continue
        add = long[long["country"] == cc][["t", "load"]].reset_index(drop=True)
        ok = untouched and len(add) > 0 and add["t"].is_monotonic_increasing and add["t"].is_unique
        gates.append(
            {
                "country": cc,
                "status": "appended" if ok else "FAIL",
                "rows2026": len(add),
                "first": str(add["t"].min()) if len(add) else "",
                "share_of_grid": round(len(add) / 2160, 4),
                "pass": ok,
            }
        )
        print(
            f"  {cc}: <=2025 untouched={untouched} rows2026={len(add)} ({len(add) / 2160:.3%} of the Q1 grid)",
            flush=True,
        )
        if ok:
            out = pd.concat([old, add], ignore_index=True).sort_values("t").reset_index(drop=True)
            out.to_parquet(path, index=False)
    g = pd.DataFrame(gates)
    g["source_sha256"] = sha
    GATES.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(GATES, index=False)
    n_fail = int((~g["pass"]).sum())
    print(f"[load-2026] {len(g) - n_fail}/{len(g)} countries; sha256 {sha[:16]}...", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
