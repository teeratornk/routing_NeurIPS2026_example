"""Extend Eurostat quarterly real GDP (namq_10_gdp) through 2025-Q4.

The deposited T_y chain is lagged-GDP: T_y(2026) consumes the 2025 annual
increment, which the deposited gdp_quarterly.parquet (fetched to 2024-Q4)
cannot supply. Same dataset, same key, same helper as fetch_eurostat_macro;
the existing file is not touched. Gate: every (geo, period) <= 2024-Q4 in the
new file equals the deposited value (Eurostat revises chain-linked volumes,
so a mismatch is recorded, and the deposited <=2024 rows are kept in that
case so nothing before 2025 can move).

Out:  data/external/eurostat/gdp_quarterly_ext2026.parquet
      reports/revision/gdp2026_gates.csv
Run:  .venv/bin/python scripts/revision_2026_gdp_extend.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path("data/external/eurostat/gdp_quarterly.parquet")
NEW = Path("data/external/eurostat/gdp_quarterly_ext2026.parquet")
GATES = Path("reports/revision/gdp2026_gates.csv")


def _load(name):
    p = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    m = importlib.util.module_from_spec(p)
    sys.modules[name] = m
    p.loader.exec_module(m)
    return m


def main() -> int:
    fe = _load("fetch_eurostat_macro")
    df = fe._fetch_dataset_per_country(
        "namq_10_gdp", ["Q", "CLV_I15", "SCA", "B1GQ"], "2006-Q1", "2025-Q4"
    )
    df = df.rename(columns={"value": "gdp_q"})
    df["geo"] = df["geo"].replace({"UK": "GB", "EL": "GR"})
    df["period"] = pd.PeriodIndex(df["period"], freq="Q").to_timestamp()
    df = df[df["geo"].isin(fe.PANEL_COUNTRIES)].copy()
    old = pd.read_parquet(OLD)
    key = ["geo", "period"]
    m = old.merge(df, on=key, how="left", suffixes=("_old", "_new"))
    d = (m["gdp_q_old"] - m["gdp_q_new"]).abs()
    n_missing = int(m["gdp_q_new"].isna().sum())
    maxd = float(np.nanmax(d)) if len(d) else 0.0
    n2025 = df[df["period"].dt.year == 2025].groupby("geo").size()
    full2025 = sorted(n2025[n2025 == 4].index)
    gates = pd.DataFrame(
        [
            {
                "gate": "overlap<=2024 identical",
                "pass": maxd < 1e-9 and n_missing == 0,
                "detail": f"max|d|={maxd:.3g} missing={n_missing}",
            },
            {
                "gate": "2025 has four quarters",
                "pass": len(full2025) >= 19,
                "detail": f"{len(full2025)} geos: {' '.join(full2025)}",
            },
        ]
    )
    print(gates.to_string(index=False), flush=True)
    GATES.parent.mkdir(parents=True, exist_ok=True)
    gates.to_csv(GATES, index=False)
    # deposited rows kept verbatim; only periods after 2024-Q4 come from the new fetch
    add = df[df["period"] > old["period"].max()]
    out = (
        pd.concat([old, add[old.columns]], ignore_index=True)
        .sort_values(key)
        .reset_index(drop=True)
    )
    out.to_parquet(NEW, index=False)
    print(
        f"[gdp-2026] wrote {NEW} rows={len(out)} (added {len(add)}; max period {out['period'].max().date()})",
        flush=True,
    )
    return 0 if gates["pass"].all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
