"""Append Jan-Mar 2026 ERA5-via-Open-Meteo weather to the per-country files.

Sibling of the 2025 step in docs/d224_2025_freeze.md: same producer (the
archive-api request in load_forecast.discovery.weather_etl, same centroid,
same variables, same UTC timezone), append-only, with an overlap-identity
check on 2025 that the producer has not moved. The 2025 rows on disk are never
rewritten; the check is recorded, not corrected.

Why not run_weather_etl: it rewrites a country's whole file from a fresh
fetch, which would replace the deposited 2006-2025 rows with whatever the
archive serves today.

Out:  data/processed/weather/era5_<cc>.parquet  (2026 rows appended)
      data/processed/weather/_backup_pre2026/    (the files as they were)
      reports/revision/weather2026_gates.csv
Run:  .venv/bin/python scripts/revision_2026_weather_append.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from load_forecast.discovery.weather_etl import (
    ARCHIVE_URL,
    COLUMN_RENAME,
    COUNTRY_CENTROIDS,
    HOURLY_VARIABLES,
)

WEATHER_DIR = Path("data/processed/weather")
BACKUP = WEATHER_DIR / "_backup_pre2026"
GATES = Path("reports/revision/weather2026_gates.csv")
END_2026 = pd.Timestamp("2026-03-31 23:00")
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


def fetch_year(cc: str, year: int, retries: int = 6) -> pd.DataFrame:
    """The same request _fetch_country_year issues, without its proxy handler."""
    lat, lon = COUNTRY_CENTROIDS[cc]
    params = {
        "latitude": f"{lat}",
        "longitude": f"{lon}",
        # the archive refuses end dates past today; the load window ends 2026-03-31
        "start_date": f"{year}-01-01",
        "end_date": min(pd.Timestamp(f"{year}-12-31"), END_2026).strftime("%Y-%m-%d"),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }
    url = ARCHIVE_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as exc:
            last = exc
            time.sleep(min(120, 5 * (2**attempt)))
    else:
        raise RuntimeError(f"open-meteo fetch failed for {cc} {year}: {last}")
    df = pd.DataFrame({"t": pd.to_datetime(payload["hourly"]["time"])})
    for var in HOURLY_VARIABLES:
        col = payload["hourly"].get(var)
        df[var] = (
            np.nan if col is None else pd.to_numeric(pd.Series(col), errors="coerce").to_numpy()
        )
    time.sleep(0.5)
    return df.rename(columns=COLUMN_RENAME)


def main() -> int:
    BACKUP.mkdir(exist_ok=True)
    gates = []
    cols = [COLUMN_RENAME[v] for v in HOURLY_VARIABLES]
    for cc in COUNTRIES:
        path = WEATHER_DIR / f"era5_{cc.lower()}.parquet"
        old = pd.read_parquet(path).sort_values("t").reset_index(drop=True)
        if old["t"].max() >= END_2026:
            print(f"  {cc}: already extended (max t {old['t'].max()}), skip", flush=True)
            continue
        shutil.copy2(path, BACKUP / path.name)
        fresh = pd.concat([fetch_year(cc, 2025), fetch_year(cc, 2026)], ignore_index=True)
        fresh = fresh.drop_duplicates(subset="t").sort_values("t").reset_index(drop=True)
        # overlap identity on 2025: same producer, same values?
        o25 = old[old["t"].dt.year == 2025].set_index("t")[cols]
        f25 = fresh[fresh["t"].dt.year == 2025].set_index("t")[cols].reindex(o25.index)
        d25 = float(np.nanmax(np.abs(o25.to_numpy() - f25.to_numpy())))
        n_nan_f25 = int(f25.isna().sum().sum())
        # the appended slice
        new = fresh[(fresh["t"] >= "2026-01-01") & (fresh["t"] <= END_2026)].reset_index(drop=True)
        grid = pd.date_range("2026-01-01", END_2026, freq="h")
        complete = len(new) == len(grid) and (new["t"].to_numpy() == grid.to_numpy()).all()
        n_nan_new = int(new[cols].isna().sum().sum())
        ok = complete and n_nan_new == 0
        gates.append(
            {
                "country": cc,
                "overlap2025_maxabs": d25,
                "overlap2025_nan": n_nan_f25,
                "rows2026": len(new),
                "grid_complete": complete,
                "nan2026": n_nan_new,
                "pass": ok,
            }
        )
        print(
            f"  {cc}: 2025 overlap max|d|={d25:.3g}  2026 rows={len(new)} "
            f"complete={complete} nan={n_nan_new} -> {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
        if not ok:
            continue
        out = pd.concat([old, new[old.columns]], ignore_index=True)
        out = out.drop_duplicates(subset="t").sort_values("t").reset_index(drop=True)
        assert out["t"].max() == END_2026 and len(out) == len(old) + len(new)
        out.to_parquet(path, index=False)
    g = pd.DataFrame(gates)
    GATES.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(GATES, index=False)
    n_fail = int((~g["pass"]).sum()) if len(g) else 0
    print(
        f"[weather-2026] {len(g) - n_fail}/{len(g)} countries appended; "
        f"2025 overlap max|d| across countries = {g['overlap2025_maxabs'].max():.3g}",
        flush=True,
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
