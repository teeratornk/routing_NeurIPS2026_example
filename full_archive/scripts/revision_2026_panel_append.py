"""Append Jan-Mar 2026 rows to the frozen ext2025 hourly panel (post-release window).

Sibling of revision_2025_panel_append.py (D224), which is left untouched so the
2025 window keeps reproducing. APPEND-ONLY: the ext2025 rows are reused as the
frame itself; every fitted quantity for 2026 is extracted from that deposit or
recomputed by a causal chain that must reproduce the deposit on <=2025.

What 2026 needs that 2025 did not:
  - the 2025 GDP increment, because the deployed T_y chain is lagged-GDP and
    T_y(2026) consumes dlog(GDP/N)(2025). Taken from a new Eurostat vintage
    (gdp_quarterly_ext2026.parquet) as a within-vintage increment
    log(GDP_25/N_25) - log(GDP_24/N_24), so no vintage jump enters the chain.
  - a 2026 yearly row (the chain needs the row to exist; its partial-year load
    level is never read, only the increment into it).
  - population N(2026) := N(2025), the anchor-year causal convention.
  - 2026 public holidays (Nager.Date re-pull; the deposited <=2025 rows are
    kept, since the re-pull moved six Spanish observed dates in past years).
  - holiday_count_month for 2026: schema-only (the frozen build does not read
    it); filled by the nearest rule, 24 x unique holiday dates per month.

Gates (all must PASS or nothing is written):
  G-sw, G-hod, G-rmean, G-clim, G-ode, G-schema  as in D224, against ext2025
  G-hol     : holidays file <=2025 identical to the deposit; 2026 present
  G-gdp     : finite 2025 increment for >= 19 countries
  G-ty      : lagged chain on the ext2026 yearly panel reproduces ext2025's
              T_y_d211 on <=2025 and is finite for 2026
  G-le2025  : output rows with t < 2026 are the ext2025 frame itself

Out:  data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2026.parquet
      data/feature_store/multi_resolution/yearly/clean20_yearly_panel_d211_ext2026.parquet
      data/external/holidays/clean20_public_holidays.parquet (2026 rows appended; backup kept)
      reports/revision/append2026_gates.csv
Run:  .venv/bin/python scripts/revision_2026_panel_append.py
"""

from __future__ import annotations

import contextlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEPOSIT = Path(
    "data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet"
)
OUT_PANEL = Path(
    "data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2026.parquet"
)
YEARLY = Path("data/feature_store/multi_resolution/yearly/clean20_yearly_panel_d211.parquet")
YEARLY_OUT = Path(
    "data/feature_store/multi_resolution/yearly/clean20_yearly_panel_d211_ext2026.parquet"
)
GDP_NEW = Path("data/external/eurostat/gdp_quarterly_ext2026.parquet")
HOL = Path("data/external/holidays/clean20_public_holidays.parquet")
HOL_NEW = Path("data/external/holidays/clean20_public_holidays_ext2026.parquet")
HOL_BACKUP = Path("data/external/holidays/_backup_pre2026/clean20_public_holidays.parquet")
HOURLY_DIR = Path("data/processed/multiyear")
WEATHER_DIR = Path("data/processed/weather")
GATES_OUT = Path("reports/revision/append2026_gates.csv")
WEATHER_COLS = [
    "T_air",
    "dewpoint",
    "humidity",
    "cloud_cover",
    "wind_10m",
    "precipitation",
    "ghi",
    "dhi",
]
ANOM = {
    "d_T_air_hourly": "T_air",
    "d_dewpoint_hourly": "dewpoint",
    "d_humidity_hourly": "humidity",
    "d_cloud_cover_hourly": "cloud_cover",
    "d_wind_10m_hourly": "wind_10m",
    "d_ghi_hourly": "ghi",
    "d_HDD_hourly": "__HDD__",
    "d_CDD_hourly": "__CDD__",
    "d_x_H_ode": "x_H_ode",
    "d_x_C_ode": "x_C_ode",
}
TOL = 1e-8
GRID = pd.date_range("2026-01-01", "2026-03-31 23:00", freq="h")


def gate(gates, name, ok, detail=""):
    gates.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(f"  G-{name}: {'PASS' if ok else 'FAIL'}  {detail}", flush=True)
    return ok


def extend_holidays(gates) -> pd.DataFrame:
    """Deposited <=2025 rows verbatim, 2026 rows from the re-pull. Idempotent."""
    dep = pd.read_parquet(HOL)
    if int((dep["year"] == 2026).sum()) > 0:
        gate(gates, "hol", True, f"already extended; 2026 rows={int((dep.year == 2026).sum())}")
        return dep
    new = pd.read_parquet(HOL_NEW)
    key = ["country", "date", "name", "global_holiday"]
    o = dep.sort_values(key).reset_index(drop=True)
    n = new[new["year"] <= 2025].sort_values(key).reset_index(drop=True)
    n_moved = int(pd.concat([o[key], n[key]]).drop_duplicates(keep=False).shape[0] // 2)
    add = new[new["year"] == 2026].reset_index(drop=True)
    out = pd.concat([dep, add[dep.columns]], ignore_index=True)
    ok = len(add) > 0 and add["country"].nunique() == dep["country"].nunique()
    gate(
        gates,
        "hol",
        ok,
        f"deposit rows kept={len(dep)} 2026 rows added={len(add)} "
        f"countries={add['country'].nunique()} re-pull moved {n_moved} past dates (ignored)",
    )
    if ok:
        HOL_BACKUP.parent.mkdir(exist_ok=True)
        shutil.copy2(HOL, HOL_BACKUP)
        out.to_parquet(HOL, index=False)
    return out


def extend_yearly(gates, load26_mean: pd.Series, yearly: pd.DataFrame) -> pd.DataFrame:
    """Fill the 2025 GDP increment within the new vintage; append 2026 rows."""
    g = pd.read_parquet(GDP_NEW).rename(columns={"geo": "country"})
    g["year"] = pd.to_datetime(g["period"]).dt.year
    ann = g.groupby(["country", "year"])["gdp_q"].agg(["sum", "size"]).reset_index()
    ann = ann[ann["size"] == 4].rename(columns={"sum": "gdp_new"})[["country", "year", "gdp_new"]]
    y = yearly.copy()
    pop = y.set_index(["country", "year"])["population"]
    gn = ann.set_index(["country", "year"])["gdp_new"]
    rows, n_ok = [], 0
    for cc in sorted(y["country"].unique()):
        try:
            lpc25 = np.log(gn.loc[(cc, 2025)] / pop.loc[(cc, 2025)])
            lpc24 = np.log(gn.loc[(cc, 2024)] / pop.loc[(cc, 2024)])
            d25 = float(lpc25 - lpc24)
        except KeyError:
            d25 = np.nan
        m25 = (y["country"] == cc) & (y["year"] == 2025)
        assert m25.sum() == 1, cc
        assert np.isnan(float(y.loc[m25, "dlog_gdp_per_capita"].iloc[0])), (
            f"{cc}: 2025 increment already set"
        )
        if np.isfinite(d25):
            n_ok += 1
            y.loc[m25, "gdp_annual"] = float(gn.loc[(cc, 2025)])
            y.loc[m25, "log_gdp_per_capita"] = float(lpc25)
            y.loc[m25, "dlog_gdp_per_capita"] = d25
        pop25 = float(pop.loc[(cc, 2025)])
        ld = float(load26_mean.get(cc, np.nan))
        rows.append(
            {
                "t": pd.Timestamp("2026-01-01"),
                "country": cc,
                "year": 2026,
                "population": pop25,
                "load": ld,
                "log_load_per_capita": np.log(ld / pop25) if np.isfinite(ld) else np.nan,
                "log_population": np.log(pop25),
                "crisis_2022": 0,
                "crisis_2020": 0,
            }
        )
    gate(
        gates,
        "gdp",
        n_ok >= 19,
        f"finite within-vintage 2025 increment for {n_ok}/{y['country'].nunique()}",
    )
    new = pd.DataFrame(rows)
    for c in y.columns:
        if c not in new.columns:
            new[c] = np.nan
    out = (
        pd.concat([y, new[y.columns]], ignore_index=True)
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )
    for c in y.columns:
        if out[c].dtype != y[c].dtype:
            with contextlib.suppress(TypeError, ValueError):
                out[c] = out[c].astype(y[c].dtype)
    out.to_parquet(YEARLY_OUT, index=False)
    return out


def main():
    sys.path.insert(0, "src")
    from load_forecast.data_ingest import hourly_panel_extension as hpe
    from load_forecast.data_ingest.daily_panel_extension import _d211_chain_per_country
    from load_forecast.discovery.thermal_state import integrate_first_order_ode

    gates = []
    print("[2026-append] loading ext2025 deposit ...", flush=True)
    dep = pd.read_parquet(DEPOSIT).sort_values(["country", "t"]).reset_index(drop=True)
    assert int(dep["year"].max()) == 2025, "deposit must be the ext2025 panel"
    countries = sorted(dep["country"].unique())
    cols = list(dep.columns)
    dep["__HDD__"] = np.maximum(18.0 - dep["T_air"], 0.0)
    dep["__CDD__"] = np.maximum(dep["T_air"] - 22.0, 0.0)

    # ---- frozen literals and extracted constants, gated exactly as in D224 ----
    s_dep = hpe.apply_seasonal_climatology(dep["month"].to_numpy())
    w_dep = hpe.apply_dow_climatology(dep["dow"].to_numpy())
    ds = float(np.nanmax(np.abs(s_dep - dep["S_seasonal_universal"].to_numpy())))
    dw = float(np.nanmax(np.abs(w_dep - dep["W_dow_universal"].to_numpy())))
    gate(gates, "sw", ds < TOL and dw < TOL, f"maxS={ds:.2e} maxW={dw:.2e}")
    prof = {
        yr: dep[dep.year == yr].groupby("hod")["H_hod_universal"].mean().sort_index()
        for yr in (2023, 2024)
    }
    dh = float((prof[2023] - prof[2024]).abs().max())
    spr = float(
        (
            dep[dep.year == 2024].groupby("hod")["H_hod_universal"].agg(lambda v: v.max() - v.min())
        ).max()
    )
    gate(gates, "hod", dh < TOL and spr < 1e-12, f"d23v24={dh:.2e} spread={spr:.2e}")
    h24 = prof[2024]
    rmeans = dep.groupby("country")["R_hour_country_mean"]
    gate(
        gates,
        "rmean",
        int(rmeans.nunique().max()) == 1,
        f"nunique_max={int(rmeans.nunique().max())}",
    )
    rm_cc = rmeans.first()
    clims, ok_all = {}, True
    for acol, bcol in ANOM.items():
        c23 = (
            dep[dep.year == 2023]
            .assign(clim=lambda d, b=bcol, a=acol: d[b] - d[a])
            .groupby(["country", "month", "hod"])["clim"]
            .mean()
        )
        c24 = (
            dep[dep.year == 2024]
            .assign(clim=lambda d, b=bcol, a=acol: d[b] - d[a])
            .groupby(["country", "month", "hod"])["clim"]
            .mean()
        )
        dmax = float((c23 - c24).abs().max())
        ok_all &= dmax < 1e-6
        clims[acol] = c24
    gate(gates, "clim", ok_all, f"{len(ANOM)} climatologies")

    # ---- holidays, yearly panel, T_y chain ----
    hol = extend_holidays(gates)
    hol["date"] = pd.to_datetime(hol["date"])
    hol26 = hol[hol["year"] == 2026]
    hcm26 = (
        hol26.assign(month=hol26["date"].dt.month).groupby(["country", "month"])["date"].nunique()
        * 24.0
    )
    load26 = {}
    for cc in countries:
        lo = pd.read_parquet(HOURLY_DIR / f"hourly_{cc.lower()}.parquet")
        lo = lo[(lo.t >= "2026-01-01") & (lo.t <= GRID[-1])]
        s = lo.set_index("t")["load"].reindex(GRID)
        load26[cc] = s.where(s > 0)
    load26_mean = pd.Series({cc: float(v.mean()) for cc, v in load26.items()})
    yearly = pd.read_parquet(YEARLY)
    yearly_ext = extend_yearly(gates, load26_mean, yearly)
    ty = _d211_chain_per_country(yearly_ext, gdp_mode="lagged")
    tym = dep[["country", "year"]].merge(ty, on=["country", "year"], how="left")
    dty = float(np.nanmax(np.abs(tym["T_y_d211"].to_numpy() - dep["T_y_d211"].to_numpy())))
    ty26 = ty[ty.year == 2026]
    fin26 = int(ty26["T_y_d211"].notna().sum())
    gate(
        gates,
        "ty",
        dty < 1e-6 and fin26 >= 19,
        f"max<=2025 diff={dty:.2e} finite2026={fin26}/{len(countries)}",
    )
    ty26v = ty26.set_index("country")["T_y_d211"]
    pop25 = yearly[yearly.year == 2025].set_index("country")["population"]

    # ---- 2026 rows ----
    print("[2026-append] building 2026 rows + ODE continuity ...", flush=True)
    new_rows, ode_maxdiff, completeness = [], 0.0, {}
    for cc in countries:
        d_cc = dep[dep.country == cc]
        we = pd.read_parquet(WEATHER_DIR / f"era5_{cc.lower()}.parquet")
        we26 = we[(we.t >= "2026-01-01") & (we.t <= GRID[-1])].set_index("t").reindex(GRID)
        assert not we26[WEATHER_COLS].isna().any().any(), f"{cc}: weather gap"
        nf = pd.DataFrame({"t": GRID, "load": load26[cc].to_numpy()})
        completeness[cc] = float(nf["load"].notna().mean())
        for c in WEATHER_COLS:
            nf[c] = we26[c].to_numpy()
        nf["country"] = cc
        ts = pd.DatetimeIndex(nf["t"])
        nf["date"] = ts.normalize()
        nf["year"] = ts.year
        nf["month"] = ts.month
        nf["dow"] = ts.dayofweek
        nf["doy"] = ts.dayofyear
        nf["hod"] = ts.hour
        nf["is_weekend"] = nf["dow"] >= 5
        if dep["is_weekend"].dtype != bool:
            nf["is_weekend"] = nf["is_weekend"].astype(dep["is_weekend"].dtype)
        nf["crisis_2022_hour"] = 0.0
        nf["crisis_2020_hour"] = 0.0
        p25 = float(pop25.loc[cc])  # N(2026) := N(2025)
        nf["population"] = p25
        nf["log_load_per_capita_hourly"] = np.log(nf["load"] / p25)
        nf["T_y_d211"] = float(ty26v.loc[cc])
        nf["S_seasonal_universal"] = hpe.apply_seasonal_climatology(nf["month"].to_numpy())
        nf["W_dow_universal"] = hpe.apply_dow_climatology(nf["dow"].to_numpy())
        nf["H_hod_universal"] = h24.reindex(nf["hod"]).to_numpy()
        resid3 = (
            nf["log_load_per_capita_hourly"]
            - nf["T_y_d211"]
            - nf["S_seasonal_universal"]
            - nf["W_dow_universal"]
        )
        if "log_load_pc_resid_3level" in cols:
            nf["log_load_pc_resid_3level"] = resid3
        nf["R_hour_raw"] = resid3 - nf["H_hod_universal"]
        nf["R_hour_country_mean"] = float(rm_cc.loc[cc])
        nf["R_hour_centered"] = nf["R_hour_raw"] - nf["R_hour_country_mean"]
        t_full = np.concatenate([d_cc["T_air"].to_numpy(), nf["T_air"].to_numpy()])
        xh = integrate_first_order_ode(t_full, tau=168.0, threshold=18.0, direction="below")
        xc = integrate_first_order_ode(t_full, tau=168.0, threshold=22.0, direction="above")
        n_dep = len(d_cc)
        ode_maxdiff = max(
            ode_maxdiff,
            float(np.max(np.abs(xh[:n_dep] - d_cc["x_H_ode"].to_numpy()))),
            float(np.max(np.abs(xc[:n_dep] - d_cc["x_C_ode"].to_numpy()))),
        )
        nf["x_H_ode"] = xh[n_dep:]
        nf["x_C_ode"] = xc[n_dep:]
        nf["__HDD__"] = np.maximum(18.0 - nf["T_air"], 0.0)
        nf["__CDD__"] = np.maximum(nf["T_air"] - 22.0, 0.0)
        key = pd.MultiIndex.from_arrays([nf["country"], nf["month"], nf["hod"]])
        for acol, bcol in ANOM.items():
            nf[acol] = nf[bcol].to_numpy() - clims[acol].reindex(key).to_numpy()
        nf["holiday_count_month"] = (
            hcm26.reindex(pd.MultiIndex.from_arrays([nf["country"], nf["month"]]))
            .fillna(0.0)
            .to_numpy()
        )
        new_rows.append(nf)
    gate(gates, "ode", ode_maxdiff < 1e-6, f"max<=2025 diff={ode_maxdiff:.2e}")

    new = pd.concat(new_rows, ignore_index=True)
    dep = dep.drop(columns=["__HDD__", "__CDD__"])
    new = new.drop(columns=["__HDD__", "__CDD__"])
    missing = [c for c in cols if c not in new.columns]
    extra = [c for c in new.columns if c not in cols]
    gate(gates, "schema", not missing and not extra, f"missing={missing} extra={extra}")
    new = new[cols]
    for c in cols:
        if new[c].dtype != dep[c].dtype:
            with contextlib.suppress(TypeError, ValueError):
                new[c] = new[c].astype(dep[c].dtype)
    out = (
        pd.concat([dep, new], ignore_index=True)
        .sort_values(["country", "t"])
        .reset_index(drop=True)
    )
    le = out[out["t"] < "2026-01-01"].reset_index(drop=True)
    same = le.shape == dep.shape and all(
        (le[c].to_numpy() == dep[c].to_numpy()).all()
        if not np.issubdtype(le[c].dtype, np.floating)
        else np.allclose(le[c].to_numpy(), dep[c].to_numpy(), equal_nan=True, rtol=0, atol=0)
        for c in cols
    )
    gate(gates, "le2025", same, f"rows<2026 == ext2025 frame: {le.shape}")
    worst = min(completeness.items(), key=lambda kv: kv[1])
    print(
        f"  2026 load completeness: min {worst[0]} {worst[1]:.4%}; "
        f"panel-19 min {min(v for k, v in completeness.items() if k != 'MK'):.4%}",
        flush=True,
    )

    gdf = pd.DataFrame(gates)
    n_fail = int((~gdf["pass"]).sum())
    print(f"\n[2026-append] gates: {len(gdf) - n_fail}/{len(gdf)} PASS", flush=True)
    GATES_OUT.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_csv(GATES_OUT, index=False)
    if n_fail:
        print("[2026-append] NOT writing extended panel (gate failure).", flush=True)
        return 1
    out.to_parquet(OUT_PANEL, index=False)
    n26 = int((out.year == 2026).sum())
    nan26 = int(out.loc[out.year == 2026, "load"].isna().sum())
    print(
        f"[2026-append] wrote {OUT_PANEL} rows={len(out)} (2026: {n26}, NaN-load hours: {nan26})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
