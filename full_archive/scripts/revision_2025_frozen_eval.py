"""D224: frozen prospective evaluation on the 2025 window (pre-registered).

Protocol: docs/d224_2025_freeze.md (committed 27b263a BEFORE any 2025 outcome
was computed). NOTHING here is fitted on 2025: the panel extension is
append-only with constants extracted from the deposited artifact
(revision_2025_panel_append.py), all layer fits are <=2017 (state <=2014),
lambda is the published constant, and LightGBM trains on <=2017 only.

Gate (must PASS before the 2025 numbers are read):
  G-2024: on the EXTENDED panel restricted to the published 2018-2024 grid the
  frozen build reproduces deployed 0.9729/2.4137/3.4125/5.4528/6.4509 and
  5L-Core 2.0684/2.5341/3.5730/6.3046/6.4621 panel-medians (tol 0.005; the
  <=2024 rows are byte-identical and every fit is <=2017, so this must hold).

2025 estimand: panel-median MdAPE over the 19 filter-passing countries
(MK excluded by the pre-registered completeness rule), rotating stride-24
grid with test_years=(2025,), horizons 1/24/168/720/8760. Models: deployed
6L-State-CalEx (primary), 5L-Core, seasonal naive, LightGBM default <=2017.
Sensitivity: population persistence N2025:=N2024 for the deployed model.

Out: reports/revision/frozen2025_eval.csv, frozen2025_cells.csv,
     frozen2025_gates.csv (+ feature-store mirror)
Run:  .venv/bin/python scripts/revision_2025_frozen_eval.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from load_forecast.eval.common_protocol import (  # noqa: E402
    DEFAULT_TEST_YEARS,
    evaluate_method,
    panel_summary,
)

OUT = Path("reports/revision")
STORE = Path("data/feature_store/multi_resolution/revision_uq_h8760")
EXT_PANEL = Path("data/feature_store/multi_resolution/hourly/"
                 "clean20_hourly_panel_d214_ode_ext2025.parquet")
LAM_PUBLISHED = {1: 1.0, 24: 0.25, 168: 0.25, 720: 0.75}
H = (1, 24, 168, 720, 8760)
EU19 = ["AT", "BE", "BG", "CH", "CZ", "DE", "ES", "FR", "GR", "HR",
        "HU", "IT", "LU", "NL", "PL", "PT", "RO", "SI", "SK"]  # MK: filter FAIL
G_2024 = {
    # deployed anchors are 4-decimal exact from every D223 gate run;
    # core5 h<=168 are the published 2-decimal table values, 720/8760 exact
    # from the D223l in-domain reference rows.
    "deployed": {1: 0.9729, 24: 2.4137, 168: 3.4125, 720: 5.4528, 8760: 6.4509},
    "core5":    {1: 2.07, 24: 2.53, 168: 3.57, 720: 6.3046, 8760: 6.4621},
}
TOL = 0.006


def _load(name):
    p = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    m = importlib.util.module_from_spec(p)
    p.loader.exec_module(m)
    return m


def _summ(fn, sub, horizons, test_years):
    cells = evaluate_method(fn, sub, horizons=tuple(horizons),
                            test_years=tuple(test_years))
    s = panel_summary(cells).set_index("horizon")
    med = {int(h): float(s.loc[h, "panel_median_mape"]) for h in horizons
           if h in s.index}
    return med, cells


def build_frozen(panel, yearly_path=None, no_gdp=False):
    """The canonical frozen build (== revision_uq_h8760.build_deployed).

    yearly_path: the yearly panel the frozen-2017 T_y chain is re-chained on.
    None is the deposited D211 panel. The 2026 window passes the ext2026 panel,
    whose <=2025 rows reproduce the deposit and whose 2026 row exists so the
    chain can take its lagged-GDP step into 2026.
    """
    mod = _load("d215_common_protocol_panel")
    hol = _load("exp_holiday_layer")
    rs = _load("exp_residual_state")
    stack = _load("exp_deployable_stack")
    c2 = _load("revision_year_frozen2017")
    dfp = mod.attach_four_scale_columns(panel)
    ty = c2.build_Ty_frozen2017(pd.read_parquet(yearly_path or c2.YEARLY), no_gdp=no_gdp)
    dfp = dfp.merge(ty, on=["country", "year"], how="left")
    dfp["T_y_d211"] = dfp["T_y_2017"].where(
        dfp["T_y_2017"].notna(), dfp["T_y_d211"]).to_numpy()
    base_fn = mod.make_four_scale_method(dfp, weather="causal")
    dfp2 = hol.attach_holiday_layer(dfp)
    dfp2 = rs.attach_state(dfp2)
    base_hol_fn = hol.make_method(dfp2, ("hol_anom", "bridge_anom"))
    panel_h = dfp2.copy()
    panel_h["e"] = panel_h["e"] - panel_h["hol_anom"] - panel_h["bridge_anom"]
    panel_h = stack._recompute_roll(panel_h)
    models = rs.build_models(panel_h, "global_gated", train_max=2014)
    deployed_fn = rs.make_state_method(panel_h, base_hol_fn, models,
                                       "global_gated", lam=LAM_PUBLISHED)
    return mod, dfp, base_fn, panel_h, deployed_fn


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    STORE.mkdir(parents=True, exist_ok=True)
    gates, rows, allcells = [], [], []

    print("[D224] loading extended panel ...", flush=True)
    panel = (pd.read_parquet(EXT_PANEL)
             .sort_values(["country", "t"]).reset_index(drop=True))
    ymax = int(panel["year"].max())
    assert ymax == 2025, f"extended panel ends {ymax}, expected 2025"
    n25 = int((panel["year"] == 2025).sum())
    print(f"  rows 2025: {n25} across {panel.loc[panel.year == 2025, 'country'].nunique()} cc",
          flush=True)

    print("[D224] frozen build on extended panel ...", flush=True)
    mod, dfp, base_fn, panel_h, deployed_fn = build_frozen(panel)

    # ---------------- G-2024 reproduction gate (published grid) ----------------
    print("[D224] G-2024: reproduce published 2018-2024 panel-medians ...", flush=True)
    m_dep, _ = _summ(deployed_fn, panel_h, H, DEFAULT_TEST_YEARS)
    m_core, _ = _summ(base_fn, dfp, H, DEFAULT_TEST_YEARS)
    ok_all = True
    for name, got in (("deployed", m_dep), ("core5", m_core)):
        for h, pub in G_2024[name].items():
            ok = abs(got[h] - pub) <= TOL
            ok_all &= ok
            gates.append({"gate": f"G2024-{name}", "h": h, "printed": pub,
                          "recomputed": round(got[h], 4), "pass": ok})
            print(f"  {name} h={h}: {got[h]:.4f} vs {pub} -> "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
    if not ok_all:
        pd.DataFrame(gates).to_csv(OUT / "frozen2025_gates.csv", index=False)
        print("[D224] GATE FAILURE -- 2025 numbers NOT computed. Fix drift first.",
              flush=True)
        return

    # ------------------------- UNBLINDING POINT -------------------------
    print("[D224] gates clean -> computing the 2025 window (unblinding) ...",
          flush=True)
    naive_fn = mod.seasonal_naive
    r4 = _load("exp_run4_baselines")
    lgbm_fn = r4.make_lgbm_method(dfp)  # lazy per-(cc,h) fits, targets <=2017 only

    ev19_dep = panel_h[panel_h["country"].isin(EU19)]
    ev19_base = dfp[dfp["country"].isin(EU19)]
    for label, fn, sub in (("deployed_6L", deployed_fn, ev19_dep),
                           ("core_5L", base_fn, ev19_base),
                           ("seasonal_naive", naive_fn, ev19_base),
                           ("lgbm_default_le2017", lgbm_fn, ev19_base)):
        med, cells = _summ(fn, sub, H, (2025,))
        cells["model"] = label
        allcells.append(cells)
        rows.append({"model": label,
                     **{f"h{h}": round(med.get(h, np.nan), 4) for h in H}})
        print(f"  2025 {label}: "
              + " ".join(f"h{h}={med.get(h, float('nan')):.3f}" for h in H),
              flush=True)

    # population persistence sensitivity (deployed only)
    pop24 = (panel[panel.year == 2024].groupby("country")["population"].last())
    pp = panel.copy()
    m25 = pp["year"] == 2025
    pp.loc[m25, "population"] = pp.loc[m25, "country"].map(pop24).to_numpy()
    pp.loc[m25, "log_load_per_capita_hourly"] = np.log(
        pp.loc[m25, "load"].to_numpy() / pp.loc[m25, "population"].to_numpy())
    _, _, _, pph, dep_pp = build_frozen(pp)
    med, _ = _summ(dep_pp, pph[pph["country"].isin(EU19)], H, (2025,))
    rows.append({"model": "deployed_6L_pop_persist",
                 **{f"h{h}": round(med.get(h, np.nan), 4) for h in H}})
    print("  2025 deployed (pop-persist): "
          + " ".join(f"h{h}={med.get(h, float('nan')):.3f}" for h in H), flush=True)

    rdf = pd.DataFrame(rows)
    gdf = pd.DataFrame(gates)
    cdf = pd.concat(allcells, ignore_index=True)
    rdf.to_csv(OUT / "frozen2025_eval.csv", index=False)
    gdf.to_csv(OUT / "frozen2025_gates.csv", index=False)
    cdf.to_csv(OUT / "frozen2025_cells.csv", index=False)
    print("\n=== D224 frozen 2025 window (pre-registered; report as-is) ===")
    print(rdf.to_string(index=False), flush=True)
    for f in ("frozen2025_eval.csv", "frozen2025_gates.csv", "frozen2025_cells.csv"):
        (STORE / f).write_bytes((OUT / f).read_bytes())
    print(f"[D224] artifacts -> {OUT} (mirrored)", flush=True)


if __name__ == "__main__":
    main()
