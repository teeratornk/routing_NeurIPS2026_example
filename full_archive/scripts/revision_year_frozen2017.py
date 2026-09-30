"""Reviewer C2 (option a): refit the operational year coefficient on <=2017.

The deployed year layer T_y_d211 chains beta_GDP=0.713 (a full-SAMPLE PySR fit,
uses 2018-2024 data) with delta=0.0136 and a 2022 crisis dummy. A reviewer noted
that this full-sample coefficient enters the scored 2018-2024 operational forecast
at the ~8% of h=720 origins that cross a calendar-year boundary, so "every
coefficient is fit through 2017" is not literally true.

This rebuilds T_y with the <=2017 GDP-only refit (beta=0.53, delta=0.0035, no 2022
crisis dummy since 2022 is post-2017), keeping the deployed causal (lagged) GDP
mode, then re-runs the full deployable stack (5L-Core base, +holiday, +state, and
the deployed 6L) under the common protocol. Isolated year-layer swap: H_c, R_eq,
S, W held at their published fits (matches the SI leakage-swap methodology).

Writes exp_deployable_stack_frozen2017/{stack_pivot.csv,stack_cells.parquet} and
prints deltas vs the published headline (2.08/2.53/3.57/6.29 base;
0.97/2.41/3.41/5.44 deployed).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from load_forecast.eval.common_protocol import (
    DEFAULT_HORIZONS,
    DEFAULT_TEST_YEARS,
    bootstrap_panel_ci,
    evaluate_method,
    panel_summary,
)

# <=2017 GDP-only refit == the annual scenario forecaster (paper sec:when).
BETA_2017 = 0.53
DELTA_2017 = 0.0035
YEARLY = Path("data/feature_store/multi_resolution/yearly/clean20_yearly_panel_d211.parquet")
OUT = Path("data/feature_store/multi_resolution/exp_deployable_stack_frozen2017")
PUBLISHED = {"base": (2.08, 2.53, 3.57, 6.29),
             "+holiday+bridge+state": (0.97, 2.41, 3.41, 5.44)}


def _load(name):
    p = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    m = importlib.util.module_from_spec(p)
    p.loader.exec_module(m)
    return m


def build_Ty_frozen2017(yearly: pd.DataFrame, no_gdp: bool = False) -> pd.DataFrame:
    """Re-chain T_y with the <=2017 GDP-only coefficient (beta=0.53, delta=0.0035,
    no crisis dummy), causal (lagged) GDP. Same anchor level as the deployed chain,
    so only the year-over-year increments into >=2018 change. Returns
    [country, year, T_y_2017]."""
    rows = []
    for cc, g in yearly.sort_values(["country", "year"]).groupby("country"):
        g = g.dropna(subset=["log_load_per_capita"]).reset_index(drop=True)
        if len(g) == 0:
            continue
        dlog = g["dlog_gdp_per_capita"].to_numpy(float)
        T = float(g.loc[0, "log_load_per_capita"])
        rows.append({"country": cc, "year": int(g.loc[0, "year"]), "T_y_2017": T})
        for i in range(1, len(g)):
            # lagged / causal; no_gdp drops the term, leaving the drift alone
            dhat = 0.0 if no_gdp else (dlog[i - 1] if not np.isnan(dlog[i - 1]) else 0.0)
            T = T + BETA_2017 * dhat - DELTA_2017                      # no crisis dummy
            rows.append({"country": cc, "year": int(g.loc[i, "year"]), "T_y_2017": T})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mod = _load("d215_common_protocol_panel")
    hol = _load("exp_holiday_layer")
    rs = _load("exp_residual_state")
    stack = _load("exp_deployable_stack")  # reuse _select_lambda, _recompute_roll

    print("[c2] loading panel + four-scale (published oracle-T_y fits) ...", flush=True)
    panel = pd.read_parquet(mod.PANEL).sort_values(["country", "t"]).reset_index(drop=True)
    panel = mod.attach_four_scale_columns(panel)

    # ---- swap in the <=2017-refit year layer (isolated) --------------------
    yearly = pd.read_parquet(YEARLY)
    ty = build_Ty_frozen2017(yearly)
    n_before = panel["T_y_d211"].to_numpy().copy()
    panel = panel.merge(ty, on=["country", "year"], how="left")
    panel["T_y_d211"] = panel["T_y_2017"].where(panel["T_y_2017"].notna(),
                                                panel["T_y_d211"]).to_numpy()
    changed = np.nanmean(np.abs(panel["T_y_d211"].to_numpy() - n_before) > 1e-9)
    print(f"  T_y rows changed by the refit: {100*changed:.1f}% "
          f"(beta 0.713->{BETA_2017}, delta 0.0136->{DELTA_2017}, no crisis)")

    panel = hol.attach_holiday_layer(panel)
    panel = rs.attach_state(panel)

    base_fn = mod.make_four_scale_method(panel, weather="causal")
    base_hol_fn = hol.make_method(panel, ("hol_anom", "bridge_anom"))
    models_base = rs.build_models(panel, "global_gated", train_max=2014)
    panel_h = panel.copy()
    panel_h["e"] = panel_h["e"] - panel_h["hol_anom"] - panel_h["bridge_anom"]
    panel_h = stack._recompute_roll(panel_h)
    models_hol = rs.build_models(panel_h, "global_gated", train_max=2014)
    # C2 fix isolates the YEAR COEFFICIENT: the state gate lambda is a separate
    # frozen (2015-2017-validated) hyperparameter and is held at the PUBLISHED
    # values, not re-selected (re-selection flips the brittle h=168 gate on the
    # infinitesimally-perturbed validation window, a confound unrelated to the fix).
    lam_base = {1: 1.0, 24: 0.0, 168: 0.25, 720: 0.5}
    lam_hol = {1: 1.0, 24: 0.25, 168: 0.25, 720: 0.75}

    variants = {
        "base": base_fn,
        "+holiday+bridge": base_hol_fn,
        "+state(gated)": rs.make_state_method(panel, base_fn, models_base,
                                              "global_gated", lam=lam_base),
        "+holiday+bridge+state": rs.make_state_method(panel_h, base_hol_fn, models_hol,
                                                      "global_gated", lam=lam_hol),
    }
    cells_all, summ_all = [], []
    for name, fn in variants.items():
        use = panel_h if ("holiday" in name and "state" in name) else panel
        cells = evaluate_method(fn, use, horizons=DEFAULT_HORIZONS, test_years=DEFAULT_TEST_YEARS)
        cells["variant"] = name
        cells_all.append(cells)
        summ = panel_summary(cells).merge(bootstrap_panel_ci(cells, B=500, seed=42), on="horizon")
        summ["variant"] = name
        summ_all.append(summ)

    cells_df = pd.concat(cells_all, ignore_index=True)
    summ_df = pd.concat(summ_all, ignore_index=True)
    cells_df.to_parquet(OUT / "stack_cells.parquet", index=False)
    summ_df.to_csv(OUT / "stack_summary.csv", index=False)
    piv = summ_df.pivot_table(index="variant", columns="horizon",
                              values="panel_median_mape").reindex(list(variants))
    piv.to_csv(OUT / "stack_pivot.csv")
    print("\n=== frozen-2017 year-coefficient stack: panel-median MdAPE (%) ===")
    print(piv.round(4).to_string())
    print("\n=== delta vs PUBLISHED (full-sample beta=0.713) ===")
    for name, pub in PUBLISHED.items():
        got = [round(float(piv.loc[name, h]), 4) for h in DEFAULT_HORIZONS]
        d = [round(g - p, 4) for g, p in zip(got, pub)]
        print(f"  {name:24s} pub={pub}  new={tuple(round(x,2) for x in got)}  delta(pp)={d}")
    print(f"\nArtifacts -> {OUT}")


if __name__ == "__main__":
    main()
