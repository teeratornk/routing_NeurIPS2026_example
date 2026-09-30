"""TS-LIMITS Phase 0: emit per-origin records for the deployed structural model.

Half of the paired data needed for the decline-frontier analysis. The foundation
models are the expensive half and run separately on GPU; this side is closed-form
and runs on CPU in minutes.

Gate before any routing number is computed (same discipline as D223/D224): the
per-origin APEs must reproduce the published per-cell medians in
si:tab:frozen2025 for the deployed model. If they do not, the origin grid has
drifted and the routing analysis would silently rest on a different comparison
than the paper's. Abort on failure.

Also emits the request-time gate features. No feature may use the realized
outcome or either model's output. Two groups are emitted: quantities measured at
or before the anchor, and quantities evaluated at the forecast hour that are
settled before the request is issued (civil calendar, published holidays, a
climatology fitted through 2017). The forecast-side group exists because the
foundation models are univariate: whatever the structural model knows about
holidays, the FM cannot see, so a gate denied those covariates is being asked to
predict a difference from features that do not contain it.
_assert_causal_features and FWD_SOURCE_COLS enforce this rather than leaving it
to prose.

Out: reports/tslimits/per_origin_structural_{window}.parquet
     reports/tslimits/per_origin_structural_gates_{window}.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_per_origin_structural.py --window dev
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

sys.path.insert(0, os.getcwd())

from load_forecast.eval.common_protocol import evaluate_method_per_origin

OUT = Path("reports/tslimits")
EXT_PANEL = Path(
    "data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet"
)
# The post-release window. Its panel is append-only on ext2025 (every row before
# 2026 is the ext2025 frame itself, gated in revision_2026_panel_append.py), and
# the frozen-2017 T_y chain needs the yearly panel that carries the 2025 GDP
# increment and a 2026 row.
EXT_PANEL_2026 = Path(
    "data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2026.parquet"
)
YEARLY_2026 = Path(
    "data/feature_store/multi_resolution/yearly/clean20_yearly_panel_d211_ext2026.parquet"
)
H = (1, 24, 168, 720)
# The long-horizon rows are emitted separately, not appended to H. Adding them to
# the shared artifact would break the one-to-one join assertion in every
# downstream script until the foundation-model side exists at that horizon, so
# each lives in its own file and the routing analysis stays on the four paired
# horizons. One year is the control: it is the only long lead with a published
# value, so it is scored alongside two and three years to back them.
YEAR_CONTROL = 8760
LONG_LEADS = (8760, 17520, 26280)
EU19 = (
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
    "NL",
    "PL",
    "PT",
    "RO",
    "SI",
    "SK",
)  # MK excluded by the pre-registered completeness filter (D224)

# Published deployed-model panel medians (si:tab:frozen2025), per window. The
# development row is the "same 19 countries" parenthesised row, not the
# full-panel headline, because this script scores the 19-country subset.
PUBLISHED = {
    "2025": {1: 1.07, 24: 2.59, 168: 3.68, 720: 5.67, 8760: 5.29},
    "dev": {1: 0.92, 24: 2.38, 168: 3.29, 720: 5.32},
    # No published 2026 value exists. The 2026 window scores 2025 and 2026 in one
    # process from the ext2026 panel and gates on the 2025 rows reproducing the
    # published values; only the 2026 rows are written.
    "2026": {1: 1.07, 24: 2.59, 168: 3.68, 720: 5.67},
    # Long leads scored on 2024 TARGETS (origins in 2023 and 2022), the record a
    # 2024-selected country-static policy needs at those leads. No published
    # value exists; the gate is identity with the dev artifact at h=720.
    "dev-long": {},
}
WINDOW_YEARS = {
    "2025": (2025,),
    "dev": (2018, 2019, 2020, 2021, 2022, 2023, 2024),
    "2026": (2025, 2026),
    "dev-long": (2024,),
}
DEVLONG_H = (720, 8760, 17520)
# rows the gate is computed on, and rows written out, per window
GATE_YEAR = {"2025": None, "dev": None, "2026": 2025, "dev-long": None}
WRITE_YEAR = {"2025": None, "dev": None, "2026": 2026, "dev-long": None}  # rows written out
# The ext2025 panel is append-only, so its 2018-2024 rows are byte-identical to
# the pre-2025 panel the FM dev jobs read; one panel serves both windows.
TOL = 0.006

# Features the gate may use, in three groups that need different justifications.
#
# CHEAP_FEATURES is the original set: the request's own horizon and anchor
# calendar, plus statistics of the load series up to and including the anchor.
#
# ANCHOR_EXTRA is measured at the anchor too, so it needs no separate argument:
# an operator issuing the request knows the current temperature and whether
# today is a holiday.
#
# KNOWN_FUTURE_FEATURES is evaluated at the FORECAST timestamp. That is
# admissible only because each one is fixed before the request is issued rather
# than observed after it:
#   fwd_hour/dow/month/doy   arithmetic on a timestamp the request already names
#   fwd_is_hol/fwd_is_bridge the national public-holiday calendar, published
#                            years ahead (data/external/holidays)
#   fwd_temp_clim and its    the <=2017 (country, month, hour) temperature
#   heating/cooling terms    climatology, which is a calendar function
#
# The REALIZED temperature at the forecast hour is deliberately excluded.
# FWD_SOURCE_COLS below fixes what is read at the forecast timestamp, and
# FWD_PERMITTED_SOURCES is the allowlist it must sit inside. Two of those
# entries are DERIVED from realized panel columns (temp_clim comes from T_air
# minus its own anomaly), so the naming and the lists are not sufficient on
# their own; _attach_gate_features additionally asserts that the result is
# constant within (country, month, hour), which is what makes it a calendar
# function rather than weather. The deployed structural model discards that same
# term (make_four_scale_method subtracts R_eq_w as oracle information) and this
# study has no numerical weather prediction product, so a gate using it would
# not describe anything deployable.
CHEAP_FEATURES = (
    "horizon",
    "anchor_hour",
    "anchor_dow",
    "anchor_month",
    "anchor_load",
    "recent_resid_vol",
    "recent_load_range",
)
ANCHOR_EXTRA = (
    "anchor_temp",
    "anchor_hdd_anom",
    "anchor_cdd_anom",
    "anchor_is_hol",
    "anchor_is_bridge",
)
KNOWN_FUTURE_FEATURES = (
    "fwd_hour",
    "fwd_dow",
    "fwd_month",
    "fwd_doy",
    "fwd_is_hol",
    "fwd_is_bridge",
    "fwd_temp_clim",
    "fwd_hdd_clim",
    "fwd_cdd_clim",
)
IDENTITY_FEATURES = ("country_code",)
GATE_FEATURES = CHEAP_FEATURES + ANCHOR_EXTRA + KNOWN_FUTURE_FEATURES + IDENTITY_FEATURES


def categorical_mask(cols: Sequence[str]) -> NDArray[np.bool_]:
    """Which of `cols` are unordered identities rather than ordered quantities.

    `country_code` is an alphabetical rank (AT=0 ... SK=18), so without this a
    tree splits it as if Austria were nearer to Belgium than to Slovakia and a
    linear model fits one coefficient to "alphabetical position". Every other
    gate feature is genuinely ordered -- hour, month and day-of-week are cyclic
    but monotone within their range, which trees handle by splitting twice.

    Passing this as `categorical_features` is what actually gives a learner
    country identity. Without it the paper's claim that the gate HAS country
    identity, and beats a per-country fixed rule anyway, rests on an encoding
    the learner cannot use.
    """
    return np.array([c in IDENTITY_FEATURES for c in cols], dtype=bool)


# Columns read at the ANCHOR timestamp and at the FORECAST timestamp. Keeping
# these as explicit lists is the actual causality guarantee: no realized weather
# and no load value may appear in the forecast-side list.
ANCHOR_SOURCE_COLS = {
    "T_air": "anchor_temp",
    "d_HDD_hourly": "anchor_hdd_anom",
    "d_CDD_hourly": "anchor_cdd_anom",
    "is_hol": "anchor_is_hol",
    "is_bridge": "anchor_is_bridge",
}
FWD_SOURCE_COLS = {
    "temp_clim": "fwd_temp_clim",
    "hdd_clim": "fwd_hdd_clim",
    "cdd_clim": "fwd_cdd_clim",
    "is_hol": "fwd_is_hol",
    "is_bridge": "fwd_is_bridge",
}
# An ALLOWLIST, not a denylist. A denylist of banned columns passes anything a
# future developer forgets to ban, and the realistic edit -- declare a feature in
# KNOWN_FUTURE_FEATURES and register its source, exactly as fwd_temp_clim
# legitimately does -- was accepted for R_hour_centered (the deployed model's own
# residual), d_ghi_hourly (realized irradiance) and log_load_per_capita_hourly
# (the log of the answer). Each entry here has to earn its place in prose.
FWD_PERMITTED_SOURCES = {
    # civil calendar and the published national holiday table, fixed years ahead
    "is_hol",
    "is_bridge",
    # the <=2017 (country, month, hour) climatology, asserted below to be a
    # calendar function rather than realized weather
    "temp_clim",
    "hdd_clim",
    "cdd_clim",
}
FORBIDDEN_SUBSTRINGS = ("y_true", "target", "fm_", "chronos", "timesfm", "ape")


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_causal_features(feature_names: tuple[str, ...]) -> None:
    """Fail loudly if a gate feature could leak the target or the FM output.

    Two checks, because the name-based one is a tripwire and not a proof. The
    substring scan catches an accidentally copied outcome column; the source
    scan is the guarantee that matters, since it fixes exactly which panel
    columns may be read at the forecast timestamp.
    """
    bad = [
        f
        for f in feature_names
        if any(s in f.lower() for s in FORBIDDEN_SUBSTRINGS) and f != "horizon"
    ]
    if bad:
        raise AssertionError(f"non-causal gate features: {bad}")
    ungranted = sorted(set(FWD_SOURCE_COLS) - FWD_PERMITTED_SOURCES)
    if ungranted:
        raise AssertionError(
            f"forecast-time features read sources that are not on the allowlist: {ungranted}"
        )
    declared = set(CHEAP_FEATURES) | set(ANCHOR_EXTRA) | set(KNOWN_FUTURE_FEATURES)
    declared |= set(IDENTITY_FEATURES)
    undeclared = sorted(set(feature_names) - declared)
    if undeclared:
        raise AssertionError(f"gate features with no causality justification: {undeclared}")


def _attach_gate_features(
    origins: pd.DataFrame, panel: pd.DataFrame, panel_h: pd.DataFrame
) -> pd.DataFrame:
    """Add request-time features.

    Anchor-side features are evaluated at or before the anchor. Forecast-side
    features are evaluated at the target hour but are restricted to quantities
    settled before the request is issued: the civil calendar, the published
    holiday list, and a climatology fitted on data through 2017.
    """
    out = origins.copy()
    out["anchor_t"] = pd.to_datetime(out["anchor_t"])
    out["target_t"] = pd.to_datetime(out["target_t"])

    lead = (out["target_t"] - out["anchor_t"]) / pd.Timedelta(hours=1)
    off = int((lead != out["horizon"].astype(float)).sum())
    if off:
        raise AssertionError(f"{off} records whose target is not exactly h hours after the anchor")

    a = out["anchor_t"]
    out["anchor_hour"] = a.dt.hour
    out["anchor_dow"] = a.dt.dayofweek
    out["anchor_month"] = a.dt.month

    f = out["target_t"]
    out["fwd_hour"] = f.dt.hour
    out["fwd_dow"] = f.dt.dayofweek
    out["fwd_month"] = f.dt.month
    out["fwd_doy"] = f.dt.dayofyear

    out["country_code"] = pd.Categorical(out["country"], categories=list(EU19)).codes
    if int((out["country_code"] < 0).sum()):
        raise AssertionError("origin grid contains a country outside the EU19 filter")

    # Rolling statistics over the 168 hours ending AT the anchor (inclusive),
    # computed per country on the panel, then joined back by (country, anchor_t).
    stats = []
    for cc, g in panel.groupby("country"):
        g = g.sort_values("t").reset_index(drop=True)
        load = g["load"].astype(float)
        d1 = load.diff()
        stats.append(
            pd.DataFrame(
                {
                    "country": cc,
                    "anchor_t": pd.to_datetime(g["t"]),
                    "anchor_load": load.to_numpy(),
                    "recent_resid_vol": d1.rolling(168, min_periods=24).std().to_numpy(),
                    "recent_load_range": (
                        load.rolling(168, min_periods=24).max()
                        - load.rolling(168, min_periods=24).min()
                    ).to_numpy(),
                }
            )
        )
    feats = pd.concat(stats, ignore_index=True)
    out = out.merge(feats, on=["country", "anchor_t"], how="left", validate="many_to_one")

    # Calendar and thermal covariates. panel_h is the frozen build's panel, so
    # is_hol/is_bridge are the very flags the deployed model's holiday layer
    # uses; the gate is not being handed a private calendar of its own.
    src = panel_h[
        [
            "country",
            "t",
            "T_air",
            "d_T_air_hourly",
            "d_HDD_hourly",
            "d_CDD_hourly",
            "is_hol",
            "is_bridge",
        ]
    ].copy()
    src["t"] = pd.to_datetime(src["t"])
    # The <=2017 (country, month, hour) temperature climatology, recovered
    # exactly as the level minus its own anomaly, then put through the same
    # 18/22 C thresholds the structural model's degree-day layer uses. This is a
    # calendar function: it does not know what the weather actually did.
    src["temp_clim"] = src["T_air"] - src["d_T_air_hourly"]
    # The claim that this is a calendar function and not realized weather rests
    # on a property of the UPSTREAM panel: that d_T_air_hourly is an anomaly
    # against a (country, month, hour) climatology. This script cannot assume
    # that. If the panel ever switched to, say, a per-year climatology, the
    # arithmetic here would silently start carrying year-specific weather and
    # every name-based check would still pass. So verify it.
    spread = (
        src.groupby(["country", src["t"].dt.month, src["t"].dt.hour])["temp_clim"]
        .agg(lambda v: float(v.max() - v.min()))
        .max()
    )
    if spread > 1e-6:
        raise AssertionError(
            f"temp_clim varies by {spread:.4g} K within (country, month, hour): "
            "it is not a climatology, so the forecast-time features would carry "
            "realized weather"
        )
    src["hdd_clim"] = np.maximum(18.0 - src["temp_clim"], 0.0)
    src["cdd_clim"] = np.maximum(src["temp_clim"] - 22.0, 0.0)

    anchor_side = src[["country", "t", *ANCHOR_SOURCE_COLS]].rename(
        columns={"t": "anchor_t", **ANCHOR_SOURCE_COLS}
    )
    out = out.merge(anchor_side, on=["country", "anchor_t"], how="left", validate="many_to_one")

    fwd_side = src[["country", "t", *FWD_SOURCE_COLS]].rename(
        columns={"t": "target_t", **FWD_SOURCE_COLS}
    )
    out = out.merge(fwd_side, on=["country", "target_t"], how="left", validate="many_to_one")

    for c in ("anchor_is_hol", "anchor_is_bridge", "fwd_is_hol", "fwd_is_bridge"):
        out[c] = out[c].astype(float)
    return out


def _causal_population(panel: pd.DataFrame) -> pd.DataFrame:
    """Substitute N2025 := N2024, the population known at forecast time.

    This is the paper's headline convention. Running build_frozen on the raw
    extended panel instead uses the *realized* 2025 population, which is the
    published sensitivity row (1.07/2.61/3.67/5.74), not the headline
    (1.07/2.59/3.68/5.67). The gate below distinguishes them; getting this
    wrong silently would build the routing analysis on the wrong baseline.
    """
    pop24 = panel[panel["year"] == 2024].groupby("country")["population"].last()
    out = panel.copy()
    m25 = out["year"] == 2025
    out.loc[m25, "population"] = out.loc[m25, "country"].map(pop24).to_numpy()
    out.loc[m25, "log_load_per_capita_hourly"] = np.log(
        out.loc[m25, "load"].to_numpy() / out.loc[m25, "population"].to_numpy()
    )
    # 2026 rows, when present, were written with N(2026) := N(2025) by the
    # append script (same anchor-year convention); assert rather than trust.
    m26 = out["year"] == 2026
    if m26.any():
        pop25 = panel[panel["year"] == 2025].groupby("country")["population"].last()
        want = out.loc[m26, "country"].map(pop25).to_numpy()
        assert np.allclose(out.loc[m26, "population"].to_numpy(), want), (
            "2026 population is not N(2025)"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--window",
        choices=sorted(PUBLISHED),
        default="2025",
        help="2025 = the frozen holdout; dev = the 2018-2024 window the gate is fitted on",
    )
    ap.add_argument(
        "--year-scale",
        action="store_true",
        help="emit one long-horizon row instead of the four paired horizons, to a separate file",
    )
    ap.add_argument(
        "--horizon-hours",
        type=int,
        default=YEAR_CONTROL,
        choices=LONG_LEADS,
        help="the lead to emit under --year-scale: 8760 (one year), 17520 (two), "
        "26280 (three). Restricted on purpose: the gate below only has a published "
        "value at 8760, so a typo'd lead would score, clear that gate on the control "
        "alone, and write a plausible-looking artifact nobody asked for.",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=24,
        help="origin stride in hours. 24 is the published grid; 1 is the dense-grid sensitivity, "
        "which scores every hour of the window, tags the artifact _stride1 and records the "
        "published gate without enforcing it (the grids differ by construction).",
    )
    ap.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="restrict the paired horizons (dense-grid sensitivity runs one cell)",
    )
    ap.add_argument(
        "--no-gdp",
        action="store_true",
        help="drop the lagged-GDP term from the frozen-2017 year chain (drift only). "
        "The published gate then fails by construction, so it is recorded and not enforced, "
        "and the artifact carries a _nogdp tag.",
    )
    args = ap.parse_args()
    window: str = args.window
    years = WINDOW_YEARS[window]
    hh = int(args.horizon_hours)
    if args.year_scale and window != "2025":
        raise SystemExit("--year-scale is defined on the frozen 2025 window only")
    if not args.year_scale and hh != YEAR_CONTROL:
        raise SystemExit("--horizon-hours applies under --year-scale only")
    # Score the one-year control alongside whatever lead was asked for. The gate
    # below has a published value at 8760 and none beyond it, so reproducing 5.29
    # in the same process is what backs a two- or three-year artifact. The extra
    # horizon costs seconds and only the requested lead is written out.
    horizons = tuple(sorted({YEAR_CONTROL, hh})) if args.year_scale else H
    if window == "dev-long":
        horizons = DEVLONG_H
    if args.year_scale:
        tag = f"{window}_year" if hh == YEAR_CONTROL else f"{window}_h{hh}"
    else:
        tag = window.replace("-", "")
    if args.no_gdp:
        tag = f"{tag}_nogdp"
    if args.stride != 24:
        tag = f"{tag}_stride{args.stride}"
    if args.horizons and not args.year_scale:
        horizons = tuple(sorted(set(args.horizons)))
    published = {h: v for h, v in PUBLISHED[window].items() if h in horizons}
    assert published or window == "dev-long", f"no published gate value among horizons {horizons}"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[tslimits] window={window} years={years}", flush=True)
    panel_path = EXT_PANEL_2026 if window == "2026" else EXT_PANEL
    panel = pd.read_parquet(panel_path).sort_values(["country", "t"]).reset_index(drop=True)

    print("[tslimits] building the frozen deployed model (causal population) ...", flush=True)
    frozen = _load("revision_2025_frozen_eval")
    _mod, _dfp, _base_fn, panel_h, deployed_fn = frozen.build_frozen(
        _causal_population(panel),
        yearly_path=YEARLY_2026 if window == "2026" else None,
        no_gdp=args.no_gdp,
    )

    sub = panel_h[panel_h["country"].isin(EU19)]
    print("[tslimits] scoring per-origin on the 2025 window ...", flush=True)
    origins = evaluate_method_per_origin(
        deployed_fn, sub, horizons=horizons, test_years=years, stride=args.stride
    )
    print(f"  retained {len(origins):,} scored origins", flush=True)

    # ---- gate: reproduce the published per-cell medians before printing anything ----
    gy = GATE_YEAR[window]
    gate_rows = origins if gy is None else origins[origins["test_year"] == gy]
    panel_median = gate_rows.groupby(["country", "test_year", "horizon"])["ape"].median()
    got = panel_median.groupby("horizon").median()
    gates, ok_all = [], True
    for h, pub in published.items():
        got_h = float(got.loc[h])
        ok = abs(got_h - pub) <= TOL
        ok_all &= ok
        gates.append(
            {
                "gate": f"si:tab:frozen2025[{tag}]",
                "h": h,
                "printed": pub,
                "recomputed": round(got_h, 4),
                "pass": ok,
            }
        )
        print(f"  h={h}: {got_h:.4f} vs published {pub} -> {'PASS' if ok else 'FAIL'}", flush=True)
    if window == "dev-long":
        # identity with the deposited dev artifact on the shared cell: same
        # origins, same APEs, to float precision
        ref = pd.read_parquet(OUT / "per_origin_structural_dev.parquet")
        ref = ref[(ref["test_year"] == 2024) & (ref["horizon"] == 720)]
        ref = ref[["country", "anchor_t", "ape"]]
        got720 = origins[origins["horizon"] == 720][["country", "anchor_t", "ape"]]
        j = ref.merge(got720, on=["country", "anchor_t"], how="outer", suffixes=("_ref", "_got"))
        d720 = float(np.nanmax((j["ape_ref"] - j["ape_got"]).abs())) if len(j) else np.nan
        ok = len(j) == len(ref) == len(got720) and d720 < 1e-9
        ok_all &= ok
        gates.append({"gate": "identity with per_origin_structural_dev[2024, h=720]", "h": 720,
                      "printed": len(ref), "recomputed": len(got720), "pass": ok})
        print(f"  h=720 on 2024: {len(got720)} origins vs dev artifact {len(ref)}, "
              f"max|dAPE|={d720:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    pd.DataFrame(gates).to_csv(OUT / f"per_origin_structural_gates_{tag}.csv", index=False)
    if not ok_all and (args.no_gdp or args.stride != 24):
        print("[tslimits] gate recorded, not enforced: the no-GDP arm differs from the published "
              "model by design.", flush=True)
    elif not ok_all:
        print("[tslimits] GATE FAILURE -- origin grid drifted; nothing written.", flush=True)
        return 1
    if window == "dev-long":
        origins = origins[origins["horizon"].isin((8760, 17520))].reset_index(drop=True)
        print(f"[tslimits] identity gate passed; keeping {len(origins):,} long-lead rows on 2024 "
              f"targets", flush=True)

    if args.year_scale and hh != YEAR_CONTROL:
        origins = origins[origins["horizon"] == hh].reset_index(drop=True)
        print(f"[tslimits] control passed; keeping {len(origins):,} rows at h={hh}", flush=True)
    if WRITE_YEAR[window] is not None:
        origins = origins[origins["test_year"] == WRITE_YEAR[window]].reset_index(drop=True)
        print(f"[tslimits] gate passed on {GATE_YEAR[window]}; keeping {len(origins):,} rows of "
              f"{WRITE_YEAR[window]}", flush=True)

    _assert_causal_features(GATE_FEATURES)
    enriched = _attach_gate_features(origins, panel, panel_h)
    missing = enriched[list(GATE_FEATURES)].isna().sum()
    print(f"[tslimits] gate-feature NaNs:\n{missing.to_string()}", flush=True)
    n_bad = int(missing.drop(labels=["recent_resid_vol", "recent_load_range"]).sum())
    if n_bad:
        raise AssertionError(f"{n_bad} unmatched gate-feature values; the merges lost rows")

    enriched.to_parquet(OUT / f"per_origin_structural_{tag}.parquet", index=False)
    print(f"[tslimits] wrote {OUT}/per_origin_structural_{tag}.parquet", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
