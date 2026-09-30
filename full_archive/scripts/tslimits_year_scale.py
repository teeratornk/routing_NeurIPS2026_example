"""TS-LIMITS: the multi-year rows, and what reaching them by rollout costs.

The routing analysis lives at h in {1, 24, 168, 720}, where both forecasters
answer natively. This script covers the horizons the operator actually plans on,
one to three years, where they do not.

The asymmetry is one of RESOLUTION, not of evaluation cleverness. The structural
composition answers a year-ahead question with a year-scale step: `predict` is a
single evaluation at `anchor_pos + h` (d215_common_protocol_panel.py:123), so h
enters once as an array index and the cost is O(1) in the horizon; measured
single-request latency is flat at 111-112 us from h=1 to h=720. A foundation
model has no annual mode. To answer the same question it must synthesise every
hourly value in between: Chronos-2's single-shot prediction length is 1024 steps
(64 output patches x 16), so the leads here cost nine, eighteen and twenty-six
chained autoregressive unroll passes, against exactly one evaluation throughout.
One horizon would be a dot; three make the compounding visible as a trend.

TimesFM is absent from these rows on purpose. Its continuous quantile head is
capped at 1024 steps and raises above it, and that head is the configuration
behind every published TimesFM number in this study, so a multi-year column for
it would be a protocol change rather than a longer run. The cap is itself the
finding.

Seasonal naive is carried at every horizon, not just the short ones. It is the
honest third column: past a year it BEATS the composition, and by a margin that
WIDENS with the lead, so the multi-year claim is reach and cost, never accuracy.

Out: reports/tslimits/year_scale.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_year_scale.py
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from load_forecast.eval.common_protocol import evaluate_method_per_origin
from load_forecast.eval.routing import panel_median

OUT = Path("reports/tslimits")
EXT_PANEL = Path(
    "data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet"
)
PAIRED = (1, 24, 168, 720)
# The long leads, each with its own structural artifact and its own FM run,
# because none of them is native to either model's paired-horizon records.
LONG = (8760, 17520, 26280)
EU19 = (
    "AT", "BE", "BG", "CH", "CZ", "DE", "ES", "FR", "GR", "HR",
    "HU", "IT", "LU", "NL", "PL", "PT", "RO", "SI", "SK",
)
# Published seasonal-naive panel medians on the frozen 2025 window
# (reports/revision/frozen2025_eval.csv). The naive column is gated against
# these before it is used anywhere, same discipline as the structural side.
# Two and three years have no published value; they are gated indirectly, by
# requiring the published leads to reproduce in the same process.
PUBLISHED_NAIVE = {1: 3.9204, 24: 3.9204, 168: 3.9204, 720: 7.8821, 8760: 4.6893}
# Chronos-2's chained-pass count is a free parameter (max_output_patches), and the
# library default is not its best setting at either long lead. The cap below was
# selected on the 2024 window and applied unchanged to 2025, the same freeze the
# escalation threshold gets; picking it on 2025 would be test-set selection.
# Appendix sec:ablation carries the sweep. Short horizons are single-pass and
# unaffected, so they have no entry.
FROZEN_CAP = {8760: 48, 17520: 56}
TOL = 0.006
# Country-cluster bootstrap, same B and seed as the routing analysis. Countries
# are the resampling unit because errors are correlated within a country and
# nearly independent across them; a request-level bootstrap would be far too
# narrow. Nineteen clusters is few, so these are wide by construction.
B = 2000
SEED = 0


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _struct_path(h: int) -> Path:
    """One year keeps its original filename so nothing downstream moves."""
    return OUT / (
        "per_origin_structural_2025_year.parquet"
        if h == 8760
        else f"per_origin_structural_2025_h{h}.parquet"
    )


def _fm_files(h: int) -> list[str]:
    stem = "tsl_chronos_year_q" if h == 8760 else f"tsl_chronos_h{h}_q"
    return sorted(glob.glob(f"reports/tslimits/fm/{stem}_*.parquet"))


def _fm_long(h: int) -> pd.DataFrame:
    files = _fm_files(h)
    if not files:
        raise FileNotFoundError(f"no Chronos records at h={h}")
    fm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fm = fm.rename(columns={"cc": "country", "origin_ts": "anchor_t"})
    fm["anchor_t"] = pd.to_datetime(fm["anchor_t"])
    fm["ape_fm"] = 100.0 * (fm["q50"] - fm["actual"]).abs() / fm["actual"].abs()
    return fm[["country", "horizon", "anchor_t", "actual", "ape_fm"]]


def _available_long() -> list[int]:
    """Long leads with both sides on disk, reported rather than silently dropped."""
    have = []
    for h in LONG:
        why = []
        if not _struct_path(h).exists():
            why.append("structural")
        n_fm = len(_fm_files(h))
        if n_fm < len(EU19):
            # a GPU job still in flight leaves a partial country set. Including it
            # would build a row from four countries and blow up later in the merge
            # with a confusing message, so it is excluded here with a clear one.
            why.append(f"chronos ({n_fm}/{len(EU19)} countries)")
        if why:
            print(f"[year] EXCLUDED h={h}: missing {' + '.join(why)}", flush=True)
        else:
            have.append(h)
    if 8760 not in have:
        raise FileNotFoundError("the one-year row is required; it anchors the ladder")
    return have


def main() -> int:
    panel = pd.read_parquet(EXT_PANEL).sort_values(["country", "t"]).reset_index(drop=True)
    sub = panel[panel["country"].isin(EU19)]
    mod = _load("d215_common_protocol_panel")
    long_h = _available_long()
    horizons = (*PAIRED, *long_h)
    print(f"[year] horizons: {horizons}", flush=True)

    # ---- seasonal naive on the same grid, gated before use ----
    # seasonal_naive steps m = ceil(h/168) WHOLE WEEKS back, so hour-of-week is
    # preserved at every lead: three years is 157 weeks (26,376 h), not a flat
    # 26,280. A flat multiple of 8760 breaks weekday alignment (8,760 h is 52.14
    # weeks) and inflates the naive error by well over a point.
    print("[year] scoring causal seasonal naive at every horizon ...", flush=True)
    nv = evaluate_method_per_origin(
        mod.seasonal_naive, sub, horizons=horizons, test_years=(2025,)
    )
    nv = nv.dropna(subset=["ape"])
    naive_med = {
        int(h): panel_median(g["ape"], g["country"]) for h, g in nv.groupby("horizon")
    }
    ungated = sorted(set(PUBLISHED_NAIVE) - set(naive_med))
    assert not ungated, f"published naive leads went unscored: {ungated}"
    ok = True
    for h, pub in PUBLISHED_NAIVE.items():
        got = naive_med[h]
        good = abs(got - pub) <= TOL
        ok &= good
        print(f"  naive h={h}: {got:.4f} vs published {pub} -> {'PASS' if good else 'FAIL'}")
    for h in long_h:
        if h not in PUBLISHED_NAIVE:
            print(f"  naive h={h}: {naive_med[h]:.4f} (new lead, no published value)")
    if not ok:
        print("[year] GATE FAILURE -- naive column not written.", flush=True)
        return 1
    # persisted, so a direct Chronos-vs-naive contrast at the long leads is
    # post-processing rather than a re-score
    nv.to_parquet(OUT / "per_origin_seasonal_naive_2025.parquet", index=False)

    # ---- structural: four paired horizons plus one artifact per long lead ----
    st_all = pd.concat(
        [pd.read_parquet(OUT / "per_origin_structural_2025.parquet")]
        + [pd.read_parquet(_struct_path(h)) for h in long_h],
        ignore_index=True,
    )
    st_med = {
        int(h): panel_median(g["ape"], g["country"]) for h, g in st_all.groupby("horizon")
    }

    # ---- Chronos at the frozen cap, where one was selected ----
    tuned: dict[int, pd.DataFrame] = {}
    for h, cap in FROZEN_CAP.items():
        f = OUT / f"rollout_ablation_h{h}_cap{cap}.parquet"
        assert f.exists(), f"h={h}: frozen cap {cap} has no 2025 record at {f}"
        t = pd.read_parquet(f)
        assert t["cc"].nunique() == len(EU19), f"h={h} cap={cap}: {t['cc'].nunique()} countries"
        assert not t["ape"].isna().any(), f"h={h} cap={cap}: NaN ape"
        tuned[h] = t.rename(columns={"cc": "country"})

    # ---- Chronos: paired horizons from the existing records, long leads from the new runs ----
    paired_files = sorted(glob.glob("reports/tslimits/fm/tsl_chronos_q_*.parquet"))
    ch = pd.concat([pd.read_parquet(f) for f in paired_files], ignore_index=True)
    ch = ch.rename(columns={"cc": "country", "origin_ts": "anchor_t"})
    ch["ape_fm"] = 100.0 * (ch["q50"] - ch["actual"]).abs() / ch["actual"].abs()
    cols = ["country", "horizon", "ape_fm"]
    ch = pd.concat(
        [ch[cols]] + [_fm_long(h)[cols] for h in long_h], ignore_index=True
    )
    ch_med = {
        int(h): panel_median(g["ape_fm"], g["country"]) for h, g in ch.groupby("horizon")
    }

    # ---- every long join must be one-to-one, same discipline as every other horizon ----
    for h in long_h:
        sy = pd.read_parquet(_struct_path(h))
        j = sy.merge(
            _fm_long(h), on=["country", "horizon", "anchor_t"], how="inner", validate="one_to_one"
        )
        assert len(j) == len(sy), f"h={h} join kept {len(j)} of {len(sy)}"
        assert float((j["y_true"] - j["actual"]).abs().max()) == 0.0, f"h={h} targets disagree"
        lead = (
            pd.to_datetime(sy["target_t"]) - pd.to_datetime(sy["anchor_t"])
        ) / pd.Timedelta(hours=1)
        assert bool((lead == h).all()), f"a record at h={h} is not exactly {h} h ahead"
        # naive is scored on its own pass over the panel, so confirm it landed on
        # the same grid rather than assuming it: comparing country medians over two
        # different origin sets would be quietly wrong, not loudly wrong.
        n_nv = int((nv["horizon"] == h).sum())
        assert n_nv == len(sy), f"h={h}: naive scored {n_nv} origins, structural {len(sy)}"
        print(f"  join h={h}: {len(j):,} one-to-one, targets identical, lead exact", flush=True)

    # ---- country-cluster intervals on the two contrasts ----
    # The estimand is the panel median, so each country contributes its own
    # median and the resample is over countries. Both contrasts are read off the
    # SAME draw, which is what makes their difference interpretable.
    rng = np.random.default_rng(SEED)
    ci: dict[int, dict[str, tuple[float, float]]] = {}
    for h in horizons:
        cols_h = {
            "structural": st_all[st_all["horizon"] == h].groupby("country")["ape"].median(),
            "chronos2": ch[ch["horizon"] == h].groupby("country")["ape_fm"].median(),
            "seasonal_naive": nv[nv["horizon"] == h].groupby("country")["ape"].median(),
        }
        # the tuned column exists only where a cap was frozen; elsewhere it IS the
        # default, so the contrast and its interval coincide by construction
        cols_h["chronos2_tuned"] = (
            tuned[h].groupby("country")["ape"].median() if h in tuned else cols_h["chronos2"]
        )
        med = pd.concat(cols_h, axis=1)
        assert not med.isna().to_numpy().any(), f"h={h}: a country is missing a method"
        assert len(med) == len(EU19), f"h={h}: {len(med)} countries, expected {len(EU19)}"
        arr = med[["structural", "chronos2", "seasonal_naive", "chronos2_tuned"]].to_numpy()
        # the point estimate must be the same panel median the table reports
        assert abs(float(np.median(arr[:, 0])) - st_med[h]) < 1e-9, f"h={h} structural mismatch"
        assert abs(float(np.median(arr[:, 1])) - ch_med[h]) < 1e-9, f"h={h} chronos mismatch"
        assert abs(float(np.median(arr[:, 2])) - naive_med[h]) < 1e-9, f"h={h} naive mismatch"
        fm_d, nv_d, tu_d, tn_d = np.empty(B), np.empty(B), np.empty(B), np.empty(B)
        for bi in range(B):
            q = np.median(arr[rng.integers(0, len(arr), len(arr))], axis=0)
            fm_d[bi] = q[1] - q[0]
            nv_d[bi] = q[2] - q[0]
            tu_d[bi] = q[3] - q[0]
            tn_d[bi] = q[3] - q[2]  # tuned Chronos-2 minus seasonal naive, same draw
        ci[h] = {
            "fm": (float(np.percentile(fm_d, 2.5)), float(np.percentile(fm_d, 97.5))),
            "naive": (float(np.percentile(nv_d, 2.5)), float(np.percentile(nv_d, 97.5))),
            "tuned": (float(np.percentile(tu_d, 2.5)), float(np.percentile(tu_d, 97.5))),
            "tuned_naive": (float(np.percentile(tn_d, 2.5)), float(np.percentile(tn_d, 97.5))),
        }

    tuned_med = {
        h: panel_median(t["ape"], t["country"]) for h, t in tuned.items()
    }
    rows = []
    for h in horizons:
        rows.append(
            {
                "horizon": h,
                "structural": round(st_med[h], 4),
                "chronos2": round(ch_med[h], 4),
                "chronos2_tuned": round(tuned_med.get(h, ch_med[h]), 4),
                "tuned_cap": FROZEN_CAP.get(h, 64),
                "tuned_passes": (
                    int(np.ceil(h / (16 * FROZEN_CAP[h]))) if h in FROZEN_CAP
                    else int(np.ceil(h / 1024))
                ),
                "seasonal_naive": round(naive_med[h], 4),
                "fm_minus_structural_pp": round(ch_med[h] - st_med[h], 4),
                "fm_tuned_minus_structural_pp": round(
                    tuned_med.get(h, ch_med[h]) - st_med[h], 4
                ),
                "fm_tuned_minus_structural_lo": round(ci[h]["tuned"][0], 4),
                "fm_tuned_minus_structural_hi": round(ci[h]["tuned"][1], 4),
                "fm_minus_structural_lo": round(ci[h]["fm"][0], 4),
                "fm_minus_structural_hi": round(ci[h]["fm"][1], 4),
                "naive_minus_structural_pp": round(naive_med[h] - st_med[h], 4),
                "naive_minus_structural_lo": round(ci[h]["naive"][0], 4),
                "naive_minus_structural_hi": round(ci[h]["naive"][1], 4),
                # the contrast that needs no structural model: tuned Chronos-2
                # against the annual baseline, directly
                "tuned_minus_naive_pp": round(tuned_med.get(h, ch_med[h]) - naive_med[h], 4),
                "tuned_minus_naive_lo": round(ci[h]["tuned_naive"][0], 4),
                "tuned_minus_naive_hi": round(ci[h]["tuned_naive"][1], 4),
                # 1024 = Chronos-2 single-shot length (64 output patches x 16)
                "chronos_unroll_passes": int(np.ceil(h / 1024)),
                "structural_evaluations": 1,
            }
        )
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "year_scale.csv", index=False)
    print("\n" + res.to_string(index=False), flush=True)

    gaps = ", ".join(
        f"{int(np.ceil(h / 1024))} passes -> {ch_med[h] - st_med[h]:+.2f} pp "
        f"[{ci[h]['fm'][0]:+.2f}, {ci[h]['fm'][1]:+.2f}]"
        for h in long_h
    )
    print(f"\n[year] the FM gap grows with the pass count: {gaps}", flush=True)
    worse = [h for h in long_h if naive_med[h] < st_med[h]]
    if worse:
        detail = ", ".join(f"h={h} {naive_med[h]:.2f} vs {st_med[h]:.2f}" for h in worse)
        print(
            f"[year] the composition is NOT the most accurate at {len(worse)} of"
            f" {len(long_h)} long leads: seasonal naive is ({detail}). The multi-year"
            " claim is reach and cost, not accuracy.",
            flush=True,
        )
    print(f"[year] wrote {OUT}/year_scale.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
