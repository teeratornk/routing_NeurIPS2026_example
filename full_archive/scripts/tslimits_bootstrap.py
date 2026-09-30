"""TS-LIMITS: intervals on every quantity the paper compares.

Point estimates alone cannot separate "the policy gains nothing" from "the panel
is 19 countries and everything is noisy". Each quantity below is differenced
against best-fixed INSIDE each resample, so the interval describes the spread of
the difference rather than the difference of two spreads.

    oracle          best_fixed - per-request minimum. Reported to show what the
                    quantity does, not as headroom: tslimits_noisefloor.py shows
                    a difficulty-matched random pairing reproduces it.
    ceiling         best_fixed - cross-fitted E[margin|x] policy. The honest
                    headroom, and NOT deployable: the cross-fitting runs on 2025
                    itself. Cheap and enriched feature sets.
    frozen          best_fixed - the deployable policy: fitted on 2018-2023,
                    with a NUMERIC score threshold chosen on 2024 and applied
                    per request to 2025, so no request's decision depends on the
                    other requests scored beside it. The realised 2025
                    escalation fraction is reported as an outcome and differs
                    from the 2024 target. Cheap and enriched feature sets.

Resampling countries, not requests. Requests within a country share a model,
a calendar and a load level, so a cell bootstrap treats ~1,460 correlated
origins per country-horizon as independent and returns intervals that are far
too narrow. Countries are the unit that actually varies.

Every policy's per-request decision is computed ONCE and then held fixed while
countries are resampled. Refitting inside the replicate would ask a different
question (how variable is the whole pipeline) and would not be what the table
reports.

Out: reports/tslimits/bootstrap_ci_{estimand}.csv, one file per estimand so the
     two cannot overwrite each other
Run: TSLIMITS_ESTIMAND=pooled_mean  PYTHONPATH=src .venv/bin/python scripts/tslimits_bootstrap.py
     TSLIMITS_ESTIMAND=panel_median PYTHONPATH=src .venv/bin/python scripts/tslimits_bootstrap.py
"""

from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from load_forecast.eval.routing import ESTIMANDS, escalate_above, frozen_threshold

OUT = Path("reports/tslimits")
# Two policy families. "budget" is the published one, both checkpoints at a
# common 2048-hour context. "native" gives each checkpoint its usable ceiling
# (Chronos-2 8192; TimesFM-2.5 16256/16128/15616 at h<=24/168/720, since the
# library enforces context + block(h) <= 16384), on BOTH windows, so c-static
# and the router are re-selected on records of the same configuration they
# are scored on. Every artifact of the native family carries FAMILY_SFX.
FM_FAMILY = os.environ.get("TSLIMITS_FM_FAMILY", "budget")
FAMILY_SFX = "" if FM_FAMILY == "budget" else f"_{FM_FAMILY}"
FM_PATTERNS_BY_FAMILY = {
    "budget": {
        "Chronos-2-Uni-ZS": (
            "reports/tslimits/fm/tsl_chronos_q_*.parquet",
            "reports/tslimits/fm/tsldev_chronos_q_*.parquet",
        ),
        "TimesFM-2.5-Uni-ZS": (
            "reports/tslimits/fm/tsl_timesfm_q_*.parquet",
            "reports/tslimits/fm/tsldev_timesfm_q_*.parquet",
        ),
    },
    "native": {
        "Chronos-2-Uni-ZS": (
            (
                "reports/tslimits/fm/tsl_chronos_ctx8192_h1h24_q_*.parquet",
                "reports/tslimits/fm/tsl_chronos_ctx8192_q_*.parquet",
            ),
            (
                "reports/tslimits/fm/tsldev_chronos_ctx8192_h1h24_q_*.parquet",
                "reports/tslimits/fm/tsldev_chronos_ctx8192_q_*.parquet",
            ),
        ),
        "TimesFM-2.5-Uni-ZS": (
            (
                "reports/tslimits/fm/tsl_timesfm_ctx16256_q_*.parquet",
                "reports/tslimits/fm/tsl_timesfm_ctxnative_q_*.parquet",
            ),
            (
                "reports/tslimits/fm/tsldev_timesfm_ctx16256_q_*.parquet",
                "reports/tslimits/fm/tsldev_timesfm_ctxnative_q_*.parquet",
            ),
        ),
    },
}
FM_PATTERNS = FM_PATTERNS_BY_FAMILY[FM_FAMILY]


def _emitter() -> Any:
    p = Path("scripts") / "tslimits_per_origin_structural.py"
    spec = importlib.util.spec_from_file_location("tslimits_per_origin_structural", p)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_EM = _emitter()


def _blocked() -> Any:
    q = Path("scripts") / "tslimits_ceiling_blocked.py"
    sp = importlib.util.spec_from_file_location("tslimits_ceiling_blocked", q)
    assert sp is not None and sp.loader is not None
    mm = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mm)
    return mm


# one definition of the fold scheme, imported rather than re-inlined: the
# duplicated copy is how the day-of-year bug reached two files at once
_EM_SPLITS = _blocked()._splits
CHEAP: tuple[str, ...] = _EM.CHEAP_FEATURES
ENRICHED: tuple[str, ...] = _EM.GATE_FEATURES
FIT_YEARS = (2018, 2019, 2020, 2021, 2022, 2023)
VAL_YEARS = (2024,)
# The escalation-budget grid the 2024 selection searches. 0.05 is the published
# step; TSLIMITS_BUDGET_STEP=0.01 is the sensitivity a reviewer asked for, and
# every artifact of a non-default grid carries GRID_SFX so it cannot overwrite
# the deployed one.
BUDGET_STEP = float(os.environ.get("TSLIMITS_BUDGET_STEP", "0.05"))
BUDGETS = tuple(round(BUDGET_STEP * i, 4) for i in range(round(1.0 / BUDGET_STEP) + 1))
GRID_SFX = "" if abs(BUDGET_STEP - 0.05) < 1e-12 else f"_grid{round(BUDGET_STEP * 100):02d}"
B = 2000
N_FOLDS = 5
SEED = 0
# Both estimands are written to disk under their own names. The paper compares
# them directly, so citing one from an artifact the other overwrote would leave a
# headline number that nothing in the repo reproduces.
ESTIMAND = os.environ.get("TSLIMITS_ESTIMAND", "pooled_mean")
# Pre-specified before any result was read: a routing gain below this is
# operationally negligible, so an upper confidence bound beneath it licenses
# "equivalent to the fixed choice" rather than the weaker "not demonstrated".
MARGIN_PP = 0.10


def _load_fm(pattern: str | tuple[str, ...]) -> pd.DataFrame:
    """One glob, or several whose horizon families are disjoint (native arms)."""
    patterns = (pattern,) if isinstance(pattern, str) else pattern
    files = sorted(f for pat in patterns for f in glob.glob(pat))
    if not files:
        raise FileNotFoundError(pattern)
    fm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fm = fm.rename(columns={"cc": "country", "origin_ts": "anchor_t"})
    fm["anchor_t"] = pd.to_datetime(fm["anchor_t"])
    fm["ape_fm"] = 100.0 * (fm["q50"] - fm["actual"]).abs() / fm["actual"].abs()
    return fm[["country", "horizon", "anchor_t", "actual", "ape_fm"]]


def _paired(window: str, pattern: str) -> pd.DataFrame:
    st = pd.read_parquet(OUT / f"per_origin_structural_{window}.parquet")
    st = st.rename(columns={"ape": "ape_st"})
    st["anchor_t"] = pd.to_datetime(st["anchor_t"])
    j = st.merge(
        _load_fm(pattern), on=["country", "horizon", "anchor_t"], how="inner", validate="one_to_one"
    )
    assert len(j) == len(st), f"{window}: joined {len(j)} of {len(st)}"
    gap = float((j["y_true"] - j["actual"]).abs().max())
    assert gap == 0.0, f"{window}: the two sides disagree on the target by {gap}"
    return j.reset_index(drop=True)


def _top_k(score: np.ndarray, fraction: float) -> np.ndarray:
    """Top-`fraction` by score; rank-based so the count is exact under ties."""
    n = score.size
    k = round(fraction * n)
    m = np.zeros(n, dtype=bool)
    if k <= 0:
        return m
    if k >= n:
        return ~m
    m[np.argsort(-score, kind="stable")[:k]] = True
    return m


def _serve(df: pd.DataFrame, mask: np.ndarray) -> float:
    served = np.where(mask, df["ape_fm"].to_numpy(), df["ape_st"].to_numpy())
    return ESTIMANDS[ESTIMAND](pd.Series(served), df["country"])


# Which learner the gate uses. The deployed router is the boosted tree; the
# environment switch lets the same scripts refit every arm with a lighter class
# so the router's gain can be shown not to depend on the learner.
GATE_LEARNER = os.environ.get("TSLIMITS_GATE", "hgb")


def _reg(cols: list[str] | None = None) -> Any:
    """Gate learner. Pass the feature columns so country identity is usable.

    `early_stopping` is left at sklearn's 'auto', which switches ON above 10,000
    samples; the fit slices here are ~41.6k rows, so the gate does hold out an
    internal 10% validation split. That is stated rather than changed, since
    every reported number was produced under it.

    Both classes get country identity in a form they can use: a categorical
    split for the tree, a one-hot block for the linear model, since the raw
    column is an alphabetical rank.
    """
    cat = _EM.categorical_mask(cols) if cols is not None else None
    if GATE_LEARNER == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=300, random_state=SEED, categorical_features=cat
        )
    if GATE_LEARNER == "shallow":
        return HistGradientBoostingRegressor(
            max_iter=50, max_depth=2, random_state=SEED, categorical_features=cat
        )
    if GATE_LEARNER == "ridge":
        if cat is None or not cat.any():
            return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
        id_cols = [c for c, m in zip(cols, cat, strict=True) if m]
        num_cols = [c for c, m in zip(cols, cat, strict=True) if not m]
        pre = ColumnTransformer(
            [
                ("id", OneHotEncoder(handle_unknown="ignore", sparse_output=False), id_cols),
                ("num", StandardScaler(), num_cols),
            ]
        )
        return make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 4, 25)))
    raise ValueError(f"unknown TSLIMITS_GATE {GATE_LEARNER!r}")


def _ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main() -> int:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []

    for fm_label, (test_pat, dev_pat) in FM_PATTERNS.items():
        dev = _paired("dev", dev_pat)
        test = _paired("2025", test_pat)
        fit = dev[dev["test_year"].isin(FIT_YEARS)]
        val = dev[dev["test_year"].isin(VAL_YEARS)]

        for h in sorted(test["horizon"].unique()):
            f_h = fit[fit["horizon"] == h].reset_index(drop=True)
            v_h = val[val["horizon"] == h].reset_index(drop=True)
            t_h = test[test["horizon"] == h].reset_index(drop=True)

            st = t_h["ape_st"].to_numpy()
            fm = t_h["ape_fm"].to_numpy()
            margin = st - fm
            served = {"st": st, "fm": fm, "orc": np.minimum(st, fm)}
            budgets: dict[str, float] = {}
            thresholds: dict[str, float] = {}
            realized: dict[str, float] = {}

            for tag, cols in (("cheap", CHEAP), ("enr", ENRICHED)):
                c = list(cols)
                # Deployable: fitted on 2018-2023, threshold chosen on 2024.
                #
                # What freezes is a NUMBER on the score scale, not a fraction.
                # Taking the top-f of the scored 2025 window would make request
                # i's decision depend on the other ~6,933 requests in that year,
                # which presupposes the whole request pool is known in advance.
                # An online router compares one score against one number, so the
                # realised 2025 fraction is an OUTCOME and is reported as such.
                reg = _reg(c).fit(f_h[c], f_h["ape_st"] - f_h["ape_fm"])
                s_val = np.asarray(reg.predict(v_h[c]), dtype=float)
                s_test = np.asarray(reg.predict(t_h[c]), dtype=float)
                b_dep = min(
                    (_serve(v_h, escalate_above(s_val, frozen_threshold(s_val, b))), b)
                    for b in BUDGETS
                )[1]
                tau = frozen_threshold(s_val, b_dep)
                mask_dep = escalate_above(s_test, tau)
                served[f"frozen_{tag}"] = np.where(mask_dep, fm, st)
                budgets[f"frozen_{tag}"] = b_dep
                thresholds[f"frozen_{tag}"] = tau
                realized[f"frozen_{tag}"] = float(mask_dep.mean())

                # Ceiling: cross-fitted on 2025 itself, budget also read off 2025.
                # Folds are CONTIGUOUS TIME BLOCKS, not shuffled. Origins are one
                # per country per day across a single year, so a shuffled fold
                # lets a held-out origin borrow from the days on either side of
                # it; that estimates a within-year interpolation rather than
                # anything a router could achieve, and it inflated this quantity
                # by an order of magnitude (median 0.213 -> 0.000 pp, out-of-fold
                # R^2 positive in 8/8 cells -> negative in 8/8). A router always
                # extrapolates forward in time.
                # Rank on the timestamp, not day-of-year: anchors start in
                # December of the preceding year, so day-of-year wraps and a
                # "block" would straddle both Decembers.
                # Embargoed contiguous-time-block folds, matching
                # tslimits_ceiling_blocked.py exactly. Blocks are cut on the
                # anchor but the outcome lands h hours later, so training rows
                # whose TARGET falls inside the held-out block have seen the
                # period they are scored on; without the embargo the ceiling is
                # inflated by ~64% at h=168 and the null is anticonservative.
                cv = _EM_SPLITS(t_h["anchor_t"], t_h["target_t"], N_FOLDS)
                oof = np.asarray(
                    cross_val_predict(_reg(c), t_h[c], margin, cv=cv, n_jobs=1), dtype=float
                )
                b_ceil = min((_serve(t_h, _top_k(oof, b)), b) for b in BUDGETS)[1]
                served[f"ceiling_{tag}"] = np.where(_top_k(oof, b_ceil), fm, st)
                budgets[f"ceiling_{tag}"] = b_ceil

            # Two aggregation shapes, because the estimands differ in kind.
            #
            # panel_median is a median over countries of within-country medians,
            # so a country contributes one number and the matrix holds medians.
            #
            # pooled_mean is a mean over REQUESTS, and country sizes are not
            # equal here -- 363 to 365 origins per country per horizon, because
            # the rotating grid does not divide the year evenly. An unweighted
            # mean of per-country means is therefore NOT the pooled mean, which
            # an assertion caught rather than a reviewer. The matrix holds sums
            # and a separate count vector, and a resample aggregates as
            # sum(sums)/sum(counts), which is exact.
            names = list(served)
            frame = pd.DataFrame({"c": t_h["country"].to_numpy(), **served})
            if ESTIMAND == "pooled_mean":
                mat = frame.groupby("c")[names].sum().to_numpy()
                cnt = frame.groupby("c")[names[0]].size().to_numpy().astype(float)
            else:
                mat = frame.groupby("c")[names].median().to_numpy()
                cnt = np.ones(mat.shape[0])
            ccs = frame["c"].unique()

            def panel(rows: np.ndarray, mat: np.ndarray = mat, cnt: np.ndarray = cnt) -> np.ndarray:
                """Aggregate the country matrix over a (possibly resampled) index.

                mat and cnt are bound as defaults rather than captured: this
                closure is redefined every horizon, and a late-binding capture
                would silently describe whichever cell the loop reached last.
                The same hazard is documented in tslimits_figures._captured.
                """
                if ESTIMAND == "pooled_mean":
                    return mat[rows].sum(axis=0) / cnt[rows].sum()
                return np.median(mat[rows], axis=0)

            all_rows = np.arange(mat.shape[0])
            s0 = panel(all_rows)
            for k, v in served.items():
                direct = ESTIMANDS[ESTIMAND](pd.Series(v), t_h["country"])
                assert abs(direct - s0[names.index(k)]) < 1e-9, (
                    f"country matrix does not reproduce the {ESTIMAND} for {k}"
                )
            # Which fixed policy is the baseline is decided ONCE on the observed
            # data, not re-chosen inside each replicate: re-selecting the minimum
            # per replicate is a winner's curse that biases every measured gain
            # toward zero, i.e. toward this paper's own conclusion.
            i_st, i_fm = names.index("st"), names.index("fm")
            which = i_st if s0[i_st] <= s0[i_fm] else i_fm
            best0 = s0[which]

            draws = {k: np.empty(B) for k in names if k not in ("st", "fm")}
            for b in range(B):
                idx = rng.integers(0, mat.shape[0], mat.shape[0])
                q = panel(idx)
                for k in draws:
                    draws[k][b] = q[which] - q[names.index(k)]

            gap0 = best0 - s0[names.index("orc")]
            row: dict[str, Any] = {
                "fm": fm_label,
                "horizon": int(h),
                "n_countries": len(ccs),
                "best_fixed": round(best0, 4),
                "fixed_choice": "cheap" if which == i_st else "expensive",
            }
            for k, d in draws.items():
                lo, hi = _ci(d)
                pt = best0 - s0[names.index(k)]
                row[f"{k}_gain_pp"] = round(pt, 4)
                row[f"{k}_lo"] = round(lo, 4)
                row[f"{k}_hi"] = round(hi, 4)
                row[f"{k}_sig"] = bool(lo > 0)
                # One-sided 95% upper bound, and the verdict the reviewer asked
                # for: "not demonstrated" is weaker than "negligible", and only
                # a bound below the pre-specified margin licenses the latter.
                ub = float(np.percentile(d, 95))
                row[f"{k}_upper95"] = round(ub, 4)
                row[f"{k}_verdict"] = (
                    "gain" if lo > 0 else ("equivalent" if ub < MARGIN_PP else "not demonstrated")
                )
                if k != "orc":
                    row[f"{k}_frac_of_oracle"] = round(pt / max(gap0, 1e-9), 4)
                    row[f"{k}_hi_frac_of_oracle"] = round(hi / max(gap0, 1e-9), 4)
                if k in budgets:
                    row[f"{k}_budget"] = budgets[k]
                if k in thresholds:
                    # the frozen number, and what fraction it actually escalated
                    row[f"{k}_threshold"] = round(thresholds[k], 6)
                    row[f"{k}_realized_frac"] = round(realized[k], 4)
            rows.append(row)
            print(
                f"[boot] {fm_label:19s} h={h:<4} oracle {row['orc_gain_pp']:.3f} "
                f"[{row['orc_lo']:.3f},{row['orc_hi']:.3f}] | "
                + " | ".join(
                    f"{k} {row[f'{k}_gain_pp']:+.3f} "
                    f"[{row[f'{k}_lo']:+.3f},{row[f'{k}_hi']:+.3f}]"
                    f"{'*' if row[f'{k}_sig'] else ' '}"
                    for k in ("frozen_cheap", "frozen_enr", "ceiling_cheap", "ceiling_enr")
                ),
                flush=True,
            )

    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"bootstrap_ci_{ESTIMAND}{FAMILY_SFX}{GRID_SFX}.csv", index=False)
    print("", flush=True)
    for k in ("orc", "frozen_cheap", "frozen_enr", "ceiling_cheap", "ceiling_enr"):
        print(
            f"[boot] {k:14s}: significant in {int(d[f'{k}_sig'].sum())}/{len(d)} cells, "
            f"median gain {d[f'{k}_gain_pp'].median():+.3f} pp",
            flush=True,
        )
    print(f"[boot] wrote {OUT}/bootstrap_ci_{ESTIMAND}{FAMILY_SFX}{GRID_SFX}.csv (B={B}, countries)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
