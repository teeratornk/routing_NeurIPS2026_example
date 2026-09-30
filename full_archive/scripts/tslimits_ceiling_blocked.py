"""TS-LIMITS: the conditional ceiling under temporally honest cross-fitting,
plus a permutation test of the ceiling that actually has power.

Two corrections to the earlier analysis, both found by adversarial review and
both reproduced here.

FOLD SCHEME. tslimits_ceiling_conditional.py cross-fits E[margin|x] with
KFold(shuffle=True) over origins that are one per country per day across a
single year. A shuffled fold lets a held-out origin borrow from the days on
either side of it, so the "ceiling" it estimates is a within-year interpolation
and not a bound on anything a deployed router could achieve. Replacing the
folds with five contiguous time blocks turns the out-of-fold R^2 negative in
every cell and collapses the ceiling. The blocked number is the honest one: a
router is always extrapolating forward in time, never interpolating between
days it has already seen.

THE NULL. tslimits_noisefloor.py permutes the FM's errors within
(country, decile of the structural error) and finds the oracle gap unchanged,
which it read as "no routable structure". That test has no power. The oracle
gap is a functional of the within-country DISTRIBUTION of min(st, fm), and
permuting within difficulty strata preserves that distribution by construction,
so the statistic is compared against itself. On data carrying a planted,
single-feature routable signal worth over 1 pp, the test still fails to reject.

The test here targets the quantity that matters instead. Permuting the
(st, fm) pair jointly across requests within a country destroys every
association between the request-time features and the margin while leaving
each country's joint error distribution, and therefore best-fixed and the
oracle gap, exactly intact. Only x-measurable routability is destroyed. If the
observed ceiling sits inside that null, no policy measurable in x has anything
to work with; if it sits above, something is there.

Out: reports/tslimits/ceiling_blocked.csv
     reports/tslimits/ceiling_blocked_null.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_ceiling_blocked.py
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from load_forecast.eval.routing import fixed_and_oracle, panel_median

OUT = Path("reports/tslimits")
FM_PATTERNS = {
    "Chronos-2-Uni-ZS": "reports/tslimits/fm/tsl_chronos_q_*.parquet",
    "TimesFM-2.5-Uni-ZS": "reports/tslimits/fm/tsl_timesfm_q_*.parquet",
}
BUDGETS = tuple(round(0.05 * i, 2) for i in range(21))
N_FOLDS = 5
N_PERM = 99  # p resolves to 0.01; 25 could not resolve below 0.04
SEED = 0
# Which estimand this script reports. Stated here rather than inherited from a
# default, so a reader can see it without opening routing.py, and so the switch
# to the additive estimand is a one-line auditable change per script.
ESTIMAND = "panel_median"


def _emitter() -> Any:
    p = Path("scripts") / "tslimits_per_origin_structural.py"
    spec = importlib.util.spec_from_file_location("tslimits_per_origin_structural", p)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_EM = _emitter()
CHEAP: tuple[str, ...] = _EM.CHEAP_FEATURES
ENRICHED: tuple[str, ...] = _EM.GATE_FEATURES


def _load_fm(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    fm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fm = fm.rename(columns={"cc": "country", "origin_ts": "anchor_t"})
    fm["anchor_t"] = pd.to_datetime(fm["anchor_t"])
    fm["ape_fm"] = 100.0 * (fm["q50"] - fm["actual"]).abs() / fm["actual"].abs()
    return fm[["country", "horizon", "anchor_t", "actual", "ape_fm"]]


def _paired(pattern: str) -> pd.DataFrame:
    st = pd.read_parquet(OUT / "per_origin_structural_2025.parquet")
    st = st.rename(columns={"ape": "ape_st"})
    st["anchor_t"] = pd.to_datetime(st["anchor_t"])
    j = st.merge(
        _load_fm(pattern), on=["country", "horizon", "anchor_t"], how="inner", validate="one_to_one"
    )
    assert len(j) == len(st), f"joined {len(j)} of {len(st)}"
    assert float((j["y_true"] - j["actual"]).abs().max()) == 0.0
    return j.reset_index(drop=True)


def _top_k(score: np.ndarray, fraction: float) -> np.ndarray:
    n = score.size
    k = round(fraction * n)
    m = np.zeros(n, dtype=bool)
    if k <= 0:
        return m
    if k >= n:
        return ~m
    m[np.argsort(-score, kind="stable")[:k]] = True
    return m


def _reg(learner: str = "hgb", cols: list[str] | None = None) -> Any:
    """The estimator for E[margin|x], in one of two very different classes.

    What this script reports is an estimate WITHIN a function class, not a
    supremum over every policy measurable in x. An adversarial review made that
    concrete: a one-line rule on country identity beat the gradient-boosted
    estimate inside its own feature set. Carrying a second, additive and heavily
    regularised class is the cheapest honest way to show the caveat instead of
    asserting it -- where the two agree, the answer is unlikely to be an
    artifact of either.

    Both classes are given country identity in a form they can use: a
    categorical split for the tree, a one-hot column block for the linear model.
    `country_code` is an alphabetical rank, so without this the ridge fits a
    single coefficient on "how late in the alphabet the country sorts", which is
    not a country effect and would make the second class a straw man.
    """
    cat = _EM.categorical_mask(cols) if cols is not None else None
    if learner == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=300, random_state=SEED, categorical_features=cat
        )
    if learner == "ridge":
        if cat is None or not cat.any():
            return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
        pre = ColumnTransformer(
            [
                ("id", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat)),
                ("num", StandardScaler(), list(~cat)),
            ]
        )
        return make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 4, 25)))
    raise ValueError(f"unknown learner {learner!r}")


def _blocks(anchor: pd.Series, n: int) -> np.ndarray:
    """`n` contiguous blocks in ACTUAL time over the scored window.

    Day-of-year is wrong here, and silently so. Every paired horizon anchors
    from December of the preceding year (h=720 spans 2024-12-02 to 2025-12-01),
    so day-of-year wraps and the last "block" ends up holding both Decembers,
    which is not contiguous in time and not a clean forward split. Rank on the
    timestamp itself.
    """
    t = pd.to_datetime(anchor).astype("int64").to_numpy()
    span = int(t.max() - t.min()) + 1
    return np.minimum((t - t.min()) * n // span, n - 1).astype(np.int64)


def _splits(anchor: pd.Series, target: pd.Series, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """`n` contiguous time blocks, with the training side embargoed.

    Blocks are cut on the ANCHOR, but the outcome is realized `h` hours later.
    A training row whose TARGET falls inside the held-out block has therefore
    already seen the period it is being scored on. Dropping those rows costs
    133 rows at h=168 and 570 at h=720, and it matters: with a dropped-random
    -rows control to separate leak from shrinkage, removing them takes the
    Chronos h=168 ceiling from 0.107 to 0.038 pp while the control leaves it at
    0.105. Without the embargo the ceiling is inflated, and because the
    permutation null cannot exploit the leak, its p-values are anticonservative.
    """
    blk = _blocks(anchor, n)
    a = pd.to_datetime(anchor).astype("int64").to_numpy()
    g = pd.to_datetime(target).astype("int64").to_numpy()
    out = []
    for k in range(n):
        te = np.flatnonzero(blk == k)
        lo, hi = a[te].min(), a[te].max()
        tr = np.flatnonzero((blk != k) & ~((g >= lo) & (g <= hi)))
        out.append((tr, te))
    return out


def _ceiling(
    st: np.ndarray,
    fm: np.ndarray,
    cc: pd.Series,
    feat: pd.DataFrame,
    cv: Any,
    learner: str = "hgb",
) -> tuple[float, float, float]:
    """Best top-k gain over the budget grid using out-of-fold E[margin|x]."""
    y = st - fm
    oof = np.asarray(
        cross_val_predict(_reg(learner, list(feat.columns)), feat, y, cv=cv, n_jobs=1),
        dtype=float,
    )
    ref = fixed_and_oracle(pd.Series(st), pd.Series(fm), cc, ESTIMAND)
    best = min(
        panel_median(pd.Series(np.where(_top_k(oof, b), fm, st)), cc) for b in BUDGETS
    )
    r2 = 1.0 - float(np.sum((y - oof) ** 2)) / float(np.sum((y - y.mean()) ** 2))
    return ref["best_fixed"] - best, r2, ref["oracle_gain_pp"]


def _classes_only() -> int:
    """Just the two-function-class comparison, which needs no permutation null.

    The null loop is 99 draws x 5 folds x 24 configurations of gradient boosting
    and runs for hours; the class comparison is 16 cross-fits. Separating them
    means the paper's "a different function class gives a different number"
    claim has an artifact without blocking on the null, which is a different
    claim with a different artifact.
    """
    rows: list[dict[str, Any]] = []
    for fm_label, pattern in FM_PATTERNS.items():
        test = _paired(pattern)
        for h in sorted(test["horizon"].unique()):
            t = test[test["horizon"] == h].reset_index(drop=True)
            st, fm = t["ape_st"].to_numpy(), t["ape_fm"].to_numpy()
            fx = t[list(ENRICHED)]
            cv = _splits(t["anchor_t"], t["target_t"], N_FOLDS)
            row: dict[str, Any] = {"fm": fm_label, "horizon": int(h), "estimand": ESTIMAND}
            for learner in ("hgb", "ridge"):
                c_blk, r2_blk, gap = _ceiling(st, fm, t["country"], fx, cv, learner)
                row[f"{learner}_blocked_pp"] = round(c_blk, 4)
                row[f"{learner}_r2_blocked"] = round(r2_blk, 4)
                row["oracle_gain_pp"] = round(gap, 4)
            row["abs_class_difference_pp"] = round(
                abs(row["hgb_blocked_pp"] - row["ridge_blocked_pp"]), 4
            )
            rows.append(row)
            print(
                f"[classes] {fm_label:19s} h={int(h):<4} hgb {row['hgb_blocked_pp']:.4f} "
                f"(R2 {row['hgb_r2_blocked']:+.3f})  ridge {row['ridge_blocked_pp']:.4f} "
                f"(R2 {row['ridge_r2_blocked']:+.3f})",
                flush=True,
            )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "ceiling_classes.csv", index=False)
    print(
        f"\n[classes] hgb median {d.hgb_blocked_pp.median():.4f} pp, "
        f"ridge median {d.ridge_blocked_pp.median():.4f} pp, "
        f"max |difference| {d.abs_class_difference_pp.max():.4f} pp, "
        f"worst ridge R2 {d.ridge_r2_blocked.min():+.2f}",
        flush=True,
    )
    print(f"[classes] wrote {OUT}/ceiling_classes.csv", flush=True)
    return 0


def main() -> int:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    nulls: list[dict[str, Any]] = []

    for fm_label, pattern in FM_PATTERNS.items():
        test = _paired(pattern)
        print(f"\n[blocked] {fm_label}", flush=True)

        for h in sorted(test["horizon"].unique()):
            t = test[test["horizon"] == h].reset_index(drop=True)
            st, fm = t["ape_st"].to_numpy(), t["ape_fm"].to_numpy()
            cc = t["country"]

            # COUNTRY is not a router: it is a per-country FIXED choice, and it
            # is reported to show how much of the residual "signal" is just
            # that. An adversarial review found the permutation null preserves
            # E[margin|country] exactly, so country-measurable policies score
            # identically under it -- which is why this baseline has to be read
            # beside the null rather than through it.
            variants = (
                ("country", ("country_code",)),
                ("cheap", CHEAP),
                ("enriched", ENRICHED),
            )
            for fs, feats in variants:
                fx = t[list(feats)]
                cv = _splits(t["anchor_t"], t["target_t"], N_FOLDS)
                rand_cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
                c_rand, r2_rand, gap = _ceiling(st, fm, cc, fx, rand_cv)
                c_blk, r2_blk, _ = _ceiling(st, fm, cc, fx, cv)

                # Permutation null on the blocked ceiling: shuffle the (st, fm)
                # pair jointly within country, so each country's joint error
                # distribution and hence best-fixed and the oracle gap are
                # untouched, and only the association with x is destroyed.
                draws = np.empty(N_PERM)
                for r in range(N_PERM):
                    stp, fmp = st.copy(), fm.copy()
                    for c in cc.unique():
                        m = np.flatnonzero((cc == c).to_numpy())
                        p = rng.permutation(m)
                        stp[m], fmp[m] = st[p], fm[p]
                    ref_p = fixed_and_oracle(pd.Series(stp), pd.Series(fmp), cc, ESTIMAND)
                    assert abs(ref_p["oracle_gain_pp"] - gap) < 1e-9, "null moved the gap"
                    draws[r] = _ceiling(stp, fmp, cc, fx, cv)[0]
                p_ge = float((draws >= c_blk).mean())

                rows.append(
                    {
                        "fm": fm_label,
                        "horizon": int(h),
                        "features": fs,
                        "learner": "hgb",
                        "oracle_gain_pp": round(gap, 4),
                        "ceiling_random_pp": round(c_rand, 4),
                        "r2_random": round(r2_rand, 4),
                        "ceiling_blocked_pp": round(c_blk, 4),
                        "r2_blocked": round(r2_blk, 4),
                        "null_mean_pp": round(float(draws.mean()), 4),
                        "null_hi_pp": round(float(np.percentile(draws, 97.5)), 4),
                        "p_null_ge_obs": round(p_ge, 3),
                    }
                )
                for v in draws:
                    nulls.append(
                        {
                            "fm": fm_label,
                            "horizon": int(h),
                            "features": fs,
                            "learner": "hgb",
                            "draw": round(v, 6),
                        }
                    )
                print(
                    f"  h={h:<4} {fs:9s} gap {gap:.3f} | random {c_rand:.3f} (R2 {r2_rand:+.3f})"
                    f" | blocked {c_blk:.3f} (R2 {r2_blk:+.3f})"
                    f" | null {draws.mean():.3f} [<= {np.percentile(draws, 97.5):.3f}]"
                    f" p={p_ge:.2f}",
                    flush=True,
                )

    # Second function class, on the enriched features only. No permutation null
    # here: the point is whether the benchmark is class-dependent, and that is
    # answered by comparing the two point estimates under identical folds.
    print("\n[blocked] second function class (ridge) for comparison", flush=True)
    for fm_label, pattern in FM_PATTERNS.items():
        test = _paired(pattern)
        for h in sorted(test["horizon"].unique()):
            t = test[test["horizon"] == h].reset_index(drop=True)
            st, fm = t["ape_st"].to_numpy(), t["ape_fm"].to_numpy()
            fx = t[list(ENRICHED)]
            cv = _splits(t["anchor_t"], t["target_t"], N_FOLDS)
            c_blk, r2_blk, gap = _ceiling(st, fm, t["country"], fx, cv, "ridge")
            c_rand, r2_rand, _ = _ceiling(
                st, fm, t["country"], fx,
                KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED), "ridge",
            )
            rows.append(
                {
                    "fm": fm_label,
                    "horizon": int(h),
                    "features": "enriched",
                    "learner": "ridge",
                    "oracle_gain_pp": round(gap, 4),
                    "ceiling_random_pp": round(c_rand, 4),
                    "r2_random": round(r2_rand, 4),
                    "ceiling_blocked_pp": round(c_blk, 4),
                    "r2_blocked": round(r2_blk, 4),
                    "null_mean_pp": None,
                    "null_hi_pp": None,
                    "p_null_ge_obs": None,
                }
            )
            print(f"  h={h:<4} ridge blocked {c_blk:.4f} (R2 {r2_blk:+.3f})", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "ceiling_blocked.csv", index=False)
    pd.DataFrame(nulls).to_csv(OUT / "ceiling_blocked_null.csv", index=False)
    for fs in ("country", "cheap", "enriched"):
        s = d[d["features"] == fs]
        print(
            f"\n[blocked] {fs:9s}: random-fold ceiling median {s.ceiling_random_pp.median():.3f} pp"
            f" (R2>0 in {int((s.r2_random > 0).sum())}/{len(s)})"
            f" -> blocked median {s.ceiling_blocked_pp.median():.3f} pp"
            f" (R2>0 in {int((s.r2_blocked > 0).sum())}/{len(s)});"
            f" ceiling above its own null in {int((s.p_null_ge_obs < 0.05).sum())}/{len(s)} cells",
            flush=True,
        )
    print(f"\n[blocked] wrote {OUT}/ceiling_blocked.csv (+ _null.csv, {N_PERM} draws/cell)")
    return 0


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument(
        "--classes-only",
        action="store_true",
        help="only the two-function-class comparison, skipping the permutation nulls",
    )
    _args = _ap.parse_args()
    raise SystemExit(_classes_only() if _args.classes_only else main())
