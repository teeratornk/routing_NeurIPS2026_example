"""TS-LIMITS: what could a policy measurable in the features actually reach?

The per-request oracle escalates on the realized minimum, so it exploits noise
that no rule can see, and reporting a policy's shortfall against it mostly
measures how noisy the two error series are. The quantity a router can be held
to is the conditional one: escalate where the EXPECTED margin given the
request-time features is positive.

Three numbers per cell, and the distances between them are the point:

  deployable   E[margin|x] fitted on 2018-2023, budget chosen on 2024, applied
               to 2025. What a deployed router gets.
  ceiling      E[margin|x] cross-fitted on 2025 itself, out of fold, with the
               budget also read off 2025. NOT deployable and not offered as a
               result: it is the best a policy measurable in x could reach given
               unlimited data from the test distribution.
  oracle       the per-request minimum.

  deployable -> ceiling   generalisation loss: signal that exists in x but does
                          not survive the transfer from 2018-2023 to 2025
  ceiling -> oracle       information loss: margin variation that x cannot
                          resolve at all

If the ceiling is near zero the features do not contain the answer and no amount
of training data or model capacity fixes it. If the ceiling is large and the
deployable number is not, the problem is drift, which is a different paper.

The ceiling is deliberately generous. Origins are one per day and adjacent days
are correlated, so random cross-fitting lets a fold borrow strength from its
neighbours; that inflates the ceiling. An upper bound that is too high is still
an upper bound, and it makes a negative reading harder to dismiss.

One caveat stated rather than buried: thresholding E[margin|x] is exactly
optimal for a pooled-sum objective, and only near-optimal for the panel median
the paper reports. The budget sweep over the cross-fitted ranking absorbs most
of the difference.

Out: reports/tslimits/ceiling_conditional.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_ceiling_conditional.py
"""

from __future__ import annotations

import glob
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

from load_forecast.eval.routing import fixed_and_oracle, panel_median

OUT = Path("reports/tslimits")
FM_PATTERNS = {
    "Chronos-2-Uni-ZS": (
        "reports/tslimits/fm/tsl_chronos_q_*.parquet",
        "reports/tslimits/fm/tsldev_chronos_q_*.parquet",
    ),
    "TimesFM-2.5-Uni-ZS": (
        "reports/tslimits/fm/tsl_timesfm_q_*.parquet",
        "reports/tslimits/fm/tsldev_timesfm_q_*.parquet",
    ),
}
FIT_YEARS = (2018, 2019, 2020, 2021, 2022, 2023)
VAL_YEARS = (2024,)
BUDGETS = tuple(round(0.05 * i, 2) for i in range(21))
N_FOLDS = 5
SEED = 0
# Which estimand this script reports. Stated here rather than inherited from a
# default, so a reader can see it without opening routing.py, and so the switch
# to the additive estimand is a one-line auditable change per script.
ESTIMAND = "panel_median"


def _emitter() -> Any:
    """Import the per-origin script so the feature lists have one definition."""
    p = Path("scripts") / "tslimits_per_origin_structural.py"
    spec = importlib.util.spec_from_file_location("tslimits_per_origin_structural", p)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_EM = _emitter()
CHEAP: tuple[str, ...] = _EM.CHEAP_FEATURES
ENRICHED: tuple[str, ...] = _EM.GATE_FEATURES
FEATURE_SETS = {"cheap": CHEAP, "enriched": ENRICHED}


def _load_fm(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
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
    return panel_median(pd.Series(served), df["country"])


def _reg() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, random_state=SEED)


def main() -> int:
    rows: list[dict[str, Any]] = []
    print(f"[cond] cheap: {len(CHEAP)} features; enriched: {len(ENRICHED)}", flush=True)

    for fm_label, (test_pat, dev_pat) in FM_PATTERNS.items():
        dev = _paired("dev", dev_pat)
        test = _paired("2025", test_pat)
        fit = dev[dev["test_year"].isin(FIT_YEARS)]
        val = dev[dev["test_year"].isin(VAL_YEARS)]
        print(f"\n[cond] {fm_label}", flush=True)

        for h in sorted(test["horizon"].unique()):
            f_h, v_h, t_h = (
                d[d["horizon"] == h].reset_index(drop=True) for d in (fit, val, test)
            )
            ref = fixed_and_oracle(t_h["ape_st"], t_h["ape_fm"], t_h["country"], ESTIMAND)
            margin_test = (t_h["ape_st"] - t_h["ape_fm"]).to_numpy()

            for fs_name, feats in FEATURE_SETS.items():
                cols = list(feats)

                # ---- deployable: fitted on 2018-2023, budget on 2024 ----
                dep = _reg().fit(f_h[cols], f_h["ape_st"] - f_h["ape_fm"])
                s_val = np.asarray(dep.predict(v_h[cols]), dtype=float)
                s_test = np.asarray(dep.predict(t_h[cols]), dtype=float)
                budget = min((_serve(v_h, _top_k(s_val, b)), b) for b in BUDGETS)[1]
                deployable = _serve(t_h, _top_k(s_test, budget))
                dep_nat = _serve(t_h, s_test > 0.0)

                # ---- ceiling: cross-fitted on 2025 itself, out of fold ----
                kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
                oof = np.asarray(
                    cross_val_predict(_reg(), t_h[cols], margin_test, cv=kf, n_jobs=1),
                    dtype=float,
                )
                curve = [(b, _serve(t_h, _top_k(oof, b))) for b in BUDGETS]
                c_budget, c_ape = min(curve, key=lambda t: t[1])
                ceil_nat = _serve(t_h, oof > 0.0)

                rows.append(
                    {
                        "fm": fm_label,
                        "horizon": int(h),
                        "features": fs_name,
                        "n_features": len(cols),
                        "best_fixed": round(ref["best_fixed"], 4),
                        "oracle_gain_pp": round(ref["oracle_gain_pp"], 4),
                        "deployable_budget": budget,
                        "deployable_gain_pp": round(ref["best_fixed"] - deployable, 4),
                        "deployable_natural_gain_pp": round(ref["best_fixed"] - dep_nat, 4),
                        "ceiling_budget": c_budget,
                        "ceiling_gain_pp": round(ref["best_fixed"] - c_ape, 4),
                        "ceiling_natural_gain_pp": round(ref["best_fixed"] - ceil_nat, 4),
                        "generalisation_loss_pp": round(
                            (ref["best_fixed"] - c_ape) - (ref["best_fixed"] - deployable), 4
                        ),
                        "information_loss_pp": round(
                            ref["oracle_gain_pp"] - (ref["best_fixed"] - c_ape), 4
                        ),
                        # can the cross-fitted regressor rank margins at all?
                        "oof_rho": round(float(spearmanr(oof, margin_test).statistic), 4),
                        "oof_r2": round(
                            1.0
                            - float(np.sum((margin_test - oof) ** 2))
                            / float(np.sum((margin_test - margin_test.mean()) ** 2)),
                            4,
                        ),
                        "deployed_rho": round(
                            float(spearmanr(s_test, margin_test).statistic), 4
                        ),
                    }
                )
            done = [r for r in rows if r["fm"] == fm_label and r["horizon"] == h]
            print(
                f"  h={h:<4} oracle {ref['oracle_gain_pp']:.3f} pp | "
                + " | ".join(
                    f"{r['features']}: deploy {r['deployable_gain_pp']:+.3f} "
                    f"ceiling {r['ceiling_gain_pp']:+.3f} "
                    f"(oof rho {r['oof_rho']:+.3f}, R2 {r['oof_r2']:+.3f})"
                    for r in done
                ),
                flush=True,
            )

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "ceiling_conditional.csv", index=False)

    print("\n=== gain vs best-fixed (pp); ceiling is not deployable ===", flush=True)
    print(
        res.pivot_table(
            index=["fm", "horizon"],
            columns="features",
            values=["deployable_gain_pp", "ceiling_gain_pp"],
        )
        .round(3)
        .to_string(),
        flush=True,
    )
    for fs in FEATURE_SETS:
        s = res[res["features"] == fs]
        print(
            f"\n[cond] {fs:9s}: deployable beats best-fixed in "
            f"{int((s.deployable_gain_pp > 0).sum())}/{len(s)} cells "
            f"(median {s.deployable_gain_pp.median():+.3f} pp); "
            f"ceiling positive in {int((s.ceiling_gain_pp > 0).sum())}/{len(s)} "
            f"(median {s.ceiling_gain_pp.median():+.3f} pp); "
            f"median OOF rho {s.oof_rho.median():+.3f}",
            flush=True,
        )
    print(f"\n[cond] wrote {OUT}/ceiling_conditional.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
