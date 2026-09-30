"""TS-LIMITS Phase 2b: why is the oracle gap unreachable?

The frozen policies capture at most ~10-13% of a 0.22-1.89 pp per-request
oracle gap, and nothing at all in 18 of 24 cells. A negative result is only
worth reporting if it is explained, so this separates the candidate causes:

  1. No signal. The request-time features simply do not predict which model
     wins. Measured as AUC of the frozen classifier on 2025.
  2. Signal, wrong target. The gate predicts *which* model wins but not *by how
     much*, so it spends escalations on requests where the win is negligible
     and misses the few where it is large. Measured as Spearman rank
     correlation between the gate score and the realized win margin, and as the
     concentration of total achievable gain in the tail of |margin|.
  3. Estimand insensitivity. The panel median is a median of medians and barely
     moves when a minority of requests are re-served. Tested by recomputing the
     same frozen comparison under pooled mean APE and per-country mean, where a
     re-served request always moves the number.

Cause 2 and cause 3 have opposite implications for the paper: 2 says routing is
hard, 3 says our yardstick cannot see routing. Both are reported.

Out: reports/tslimits/why_diagnostics.csv, reports/tslimits/why_concentration.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_why.py
"""

from __future__ import annotations

import glob
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from load_forecast.eval.routing import panel_median

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


def _emitter() -> Any:
    """Import the per-origin script so the feature list has one definition."""
    p = Path("scripts") / "tslimits_per_origin_structural.py"
    spec = importlib.util.spec_from_file_location("tslimits_per_origin_structural", p)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# The enriched set, so the diagnosis describes the gate the paper actually
# reports rather than a weaker one.
FEATURES: tuple[str, ...] = _emitter().GATE_FEATURES
FIT_YEARS = (2018, 2019, 2020, 2021, 2022, 2023)
SEED = 0


def _load_fm(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    fm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fm = fm.rename(columns={"cc": "country", "origin_ts": "anchor_t"})
    fm["anchor_t"] = pd.to_datetime(fm["anchor_t"])
    fm["ape_fm"] = 100.0 * (fm["q50"] - fm["actual"]).abs() / fm["actual"].abs()
    return fm[["country", "horizon", "anchor_t", "ape_fm"]]


def _paired(window: str, pattern: str) -> pd.DataFrame:
    st = pd.read_parquet(OUT / f"per_origin_structural_{window}.parquet")
    st = st.rename(columns={"ape": "ape_st"})
    st["anchor_t"] = pd.to_datetime(st["anchor_t"])
    j = st.merge(
        _load_fm(pattern), on=["country", "horizon", "anchor_t"], how="inner", validate="one_to_one"
    )
    assert len(j) == len(st), f"{window}: joined {len(j)} of {len(st)}"
    return j.reset_index(drop=True)


def _pooled_mean(v: pd.Series, _c: pd.Series) -> float:
    return float(np.mean(np.asarray(v, dtype=float)))


def _country_mean(v: pd.Series, c: pd.Series) -> float:
    f = pd.DataFrame({"v": np.asarray(v, dtype=float), "c": np.asarray(c)})
    return float(f.groupby("c")["v"].mean().median())


ESTIMANDS = {
    "panel_median": panel_median,
    "pooled_mean": _pooled_mean,
    "country_mean": _country_mean,
}


def main() -> int:
    diag: list[dict[str, Any]] = []
    conc: list[dict[str, Any]] = []

    for fm_label, (test_pat, dev_pat) in FM_PATTERNS.items():
        dev = _paired("dev", dev_pat)
        test = _paired("2025", test_pat)
        fit = dev[dev["test_year"].isin(FIT_YEARS)]

        for h in sorted(test["horizon"].unique()):
            f_h = fit[fit["horizon"] == h]
            t_h = test[test["horizon"] == h].reset_index(drop=True)

            # categorical_features so the classifier reads country identity as a
            # label rather than an alphabetical rank, matching the regressor gate
            # reported in the body tables
            clf = HistGradientBoostingClassifier(
                max_iter=300,
                random_state=SEED,
                categorical_features=_emitter().categorical_mask(list(FEATURES)),
            )
            clf.fit(f_h[list(FEATURES)], (f_h["ape_fm"] < f_h["ape_st"]).astype(int))
            p = clf.predict_proba(t_h[list(FEATURES)])[:, 1]

            margin = (t_h["ape_st"] - t_h["ape_fm"]).to_numpy()  # >0 means FM wins
            won = (margin > 0).astype(int)

            # 1. is there any signal at all?
            auc = float(roc_auc_score(won, p)) if won.min() != won.max() else float("nan")
            # 2. does the score rank the *size* of the win, not just its sign?
            rho_sign = float(spearmanr(p, won).statistic)
            rho_margin = float(spearmanr(p, margin).statistic)

            # where does the achievable gain actually live?
            gain = np.clip(margin, 0.0, None)  # only wins are exploitable
            order = np.argsort(-gain)
            tot = gain.sum()
            n = gain.size
            shares = {
                f"top{int(q * 100)}pct_share_of_gain": round(
                    float(gain[order[: max(1, int(q * n))]].sum() / max(tot, 1e-12)), 4
                )
                for q in (0.01, 0.05, 0.10, 0.25)
            }
            conc.append({"fm": fm_label, "horizon": int(h), "n": n, **shares})

            # 3. is the negative an artifact of the estimand?
            row: dict[str, Any] = {
                "fm": fm_label,
                "horizon": int(h),
                "auc": round(auc, 4),
                "spearman_score_vs_win": round(rho_sign, 4),
                "spearman_score_vs_margin": round(rho_margin, 4),
                "fm_win_share": round(float(won.mean()), 4),
            }
            for est_name, est in ESTIMANDS.items():
                st_v = est(t_h["ape_st"], t_h["country"])
                fm_v = est(t_h["ape_fm"], t_h["country"])
                orc = est(pd.Series(np.minimum(t_h["ape_st"], t_h["ape_fm"])), t_h["country"])
                best = min(st_v, fm_v)
                # the learned gate at its natural threshold, under this estimand
                served = np.where(p > 0.5, t_h["ape_fm"], t_h["ape_st"])
                pol = est(pd.Series(served), t_h["country"])
                row[f"{est_name}_oracle_gain_pp"] = round(best - orc, 4)
                row[f"{est_name}_learned_gain_pp"] = round(best - pol, 4)
                row[f"{est_name}_gap_closed"] = round((best - pol) / max(best - orc, 1e-9), 4)
            diag.append(row)
            print(
                f"[why] {fm_label:19s} h={h:<4} AUC {auc:.3f}  "
                f"rho(score,margin) {rho_margin:+.3f}  "
                f"top5% holds {shares['top5pct_share_of_gain']:.0%} of gain  "
                f"gap closed: median {row['panel_median_gap_closed']:+.1%} "
                f"pooled {row['pooled_mean_gap_closed']:+.1%}",
                flush=True,
            )

    d = pd.DataFrame(diag)
    d.to_csv(OUT / "why_diagnostics.csv", index=False)
    pd.DataFrame(conc).to_csv(OUT / "why_concentration.csv", index=False)

    print("\n=== is the negative an artifact of the panel median? ===", flush=True)
    cols = [c for c in d.columns if c.endswith("gap_closed")]
    print(d[["fm", "horizon", "auc", *cols]].to_string(index=False), flush=True)
    print(f"\n[why] wrote {OUT}/why_diagnostics.csv and why_concentration.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
