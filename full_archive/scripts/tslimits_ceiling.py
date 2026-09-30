"""TS-LIMITS Phase 2d: would a signal we cannot afford do any better?

The obvious objection to the negative result is that our gate sees only seven
cheap features, so we may have shown that OUR gate fails rather than that
request-time routing fails. This closes that objection by handing the gate a
signal it has no honest right to: the foundation model's own predictive
interval width, (q90 - q10) / |q50|, on the very request being routed.

That signal is not deployable, and its undeployability is the point. Obtaining
it requires making the FM call the router exists to avoid, so a policy built on
it is a CEILING, not a candidate. Two readings, both useful:

  it also fails   the information needed to route is absent from the FM's own
                  uncertainty too, and the negative result is a property of the
                  problem rather than of our feature engineering
  it succeeds     the gate is priced: escalation pays only if the FM's
                  uncertainty can be predicted without running the FM, which is
                  a sharper and more interesting open problem

The structural model's interval width is deliberately not tested as a rival
signal: in the deployed configuration it is a deterministic function of
(country, horizon, month, hour), so it carries nothing the calendar rule does
not already have.

Protocol is identical to the deployable experiment, because a ceiling computed
under looser rules would not be comparable: cells and classifiers are fitted on
2018-2023, the escalation budget is chosen on 2024, and 2025 is scored once.

Out: reports/tslimits/ceiling_privileged.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_ceiling.py
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

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
CHEAP = (
    "horizon",
    "anchor_hour",
    "anchor_dow",
    "anchor_month",
    "anchor_load",
    "recent_resid_vol",
    "recent_load_range",
)
PRIVILEGED = (*CHEAP, "fm_width")
FIT_YEARS = (2018, 2019, 2020, 2021, 2022, 2023)
VAL_YEARS = (2024,)
BUDGETS = tuple(round(0.05 * i, 2) for i in range(21))
SEED = 0
# Which estimand this script reports. Stated here rather than inherited from a
# default, so a reader can see it without opening routing.py, and so the switch
# to the additive estimand is a one-line auditable change per script.
ESTIMAND = "panel_median"


def _load_fm(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    fm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fm = fm.rename(columns={"cc": "country", "origin_ts": "anchor_t"})
    fm["anchor_t"] = pd.to_datetime(fm["anchor_t"])
    fm["ape_fm"] = 100.0 * (fm["q50"] - fm["actual"]).abs() / fm["actual"].abs()
    # relative predictive interval width, the privileged signal
    fm["fm_width"] = (fm["q90"] - fm["q10"]).abs() / fm["q50"].abs().clip(lower=1e-9)
    return fm[["country", "horizon", "anchor_t", "actual", "ape_fm", "fm_width"]]


def _paired(window: str, pattern: str) -> pd.DataFrame:
    st = pd.read_parquet(OUT / f"per_origin_structural_{window}.parquet")
    st = st.rename(columns={"ape": "ape_st"})
    st["anchor_t"] = pd.to_datetime(st["anchor_t"])
    j = st.merge(
        _load_fm(pattern), on=["country", "horizon", "anchor_t"], how="inner", validate="one_to_one"
    )
    assert len(j) == len(st), f"{window}: joined {len(j)} of {len(st)}"
    gap = float((j["y_true"] - j["actual"]).abs().max())
    assert gap == 0.0, f"{window}: targets disagree by {gap}"
    n_bad = int(j["fm_width"].isna().sum())
    assert n_bad == 0, f"{window}: {n_bad} NaN interval widths"
    return j.reset_index(drop=True)


def _serve(df: pd.DataFrame, mask: np.ndarray) -> float:
    served = np.where(mask, df["ape_fm"].to_numpy(), df["ape_st"].to_numpy())
    return panel_median(pd.Series(served), df["country"])


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


def _fit_score(fit: pd.DataFrame, tgt: pd.DataFrame, feats: tuple[str, ...]) -> np.ndarray:
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
    clf.fit(fit[list(feats)], (fit["ape_fm"] < fit["ape_st"]).astype(int))
    return np.asarray(clf.predict_proba(tgt[list(feats)])[:, 1], dtype=float)


def main() -> int:
    rows: list[dict[str, Any]] = []

    for fm_label, (test_pat, dev_pat) in FM_PATTERNS.items():
        dev = _paired("dev", dev_pat)
        test = _paired("2025", test_pat)
        fit = dev[dev["test_year"].isin(FIT_YEARS)]
        val = dev[dev["test_year"].isin(VAL_YEARS)]
        print(f"\n[ceiling] {fm_label}", flush=True)

        for h in sorted(test["horizon"].unique()):
            f_h, v_h, t_h = (d[d["horizon"] == h].reset_index(drop=True) for d in (fit, val, test))
            ref = fixed_and_oracle(t_h["ape_st"], t_h["ape_fm"], t_h["country"], ESTIMAND)
            margin = (t_h["ape_st"] - t_h["ape_fm"]).to_numpy()
            won = (margin > 0).astype(int)

            variants = {
                "cheap": CHEAP,
                "privileged": PRIVILEGED,
            }
            scores = {
                k: (_fit_score(f_h, v_h, v), _fit_score(f_h, t_h, v)) for k, v in variants.items()
            }
            # the raw privileged signal on its own, direction learned on dev
            scores["fm_width_only"] = (
                _fit_score(f_h, v_h, ("fm_width",)),
                _fit_score(f_h, t_h, ("fm_width",)),
            )

            for name, (s_val, s_test) in scores.items():
                budget = min((_serve(v_h, _top_k(s_val, b)), b) for b in BUDGETS)[1]
                frozen = _serve(t_h, _top_k(s_test, budget))
                nat = _serve(t_h, s_test > 0.5)
                auc = float(roc_auc_score(won, s_test)) if won.min() != won.max() else float("nan")
                rows.append(
                    {
                        "fm": fm_label,
                        "horizon": int(h),
                        "policy": name,
                        "auc": round(auc, 4),
                        "rho_score_margin": round(float(spearmanr(s_test, margin).statistic), 4),
                        "best_fixed": round(ref["best_fixed"], 4),
                        "oracle": round(ref["oracle"], 4),
                        "oracle_gain_pp": round(ref["oracle_gain_pp"], 4),
                        "frozen_budget": budget,
                        "frozen_gain_pp": round(ref["best_fixed"] - frozen, 4),
                        "frozen_gap_closed": round(
                            (ref["best_fixed"] - frozen) / max(ref["oracle_gain_pp"], 1e-9), 4
                        ),
                        "natural_gain_pp": round(ref["best_fixed"] - nat, 4),
                    }
                )
            # does the FM's own uncertainty even track where the gain is?
            rho_w = float(spearmanr(t_h["fm_width"], np.clip(margin, 0, None)).statistic)
            done = [r for r in rows if r["fm"] == fm_label and r["horizon"] == h]
            print(
                f"  h={h:<4} gap {ref['oracle_gain_pp']:.3f} pp | "
                + "  ".join(
                    f"{r['policy']}: AUC {r['auc']:.3f} gain {r['frozen_gain_pp']:+.3f}"
                    for r in done
                )
                + f" | rho(width,gain) {rho_w:+.3f}",
                flush=True,
            )

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "ceiling_privileged.csv", index=False)

    print("\n=== does the privileged signal beat the cheap one? ===", flush=True)
    piv = d.pivot_table(index=["fm", "horizon"], columns="policy", values="frozen_gain_pp").round(3)
    print(piv.to_string(), flush=True)
    print("\nAUC:", flush=True)
    print(
        d.pivot_table(index=["fm", "horizon"], columns="policy", values="auc").round(3).to_string(),
        flush=True,
    )
    for pol in ("cheap", "privileged", "fm_width_only"):
        sub = d[d["policy"] == pol]
        n_win = int((sub.frozen_gain_pp > 0).sum())
        print(
            f"\n[ceiling] {pol:14s}: beats best-fixed in {n_win}/{len(sub)} cells, "
            f"median gap closed {sub.frozen_gap_closed.median():.1%}, "
            f"mean AUC {sub.auc.mean():.3f}",
            flush=True,
        )
    print(f"\n[ceiling] wrote {OUT}/ceiling_privileged.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
