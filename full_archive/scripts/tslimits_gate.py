"""TS-LIMITS Phase 2: can any request-time policy capture the oracle gap?

Phase 0 measured a per-request oracle gap of 0.22-1.89 pp at every horizon and
showed the obvious heuristic (escalate when recent residual volatility is high)
is *worse* than never escalating. This asks whether any request-time policy
reaches that budget.

Protocol, and the reason for it. Reporting the best point on a 2025 frontier
would be selection on the test set: with a free budget parameter almost any
score looks good in hindsight. So every policy here is frozen end to end before
2025 is scored.

    fit    2018-2023   rule cells and the classifier
    val    2024        the escalation budget, and nothing else
    test   2025        scored once, with model and budget already fixed

Three operating points are reported per policy, and they answer different
questions:

    frozen      budget picked on val, applied to 2025. The deployable number,
                and the one the paper's claim rests on.
    natural     the policy's own decision rule (dev margin > 0, or p > 0.5),
                no budget at all. Deployable without a tuning step.
    frontier    the best point on the 2025 curve. NOT deployable: it is chosen
                with the answers in hand. Reported only as the ceiling any
                budget-tuning could have reached, and labelled as such.

Policies are evaluated within a horizon, since a per-horizon choice is already
the best_fixed baseline; a gate has to beat that, not rediscover it.

Out: reports/tslimits/gate_policies.csv, reports/tslimits/gate_frontier.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_gate.py
"""

from __future__ import annotations

import glob
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

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

def _emitter() -> Any:
    """Import the per-origin script so the feature lists have one definition."""
    p = Path("scripts") / "tslimits_per_origin_structural.py"
    spec = importlib.util.spec_from_file_location("tslimits_per_origin_structural", p)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_EM = _emitter()
# The original seven: horizon, anchor calendar, and load statistics ending at
# the anchor.
FEATURES: tuple[str, ...] = _EM.CHEAP_FEATURES
# Everything a request-time router could legitimately know, including the
# covariates the structural model uses and the univariate FMs cannot see: the
# published holiday calendar at the forecast hour, the forecast hour's own
# calendar attributes, a temperature climatology, the temperature measured at
# the anchor, and country identity. Denying the gate these while attributing the
# fixed-comparison result to exactly this asymmetry would not be a fair test.
ENRICHED: tuple[str, ...] = _EM.GATE_FEATURES
# Anything matching these in a feature name would let the gate see the target
# or the foundation model's own output, which would make the whole result void.
FORBIDDEN = ("y_true", "actual", "ape", "chronos", "timesfm", "fm_", "y_pred")
FIT_YEARS = (2018, 2019, 2020, 2021, 2022, 2023)
VAL_YEARS = (2024,)
BUDGETS = tuple(round(0.05 * i, 2) for i in range(21))
SEED = 0
# Which estimand this script reports. Stated here rather than inherited from a
# default, so a reader can see it without opening routing.py, and so the switch
# to the additive estimand is a one-line auditable change per script.
ESTIMAND = "panel_median"


def _assert_request_time(cols: tuple[str, ...]) -> None:
    bad = [c for c in cols if any(f in c.lower() for f in FORBIDDEN)]
    if bad:
        raise AssertionError(f"gate features leak the target or the FM output: {bad}")


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
    assert len(j) == len(st), f"{window}: joined {len(j)} of {len(st)} structural origins"
    gap = float((j["y_true"] - j["actual"]).abs().max())
    assert gap == 0.0, f"{window}: the two sides disagree on the target by {gap}"
    return j.reset_index(drop=True)


def _cells(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        daypart=df["anchor_hour"] // 6,
        weekend=(df["anchor_dow"] >= 5).astype(int),
    )


def _serve(df: pd.DataFrame, mask: np.ndarray) -> float:
    """Panel-median APE when `mask` requests are served by the FM."""
    served = np.where(mask, df["ape_fm"].to_numpy(), df["ape_st"].to_numpy())
    return panel_median(pd.Series(served), df["country"])


def _top_k_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    """Top-`fraction` by score; rank-based so the count is exact under ties."""
    n = score.size
    k = round(fraction * n)
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    if k >= n:
        return ~mask
    mask[np.argsort(-score, kind="stable")[:k]] = True
    return mask


def _rule_score(fit: pd.DataFrame, target: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Mean win margin per cell, learned on `fit`, looked up for `target`.

    Positive means the FM won that cell on the development window. Cells unseen
    in fit get the minimum observed margin, i.e. never escalate: the
    conservative default, since an unseen cell is not evidence of a win.
    """
    margin = fit.assign(gain=fit["ape_st"] - fit["ape_fm"]).groupby(keys)["gain"].mean()
    s = target[keys].merge(margin.rename("s").reset_index(), on=keys, how="left")["s"].to_numpy()
    if np.isnan(s).any():
        s = np.where(np.isnan(s), np.nanmin(s) - 1.0, s)
    return np.asarray(s, dtype=float)


def main() -> int:
    _assert_request_time(FEATURES)
    _assert_request_time(ENRICHED)
    rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []

    for fm_label, (test_pat, dev_pat) in FM_PATTERNS.items():
        dev = _cells(_paired("dev", dev_pat))
        test = _cells(_paired("2025", test_pat))
        fit = dev[dev["test_year"].isin(FIT_YEARS)]
        val = dev[dev["test_year"].isin(VAL_YEARS)]
        print(
            f"\n[gate] {fm_label}: fit {len(fit):,} / val {len(val):,} / test {len(test):,}",
            flush=True,
        )

        for h in sorted(test["horizon"].unique()):
            fit_h, val_h, test_h = (
                d[d["horizon"] == h].reset_index(drop=True) for d in (fit, val, test)
            )
            ref = fixed_and_oracle(test_h["ape_st"], test_h["ape_fm"], test_h["country"], ESTIMAND)

            builders: dict[str, Any] = {
                "calendar": lambda src, tgt: _rule_score(src, tgt, ["daypart", "weekend"]),
                "volatility": lambda src, tgt: tgt["recent_resid_vol"].to_numpy(dtype=float),
            }

            def _learned(
                src: pd.DataFrame, tgt: pd.DataFrame, cols: tuple[str, ...] = FEATURES
            ) -> np.ndarray:
                clf = HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
                clf.fit(src[list(cols)], (src["ape_fm"] < src["ape_st"]).astype(int))
                return np.asarray(clf.predict_proba(tgt[list(cols)])[:, 1], dtype=float)

            def _learned_margin(
                src: pd.DataFrame, tgt: pd.DataFrame, cols: tuple[str, ...] = FEATURES
            ) -> np.ndarray:
                """Regress the win MARGIN, not its sign.

                The exploitable gain is concentrated in a thin tail, so a
                classifier trained on "which model wins" optimises the wrong
                target: it treats a 0.01 pp win and a 10 pp win as the same
                label. This is the obvious rejoinder to the diagnosis and it
                costs one extra fit, so we run it rather than argue about it.
                """
                reg = HistGradientBoostingRegressor(max_iter=300, random_state=SEED)
                reg.fit(src[list(cols)], src["ape_st"] - src["ape_fm"])
                return np.asarray(reg.predict(tgt[list(cols)]), dtype=float)

            builders["learned"] = _learned
            builders["learned_margin"] = _learned_margin
            # Same two learners on the enriched feature set. Reported beside the
            # cheap ones rather than replacing them, so the effect of the extra
            # covariates is the difference between two otherwise identical runs.
            builders["learned_enriched"] = (
                lambda src, tgt: _learned(src, tgt, ENRICHED)
            )
            builders["learned_margin_enriched"] = (
                lambda src, tgt: _learned_margin(src, tgt, ENRICHED)
            )

            for name, build in builders.items():
                s_val, s_test = build(fit_h, val_h), build(fit_h, test_h)

                # budget chosen on val only, then frozen
                v = [(_serve(val_h, _top_k_mask(s_val, b)), b) for b in BUDGETS]
                budget = min(v)[1]
                frozen = _serve(test_h, _top_k_mask(s_test, budget))

                # the policy's own rule, no budget
                # The volatility cut is the median of the FIT window, not of
                # s_test: a threshold read off the test scores is not a policy
                # anyone could have deployed.
                # A margin regressor's own rule is "escalate where the expected
                # margin is positive", so its cut is 0. An earlier version gave
                # it 0.5 by matching the classifier's prefix, which silently
                # demanded a half-point expected gain before escalating.
                if name.startswith("learned_margin"):
                    cut = 0.0
                elif name.startswith("learned"):
                    cut = 0.5
                elif name == "volatility":
                    cut = float(np.median(build(fit_h, fit_h)))
                else:
                    cut = 0.0
                nat_mask = s_test > cut
                natural = _serve(test_h, nat_mask)

                # 2025 curve; its minimum is chosen with the answers in hand
                curve = [(b, _serve(test_h, _top_k_mask(s_test, b))) for b in BUDGETS]
                curves.append(
                    pd.DataFrame(curve, columns=["escalated_fraction", "panel_median_ape"]).assign(
                        fm=fm_label, horizon=h, policy=name
                    )
                )
                best_frac, best_ape = min(curve, key=lambda t: t[1])

                rows.append(
                    {
                        "fm": fm_label,
                        "horizon": int(h),
                        "policy": name,
                        "best_fixed": round(ref["best_fixed"], 4),
                        "oracle": round(ref["oracle"], 4),
                        "oracle_gain_pp": round(ref["oracle_gain_pp"], 4),
                        "frozen_budget": budget,
                        "frozen_ape": round(frozen, 4),
                        "frozen_gain_pp": round(ref["best_fixed"] - frozen, 4),
                        "frozen_gap_closed": round(
                            (ref["best_fixed"] - frozen) / max(ref["oracle_gain_pp"], 1e-9), 4
                        ),
                        "natural_frac": round(float(nat_mask.mean()), 4),
                        "natural_ape": round(natural, 4),
                        "natural_gain_pp": round(ref["best_fixed"] - natural, 4),
                        "oracle_sel_frac": best_frac,
                        "oracle_sel_ape": round(best_ape, 4),
                        "oracle_sel_gain_pp": round(ref["best_fixed"] - best_ape, 4),
                    }
                )
            done = [r for r in rows if r["fm"] == fm_label and r["horizon"] == h]
            print(
                f"  h={h:<4} fixed {ref['best_fixed']:.3f} oracle {ref['oracle']:.3f} "
                f"(gain {ref['oracle_gain_pp']:.3f} pp) | "
                + "  ".join(f"{r['policy']}: {r['frozen_gain_pp']:+.3f}" for r in done),
                flush=True,
            )

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "gate_policies.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "gate_frontier.csv", index=False)

    print("\n=== frozen policies vs best fixed (positive = policy wins) ===", flush=True)
    piv = res.pivot_table(index=["fm", "horizon"], columns="policy", values="frozen_gain_pp").round(
        3
    )
    print(piv.to_string(), flush=True)
    n_win = int((res["frozen_gain_pp"] > 0).sum())
    print(
        f"\n[gate] {n_win}/{len(res)} frozen policy-horizon cells beat best-fixed; "
        f"median gap closed {res['frozen_gap_closed'].median():.1%}",
        flush=True,
    )
    print(f"[gate] wrote {OUT}/gate_policies.csv and gate_frontier.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
