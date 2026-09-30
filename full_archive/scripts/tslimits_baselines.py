"""TS-LIMITS: the baselines a routing claim has to beat, all frozen honestly.

Three reviewer items share one fit/validate/test skeleton, so they share one
script rather than three copies of it.

FROZEN STATIC CHOICE (item 4). `fixed_and_oracle`'s `best_fixed` is the better
of always-structural and always-FM computed ON the scored window, so on 2025 it
is a HINDSIGHT quantity. It is the right reference for decomposing the oracle
gap and the wrong one for any practical claim, because nobody could have
selected it before 2025. The deployable analogue picks per horizon on 2024 and
applies that choice unchanged. With only two candidates an argmin flips on
hundredths of a point, so the pick uses a parsimony margin: stay with the
structural model unless the FM wins on 2024 by more than MARGIN_PP. That idiom
is borrowed from exp_deployable_stack._select_lambda, which does the same for a
shrinkage constant.

FROZEN THRESHOLD (item 5) IS NOT HERE. This docstring used to describe that
analysis as though the script performed it; it never did, and the claim survived
until an audit compared every docstring against its own main(). The work now
lives where the escalation decision is actually made: `frozen_threshold` and
`escalate_above` in load_forecast.eval.routing, called from
tslimits_bootstrap.py and tslimits_country_static.py, which report the 2024
target fraction, the frozen number, and the realised 2025 fraction side by side.

FIXED CONVEX BLEND (item 8). Unpredictable variation can still be useful
variation: a blend exploits diversity without predicting the winner, which is
exactly what hard routing cannot do. Per horizon,

    y_blend = alpha_h * y_FM + (1 - alpha_h) * y_struct

with alpha_h fitted on 2018-2023 by the no-intercept least-squares slope that
minimises squared error, clipped to [0, 1], and validated on 2024 before 2025 is
touched. The estimator is the one already used four times in this repo
(d125d_entsoe_blend._fit_gamma and relatives); it is written once here rather
than copied a fifth time.

The blend always pays the FM's cost, so it does not answer "when to pay". It
answers whether paying buys anything at all when the router cannot say where.

Out: reports/tslimits/baselines.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_baselines.py
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from load_forecast.eval.routing import ESTIMANDS, fixed_and_oracle, weighted_mape

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
# Stay with the cheap model unless the expensive one wins on validation by more
# than this. Two candidates and one noisy year would otherwise let the choice
# flip on a hundredth of a point.
MARGIN_PP = 0.05
ESTIMAND = "pooled_mean"


def _load_fm(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    fm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fm = fm.rename(columns={"cc": "country", "origin_ts": "anchor_t"})
    fm["anchor_t"] = pd.to_datetime(fm["anchor_t"])
    fm["ape_fm"] = 100.0 * (fm["q50"] - fm["actual"]).abs() / fm["actual"].abs()
    return fm[["country", "horizon", "anchor_t", "actual", "q50", "ape_fm"]]


def _paired(window: str, pattern: str) -> pd.DataFrame:
    st = pd.read_parquet(OUT / f"per_origin_structural_{window}.parquet")
    st = st.rename(columns={"ape": "ape_st"})
    st["anchor_t"] = pd.to_datetime(st["anchor_t"])
    j = st.merge(
        _load_fm(pattern), on=["country", "horizon", "anchor_t"], how="inner", validate="one_to_one"
    )
    assert len(j) == len(st), f"{window}: joined {len(j)} of {len(st)}"
    assert float((j["y_true"] - j["actual"]).abs().max()) == 0.0, f"{window}: targets disagree"
    return j.reset_index(drop=True)


def fit_alpha(y_true: np.ndarray, y_cheap: np.ndarray, y_exp: np.ndarray) -> float:
    """Convex weight on the expensive forecast, least squares, clipped to [0,1].

    Minimising ||y - (c + a*(e - c))||^2 over a gives the no-intercept slope of
    the residual (y - c) on the gap (e - c). Clipping keeps it a genuine convex
    combination; an unclipped weight can extrapolate beyond both forecasts.
    """
    gap = y_exp - y_cheap
    v = float((gap**2).mean())
    if v < 1e-12:
        return 0.0
    a = float(((y_true - y_cheap) * gap).mean() / v)
    return float(min(1.0, max(0.0, a)))


def _ape(y_true: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    return 100.0 * np.abs(y_hat - y_true) / np.abs(y_true)


def main() -> int:
    agg = ESTIMANDS[ESTIMAND]
    rows: list[dict[str, Any]] = []

    for fm_label, (test_pat, dev_pat) in FM_PATTERNS.items():
        dev = _paired("dev", dev_pat)
        test = _paired("2025", test_pat)
        fit = dev[dev["test_year"].isin(FIT_YEARS)]
        val = dev[dev["test_year"].isin(VAL_YEARS)]
        print(f"\n[base] {fm_label}", flush=True)

        for h in sorted(test["horizon"].unique()):
            f_h, v_h, t_h = (
                d[d["horizon"] == h].reset_index(drop=True) for d in (fit, val, test)
            )
            ref = fixed_and_oracle(t_h["ape_st"], t_h["ape_fm"], t_h["country"], ESTIMAND)

            # --- item 4: frozen static choice, selected on 2024 only ---
            v_st, v_fm = agg(v_h["ape_st"], v_h["country"]), agg(v_h["ape_fm"], v_h["country"])
            pick = "expensive" if (v_st - v_fm) > MARGIN_PP else "cheap"
            frozen_static = agg(
                t_h["ape_fm"] if pick == "expensive" else t_h["ape_st"], t_h["country"]
            )

            # --- item 8: fixed convex blend, alpha on 2018-2023, frozen ---
            alpha = fit_alpha(
                f_h["y_true"].to_numpy(), f_h["y_pred"].to_numpy(), f_h["q50"].to_numpy()
            )
            def blended(d: pd.DataFrame, a: float = alpha) -> np.ndarray:
                return _ape(
                    d["y_true"].to_numpy(),
                    (1.0 - a) * d["y_pred"].to_numpy() + a * d["q50"].to_numpy(),
                )
            blend_val = agg(pd.Series(blended(v_h)), v_h["country"])
            blend_test = agg(pd.Series(blended(t_h)), t_h["country"])
            # Fitting alpha is not a decision. The deployable question is
            # whether to ship the blend at all, and that is decided on 2024
            # against the same frozen static choice, before 2025 is touched.
            # Reporting the 2025 gain of a blend that lost on validation would
            # be selection on the test window by another name.
            v_static = agg(
                v_h["ape_fm"] if pick == "expensive" else v_h["ape_st"], v_h["country"]
            )
            deploy_blend = bool(v_static - blend_val > MARGIN_PP)
            # alpha=0 and alpha=1 must reproduce the two fixed policies exactly
            assert np.allclose(blended(t_h, 0.0), t_h["ape_st"].to_numpy()), "alpha=0 broken"
            assert np.allclose(blended(t_h, 1.0), t_h["ape_fm"].to_numpy()), "alpha=1 broken"

            rows.append(
                {
                    "fm": fm_label,
                    "horizon": int(h),
                    "estimand": ESTIMAND,
                    "always_cheap": round(ref["always_cheap"], 4),
                    "always_expensive": round(ref["always_expensive"], 4),
                    "hindsight_best_fixed": round(ref["best_fixed"], 4),
                    "frozen_static_pick": pick,
                    "frozen_static": round(frozen_static, 4),
                    # what the hindsight baseline flatters by, i.e. the cost of
                    # having to choose before seeing the window
                    "hindsight_advantage_pp": round(frozen_static - ref["best_fixed"], 4),
                    "blend_alpha": round(alpha, 4),
                    "blend_val": round(blend_val, 4),
                    "blend_val_gain_pp": round(v_static - blend_val, 4),
                    "deploy_blend_on_2024": deploy_blend,
                    "blend_test": round(blend_test, 4),
                    "blend_vs_frozen_static_pp": round(frozen_static - blend_test, 4),
                    "blend_vs_hindsight_pp": round(ref["best_fixed"] - blend_test, 4),
                    "oracle_gain_pp": round(ref["oracle_gain_pp"], 4),
                    "wmape_cheap": round(
                        weighted_mape(t_h["ape_st"], t_h["country"], t_h["actual"]), 4
                    ),
                    "wmape_expensive": round(
                        weighted_mape(t_h["ape_fm"], t_h["country"], t_h["actual"]), 4
                    ),
                    "wmape_blend": round(
                        weighted_mape(pd.Series(blended(t_h)), t_h["country"], t_h["actual"]), 4
                    ),
                }
            )
            r = rows[-1]
            print(
                f"  h={h:<4} hindsight {r['hindsight_best_fixed']:.3f} | frozen-static "
                f"{r['frozen_static']:.3f} ({pick}) | blend a={alpha:.3f} {r['blend_test']:.3f} "
                f"-> {r['blend_vs_frozen_static_pp']:+.3f} vs frozen static "
                f"(2024 says {'ship' if deploy_blend else 'do not ship'})",
                flush=True,
            )

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "baselines.csv", index=False)
    shipped = d[d.deploy_blend_on_2024]
    print(
        f"\n[base] 2024 said ship the blend in {len(shipped)}/{len(d)} cells; where it did, the "
        f"2025 gain is {shipped.blend_vs_frozen_static_pp.median():+.3f} pp median "
        f"(min {shipped.blend_vs_frozen_static_pp.min():+.3f})"
        if len(shipped)
        else "\n[base] 2024 never said ship the blend",
        flush=True,
    )
    print(
        f"[base] blend beats the frozen static choice in "
        f"{int((d.blend_vs_frozen_static_pp > 0).sum())}/{len(d)} cells "
        f"(median {d.blend_vs_frozen_static_pp.median():+.3f} pp); "
        f"hindsight flatters the fixed baseline by a median of "
        f"{d.hindsight_advantage_pp.median():+.3f} pp",
        flush=True,
    )
    print(f"[base] wrote {OUT}/baselines.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
