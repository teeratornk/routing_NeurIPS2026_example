"""TS-LIMITS: is the estimand disagreement real, and what part of it is scoring?

The paper's central claim is that whether request-time routing pays depends on
the loss. The evidence offered for it was that the pooled-mean interval excludes
zero in 8/8 cells while the panel-median interval covers zero in 8/8. Comparing
two intervals for overlap is NOT a test of their difference, so this script
bootstraps the difference itself on shared country draws.

It also separates two effects the earlier framing ran together. The escalation
budget is selected on 2024 UNDER THE ESTIMAND, so the two runs do not evaluate
the same policy: the budgets differ in 7 of 8 cells, and at h=168 the
median-selected budget is 0.00, i.e. "route nothing", which forces a gain of
exactly zero by construction rather than by measurement. Calling that "the same
policy scored two ways" was wrong. The two effects are:

    scoring   policy_pool scored under pooled mean vs under panel median.
              One policy, one set of requests, two losses. This is the pure
              loss-dependence claim.
    total     policy_pool under pooled mean vs policy_med under panel median,
              each estimand using the budget its own validation chose. This is
              what a practitioner following each convention would deploy.

Both differences are computed inside every replicate on the SAME resampled
countries, so the interval describes the spread of the difference rather than
the difference of two spreads.

Out: reports/tslimits/paired_estimand.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_paired_estimand.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from load_forecast.eval.routing import escalate_above, frozen_threshold

OUT = Path("reports/tslimits")
B = 2000
SEED = 0
# one grid definition, in tslimits_bootstrap (TSLIMITS_BUDGET_STEP for the 0.01 sensitivity)
BUDGETS = None  # bound after _BS is loaded below


def _mod(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Reuse the bootstrap's own loaders and gate rather than re-inlining them: a
# second copy of the fold scheme is how the day-of-year bug reached two files.
_BS = _mod("tslimits_bootstrap")
BUDGETS = _BS.BUDGETS


def _agg(served: dict[str, np.ndarray], country: pd.Series) -> tuple[Any, Any, list[str]]:
    """Country matrices for both estimands, from one set of served vectors.

    pooled mean holds sums and counts, because country sizes are 363-365 and an
    unweighted mean of per-country means is therefore not the pooled mean.
    panel median holds within-country medians, one number per country.
    """
    names = list(served)
    frame = pd.DataFrame({"c": country.to_numpy(), **served})
    g = frame.groupby("c")
    sums = g[names].sum().to_numpy()
    cnt = g[names[0]].size().to_numpy().astype(float)
    meds = g[names].median().to_numpy()

    def pooled(rows: np.ndarray) -> np.ndarray:
        return sums[rows].sum(axis=0) / cnt[rows].sum()

    def median(rows: np.ndarray) -> np.ndarray:
        return np.median(meds[rows], axis=0)

    return pooled, median, names


def _ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main() -> int:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    draws_by_cell: list[dict[str, np.ndarray]] = []
    cols = list(_BS.ENRICHED)

    for fm_label, (test_pat, dev_pat) in _BS.FM_PATTERNS.items():
        dev = _BS._paired("dev", dev_pat)
        test = _BS._paired("2025", test_pat)
        fit = dev[dev["test_year"].isin(_BS.FIT_YEARS)]
        val = dev[dev["test_year"].isin(_BS.VAL_YEARS)]

        for h in sorted(test["horizon"].unique()):
            f_h = fit[fit["horizon"] == h].reset_index(drop=True)
            v_h = val[val["horizon"] == h].reset_index(drop=True)
            t_h = test[test["horizon"] == h].reset_index(drop=True)

            st = t_h["ape_st"].to_numpy()
            fm = t_h["ape_fm"].to_numpy()
            reg = _BS._reg(cols).fit(f_h[cols], f_h["ape_st"] - f_h["ape_fm"])
            s_val = np.asarray(reg.predict(v_h[cols]), dtype=float)
            s_test = np.asarray(reg.predict(t_h[cols]), dtype=float)

            # Budget selected on 2024 under each estimand separately. This is the
            # step the paper's "same policy" wording elided.
            def sel(agg_name: str, v_h: pd.DataFrame = v_h, s_val: np.ndarray = s_val) -> float:
                f = _BS.ESTIMANDS[agg_name]
                best = min(
                    (
                        f(
                            pd.Series(
                                np.where(
                                    escalate_above(s_val, frozen_threshold(s_val, b)),
                                    v_h["ape_fm"].to_numpy(),
                                    v_h["ape_st"].to_numpy(),
                                )
                            ),
                            v_h["country"],
                        ),
                        b,
                    )
                    for b in BUDGETS
                )
                return best[1]

            b_pool = sel("pooled_mean")
            b_med = sel("panel_median")

            # The country x horizon fixed rule, chosen on 2024. It is ONE policy,
            # scored under both estimands, so the baseline is held fixed while
            # only the aggregation changes -- which is what Table 1 compares
            # against and what the figure has to match.
            sel = (
                v_h.groupby("country")[["ape_fm", "ape_st"]]
                .mean()
                .pipe(lambda x: x["ape_fm"] < x["ape_st"])
            )
            use_fm = t_h["country"].map(sel).to_numpy()
            assert not pd.isna(use_fm).any(), "a 2025 country is missing from 2024"

            # The asymmetry that motivated this arm. Above, the baseline is
            # picked by a per-country MEAN whatever the scoring convention,
            # while the router's budget is re-selected per estimand. Comparing
            # the two conventions that way gives the router an estimand-matched
            # selection and denies the baseline one, so part of any "deployment
            # difference" is the protocol rather than the estimand. The
            # median-convention analogue of a per-country mean is a per-country
            # median, which is what the panel median aggregates.
            sel_med = (
                v_h.groupby("country")[["ape_fm", "ape_st"]]
                .median()
                .pipe(lambda x: x["ape_fm"] < x["ape_st"])
            )
            use_fm_med = t_h["country"].map(sel_med).to_numpy()
            assert not pd.isna(use_fm_med).any(), "a 2025 country is missing from 2024"

            served = {
                "st": st,
                "fm": fm,
                "cc": np.where(use_fm, fm, st),
                "cc_med": np.where(use_fm_med, fm, st),
                "pol_pool": np.where(
                    escalate_above(s_test, frozen_threshold(s_val, b_pool)), fm, st
                ),
                "pol_med": np.where(
                    escalate_above(s_test, frozen_threshold(s_val, b_med)), fm, st
                ),
            }
            pooled, median, names = _agg(served, t_h["country"])
            all_rows = np.arange(t_h["country"].nunique())
            p0, m0 = pooled(all_rows), median(all_rows)

            for agg_name, vec in (("pooled_mean", p0), ("panel_median", m0)):
                for k in names:
                    direct = _BS.ESTIMANDS[agg_name](pd.Series(served[k]), t_h["country"])
                    assert abs(direct - vec[names.index(k)]) < 1e-9, f"{agg_name}/{k} mismatch"

            i_cc = names.index("cc")
            w_p = w_m = i_cc  # both estimands score the SAME baseline policy
            i_ccm = names.index("cc_med")  # the estimand-matched baseline
            i_pp, i_pm = names.index("pol_pool"), names.index("pol_med")

            d_score = np.empty(B)
            d_total = np.empty(B)
            g_pool = np.empty(B)
            g_med_same = np.empty(B)
            g_med_own = np.empty(B)
            g_med_sym = np.empty(B)
            d_total_sym = np.empty(B)
            for b in range(B):
                idx = rng.integers(0, all_rows.size, all_rows.size)
                qp, qm = pooled(idx), median(idx)
                g_pool[b] = qp[w_p] - qp[i_pp]
                g_med_same[b] = qm[w_m] - qm[i_pp]
                g_med_own[b] = qm[w_m] - qm[i_pm]
                # Both sides estimand-matched: median-selected baseline against
                # the median-selected router budget.
                g_med_sym[b] = qm[i_ccm] - qm[i_pm]
                d_score[b] = g_pool[b] - g_med_same[b]
                d_total[b] = g_pool[b] - g_med_own[b]
                d_total_sym[b] = g_pool[b] - g_med_sym[b]

            # Level CIs, not only the CI on the difference. Figure 1 must show
            # both estimands scoring the IDENTICAL frozen policy, which needs an
            # interval on each level; the draws already exist here and were
            # being discarded after the subtraction.
            lo_gp, hi_gp = _ci(g_pool)
            lo_gm, hi_gm = _ci(g_med_same)
            lo_go, hi_go = _ci(g_med_own)
            lo_gs, hi_gs = _ci(g_med_sym)

            pt_pool = p0[w_p] - p0[i_pp]
            pt_med_same = m0[w_m] - m0[i_pp]
            pt_med_own = m0[w_m] - m0[i_pm]
            pt_med_sym = m0[i_ccm] - m0[i_pm]
            lo_s, hi_s = _ci(d_score)
            lo_t, hi_t = _ci(d_total)
            lo_ts, hi_ts = _ci(d_total_sym)
            draws_by_cell.append(
                {
                    "d_score": d_score.copy(),
                    "d_total": d_total.copy(),
                    "d_total_sym": d_total_sym.copy(),
                }
            )
            rows.append(
                {
                    "fm": fm_label,
                    "horizon": int(h),
                    "budget_pooled": b_pool,
                    "budget_median": b_med,
                    "gain_pooled_pp": round(pt_pool, 4),
                    "gain_pooled_lo": round(lo_gp, 4),
                    "gain_pooled_hi": round(hi_gp, 4),
                    "gain_pooled_sig": bool(lo_gp > 0),
                    "gain_median_samepolicy_pp": round(pt_med_same, 4),
                    "gain_median_samepolicy_lo": round(lo_gm, 4),
                    "gain_median_samepolicy_hi": round(hi_gm, 4),
                    "gain_median_samepolicy_sig": bool(lo_gm > 0),
                    "gain_median_ownpolicy_lo": round(lo_go, 4),
                    "gain_median_ownpolicy_hi": round(hi_go, 4),
                    "gain_median_ownpolicy_sig": bool(lo_go > 0),
                    "gain_median_ownpolicy_pp": round(pt_med_own, 4),
                    "diff_scoring_pp": round(pt_pool - pt_med_same, 4),
                    "diff_scoring_lo": round(lo_s, 4),
                    "diff_scoring_hi": round(hi_s, 4),
                    "diff_scoring_sig": bool(lo_s > 0),
                    "diff_total_pp": round(pt_pool - pt_med_own, 4),
                    "diff_total_lo": round(lo_t, 4),
                    "diff_total_hi": round(hi_t, 4),
                    "diff_total_sig": bool(lo_t > 0),
                    # Both conventions estimand-matched on BOTH sides.
                    "gain_median_symmetric_pp": round(pt_med_sym, 4),
                    "gain_median_symmetric_lo": round(lo_gs, 4),
                    "gain_median_symmetric_hi": round(hi_gs, 4),
                    "gain_median_symmetric_sig": bool(lo_gs > 0),
                    "diff_total_symmetric_pp": round(pt_pool - pt_med_sym, 4),
                    "diff_total_symmetric_lo": round(lo_ts, 4),
                    "diff_total_symmetric_hi": round(hi_ts, 4),
                    "diff_total_symmetric_sig": bool(lo_ts > 0),
                }
            )
            print(
                f"[pair] {fm_label:19s} h={h:<4} b={b_pool:.2f}/{b_med:.2f}  "
                f"pooled {pt_pool:+.3f}  median(same pol) {pt_med_same:+.3f}  "
                f"median(own pol) {pt_med_own:+.3f} | "
                f"d_score {pt_pool - pt_med_same:+.3f} [{lo_s:+.3f},{hi_s:+.3f}]"
                f"{'*' if lo_s > 0 else ' '}  "
                f"d_total {pt_pool - pt_med_own:+.3f} [{lo_t:+.3f},{hi_t:+.3f}]"
                f"{'*' if lo_t > 0 else ' '}  "
                f"d_total_sym {pt_pool - pt_med_sym:+.3f} [{lo_ts:+.3f},{hi_ts:+.3f}]"
                f"{'*' if lo_ts > 0 else ' '}",
                flush=True,
            )

    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"paired_estimand{_BS.GRID_SFX}.csv", index=False)

    # ONE aggregate test of the paper-level claim. Per-cell intervals answer
    # "does this cell differ", which is not the claim; the claim is about the
    # paper's eight cells taken together. Because every cell was resampled on the
    # SAME country draws, the eight per-draw differences can be aggregated inside
    # each replicate, which preserves the cross-cell dependence that a
    # meta-analysis over independent cells would throw away. The summary is
    # pre-specified as the median across cells, matching how every other
    # cross-cell number in the paper is reported.
    agg_rows = []
    for tag, col in (
        ("d_score", "diff_scoring_pp"),
        ("d_total", "diff_total_pp"),
        ("d_total_sym", "diff_total_symmetric_pp"),
    ):
        mat = np.vstack([c[tag] for c in draws_by_cell])  # 8 cells x B draws
        assert mat.shape == (8, B), mat.shape
        per_draw = np.median(mat, axis=0)
        lo, hi = _ci(per_draw)
        point = float(np.median([r[col] for r in rows]))
        agg_rows.append(
            {
                "contrast": tag,
                "column": col,
                "median_across_cells_pp": round(point, 4),
                "lo": round(lo, 4),
                "hi": round(hi, 4),
                "excludes_zero": bool(lo > 0),
                "n_cells": mat.shape[0],
                "n_draws": B,
            }
        )
        print(
            f"[pair] AGGREGATE {tag}: median across 8 cells "
            f"{point:+.4f} pp [{lo:+.4f}, {hi:+.4f}]"
            f"{'  excludes zero' if lo > 0 else '  includes zero'}",
            flush=True,
        )
    pd.DataFrame(agg_rows).to_csv(OUT / f"paired_aggregate{_BS.GRID_SFX}.csv", index=False)
    print("", flush=True)
    print(
        f"[pair] budgets differ in {int((d.budget_pooled != d.budget_median).sum())}/{len(d)} "
        f"cells; median-selected budget is 0.00 (route nothing) in "
        f"{int((d.budget_median == 0.0).sum())}/{len(d)}",
        flush=True,
    )
    for tag in ("scoring", "total"):
        print(
            f"[pair] {tag:8s}: difference significant in "
            f"{int(d[f'diff_{tag}_sig'].sum())}/{len(d)} cells, "
            f"median {d[f'diff_{tag}_pp'].median():+.4f} pp",
            flush=True,
        )
    print(f"[pair] wrote {OUT}/paired_estimand.csv (B={B}, shared country draws)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
