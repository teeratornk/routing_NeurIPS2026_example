"""TS-LIMITS: does the story hold on years other than the one that suggested it?

The size of the 2025 oracle gap was inspected before this study was conceived.
The routing protocol that followed is clean, but the headline is still a
post-hoc hypothesis tested on the window that generated it. That is a real
weakness and the fix is cheap, because the 2018-2024 per-origin records already
exist for both foundation models at all four horizons with row counts matching
the structural side exactly. No new inference is required.

For each test year t, everything is refitted with the same shape the 2025
analysis uses:

    fit     <= t-2      the margin model
    select  t-1         the escalation budget, and nothing else
    test    t           scored once

so t = 2020..2025 is available (2018 is the first year on disk, so t=2019 would
have an empty fit window). t=2020 has a single fit year AND is the COVID
structural break; it is reported and flagged rather than dropped, because
dropping the year that looks hardest is how a replication becomes decoration.

Reported per (year, model, horizon): the oracle gap, its closed-form
decomposition, the feature-conditional benchmark under embargoed blocked folds,
and the frozen routing gain. If the identity accounts for the gap across years
rather than only on 2025, the methodological claim stops resting on the window
that produced it.

Out: reports/tslimits/by_year.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_by_year.py
"""

from __future__ import annotations

import glob
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

from load_forecast.eval.routing import (
    ESTIMANDS,
    escalate_above,
    fixed_and_oracle,
    frozen_threshold,
)

OUT = Path("reports/tslimits")
FM_PATTERNS = {
    "Chronos-2-Uni-ZS": ("tsl_chronos_q", "tsldev_chronos_q"),
    "TimesFM-2.5-Uni-ZS": ("tsl_timesfm_q", "tsldev_timesfm_q"),
}
TEST_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
BUDGETS = tuple(round(0.05 * i, 2) for i in range(21))
N_FOLDS = 5
SEED = 0
ESTIMAND = "pooled_mean"


def _mod(name: str) -> Any:
    p = Path("scripts") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_EM = _mod("tslimits_per_origin_structural")
_BLK = _mod("tslimits_ceiling_blocked")
ENRICHED: tuple[str, ...] = _EM.GATE_FEATURES


def _load_fm(prefix: str) -> pd.DataFrame:
    files = sorted(glob.glob(f"reports/tslimits/fm/{prefix}_*.parquet"))
    if not files:
        raise FileNotFoundError(prefix)
    fm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fm = fm.rename(columns={"cc": "country", "origin_ts": "anchor_t"})
    fm["anchor_t"] = pd.to_datetime(fm["anchor_t"])
    fm["ape_fm"] = 100.0 * (fm["q50"] - fm["actual"]).abs() / fm["actual"].abs()
    return fm[["country", "horizon", "anchor_t", "actual", "ape_fm"]]


def _all_years(test_prefix: str, dev_prefix: str) -> pd.DataFrame:
    """Every year 2018-2025 in one frame, joined one-to-one per window."""
    out = []
    for window, prefix in (("dev", dev_prefix), ("2025", test_prefix)):
        st = pd.read_parquet(OUT / f"per_origin_structural_{window}.parquet")
        st = st.rename(columns={"ape": "ape_st"})
        st["anchor_t"] = pd.to_datetime(st["anchor_t"])
        j = st.merge(
            _load_fm(prefix),
            on=["country", "horizon", "anchor_t"],
            how="inner",
            validate="one_to_one",
        )
        assert len(j) == len(st), f"{window}: joined {len(j)} of {len(st)}"
        assert float((j["y_true"] - j["actual"]).abs().max()) == 0.0
        out.append(j)
    return pd.concat(out, ignore_index=True)


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


def _reg(cols: list[str] | None = None) -> HistGradientBoostingRegressor:
    """Gate learner. Pass the feature columns so country identity is usable.

    `early_stopping` is left at sklearn's 'auto', which switches ON above 10,000
    samples; the fit slices here are ~41.6k rows, so the gate does hold out an
    internal 10% validation split. That is stated rather than changed, since
    every reported number was produced under it.
    """
    cat = _EM.categorical_mask(cols) if cols is not None else None
    return HistGradientBoostingRegressor(
        max_iter=300, random_state=SEED, categorical_features=cat
    )


def main() -> int:
    agg = ESTIMANDS[ESTIMAND]
    rows: list[dict[str, Any]] = []

    for fm_label, (test_prefix, dev_prefix) in FM_PATTERNS.items():
        allp = _all_years(test_prefix, dev_prefix)
        print(f"\n[year] {fm_label}  years {sorted(allp.test_year.unique())}", flush=True)

        for t in TEST_YEARS:
            for h in sorted(allp["horizon"].unique()):
                d = allp[allp["horizon"] == h]
                f_h = d[d["test_year"] <= t - 2].reset_index(drop=True)
                v_h = d[d["test_year"] == t - 1].reset_index(drop=True)
                t_h = d[d["test_year"] == t].reset_index(drop=True)
                if len(f_h) == 0 or len(v_h) == 0 or len(t_h) == 0:
                    continue

                ref = fixed_and_oracle(t_h["ape_st"], t_h["ape_fm"], t_h["country"], ESTIMAND)

                # The replication used to clear only the horizon-only fixed
                # choice. A reviewer pointed out that the gate carries country
                # identity, so the country x horizon rule is the baseline the
                # claim has to survive -- in every year, not just 2025. Frozen
                # the same way the policy is: chosen on t-1, applied to t.
                sel = (
                    v_h.groupby("country")[["ape_fm", "ape_st"]]
                    .mean()
                    .pipe(lambda x: x["ape_fm"] < x["ape_st"])
                )
                use_fm = t_h["country"].map(sel)
                assert not use_fm.isna().any(), f"t={t}: a test country is missing from t-1"
                cc_static = agg(
                    pd.Series(
                        np.where(use_fm.to_numpy(), t_h["ape_fm"], t_h["ape_st"])
                    ),
                    t_h["country"],
                )

                def serve(df: pd.DataFrame, mask: np.ndarray) -> float:
                    v = np.where(mask, df["ape_fm"].to_numpy(), df["ape_st"].to_numpy())
                    return agg(pd.Series(v), df["country"])

                # deployable: fit <= t-2, budget on t-1, applied to t
                cols = list(ENRICHED)
                reg = _reg(cols).fit(f_h[cols], f_h["ape_st"] - f_h["ape_fm"])
                s_val = np.asarray(reg.predict(v_h[cols]), dtype=float)
                s_test = np.asarray(reg.predict(t_h[cols]), dtype=float)
                # frozen NUMBER, not a fraction: the same online-routing
                # semantics as the 2025 pipeline, so this column and Table 1
                # report the same quantity for t=2025
                budget = min(
                    (serve(v_h, escalate_above(s_val, frozen_threshold(s_val, b))), b)
                    for b in BUDGETS
                )[1]
                tau = frozen_threshold(s_val, budget)
                mask_t = escalate_above(s_test, tau)
                served_t = serve(t_h, mask_t)
                frozen_gain = ref["best_fixed"] - served_t
                frozen_gain_over_cc = cc_static - served_t

                # feature-conditional benchmark on year t, embargoed blocked folds
                cv = _BLK._splits(t_h["anchor_t"], t_h["target_t"], N_FOLDS)
                oof = np.asarray(
                    cross_val_predict(
                        _reg(cols), t_h[cols], (t_h["ape_st"] - t_h["ape_fm"]).to_numpy(),
                        cv=cv, n_jobs=1,
                    ),
                    dtype=float,
                )
                bench = ref["best_fixed"] - min(serve(t_h, _top_k(oof, b)) for b in BUDGETS)

                rows.append(
                    {
                        "fm": fm_label,
                        "test_year": t,
                        "horizon": int(h),
                        "n_fit_years": int(f_h["test_year"].nunique()),
                        "n_test": len(t_h),
                        "oracle_gain_pp": round(ref["oracle_gain_pp"], 4),
                        "mean_abs_margin": round(ref["mean_abs_margin"], 4),
                        "abs_mean_margin": round(ref["abs_mean_margin"], 4),
                        # the identity must reproduce the gap in EVERY year, not
                        # only in the one that motivated the study
                        "identity_err": round(
                            abs(ref["oracle_gain_identity"] - ref["oracle_gain_pp"]), 12
                        ),
                        "benchmark_pp": round(bench, 4),
                        "benchmark_frac_of_oracle": round(
                            bench / max(ref["oracle_gain_pp"], 1e-9), 4
                        ),
                        "frozen_gain_pp": round(frozen_gain, 4),
                        "cc_static": round(cc_static, 4),
                        "frozen_gain_over_cc_pp": round(frozen_gain_over_cc, 4),
                        "frozen_budget": budget,
                        "frozen_threshold": round(float(tau), 6),
                        "frozen_realized_frac": round(float(mask_t.mean()), 4),
                    }
                )
            done = [r for r in rows if r["fm"] == fm_label and r["test_year"] == t]
            if done:
                print(
                    f"  t={t} (fit {done[0]['n_fit_years']}y): "
                    + "  ".join(
                        f"h={r['horizon']}: gap {r['oracle_gain_pp']:.2f} "
                        f"bench {r['benchmark_pp']:.3f} frozen {r['frozen_gain_pp']:+.3f}"
                        for r in done
                    ),
                    flush=True,
                )

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "by_year.csv", index=False)
    print(
        f"\n[year] identity reproduces the oracle gap in {int((d.identity_err < 1e-9).sum())}"
        f"/{len(d)} year-model-horizon cells (max error {d.identity_err.max():.2e})",
        flush=True,
    )
    print(
        f"[year] benchmark as a fraction of the oracle gap: median "
        f"{d.benchmark_frac_of_oracle.median():.3f}, max {d.benchmark_frac_of_oracle.max():.3f}",
        flush=True,
    )
    print(
        f"[year] over the country x horizon baseline: median "
        f"{d.frozen_gain_over_cc_pp.median():+.4f} pp, "
        f"positive in {int((d.frozen_gain_over_cc_pp > 0).sum())}/{len(d)}, "
        f"above 0.10 pp in {int((d.frozen_gain_over_cc_pp > 0.10).sum())}/{len(d)}",
        flush=True,
    )
    print(
        f"[year] frozen routing gain: median {d.frozen_gain_pp.median():+.4f} pp, "
        f"positive in {int((d.frozen_gain_pp > 0).sum())}/{len(d)}, "
        f"above 0.10 pp in {int((d.frozen_gain_pp > 0.10).sum())}/{len(d)}",
        flush=True,
    )
    print(f"[year] wrote {OUT}/by_year.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
