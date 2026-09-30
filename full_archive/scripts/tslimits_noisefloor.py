"""TS-LIMITS: how much of the per-request oracle gap is manufactured by noise?

The oracle takes a per-request minimum of two error series. A minimum over two
noisy quantities is biased downward even when neither model is ever genuinely
the better choice for identifiable reasons: whichever series happens to be low
on a given request gets picked, and the winner's curse does the rest. So a
positive oracle gap is not by itself evidence that anything is routable, and
"no policy reaches the oracle" is not by itself an interesting failure.

This calibrates the gap against nulls that keep both error distributions exactly
and destroy only the per-request pairing. Under each null the marginals are
untouched, so best-fixed is unchanged by construction and every difference in
the gap comes from the oracle term alone.

  independent  permute the FM errors within a country. Removes all pairing,
               including the part that comes from days being jointly hard.
  matched      permute within (country, decile of the structural error).
               Requests are only exchanged with requests of comparable
               difficulty, so the shared "hard day" coupling survives and only
               a MODEL-SPECIFIC per-request advantage is destroyed.

The independent null is the loose one and it overstates: real errors are
positively correlated (rho is reported), and positive correlation shrinks the
min-based gap, so independence sets a higher bar than the truth. The matched
null is the one that carries the argument, because the difficulty coupling it
preserves is precisely the part of the pairing that a router cannot exploit --
a request that is hard for both models is not a request where escalating helps.

Reading it: observed / null near or below 1 means the gap is what two
uncorrelated-in-the-tails error series of these shapes would produce anyway.

Out: reports/tslimits/noise_floor.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_noisefloor.py
"""

from __future__ import annotations

import glob
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from load_forecast.eval.routing import fixed_and_oracle, panel_median

OUT = Path("reports/tslimits")
FM_PATTERNS = {
    "Chronos-2-Uni-ZS": "reports/tslimits/fm/tsl_chronos_q_*.parquet",
    "TimesFM-2.5-Uni-ZS": "reports/tslimits/fm/tsl_timesfm_q_*.parquet",
}
R = 500
SEED = 0
# Which estimand this script reports. Stated here rather than inherited from a
# default, so a reader can see it without opening routing.py, and so the switch
# to the additive estimand is a one-line auditable change per script.
ESTIMAND = "panel_median"
N_STRATA = 10


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
    gap = float((j["y_true"] - j["actual"]).abs().max())
    assert gap == 0.0, f"the two sides disagree on the target by {gap}"
    return j.reset_index(drop=True)


def _strata(st: np.ndarray, cc: np.ndarray, n: int) -> np.ndarray:
    """Label each request by (country, within-country error decile)."""
    out = np.empty(st.size, dtype=np.int64)
    for k, c in enumerate(np.unique(cc)):
        m = cc == c
        r = pd.Series(st[m]).rank(method="first").to_numpy()
        q = np.minimum((r - 1) * n // m.sum(), n - 1).astype(np.int64)
        out[m] = k * n + q
    return out


def _permute_within(x: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle x inside each group, leaving the group's multiset untouched."""
    out = x.copy()
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    bounds = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1], True])
    for lo, hi in pairwise(bounds):
        idx = order[lo:hi]
        out[idx] = x[rng.permutation(idx)]
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []

    for fm_label, pattern in FM_PATTERNS.items():
        paired = _paired(pattern)
        print(f"\n[noise] {fm_label}", flush=True)

        for h in sorted(paired["horizon"].unique()):
            d = paired[paired["horizon"] == h].reset_index(drop=True)
            st = d["ape_st"].to_numpy()
            fm = d["ape_fm"].to_numpy()
            cc = d["country"].to_numpy()
            ref = fixed_and_oracle(d["ape_st"], d["ape_fm"], d["country"], ESTIMAND)
            observed = ref["oracle_gain_pp"]
            best_fixed = ref["best_fixed"]

            groups = {
                "independent": pd.factorize(cc)[0],
                "matched": _strata(st, cc, N_STRATA),
            }
            summary: dict[str, dict[str, float]] = {}
            for name, g in groups.items():
                draws = np.empty(R)
                for r in range(R):
                    fm_p = _permute_within(fm, g, rng)
                    orc = panel_median(pd.Series(np.minimum(st, fm_p)), d["country"])
                    # best_fixed is invariant under a within-country permutation;
                    # assert it rather than assume it, since a bug in the
                    # shuffler would otherwise be invisible in the ratio.
                    if r == 0:
                        bf = fixed_and_oracle(d["ape_st"], pd.Series(fm_p), d["country"], ESTIMAND)
                        assert abs(bf["best_fixed"] - best_fixed) < 1e-9, "null moved best-fixed"
                    draws[r] = best_fixed - orc
                summary[name] = {
                    "mean": float(draws.mean()),
                    "lo": float(np.percentile(draws, 2.5)),
                    "hi": float(np.percentile(draws, 97.5)),
                    # share of null draws at least as large as what we observed
                    "p_ge": float((draws >= observed).mean()),
                }

            rho = float(spearmanr(st, fm).statistic)
            rows.append(
                {
                    "fm": fm_label,
                    "horizon": int(h),
                    "n_requests": len(d),
                    "best_fixed": round(best_fixed, 4),
                    "observed_gap_pp": round(observed, 4),
                    "rho_st_fm": round(rho, 4),
                    **{
                        f"{k}_{stat}": round(v[stat], 4)
                        for k, v in summary.items()
                        for stat in ("mean", "lo", "hi", "p_ge")
                    },
                    "observed_over_independent": round(
                        observed / max(summary["independent"]["mean"], 1e-9), 4
                    ),
                    "observed_over_matched": round(
                        observed / max(summary["matched"]["mean"], 1e-9), 4
                    ),
                }
            )
            r0 = rows[-1]
            print(
                f"  h={h:<4} observed {observed:.3f} pp | "
                f"independent {summary['independent']['mean']:.3f} "
                f"[{summary['independent']['lo']:.3f},{summary['independent']['hi']:.3f}] "
                f"ratio {r0['observed_over_independent']:.2f} | "
                f"matched {summary['matched']['mean']:.3f} "
                f"[{summary['matched']['lo']:.3f},{summary['matched']['hi']:.3f}] "
                f"ratio {r0['observed_over_matched']:.2f} | rho {rho:+.3f}",
                flush=True,
            )

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "noise_floor.csv", index=False)
    n_above = int((res["observed_gap_pp"] > res["matched_hi"]).sum())
    print(
        f"\n[noise] observed gap exceeds the MATCHED null's 97.5th percentile in "
        f"{n_above}/{len(res)} cells; median ratio to matched "
        f"{res['observed_over_matched'].median():.2f}, to independent "
        f"{res['observed_over_independent'].median():.2f}",
        flush=True,
    )
    print(f"[noise] wrote {OUT}/noise_floor.csv  (R={R} permutations per cell)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
