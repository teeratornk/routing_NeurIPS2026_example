"""TS-LIMITS: how much of the routing gain is the seed?

The country-cluster intervals condition on ONE fitted gate and its frozen
threshold. They describe heterogeneity across the 19 evaluation countries, not
variability from retraining. A reviewer asked for that second source to be shown
rather than merely disclaimed, which is fair: the gate is a gradient-boosted
model with an internal early-stopping split, so the seed moves both the fit and
the held-out split it stops on.

Refit the whole deployable pipeline under several seeds -- fit on 2018-2023,
threshold frozen on 2024, applied to 2025 -- and report the spread of the 2025
gain over the country x horizon baseline. Everything else is held fixed, so the
spread is attributable to the seed alone.

This is a sensitivity, not an interval: five seeds bound the visible wobble, they
do not estimate a sampling distribution, and the two sources are not combined.

Out: reports/tslimits/seed_sensitivity.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_seed_sensitivity.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from load_forecast.eval.routing import ESTIMANDS, escalate_above, frozen_threshold

OUT = Path("reports/tslimits")
SEEDS = (0, 1, 2, 3, 4)
ESTIMAND = "pooled_mean"


def _mod(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_BS = _mod("tslimits_bootstrap")
_EM = _mod("tslimits_per_origin_structural")


def main() -> int:
    agg = ESTIMANDS[ESTIMAND]
    cols = list(_BS.ENRICHED)
    cat = _EM.categorical_mask(cols)
    rows: list[dict[str, Any]] = []

    for fm_label, (test_pat, dev_pat) in _BS.FM_PATTERNS.items():
        dev = _BS._paired("dev", dev_pat)
        test = _BS._paired("2025", test_pat)
        fit = dev[dev["test_year"].isin(_BS.FIT_YEARS)]
        val = dev[dev["test_year"].isin(_BS.VAL_YEARS)]

        for h in sorted(test["horizon"].unique()):
            f = fit[fit["horizon"] == h].reset_index(drop=True)
            v = val[val["horizon"] == h].reset_index(drop=True)
            t = test[test["horizon"] == h].reset_index(drop=True)
            st, fmv = t["ape_st"].to_numpy(), t["ape_fm"].to_numpy()

            # the baseline the gain is measured against does not depend on the
            # seed, so it is computed once
            sel = (
                v.groupby("country")[["ape_fm", "ape_st"]]
                .mean()
                .pipe(lambda x: x["ape_fm"] < x["ape_st"])
            )
            cc = agg(
                pd.Series(np.where(t["country"].map(sel).to_numpy(), fmv, st)), t["country"]
            )

            gains, fracs = [], []
            for seed in SEEDS:
                reg = HistGradientBoostingRegressor(
                    max_iter=300, random_state=seed, categorical_features=cat
                ).fit(f[cols], f["ape_st"] - f["ape_fm"])
                sv = np.asarray(reg.predict(v[cols]), dtype=float)
                stt = np.asarray(reg.predict(t[cols]), dtype=float)
                b = min(
                    (
                        agg(
                            pd.Series(
                                np.where(
                                    escalate_above(sv, frozen_threshold(sv, q)),
                                    v["ape_fm"].to_numpy(),
                                    v["ape_st"].to_numpy(),
                                )
                            ),
                            v["country"],
                        ),
                        q,
                    )
                    for q in _BS.BUDGETS
                )[1]
                mask = escalate_above(stt, frozen_threshold(sv, b))
                gains.append(cc - agg(pd.Series(np.where(mask, fmv, st)), t["country"]))
                fracs.append(float(mask.mean()))

            g = np.asarray(gains)
            rows.append(
                {
                    "fm": fm_label,
                    "horizon": int(h),
                    "n_seeds": len(SEEDS),
                    "gain_seed0_pp": round(g[0], 4),
                    "gain_mean_pp": round(float(g.mean()), 4),
                    "gain_min_pp": round(float(g.min()), 4),
                    "gain_max_pp": round(float(g.max()), 4),
                    "gain_sd_pp": round(float(g.std(ddof=1)), 4),
                    "gain_range_pp": round(float(g.max() - g.min()), 4),
                    "all_seeds_positive": bool((g > 0).all()),
                    "frac_min": round(min(fracs), 4),
                    "frac_max": round(max(fracs), 4),
                }
            )
            r = rows[-1]
            print(
                f"[seed] {fm_label:19s} h={int(h):<4} seed0 {r['gain_seed0_pp']:+.4f}  "
                f"mean {r['gain_mean_pp']:+.4f}  range [{r['gain_min_pp']:+.4f},"
                f"{r['gain_max_pp']:+.4f}] sd {r['gain_sd_pp']:.4f}  "
                f"{'all positive' if r['all_seeds_positive'] else 'SIGN FLIPS'}",
                flush=True,
            )

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "seed_sensitivity.csv", index=False)
    print("", flush=True)
    print(
        f"[seed] gain over the country baseline is positive under every seed in "
        f"{int(d.all_seeds_positive.sum())}/{len(d)} cells; "
        f"seed-to-seed sd {d.gain_sd_pp.min():.4f}-{d.gain_sd_pp.max():.4f} pp, "
        f"widest range {d.gain_range_pp.max():.4f} pp",
        flush=True,
    )
    print(f"[seed] wrote {OUT}/seed_sensitivity.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
