"""Threshold-selection stability: how much the 2024 selection moves the budget.

The paper's intervals condition on the fitted gate and the selected threshold.
This script holds the gate fixed (the deployed fit on 2018-2023, seed 0) and
re-selects the escalation budget on perturbed 2024 validation sets, under
each metric:
  loco      leave one country out of 2024 (19 selections)
  cboot     country bootstrap of the 2024 countries (200 draws)
  quarter   each contiguous quarter of 2024 on its own (4 selections)
The gate is never refit; only the number the operator freezes moves.

Out:  reports/tslimits/threshold_stability.csv, tab_thrstab.tex
Run:  PYTHONPATH=src .venv/bin/python scripts/tslimits_threshold_stability.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from load_forecast.eval.routing import (
    CLUSTER_ESTIMANDS,
    ESTIMANDS,
    escalate_above,
    frozen_threshold,
)

OUT = Path("reports/tslimits")
SEED, N_CBOOT = 0, 200
SHORT = {"Chronos-2-Uni-ZS": "Chronos-2", "TimesFM-2.5-Uni-ZS": "TimesFM-2.5"}


def _mod(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_BS = _mod("tslimits_bootstrap")


def _select(sv: np.ndarray, v: pd.DataFrame, agg, keep: np.ndarray) -> float:
    """The deployed selection rule on the validation rows `keep`."""
    vv = v[keep]
    svv = sv[keep]
    return min(
        (
            agg(
                pd.Series(
                    np.where(
                        escalate_above(svv, frozen_threshold(svv, q)),
                        vv["ape_fm"].to_numpy(),
                        vv["ape_st"].to_numpy(),
                    )
                ),
                vv["country"],
            ),
            q,
        )
        for q in _BS.BUDGETS
    )[1]


def _select_draw(sv: np.ndarray, v: pd.DataFrame, cagg, draw: np.ndarray) -> float:
    """The deployed selection rule on a country-cluster draw with multiplicity.

    The threshold is a quantile of the resampled score vector, as the deployed
    rule takes it over the observed panel; the served losses are then
    aggregated over the drawn labels by `cagg`.
    """
    idx = np.concatenate([np.flatnonzero((v["country"] == c).to_numpy()) for c in draw])
    svd = sv[idx]
    return min(
        (
            cagg(
                pd.Series(
                    np.where(
                        escalate_above(sv, frozen_threshold(svd, q)),
                        v["ape_fm"].to_numpy(),
                        v["ape_st"].to_numpy(),
                    )
                ),
                v["country"],
                draw,
            ),
            q,
        )
        for q in _BS.BUDGETS
    )[1]


def main() -> int:
    cols = list(_BS.ENRICHED)
    rows = []
    for fm_label, (_test_pat, dev_pat) in _BS.FM_PATTERNS.items():
        dev = _BS._paired("dev", dev_pat)
        fit = dev[dev["test_year"].isin(_BS.FIT_YEARS)]
        val = dev[dev["test_year"].isin(_BS.VAL_YEARS)]
        for h in sorted(val["horizon"].unique()):
            f = fit[fit["horizon"] == h].reset_index(drop=True)
            v = val[val["horizon"] == h].reset_index(drop=True)
            reg = _BS._reg(cols).fit(f[cols], f["ape_st"] - f["ape_fm"])
            sv = np.asarray(reg.predict(v[cols]), dtype=float)
            countries = np.asarray(sorted(v["country"].unique()))
            quarter = pd.to_datetime(v["anchor_t"]).dt.quarter.to_numpy()
            rng = np.random.default_rng(SEED)
            draws = [rng.choice(countries, len(countries), replace=True) for _ in range(N_CBOOT)]
            for metric, agg in (
                ("pooled_mean", ESTIMANDS["pooled_mean"]),
                ("panel_median", ESTIMANDS["panel_median"]),
            ):
                b0 = _select(sv, v, agg, np.ones(len(v), dtype=bool))
                loco = [_select(sv, v, agg, (v["country"] != c).to_numpy()) for c in countries]
                # a bootstrap draw repeats countries. Row repetition weights the
                # pooled mean correctly but NOT the panel median, whose groupby
                # on the label collapses a country drawn k times into one group;
                # both estimands are therefore aggregated over the drawn labels.
                cb = [_select_draw(sv, v, CLUSTER_ESTIMANDS[metric], d) for d in draws]
                qs = [_select(sv, v, agg, quarter == q) for q in (1, 2, 3, 4)]
                rows.append(
                    {
                        "fm": fm_label,
                        "horizon": int(h),
                        "metric": metric,
                        "deployed_budget": b0,
                        "loco_min": min(loco),
                        "loco_max": max(loco),
                        "loco_share_same": float(np.mean(np.isclose(loco, b0))),
                        "cboot_q25": float(np.percentile(cb, 25)),
                        "cboot_q75": float(np.percentile(cb, 75)),
                        "cboot_share_same": float(np.mean(np.isclose(cb, b0))),
                        "cboot_share_zero": float(np.mean(np.isclose(cb, 0.0))),
                        "quarters": " ".join(f"{q:.2f}" for q in qs),
                    }
                )
                print(
                    f"[thr] {SHORT[fm_label]:11s} h={int(h):<4} {metric:12s} b={b0:.2f} loco "
                    f"[{min(loco):.2f},{max(loco):.2f}] "
                    f"cboot IQR [{np.percentile(cb, 25):.2f},{np.percentile(cb, 75):.2f}] same "
                    f"{np.mean(np.isclose(cb, b0)):.2f} "
                    f"quarters {' '.join(f'{q:.2f}' for q in qs)}",
                    flush=True,
                )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "threshold_stability.csv", index=False)
    tex = [
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "FM & $h$ & metric & deployed & LOCO range & bootstrap IQR & same (\\%) & quarters \\\\",
        "\\midrule",
    ]
    for r in d.itertuples():
        tex.append(
            f"{SHORT[r.fm]} & {r.horizon} & {'mean' if r.metric == 'pooled_mean' else 'median'} & "
            f"{r.deployed_budget:.2f} & "
            f"[{r.loco_min:.2f}, {r.loco_max:.2f}] & [{r.cboot_q25:.2f}, {r.cboot_q75:.2f}] & "
            f"{100 * r.cboot_share_same:.0f} & {r.quarters} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_thrstab.tex").write_text("\n".join(tex) + "\n")
    print(f"[thr] wrote {OUT}/threshold_stability.csv and tab_thrstab.tex", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
