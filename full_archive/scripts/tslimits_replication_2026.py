"""Post-release replication: the frozen 2025 policies scored on Jan-Mar 2026.

Nothing is selected here. The gate is the deployed fit (2018-2023, seed 0),
the thresholds are the numbers frozen on 2024 under each metric, and the
static baselines are the 2024 rules; each is recomputed by the deployed code
path and asserted equal to the record on disk before it touches a 2026
request. The window post-dates the weights of both checkpoints, so pretraining overlap is
excluded by construction, and it post-dates the study design, so it cannot
have motivated it.

Panel: the pre-registered completeness rule (>99.5% valid hours in the
window) drops FR (98.80%, 26 hours missing at the window start); the primary
panel is 18 countries and the 19-country panel is reported as a sensitivity.

Out:  reports/tslimits/replication_2026.csv, tab_replication2026.tex
Run:  PYTHONPATH=src .venv/bin/python scripts/tslimits_replication_2026.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from load_forecast.eval.routing import ESTIMANDS, escalate_above, frozen_threshold

OUT = Path("reports/tslimits")
SEED, B = 0, 2000
SHORT = {"Chronos-2-Uni-ZS": "Chronos-2", "TimesFM-2.5-Uni-ZS": "TimesFM-2.5"}
PAT26 = {
    "Chronos-2-Uni-ZS": "reports/tslimits/fm/tsl26_chronos_q_*.parquet",
    "TimesFM-2.5-Uni-ZS": "reports/tslimits/fm/tsl26_timesfm_q_*.parquet",
}
COMPLETENESS_MIN = 0.995
DROPPED = {"FR": 0.9880}  # from reports/revision/load2026_gates.csv
# Sensitivity arms, both CPU-only post-processing of the same frozen policies:
#   TSLIMITS_START_2026=2026-01-07  keeps targets from that date on (after the later
#                                   checkpoint revision hash); the 18-country panel
#                                   is kept for comparability although FR passes
#                                   the completeness rule on the shortened window.
#   TSLIMITS_STRUCT_2026=2026_nogdp scores the 2026 structural records without the
#                                   GDP term (tslimits_per_origin_structural.py
#                                   --window 2026 --no-gdp); gate, thresholds and
#                                   rules stay those of the deployed model.
START_2026 = os.environ.get("TSLIMITS_START_2026", "")
STRUCT_2026 = os.environ.get("TSLIMITS_STRUCT_2026", "2026")
SFX = (f"_from{START_2026[5:7]}{START_2026[8:10]}" if START_2026 else "") + (
    "_nogdp" if STRUCT_2026.endswith("nogdp") else ""
)


def _mod(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_BS = _mod("tslimits_bootstrap")


def _select(sv: np.ndarray, v: pd.DataFrame, agg) -> float:
    return min(
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


def _cluster_ci(
    a: np.ndarray, b: np.ndarray, cc: np.ndarray, rng, agg_name: str
) -> tuple[float, float, float]:
    """Point and 95% country-cluster interval of agg(a) - agg(b)."""
    frame = pd.DataFrame({"c": cc, "a": a, "b": b})
    if agg_name == "pooled_mean":
        g = frame.groupby("c")
        sums = g[["a", "b"]].sum().to_numpy()
        cnt = g["a"].size().to_numpy().astype(float)
        point = (sums[:, 0].sum() - sums[:, 1].sum()) / cnt.sum()
        d = np.empty(B)
        for bi in range(B):
            ix = rng.integers(0, len(cnt), len(cnt))
            d[bi] = (sums[ix, 0].sum() - sums[ix, 1].sum()) / cnt[ix].sum()
    else:
        med = frame.groupby("c")[["a", "b"]].median().to_numpy()
        point = float(np.median(med[:, 0]) - np.median(med[:, 1]))
        d = np.empty(B)
        for bi in range(B):
            q = np.median(med[rng.integers(0, len(med), len(med))], axis=0)
            d[bi] = q[0] - q[1]
    return float(point), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> int:
    cols = list(_BS.ENRICHED)
    cs = pd.read_csv(OUT / "country_static.csv").set_index(["fm", "horizon"])
    bcm = pd.read_csv(OUT / "bootstrap_ci_panel_median.csv").set_index(["fm", "horizon"])
    agg_mean, agg_med = ESTIMANDS["pooled_mean"], ESTIMANDS["panel_median"]
    rows = []
    for fm_label, (_test_pat, dev_pat) in _BS.FM_PATTERNS.items():
        dev = _BS._paired("dev", dev_pat)
        t26 = _BS._paired(STRUCT_2026, PAT26[fm_label])
        if START_2026:
            keep = pd.to_datetime(t26["target_t"]) >= pd.Timestamp(START_2026)
            t26 = t26[keep].reset_index(drop=True)
        fit = dev[dev["test_year"].isin(_BS.FIT_YEARS)]
        val = dev[dev["test_year"].isin(_BS.VAL_YEARS)]
        for h in sorted(t26["horizon"].unique()):
            f = fit[fit["horizon"] == h].reset_index(drop=True)
            v = val[val["horizon"] == h].reset_index(drop=True)
            t = t26[t26["horizon"] == h].reset_index(drop=True)
            reg = _BS._reg(cols).fit(f[cols], f["ape_st"] - f["ape_fm"])
            sv = np.asarray(reg.predict(v[cols]), dtype=float)
            s26 = np.asarray(reg.predict(t[cols]), dtype=float)
            b_mean, b_med = _select(sv, v, agg_mean), _select(sv, v, agg_med)
            thr_mean, thr_med = frozen_threshold(sv, b_mean), frozen_threshold(sv, b_med)
            # the frozen numbers must be the ones on disk, or this is not the deployed policy
            key = (fm_label, int(h))
            assert abs(b_mean - float(cs.loc[key, "router_budget"])) < 1e-9, f"{key}: budget drift"
            assert abs(thr_mean - float(cs.loc[key, "router_threshold"])) < 1e-5, (
                f"{key}: threshold drift"
            )
            assert abs(b_med - float(bcm.loc[key, "frozen_enr_budget"])) < 1e-9, (
                f"{key}: median budget drift"
            )
            st, fm = t["ape_st"].to_numpy(), t["ape_fm"].to_numpy()
            # 2024 static rules, unchanged
            take_fm = agg_mean(v["ape_fm"], v["country"]) < agg_mean(v["ape_st"], v["country"])
            sel = (
                v.groupby("country")[["ape_fm", "ape_st"]]
                .mean()
                .pipe(lambda x: x["ape_fm"] < x["ape_st"])
            )
            use = t["country"].map(sel).to_numpy()
            assert not pd.isna(use).any(), "a 2026 country has no 2024 rule"
            served = {
                "st": st,
                "fm": fm,
                "oracle": np.minimum(st, fm),
                "h_static": fm if take_fm else st,
                "cc_static": np.where(use, fm, st),
                "router_mean": np.where(escalate_above(s26, thr_mean), fm, st),
                "router_med": np.where(escalate_above(s26, thr_med), fm, st),
            }
            cc = t["country"].to_numpy()
            for panel_name, keep in (
                ("18", ~np.isin(cc, list(DROPPED))),
                ("19", np.ones(len(cc), dtype=bool)),
            ):
                rng = np.random.default_rng(SEED)
                c = cc[keep]
                sv_ = {k: a[keep] for k, a in served.items()}
                r: dict[str, Any] = {
                    "fm": fm_label,
                    "horizon": int(h),
                    "panel": panel_name,
                    "n": int(keep.sum()),
                    "n_countries": len(np.unique(c)),
                    "budget_mean": b_mean,
                    "budget_median": b_med,
                    "frac_mean": float(escalate_above(s26[keep], thr_mean).mean()),
                    "frac_median": float(escalate_above(s26[keep], thr_med).mean()),
                    "cc_picks_fm": int(sel.reindex(np.unique(c)).sum()),
                }
                for agg_name, agg in (("pooled_mean", agg_mean), ("panel_median", agg_med)):
                    tag = "pm" if agg_name == "pooled_mean" else "md"
                    for k in (
                        "st",
                        "fm",
                        "h_static",
                        "cc_static",
                        "router_mean",
                        "router_med",
                        "oracle",
                    ):
                        r[f"{tag}_{k}"] = round(float(agg(pd.Series(sv_[k]), c)), 4)
                    for k in ("router_mean", "router_med"):
                        pt, lo, hi = _cluster_ci(sv_["cc_static"], sv_[k], c, rng, agg_name)
                        r[f"{tag}_{k}_over_cc_pp"], r[f"{tag}_{k}_lo"], r[f"{tag}_{k}_hi"] = (
                            round(pt, 4),
                            round(lo, 4),
                            round(hi, 4),
                        )
                    pt, lo, hi = _cluster_ci(sv_["h_static"], sv_["cc_static"], c, rng, agg_name)
                    r[f"{tag}_cc_over_h_pp"], r[f"{tag}_cc_over_h_lo"], r[f"{tag}_cc_over_h_hi"] = (
                        round(pt, 4),
                        round(lo, 4),
                        round(hi, 4),
                    )
                    r[f"{tag}_headroom_pp"] = round(r[f"{tag}_cc_static"] - r[f"{tag}_oracle"], 4)
                rows.append(r)
                print(
                    f"[2026] {SHORT[fm_label]:11s} h={int(h):<4} panel {panel_name}: n={r['n']} "
                    f"router/mean {r['pm_router_mean_over_cc_pp']:+.3f} "
                    f"[{r['pm_router_mean_lo']:+.3f},{r['pm_router_mean_hi']:+.3f}] "
                    f"| median-scored {r['md_router_mean_over_cc_pp']:+.3f} | median-selected "
                    f"{r['md_router_med_over_cc_pp']:+.3f} "
                    f"| headroom {r['pm_headroom_pp']:.3f} | esc {r['frac_mean']:.2f}",
                    flush=True,
                )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"replication_2026{SFX}.csv", index=False)
    p18 = d[d["panel"] == "18"]
    tex = [
        "\\begin{tabular}{llrlrrrrr}",
        "\\toprule",
        "FM & $h$ & $n$ & best fixed & gain, pooled mean (pp) & same policy, median"
        " & median-selected, median & headroom (pp) & esc. \\\\",
        "\\midrule",
    ]
    for r in p18.itertuples():
        tex.append(
            f"{SHORT[r.fm]} & {r.horizon} & {r.n:,} & "
            f"{'FM' if r.pm_fm < r.pm_st else 'st'} {min(r.pm_fm, r.pm_st):.2f} & "
            f"${r.pm_router_mean_over_cc_pp:+.3f}$ "
            f"[{r.pm_router_mean_lo:+.3f}, {r.pm_router_mean_hi:+.3f}] & "
            f"${r.md_router_mean_over_cc_pp:+.3f}$ & ${r.md_router_med_over_cc_pp:+.3f}$ & "
            f"{r.pm_headroom_pp:.3f} & {r.frac_mean:.2f} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"tab_replication2026{SFX}.tex").write_text("\n".join(tex) + "\n")
    pos = int((p18["pm_router_mean_over_cc_pp"] > 0).sum())
    sig = int((p18["pm_router_mean_lo"] > 0).sum())
    print(
        f"[2026] 18-country panel: router over c-static positive in {pos}/8, interval excludes "
        f"zero in {sig}/8, "
        f"median gain {p18['pm_router_mean_over_cc_pp'].median():+.3f} pp (2025: +0.191); "
        f"median-scored {p18['md_router_mean_over_cc_pp'].median():+.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
