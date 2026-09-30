"""TS-LIMITS: is the "routing" gain just country-level configuration?

The gate carries `country_code` among its 22 features while the deployable
static baseline is chosen per horizon only. That is not a fair contest: a policy
allowed to know the country is being compared against one that is not, so any
gain attributable purely to "Chronos is better in Belgium" is being credited to
request-time routing. A reviewer put this first and was right to.

Four policy levels, each frozen on 2024 and applied unchanged to 2025:

    1 horizon-only static      one binary choice per horizon
    2 country x horizon static one binary choice per (country, horizon), 76 per FM
    3 router WITHOUT country   the gate on the 21 non-identity features
    4 router WITH country      the deployed gate

The number that decides the paper's framing is 4 - 2, not 4 - 1: if the
country x horizon rule absorbs the gain, the contribution is auditable
grid-and-horizon configuration and should be described that way.

Two further diagnostics separate configuration from dynamics:

    margin decomposition   D = m_{c,h} + r_{i,c,h} with m estimated on the FIT
                           years only, never on the scored window, so the split
                           is causal. Reports the share of margin VARIANCE the
                           country mean carries, and the correlation between the
                           gate's score and the residual r.

Escalation uses the frozen numeric threshold, not a top-f rank over the test
window, so levels 3 and 4 are genuinely online policies.

Out: reports/tslimits/country_static.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_country_static.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from load_forecast.eval.routing import ESTIMANDS, escalate_above, frozen_threshold

OUT = Path("reports/tslimits")
# Realized weather at the anchor, joined from reanalysis. See noweather above.
WEATHER_ANCHOR = ("anchor_temp", "anchor_hdd_anom", "anchor_cdd_anom")
B = 2000
SEED = 0
ESTIMAND = "pooled_mean"


def _mod(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_BS = _mod("tslimits_bootstrap")


def _ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def _matches_bootstrap(bcm: pd.DataFrame | None, fm: str, h: int, b: float) -> bool | None:
    """Whether the median-selected budget equals the one the bootstrap script froze."""
    if bcm is None or (fm, h) not in bcm.index:
        return None
    return bool(abs(float(bcm.loc[(fm, h), "frozen_enr_budget"]) - b) < 1e-9)


def main() -> int:
    _learner_sfx = "" if _BS.GATE_LEARNER == "hgb" else f"_{_BS.GATE_LEARNER}"
    _sfx = _learner_sfx + _BS.FAMILY_SFX + _BS.GRID_SFX
    rng = np.random.default_rng(SEED)
    agg = ESTIMANDS[ESTIMAND]
    enr = list(_BS.ENRICHED)
    plain = [c for c in enr if c != "country_code"]
    assert len(plain) == len(enr) - 1, "country_code not found in the enriched set"
    # The three anchor weather features are read from reanalysis, which publishes
    # days after the hour it describes, so they are a retrospective proxy rather
    # than something an operator could have pulled from that source at the
    # anchor. Refitting without them bounds what they were worth. The forward
    # climatology terms stay: they are a <=2017 (country, month, hour) constant,
    # which is a calendar function rather than weather.
    noweather = [c for c in enr if c not in WEATHER_ANCHOR]
    assert len(noweather) == len(enr) - len(WEATHER_ANCHOR), "weather features not found"
    # An operational router may also read the cheap model's own output before
    # deciding, which the deployed gate is deliberately denied so that its gain
    # cannot be self-knowledge. Comparing the two separates request descriptors
    # from what the structural forecast says about itself. The feature is the
    # forecast's implied change from the anchor, available before any FM call.
    operational = [*enr, "st_implied_change"]
    rows: list[dict[str, Any]] = []
    cc_rows: list[dict[str, Any]] = []
    # The bootstrap depends on nothing but the per-country matrix, so persisting
    # it makes every later interval variant (BCa, jackknife, leave-one-out) a
    # pure post-processing step instead of a refit. A few kB per cell.
    mats: dict[str, np.ndarray] = {}
    per_req: list[pd.DataFrame] = []

    _bcm_path = OUT / f"bootstrap_ci_panel_median{_BS.FAMILY_SFX}{_BS.GRID_SFX}.csv"
    _bcm = pd.read_csv(_bcm_path).set_index(["fm", "horizon"]) if _bcm_path.exists() else None
    for fm_label, (test_pat, dev_pat) in _BS.FM_PATTERNS.items():
        dev = _BS._paired("dev", dev_pat)
        test = _BS._paired("2025", test_pat)
        fit = dev[dev["test_year"].isin(_BS.FIT_YEARS)]
        val = dev[dev["test_year"].isin(_BS.VAL_YEARS)]

        for h in sorted(test["horizon"].unique()):
            f = fit[fit["horizon"] == h].reset_index(drop=True)
            v = val[val["horizon"] == h].reset_index(drop=True)
            t = test[test["horizon"] == h].reset_index(drop=True)
            for frame in (f, v, t):
                frame["st_implied_change"] = frame["y_pred"] / frame["anchor_load"] - 1.0
            st = t["ape_st"].to_numpy()
            fm = t["ape_fm"].to_numpy()

            served: dict[str, np.ndarray] = {"st": st, "fm": fm}

            # 1 horizon-only static, frozen on 2024
            take_fm = agg(v["ape_fm"], v["country"]) < agg(v["ape_st"], v["country"])
            served["horizon_static"] = fm if take_fm else st

            # 2 country x horizon static, frozen on 2024. One decision per
            # country, so 19 per (fm, horizon) rather than one.
            #
            # A reviewer objected that this baseline sees one year while the
            # router sees six, so the contest is unfair to the baseline. The
            # objection is correct about the protocol, so the fairer variants are
            # computed here rather than argued away: the same rule fitted on ALL
            # fit+selection years, and -- as an upper bound no deployable rule
            # could reach -- the same rule fitted on 2025 itself.
            def _sel(d: pd.DataFrame) -> pd.Series:
                """Per-country FM-or-structural choice on the window `d`."""
                return d.groupby("country")[["ape_fm", "ape_st"]].mean().pipe(
                    lambda x: x["ape_fm"] < x["ape_st"]
                )

            def _pick(
                d: pd.DataFrame,
                t: pd.DataFrame = t,
                fm: np.ndarray = fm,
                st: np.ndarray = st,
            ) -> np.ndarray:
                # t/fm/st bound as defaults, not captured: this closure is
                # redefined every horizon and a late-binding capture would
                # silently describe whichever cell the loop reached last.
                use = t["country"].map(_sel(d)).to_numpy()
                assert not pd.isna(use).any(), "a 2025 country is missing from the window"
                return np.where(use, fm, st)

            pick = _sel(v)
            served["cc_static"] = _pick(v)
            served["cc_static_all"] = _pick(pd.concat([f, v], ignore_index=True))
            served["cc_static_hindsight"] = _pick(t)

            # 3, 4, 5 routers, frozen threshold from 2024.
            #
            # router_country is the protocol-matched country control the reviewer
            # asked for: the SAME pipeline as the deployed gate -- fit on the fit
            # years, threshold frozen on 2024, applied per request -- but with
            # country identity as its only feature. Its score takes 19 distinct
            # values, so it is a per-country step function reached through the
            # router's own machinery rather than a differently-shaped baseline.
            budgets: dict[str, float] = {}
            fracs: dict[str, float] = {}
            thrs: dict[str, float] = {}
            for tag, cols in (
                ("router_nocc", plain),
                ("router_noweather", noweather),
                ("router_operational", operational),
                ("router_cc", enr),
                ("router_country", ["country_code"]),
            ):
                reg = _BS._reg(cols).fit(f[cols], f["ape_st"] - f["ape_fm"])
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
                thr = frozen_threshold(sv, b)
                m = escalate_above(stt, thr)
                served[tag] = np.where(m, fm, st)
                budgets[tag] = b
                thrs[tag] = thr
                fracs[tag] = float(m.mean())
                if tag == "router_cc":
                    score_cc = stt
                    # The same gate with its threshold re-selected under the
                    # panel median: the policy a median-following operator
                    # deploys, scored here under the pooled mean. This is the
                    # fourth cell of the policy-by-metric table; the other three
                    # already exist in paired_estimand.csv.
                    agg_med = ESTIMANDS["panel_median"]
                    b_med = min(
                        (
                            agg_med(
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
                    thr_med = frozen_threshold(sv, b_med)
                    m_med = escalate_above(stt, thr_med)
                    served["router_cc_medsel"] = np.where(m_med, fm, st)
                    budgets["router_cc_medsel"] = b_med
                    thrs["router_cc_medsel"] = thr_med
                    fracs["router_cc_medsel"] = float(m_med.mean())

            # 6 residual-margin router. The country mean is removed from the
            # TARGET, estimated on the fit years alone, so the learner cannot
            # reach the gain by rediscovering "the FM is better in Belgium"; the
            # mean is added back at serving time so the policy is still complete.
            # Its gain over cc_static is the cleanest statement of request-level
            # information beyond stable country and horizon effects.
            m_fit = (f["ape_st"] - f["ape_fm"]).groupby(f["country"]).mean()
            reg_r = _BS._reg(enr).fit(
                f[enr], (f["ape_st"] - f["ape_fm"]) - f["country"].map(m_fit).to_numpy()
            )
            sv_r = np.asarray(reg_r.predict(v[enr]), dtype=float) + v["country"].map(
                m_fit
            ).to_numpy()
            st_r = np.asarray(reg_r.predict(t[enr]), dtype=float) + t["country"].map(
                m_fit
            ).to_numpy()
            b_r = min(
                (
                    agg(
                        pd.Series(
                            np.where(
                                escalate_above(sv_r, frozen_threshold(sv_r, q)),
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
            m_r = escalate_above(st_r, frozen_threshold(sv_r, b_r))
            served["router_resid"] = np.where(m_r, fm, st)
            budgets["router_resid"] = b_r
            fracs["router_resid"] = float(m_r.mean())

            # country matrix once, then every contrast on the same resample
            names = list(served)
            frame = pd.DataFrame({"c": t["country"].to_numpy(), **served})
            # Per-request served losses, keyed by country AND anchor. The
            # cluster matrices collapse the time index, which is enough to
            # resample countries and nothing else; a date-block design needs the
            # rows. A few MB, and it makes every later interval variant pure
            # post-processing rather than a refit.
            per_req.append(
                pd.DataFrame(
                    {
                        # not "fm": served already has an "fm" policy column and
                        # the splat would silently overwrite the label
                        "model": fm_label,
                        "horizon": int(h),
                        "country": t["country"].to_numpy(),
                        "anchor_t": t["anchor_t"].to_numpy(),
                        # the deployed gate's score, so any later threshold
                        # variant is post-processing rather than a refit
                        "score": score_cc,
                        **served,
                    }
                )
            )
            g = frame.groupby("c")
            sums_df = g[names].sum()
            cnt_s = g[names[0]].size().astype(float)
            labels = list(sums_df.index)
            sums = sums_df.to_numpy()
            cnt = cnt_s.to_numpy().astype(float)

            def panel(ix: np.ndarray, s: np.ndarray = sums, n: np.ndarray = cnt) -> np.ndarray:
                return s[ix].sum(axis=0) / n[ix].sum()

            allr = np.arange(sums.shape[0])
            s0 = panel(allr)

            cell = f"{fm_label}|{int(h)}"
            mats[f"{cell}|sums"] = sums
            mats[f"{cell}|cnt"] = cnt
            mats[f"{cell}|labels"] = np.asarray(labels, dtype=object)
            mats[f"{cell}|names"] = np.asarray(names, dtype=object)
            for k in names:
                direct = agg(pd.Series(served[k]), t["country"])
                assert abs(direct - s0[names.index(k)]) < 1e-9, f"matrix mismatch for {k}"

            i = {k: names.index(k) for k in names}
            contrasts = {
                "cc_over_horizon": (i["horizon_static"], i["cc_static"]),
                "router_over_horizon": (i["horizon_static"], i["router_cc"]),
                "router_over_cc": (i["cc_static"], i["router_cc"]),
                "router_cc_over_nocc": (i["router_nocc"], i["router_cc"]),
                "noweather_over_cc": (i["cc_static"], i["router_noweather"]),
                "operational_over_cc": (i["cc_static"], i["router_operational"]),
                # the protocol-matched controls
                "router_over_ccall": (i["cc_static_all"], i["router_cc"]),
                "router_over_cchind": (i["cc_static_hindsight"], i["router_cc"]),
                "router_over_ccgate": (i["router_country"], i["router_cc"]),
                "ccgate_over_horizon": (i["horizon_static"], i["router_country"]),
                # the residual router, which never saw the country mean in its target
                "resid_over_cc": (i["cc_static"], i["router_resid"]),
                # the median-selected policy under the pooled mean, and the two
                # selected policies against each other
                "medsel_over_cc": (i["cc_static"], i["router_cc_medsel"]),
                "router_over_medsel": (i["router_cc_medsel"], i["router_cc"]),
            }
            draws = {k: np.empty(B) for k in contrasts}
            for bi in range(B):
                ix = rng.integers(0, sums.shape[0], sums.shape[0])
                q = panel(ix)
                for k, (a, b_) in contrasts.items():
                    draws[k][bi] = q[a] - q[b_]

            # margin decomposition, m estimated on FIT years only
            m_c = (f["ape_st"] - f["ape_fm"]).groupby(f["country"]).mean()
            d_t = (t["ape_st"] - t["ape_fm"]).to_numpy()
            resid = d_t - t["country"].map(m_c).to_numpy()
            reg_cc = _BS._reg(enr).fit(f[enr], f["ape_st"] - f["ape_fm"])
            score_t = np.asarray(reg_cc.predict(t[enr]), dtype=float)

            row: dict[str, Any] = {
                "fm": fm_label,
                "horizon": int(h),
                "n_countries": int(sums.shape[0]),
                "cc_pick_fm_n": int(pick.sum()),
                "horizon_static": round(float(s0[i["horizon_static"]]), 4),
                "cc_static": round(float(s0[i["cc_static"]]), 4),
                "router_nocc": round(float(s0[i["router_nocc"]]), 4),
                "router_noweather": round(float(s0[i["router_noweather"]]), 4),
                "router_noweather_frac": round(fracs["router_noweather"], 4),
                "router_operational": round(float(s0[i["router_operational"]]), 4),
                "router_operational_frac": round(fracs["router_operational"], 4),
                "router_cc": round(float(s0[i["router_cc"]]), 4),
                "router_budget": budgets["router_cc"],
                "router_threshold": round(thrs["router_cc"], 6),
                "router_realized_frac": round(fracs["router_cc"], 4),
                "router_budget_median": budgets["router_cc_medsel"],
                "router_threshold_median": round(thrs["router_cc_medsel"], 6),
                "router_medsel_realized_frac": round(fracs["router_cc_medsel"], 4),
                # the same selection tslimits_bootstrap.py makes under
                # TSLIMITS_ESTIMAND=panel_median; recorded, so a drift between the
                # two scripts cannot pass silently
                "medsel_budget_matches_bootstrap": _matches_bootstrap(
                    _bcm, fm_label, int(h), budgets["router_cc_medsel"]
                ),
                "margin_var": round(float(d_t.var()), 4),
                "resid_var": round(float(resid.var()), 4),
                "country_share_of_var": round(float(1.0 - resid.var() / d_t.var()), 4),
                "corr_score_resid": round(float(np.corrcoef(score_t, resid)[0, 1]), 4),
            }
            for k, (a, b_) in contrasts.items():
                pt = float(s0[a] - s0[b_])
                lo, hi = _ci(draws[k])
                row[f"{k}_pp"] = round(pt, 4)
                row[f"{k}_lo"] = round(lo, 4)
                row[f"{k}_hi"] = round(hi, 4)
                row[f"{k}_sig"] = bool(lo > 0)
            rows.append(row)

            # The same served vectors, split by country. A decomposition of
            # router_over_cc_pp above, not a new estimate: pooled mean is a mean
            # over requests, so the panel figure is the request-weighted, not the
            # unweighted, average of these.
            for ci, cc in enumerate(labels):
                cc_rows.append({
                    "fm": fm_label,
                    "horizon": int(h),
                    "country": cc,
                    "n_requests": int(cnt[ci]),
                    "cc_static": round(float(sums[ci, i["cc_static"]] / cnt[ci]), 4),
                    "router_cc": round(float(sums[ci, i["router_cc"]] / cnt[ci]), 4),
                    "router_over_cc_pp": round(
                        float((sums[ci, i["cc_static"]] - sums[ci, i["router_cc"]]) / cnt[ci]), 4),
                })

            print(
                f"[cc] {fm_label:19s} h={int(h):<4} "
                f"hor {row['horizon_static']:.3f} ccxh {row['cc_static']:.3f} "
                f"router {row['router_cc']:.3f} | "
                f"ccxh-over-hor {row['cc_over_horizon_pp']:+.3f} "
                f"router-over-ccxh {row['router_over_cc_pp']:+.3f} "
                f"[{row['router_over_cc_lo']:+.3f},{row['router_over_cc_hi']:+.3f}]"
                f"{'*' if row['router_over_cc_sig'] else ' '} "
                f"(cc picks FM in {row['cc_pick_fm_n']}/19)",
                flush=True,
            )

    # How well the other features recover the country. The paper uses this to say
    # why dropping country_code is not a valid control: the remaining features
    # reconstruct the country anyway, so the "without country" gate still has it.
    # Measured here rather than asserted.
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    _dev = _BS._paired("dev", next(iter(_BS.FM_PATTERNS.values()))[1])
    _f = [c for c in _BS.ENRICHED if c != "country_code"]
    _s = _dev[_dev["horizon"] == 168].sample(40000, random_state=0)
    _acc = cross_val_score(
        HistGradientBoostingClassifier(max_iter=300, random_state=0),
        _s[_f].to_numpy(), _s["country_code"].to_numpy(), cv=3, scoring="accuracy"
    ).mean()
    print(
        f"[cc] country recovered from the other {len(_f)} features with "
        f"{_acc:.3f} accuracy (3-fold, h=168, 40k rows)",
        flush=True,
    )

    d = pd.DataFrame(rows)
    d["country_recovery_acc"] = round(float(_acc), 4)
    # An alternative gate learner writes beside the deployed outputs, never over
    # them, so country_static.csv always means the boosted tree the paper deploys.
    d.to_csv(OUT / f"country_static{_sfx}.csv", index=False)

    np.savez(OUT / f"cluster_matrices{_sfx}.npz", **mats)
    pr = pd.concat(per_req, ignore_index=True)
    pr.to_parquet(OUT / f"served_per_request{_sfx}.parquet", index=False)
    print(
        f"[cc] persisted {len(pr):,} served rows x {len(per_req[0].columns) - 4} policies "
        f"for date-block resampling",
        flush=True,
    )
    print(f"[cc] persisted {len(mats) // 4} cluster matrices for interval variants", flush=True)

    cc = pd.DataFrame(cc_rows)
    cc.to_csv(OUT / f"country_breakdown{_sfx}.csv", index=False)
    _pos = int((cc.router_over_cc_pp > 0).sum())
    _zero = int((cc.router_over_cc_pp == 0).sum())
    print(
        f"[cc] per-country breakdown: {len(cc)} rows, gain positive in {_pos}, "
        f"exactly zero in {_zero}, negative in {len(cc) - _pos - _zero}",
        flush=True,
    )
    print("", flush=True)
    for k in (
        "cc_over_horizon",
        "router_over_horizon",
        "router_over_cc",
        "router_cc_over_nocc",
        "router_over_ccall",
        "router_over_cchind",
        "router_over_ccgate",
        "ccgate_over_horizon",
        "resid_over_cc",
    ):
        print(
            f"[cc] {k:22s}: median {d[f'{k}_pp'].median():+.4f} pp, "
            f"positive {int((d[f'{k}_pp'] > 0).sum())}/{len(d)}, "
            f"interval excludes zero {int(d[f'{k}_sig'].sum())}/{len(d)}",
            flush=True,
        )
    share = d.cc_over_horizon_pp.sum() / d.router_over_horizon_pp.sum()
    print(
        f"[cc] country x horizon configuration explains {share:.1%} of the router's "
        f"gain over horizon-only static",
        flush=True,
    )
    print(
        f"[cc] country mean carries {d.country_share_of_var.min():.1%}-"
        f"{d.country_share_of_var.max():.1%} of margin variance; "
        f"corr(score, residual) {d.corr_score_resid.min():.2f}-{d.corr_score_resid.max():.2f}",
        flush=True,
    )
    print(f"[cc] wrote {OUT}/country_static{_sfx}.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
