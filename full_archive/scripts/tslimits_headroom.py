"""TS-LIMITS: how much of the relevant routing headroom the frozen router captures.

The paper's Table 1 prints the oracle gap against hindsight best-fixed, and the
router's gain against c-static. Dividing one by the other is a mistake a reader
can make: the gap is measured from a baseline the gain is not. The relevant
headroom is c-static to oracle, and the fraction captured is

    (R_cstatic - R_router) / (R_cstatic - R_oracle).

Two further questions the same per-request records answer.

First, a 2x2 crossing of the aggregation: within-country mean or median, then
across-country mean or median. The paper compares only the two diagonal cells,
pooled mean and panel median. Crossing them says whether metric dependence comes
from within-country tail sensitivity, from equal country weighting, or from both.

Second, the long leads. No router exists there, but the records do, so the
descriptive diagnostics are cheap: the oracle gap, the fraction of requests
Chronos-2 wins, and the share of absolute margin those wins carry. Together they
say whether Chronos-2 is consistently worse or occasionally much better.

Out: reports/tslimits/headroom.csv
     reports/tslimits/aggregation_2x2.csv
     reports/tslimits/longlead_diagnostics.csv
     reports/tslimits/tab_headroom.tex, tab_agg2x2.tex, tab_longlead.tex
Run: .venv/bin/python scripts/tslimits_headroom.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("reports/tslimits")
SHORT = {"Chronos-2-Uni-ZS": "Chronos-2", "TimesFM-2.5-Uni-ZS": "TimesFM-2.5"}

# The tuned-rollout arms the paper reports at the long leads.
LONG = (
    (
        8760,
        "1 yr",
        "per_origin_structural_2025_year.parquet",
        "rollout_ablation_h8760_cap48.parquet",
    ),
    (
        17520,
        "2 yr",
        "per_origin_structural_2025_h17520.parquet",
        "rollout_ablation_h17520_cap56.parquet",
    ),
)


def _agg(x: pd.Series, country: pd.Series, within: str, across: str) -> float:
    w = x.groupby(country).mean() if within == "mean" else x.groupby(country).median()
    return float(w.mean() if across == "mean" else w.median())


def main() -> int:
    pr = pd.read_parquet(OUT / "served_per_request.parquet")
    pr["oracle"] = np.minimum(pr["st"], pr["fm"])

    # ---- headroom fraction, pooled mean ----
    rows = []
    for (m, h), g in pr.groupby(["model", "horizon"], sort=True):
        rc, rr, ro = g.cc_static.mean(), g.router_cc.mean(), g.oracle.mean()
        rows.append(
            {
                "model": m,
                "horizon": int(h),
                "r_cstatic": round(rc, 4),
                "r_router": round(rr, 4),
                "r_oracle": round(ro, 4),
                "cstatic_minus_oracle": round(rc - ro, 4),
                "gain_over_cstatic": round(rc - rr, 4),
                "headroom_captured": round((rc - rr) / (rc - ro), 4),
            }
        )
    hd = pd.DataFrame(rows)
    hd.to_csv(OUT / "headroom.csv", index=False)

    # ---- 2x2 aggregation ----
    rows = []
    for (m, h), g in pr.groupby(["model", "horizon"], sort=True):
        for within in ("mean", "median"):
            for across in ("mean", "median"):
                rc = _agg(g.cc_static, g.country, within, across)
                rr = _agg(g.router_cc, g.country, within, across)
                rows.append(
                    {
                        "model": m,
                        "horizon": int(h),
                        "within": within,
                        "across": across,
                        "gain_over_cstatic": round(rc - rr, 4),
                    }
                )
    ag = pd.DataFrame(rows)
    ag.to_csv(OUT / "aggregation_2x2.csv", index=False)

    # ---- long-lead diagnostics ----
    rows = []
    for h, label, st_file, fm_file in LONG:
        st = pd.read_parquet(OUT / st_file)[["country", "anchor_t", "ape"]].rename(
            columns={"ape": "ape_st"}
        )
        fm = pd.read_parquet(OUT / fm_file)
        fm = fm.rename(columns={"cc": "country", "origin_ts": "anchor_t"})
        fm["anchor_t"] = pd.to_datetime(fm["anchor_t"])
        fm = fm[["country", "anchor_t", "ape"]].rename(columns={"ape": "ape_fm"})
        j = st.merge(fm, on=["country", "anchor_t"], how="inner", validate="one_to_one")
        assert len(j) == len(st), f"h={h}: {len(j)} of {len(st)} matched"
        d = j.ape_st - j.ape_fm  # positive: Chronos wins
        gap = 0.5 * (d.abs().mean() - abs(d.mean()))
        rows.append(
            {
                "horizon": h,
                "label": label,
                "n": len(j),
                "r_structural": round(j.ape_st.mean(), 4),
                "r_chronos": round(j.ape_fm.mean(), 4),
                "r_oracle": round(np.minimum(j.ape_st, j.ape_fm).mean(), 4),
                "oracle_gap_pp": round(gap, 4),
                "frac_chronos_wins": round(float((d > 0).mean()), 4),
                "positive_margin_share": round(float(d[d > 0].sum() / d.abs().sum()), 4),
                "mean_win_pp": round(float(d[d > 0].mean()), 4),
                "mean_loss_pp": round(float(d[d < 0].mean()), 4),
            }
        )
    ll = pd.DataFrame(rows)
    ll.to_csv(OUT / "longlead_diagnostics.csv", index=False)

    # ---- tables ----
    t = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "FM & $h$ & c-static $-$ oracle & gain over c-static & headroom captured \\\\",
        "\\midrule",
    ]
    for r in hd.itertuples():
        t.append(
            f"{SHORT[r.model]} & {r.horizon} & {r.cstatic_minus_oracle:.3f} & "
            f"{r.gain_over_cstatic:.3f} & {r.headroom_captured:.2f} \\\\"
        )
    t += ["\\bottomrule", "\\end{tabular}", ""]
    (OUT / "tab_headroom.tex").write_text("\n".join(t))

    piv = ag.pivot_table(
        index=["model", "horizon"], columns=["within", "across"], values="gain_over_cstatic"
    ).reset_index()
    t = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "FM & $h$ & mean/mean & mean/median & median/mean & median/median \\\\",
        "\\midrule",
    ]
    for r in piv.itertuples(index=False):
        m, h = r[0], r[1]
        vals = {
            (w, a): piv.loc[(piv.model == m) & (piv.horizon == h), (w, a)].iloc[0]
            for w in ("mean", "median")
            for a in ("mean", "median")
        }
        t.append(
            f"{SHORT[m]} & {h} & ${vals[('mean', 'mean')]:+.3f}$ & "
            f"${vals[('mean', 'median')]:+.3f}$ & ${vals[('median', 'mean')]:+.3f}$ & "
            f"${vals[('median', 'median')]:+.3f}$ \\\\"
        )
    t += ["\\bottomrule", "\\end{tabular}", ""]
    (OUT / "tab_agg2x2.tex").write_text("\n".join(t))

    t = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "lead & structural & Chronos-2 & oracle & oracle gap & Chronos wins"
        " & win share of $|D|$ \\\\",
        "\\midrule",
    ]
    for r in ll.itertuples():
        t.append(
            f"{r.label} & {r.r_structural:.2f} & {r.r_chronos:.2f} & {r.r_oracle:.2f} & "
            f"{r.oracle_gap_pp:.2f} & {r.frac_chronos_wins:.0%} & "
            f"{r.positive_margin_share:.0%} \\\\".replace("%", "\\%")
        )
    t += ["\\bottomrule", "\\end{tabular}", ""]
    (OUT / "tab_longlead.tex").write_text("\n".join(t))

    print(
        f"[hr] headroom captured: median {hd.headroom_captured.median():.3f}, "
        f"range {hd.headroom_captured.min():.3f}-{hd.headroom_captured.max():.3f}",
        flush=True,
    )
    neg = ag[ag.gain_over_cstatic < 0]
    print(
        f"[hr] 2x2: gain positive in {int((ag.gain_over_cstatic > 0).sum())}/32 cells; "
        f"negatives: {[(SHORT[r.model], r.horizon, r.within, r.across) for r in neg.itertuples()]}",
        flush=True,
    )
    for r in ll.itertuples():
        print(
            f"[hr] {r.label}: oracle gap {r.oracle_gap_pp:.2f} pp, Chronos wins "
            f"{r.frac_chronos_wins:.0%} of requests carrying {r.positive_margin_share:.0%} "
            f"of |D|; mean win {r.mean_win_pp:+.2f}, mean loss {r.mean_loss_pp:+.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
