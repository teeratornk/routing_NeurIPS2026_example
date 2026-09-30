"""TS-LIMITS: emit the paper's body tables as LaTeX, straight from the artifacts.

Every number in the two body tables is regenerated here from the CSVs rather
than transcribed. Hand-copying was how a stale cold-start figure and a
best-fixed value that belonged in a different column both reached the PDF, and
it is why a "which artifact is this from" audit was needed at all.

    tab:gap       gap accounting: the identity against the directly computed
                  oracle gap, plus the router against the horizon-only static
                  baseline with its interval and the two verdicts.
    tab:policies  the four policy levels, and the contrast that decides the
                  paper's framing: the router against a per-(country, horizon)
                  fixed rule, which is the fair baseline for a gate that carries
                  country identity.

Verdicts are two separate judgements, because a statistically positive gain is
not automatically an operationally useful one:

    stat   +      the 95% country-cluster interval excludes zero
           n.d.   not demonstrated
    prac   above  lower bound above the pre-specified 0.10 pp practical margin
          open   interval straddles the margin
          below  upper bound below it, i.e. negligible

Out: reports/tslimits/tab_gap.tex, reports/tslimits/tab_policies.tex
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("reports/tslimits")
MARGIN_PP = 0.10
SHORT = {"Chronos-2-Uni-ZS": "Chronos-2", "TimesFM-2.5-Uni-ZS": "TimesFM-2.5"}


def _stat(lo: float) -> str:
    return "yes" if lo > 0 else "no"


def _prac(lo: float, up95: float) -> str:
    """Words, not inequality signs: a reader should not have to recall that
    "$<$" marks the NEGLIGIBLE cell while "$>$" marks the good one, in a row
    that already carries a signed gain and a signed delta."""
    if lo > MARGIN_PP:
        return "above"
    if up95 < MARGIN_PP:
        return "below"
    return "open"


def _rows(d: pd.DataFrame, fmt) -> list[str]:
    out = []
    for fm, g in d.groupby("fm", sort=False):
        out.append(f"\\multirow{{4}}{{*}}{{{SHORT[fm]}}}")
        for _, r in g.sort_values("horizon").iterrows():
            out.append(fmt(r))
        out.append("\\midrule")
    out.pop()  # trailing midrule
    return out


def _capselect_tex(sel_rows, patch: int, default_cap: int) -> str:
    """The 2024 selection sweep, one block per long lead."""
    out = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "cap & steps/call & passes & 2024 panel-median MdAPE & \\\\",
        "\\midrule",
    ]
    for h, cells, best in sel_rows:
        yr = "one year" if h == 8760 else "two years"
        out.append(
            f"\\multicolumn{{5}}{{l}}{{\\emph{{$h={h}$, {yr}}}}} \\\\"
        )
        for cap, passes, score in cells:
            tag = " & \\textbf{frozen}" if cap == best[0] else (
                " & default" if cap == default_cap else " &")
            bold = f"\\textbf{{{score:.2f}}}" if cap == best[0] else f"{score:.2f}"
            out.append(f"{cap} & {cap * patch} & {passes} & {bold}{tag} \\\\")
        out.append("\\midrule")
    out.pop()
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)

def main() -> int:
    b = pd.read_csv(OUT / "bootstrap_ci_pooled_mean.csv")
    c = pd.read_csv(OUT / "country_static.csv")
    m = b.merge(c, on=["fm", "horizon"], validate="one_to_one", suffixes=("", "_cc"))
    assert len(m) == 8, f"expected 8 cells, got {len(m)}"

    # The "gap (identity)" column prints one number twice, which is only honest
    # if the two are actually equal. by_year.csv carries the independently
    # computed identity error per cell, so check the 2025 rows here rather than
    # asserting a column against itself -- which is what this line used to do,
    # and it could never have failed.
    yr = pd.read_csv(OUT / "by_year.csv")
    y25 = yr[yr["test_year"] == 2025].merge(
        m[["fm", "horizon", "orc_gain_pp"]], on=["fm", "horizon"], validate="one_to_one"
    )
    assert len(y25) == 8, f"by_year has {len(y25)} of 8 cells for 2025"
    # `identity_err` compares two expressions computed from the same two arrays
    # inside one function, so it is an algebraic tautology and cannot fail. The
    # checks that can: rebuild the identity from the two independently stored
    # moments, and require by_year and the bootstrap -- separate runs, separate
    # scripts -- to agree on the gap.
    rebuilt = (y25["mean_abs_margin"] - y25["abs_mean_margin"]) / 2.0
    ident = (rebuilt - y25["oracle_gain_pp"]).abs().max()
    assert ident < 1e-3, f"(E|D|-|ED|)/2 does not reproduce the gap; max error {ident}"
    drift = (y25["oracle_gain_pp"] - y25["orc_gain_pp"]).abs().max()
    assert drift < 5e-4, f"by_year and bootstrap disagree on the oracle gap by {drift}"
    print(
        f"[tab] identity rebuilt from stored moments: max error {ident:.2e} pp; "
        f"by_year vs bootstrap gap drift {drift:.2e} pp",
        flush=True,
    )

    def gap_row(r: pd.Series) -> str:
        return (
            f" & {int(r.horizon)} & {r.horizon_static:.3f}"
            f" & {r.orc_gain_pp:.3f} ({r.orc_gain_pp:.3f})"
            f" & ${r.frozen_enr_gain_pp:+.3f}$ [{r.frozen_enr_lo:.3f}, {r.frozen_enr_hi:.3f}]"
            f" & {_stat(r.frozen_enr_lo)} & {_prac(r.frozen_enr_lo, r.frozen_enr_upper95)} \\\\"
        )

    def pol_row(r: pd.Series) -> str:
        share = r.cc_over_horizon_pp / r.router_over_horizon_pp if r.router_over_horizon_pp else 0.0
        return (
            f" & {int(r.horizon)} & {r.cc_static:.3f} & ${r.cc_over_horizon_pp:+.3f}$"
            f" & ${r.router_over_cc_pp:+.3f}$"
            f" [{r.router_over_cc_lo:.3f}, {r.router_over_cc_hi:.3f}]"
            f" & {100 * share:.0f}\\% & {100 * r.router_realized_frac:.1f} \\\\"
        )

    def thr_row(r: pd.Series) -> str:
        return (
            f" & {int(r.horizon)} & {r.router_budget:.2f} & ${r.router_threshold:+.3f}$"
            f" & {r.router_realized_frac:.4f}"
            f" & ${r.router_realized_frac - r.router_budget:+.4f}$ \\\\"
        )

    # The estimand-difference table was hand-typed once and then silently went
    # stale when the threshold and gate encoding changed. It is generated here
    # for the same reason as the others.
    pr = pd.read_csv(OUT / "paired_estimand.csv")
    assert len(pr) == 8, f"paired_estimand has {len(pr)} of 8 cells"

    def pair_row(r: pd.Series) -> str:
        return (
            f" & {int(r.horizon)} & {r.budget_pooled:.2f} & {r.budget_median:.2f}"
            f" & ${r.diff_scoring_pp:+.3f}$"
            f" & ${r.diff_scoring_lo:+.3f}$ & ${r.diff_scoring_hi:+.3f}$ \\\\"
        )

    pair = "\n".join(
        [
            "\\begin{tabular}{llrrrrr}",
            "\\toprule",
            " &  & \\multicolumn{2}{c}{budget} & \\multicolumn{1}{c}{pooled $-$ median}"
            " & \\multicolumn{2}{c}{[95\\% CI]} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-7}",
            "FM & $h$ & pooled & median & same policy & lower & upper \\\\",
            "\\midrule",
            *_rows(pr, pair_row),
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (OUT / "tab_paired.tex").write_text(pair + "\n")

    # by-year replication, also previously hand-typed and also stale: it still
    # showed two negative cells that the refreshed pipeline does not produce.
    years = [2020, 2021, 2022, 2023, 2024, 2025]

    def _block(col: str) -> list[str]:
        wide = yr.pivot_table(index=["fm", "horizon"], columns="test_year", values=col)
        assert list(wide.columns) == years, f"unexpected year columns {list(wide.columns)}"
        rows: list[str] = []
        for fm in ("Chronos-2-Uni-ZS", "TimesFM-2.5-Uni-ZS"):
            rows.append(f"\\multirow{{4}}{{*}}{{{SHORT[fm]}}}")
            for h in (1, 24, 168, 720):
                v = wide.loc[(fm, h)]
                rows.append(f" & {h} & " + " & ".join(f"${v[y]:+.3f}$" for y in years) + " \\\\")
            rows.append("\\midrule")
        rows.pop()
        return rows

    # Two panels, not one. Every practical claim in the paper is measured against
    # c-static, so showing only the hindsight best-fixed column and asserting the
    # c-static count in the caption was the weaker baseline doing the visible work.
    # The two differ in 31 of 48 cells, so the second panel is not decoration.
    byy = "\n".join(
        [
            "\\begin{tabular}{llrrrrrr}",
            "\\toprule",
            "FM & $h$ & " + " & ".join(str(y) for y in years) + " \\\\",
            "\\midrule",
            "\\multicolumn{8}{l}{\\emph{over the hindsight best-fixed configuration}} \\\\",
            "\\midrule",
            *_block("frozen_gain_pp"),
            "\\midrule",
            "\\multicolumn{8}{l}{\\emph{over c-static, the baseline the practical"
            " claims use}} \\\\",
            "\\midrule",
            *_block("frozen_gain_over_cc_pp"),
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (OUT / "tab_byyear.tex").write_text(byy + "\n")

    # Country controls, previously prose only. The body describes four ways of
    # strengthening the country baseline and reports one median each; a reader
    # cannot tell from that whether the intervals exclude zero, and for two of
    # them they do not, in one cell each. Generated rather than typed.
    cs = pd.read_csv(OUT / "country_static.csv")
    controls = (
        ("router_over_cc", "c-static, selection year only"),
        ("router_over_ccall", "c-static, fitted on all fit and selection years"),
        ("router_over_cchind", "c-static, fitted on 2025 with hindsight"),
        ("router_over_ccgate", "gate on \\texttt{country\\_code} alone (a learned c-static)"),
        ("resid_over_cc", "router trained on the country-residualised target"),
    )
    ctl: list[str] = []
    for key, label in controls:
        pp, lo, hi = cs[f"{key}_pp"], cs[f"{key}_lo"], cs[f"{key}_hi"]
        sig = int(((lo > 0) | (hi < 0)).sum())
        ctl.append(
            f"{label} & ${pp.median():+.3f}$ & {int((pp > 0).sum())} of {len(pp)}"
            f" & {sig} of {len(pp)} \\\\"
        )
    ctlt = "\n".join(
        [
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "control the router is measured against & median (pp) & positive"
            " & interval excludes 0 \\\\",
            "\\midrule",
            *ctl,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (OUT / "tab_controls.tex").write_text(ctlt + "\n")

    # Per-country decomposition of router_over_cc_pp. A co-author asked whether
    # the gain is broad or carried by a few countries; this answers it directly.
    cb = pd.read_csv(OUT / "country_breakdown.csv")
    agg_cb = (
        cb.groupby("country")["router_over_cc_pp"]
        .agg(npos=lambda x: int((x > 0).sum()), med="median", lo="min", hi="max")
        .sort_values("med", ascending=False)
    )
    cbt = "\n".join(
        [
            "\\begin{tabular}{lcrrr}",
            "\\toprule",
            "country & cells with a gain & median & min & max \\\\",
            "\\midrule",
        ]
        + [
            f"{cc} & {int(r.npos)}/8 & ${r.med:+.3f}$ & ${r.lo:+.3f}$ & ${r.hi:+.3f}$ \\\\"
            for cc, r in agg_cb.iterrows()
        ]
        + ["\\bottomrule", "\\end{tabular}"]
    )
    (OUT / "tab_countrybreak.tex").write_text(cbt + "\n")
    _share = []
    for _, gg in cb.groupby(["fm", "horizon"]):
        w = (gg.router_over_cc_pp * gg.n_requests).sort_values(ascending=False)
        _share.append(float(w.iloc[:3].sum() / w.sum()))
    print(
        f"[tab] wrote tab_countrybreak: {int((agg_cb.npos == 8).sum())}/19 countries gain in all "
        f"8 cells, {int((agg_cb.npos >= 6).sum())}/19 in at least 6, "
        f"{int((agg_cb.med < 0).sum())}/19 with a negative median; top-3 countries carry a median "
        f"{np.median(_share):.0%} of each cell's gain",
        flush=True,
    )

    # The year ladder, previously prose only. Every other claim in the paper has
    # a table; the long-horizon numbers did not, which made the strongest
    # systems result the least auditable one. PUBLISHED_H excludes h=26280: it
    # was measured and is retained in year_scale.csv, but is not in the paper.
    published_h = (1, 24, 168, 720, 8760, 17520)
    ys = pd.read_csv(OUT / "year_scale.csv").set_index("horizon").loc[list(published_h)]
    label = {1: "1\\,h", 24: "1\\,d", 168: "1\\,w", 720: "1\\,mo",
             8760: "1\\,yr", 17520: "2\\,yr"}
    yr_rows = []
    for h in published_h:
        r = ys.loc[h]
        yr_rows.append(
            f"{label[h]} & {int(h)} & {r['structural']:.2f} & {r['chronos2_tuned']:.2f}"
            f" & {r['seasonal_naive']:.2f} & ${-r['fm_tuned_minus_structural_pp']:+.2f}$"
            f" [{-r['fm_tuned_minus_structural_hi']:+.2f},"
            f" {-r['fm_tuned_minus_structural_lo']:+.2f}]"
            # the contrast that needs no structural model: naive minus FM, same
            # sign rule, so a negative value again means the FM is worse
            f" & ${-r['tuned_minus_naive_pp']:+.2f}$"
            f" [{-r['tuned_minus_naive_hi']:+.2f},"
            f" {-r['tuned_minus_naive_lo']:+.2f}] \\\\"
        )
    yrt = "\n".join(
        [
            "\\begin{tabular}{llrrrll}",
            "\\toprule",
            " & $h$ & structural & Chronos-2 & seas.\\ naive"
            " & structural $-$ FM [95\\% CI] & naive $-$ FM [95\\% CI] \\\\",
            "\\midrule",
            *yr_rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (OUT / "tab_year.tex").write_text(yrt + "\n")
    # ---- the 2024 cap-selection sweep, and the long-horizon estimand check ----
    # The appendix previously showed the 2025 sweep as evidence for a cap the
    # protocol says was chosen on 2024. That is test-window evidence for a
    # training-window decision, so the selection sweep is generated here instead.
    import glob as _glob

    from load_forecast.eval.routing import panel_median

    patch, default_cap = 16, 64
    frozen = {8760: 48, 17520: 56}
    sel_rows = []
    for h in (8760, 17520):
        fs = sorted(_glob.glob(f"{OUT}/rollout_sel2024_h{h}_cap*.parquet"))
        assert fs, f"no 2024 selection arms at h={h}"
        sel = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        caps = sorted(sel["cap"].unique(), reverse=True)
        cells = []
        for cap in caps:
            g = sel[sel["cap"] == cap]
            assert g["cc"].nunique() == 19, f"h={h} cap={cap}: {g['cc'].nunique()} countries"
            cells.append((cap, int(g["passes"].iloc[0]), panel_median(g["ape"], g["cc"])))
        best = min(cells, key=lambda c: c[2])
        assert best[0] == frozen[h], f"h={h}: 2024 argmin cap={best[0]}, frozen {frozen[h]}"
        sel_rows.append((h, cells, best))
    (OUT / "tab_capselect.tex").write_text(
        _capselect_tex(sel_rows, patch, default_cap) + "\n"
    )
    print(f"[tab] wrote tab_capselect: 2024 argmins reproduce {frozen}", flush=True)

    # The same selection grid at the native 8192 context, so the "longer
    # context does not rescue it" claim is read at a cap tuned for that context
    # rather than one inherited from 2048. No frozen value is asserted here: the
    # argmin is reported, and where it moves off 48/56 the 2025 arm at that cap
    # is what Section 6 should quote. Incomplete arms are skipped, not scored.
    sel8 = []
    for h in (8760, 17520):
        cells = []
        for f in sorted(_glob.glob(f"{OUT}/ctx8192_rollout_sel2024_h{h}_cap*.parquet")):
            g = pd.read_parquet(f)
            if g["cc"].nunique() != 19:
                print(
                    f"[tab] capselect8192 h={h}: {f.split('/')[-1]} has {g['cc'].nunique()}/19 "
                    f"countries, skipped",
                    flush=True,
                )
                continue
            cap = int(g["cap"].iloc[0])
            cells.append((cap, int(g["passes"].iloc[0]), panel_median(g["ape"], g["cc"])))
        if len(cells) < 2:
            print(
                f"[tab] capselect8192 h={h}: {len(cells)} complete arms, table not written",
                flush=True,
            )
            continue
        cells = sorted(cells, key=lambda c: -c[0])
        best = min(cells, key=lambda c: c[2])
        sel8.append((h, cells, best))
        print(
            f"[tab] capselect8192 h={h}: argmin cap={best[0]} ({best[2]:.2f}); frozen at 2048 was "
            f"{frozen[h]}",
            flush=True,
        )
    if len(sel8) == 2:
        (OUT / "tab_capselect8192.tex").write_text(_capselect_tex(sel8, patch, default_cap) + "\n")
        print("[tab] wrote tab_capselect8192", flush=True)

    # Does the long-horizon ordering survive the paper's PRIMARY (additive)
    # estimand, or is it an artifact of the median? Four metrics, both leads,
    # each with a country-cluster interval.
    #
    # The three pooled metrics are RATIOS, so a cluster bootstrap must resample
    # countries and recompute the ratio over the union of the resampled
    # countries' requests. Averaging per-country means instead would silently
    # reweight to equal-country weights and change the estimand. Four per-country
    # sufficient statistics carry all of it, plus the per-country median.
    b_ci, seed_ci = 2000, 0
    est, ci_e = {}, {}
    for h in (8760, 17520):
        st = pd.read_parquet(
            OUT
            / (
                "per_origin_structural_2025_year.parquet"
                if h == 8760
                else f"per_origin_structural_2025_h{h}.parquet"
            )
        )
        fm = pd.read_parquet(OUT / f"rollout_ablation_h{h}_cap{frozen[h]}.parquet")
        fm = fm.rename(columns={"cc": "country", "origin_ts": "anchor_t"})
        j = st[["country", "anchor_t", "ape", "y_true"]].merge(
            fm[["country", "anchor_t", "ape", "actual"]],
            on=["country", "anchor_t"],
            how="inner",
            suffixes=("_st", "_fm"),
            validate="one_to_one",
        )
        assert len(j) == len(st) == 6935, f"h={h}: join kept {len(j)} of {len(st)}"
        assert float((j["y_true"] - j["actual"]).abs().max()) == 0.0, f"h={h}: targets differ"
        j["y"] = j["y_true"].abs()

        # per-country sufficient statistics, identical weights for both models
        g = j.groupby("country")
        stats = pd.DataFrame(
            {
                "n": g.size(),
                "sum_st": g["ape_st"].sum(),
                "sum_fm": g["ape_fm"].sum(),
                "sumy": g["y"].sum(),
                "sumay_st": j.assign(w=j["ape_st"] * j["y"]).groupby("country")["w"].sum(),
                "sumay_fm": j.assign(w=j["ape_fm"] * j["y"]).groupby("country")["w"].sum(),
                "med_st": g["ape_st"].median(),
                "med_fm": g["ape_fm"].median(),
            }
        )
        assert len(stats) == 19, f"h={h}: {len(stats)} countries"
        a = stats.to_numpy(dtype=float)
        cols = {c: i for i, c in enumerate(stats.columns)}

        def metrics(m, who, cx=cols):
            """(panel median, pooled mean, wMAPE, MW MAE) over a country subset."""
            n, sy = m[:, cx["n"]].sum(), m[:, cx["sumy"]].sum()
            return (
                float(np.median(m[:, cx[f"med_{who}"]])),
                float(m[:, cx[f"sum_{who}"]].sum() / n),
                float(m[:, cx[f"sumay_{who}"]].sum() / sy),
                float(m[:, cx[f"sumay_{who}"]].sum() / 100.0 / n),
            )

        est[h] = {"structural": metrics(a, "st"), "Chronos-2": metrics(a, "fm")}
        rng_e = np.random.default_rng(seed_ci)
        draws = np.empty((b_ci, 4))
        for bi in range(b_ci):
            # NB: not `m` -- that name holds a DataFrame later in this function
            boot = a[rng_e.integers(0, len(a), len(a))]
            draws[bi] = np.subtract(metrics(boot, "fm"), metrics(boot, "st"))
        ci_e[h] = [
            (float(np.percentile(draws[:, k], 2.5)), float(np.percentile(draws[:, k], 97.5)))
            for k in range(4)
        ]

    # The panel-median column is the same quantity tab_year prints, so it takes
    # the stored interval rather than a second bootstrap of its own. Both used
    # B=2000 and seed 0, but tslimits_year_scale.py shares one default_rng(0)
    # across seven horizons while the loop above opens a fresh one per horizon,
    # so the two drew from different points in the stream and the two tables
    # disagreed in the second decimal. The point estimate was asserted; the
    # bounds were not.
    ypub = pd.read_csv(OUT / "year_scale.csv").set_index("horizon")
    for h in (8760, 17520):
        got = est[h]["Chronos-2"][0] - est[h]["structural"][0]
        assert abs(got - ypub.loc[h, "fm_tuned_minus_structural_pp"]) < 5e-3, (
            f"h={h}: panel-median gap {got:.4f} != published"
        )
        ci_e[h][0] = (
            float(ypub.loc[h, "fm_tuned_minus_structural_lo"]),
            float(ypub.loc[h, "fm_tuned_minus_structural_hi"]),
        )

    def fmt(v, k):
        return f"{v:.0f}" if k == 3 else f"{v:.2f}"

    rows_e = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        " & & panel-median & pooled-mean & load-weighted & MW MAE \\\\",
        " & & MdAPE & APE & MAPE & (MW) \\\\",
        "\\midrule",
    ]
    for h in (8760, 17520):
        lab = "1\\,yr" if h == 8760 else "2\\,yr"
        for i, who in enumerate(("structural", "Chronos-2")):
            v = est[h][who]
            rows_e.append(
                f"{lab if i == 0 else ''} & {who} & "
                + " & ".join(fmt(v[k], k) for k in range(4))
                + " \\\\"
            )
        d = np.subtract(est[h]["Chronos-2"], est[h]["structural"])
        rows_e.append(
            " & \\emph{gap} & "
            + " & ".join(
                f"\\emph{{{d[k]:+.2f}}}" if k < 3 else f"\\emph{{{d[k]:+.0f}}}" for k in range(4)
            )
            + " \\\\"
        )
        rows_e.append(
            " & \\emph{95\\% CI} & "
            + " & ".join(
                f"\\emph{{[{ci_e[h][k][0]:+.2f}, {ci_e[h][k][1]:+.2f}]}}"
                if k < 3
                else f"\\emph{{[{ci_e[h][k][0]:+.0f}, {ci_e[h][k][1]:+.0f}]}}"
                for k in range(4)
            )
            + " \\\\"
        )
        rows_e.append("\\midrule")
    rows_e.pop()
    rows_e += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_estimands.tex").write_text("\n".join(rows_e) + "\n")

    # ---- policy-by-metric, the full 2x2: selected under {mean, median} x scored by {mean, median}
    # Three cells come from the paired test (paired_estimand.csv); the fourth,
    # the median-selected policy scored under the pooled mean, from
    # tslimits_country_static.py, with its country-cluster interval.
    pe = pd.read_csv(OUT / "paired_estimand.csv").set_index(["fm", "horizon"])
    cs4 = pd.read_csv(OUT / "country_static.csv").set_index(["fm", "horizon"])
    assert "medsel_over_cc_pp" in cs4.columns, (
        "run tslimits_country_static.py first (median-selected cell)"
    )
    rows_p = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{selected under the pooled mean} & "
        "\\multicolumn{2}{c}{selected under the panel median} \\\\",
        "FM & $h$ & scored: mean & scored: median & scored: mean & scored: median \\\\",
        "\\midrule",
    ]
    cells = {k: [] for k in ("mm", "mM", "Mm", "MM")}
    for fm in ("Chronos-2-Uni-ZS", "TimesFM-2.5-Uni-ZS"):
        for h in (1, 24, 168, 720):
            r, c = pe.loc[(fm, h)], cs4.loc[(fm, h)]
            v = {
                "mm": r["gain_pooled_pp"],
                "mM": r["gain_median_samepolicy_pp"],
                "Mm": c["medsel_over_cc_pp"],
                "MM": r["gain_median_ownpolicy_pp"],
            }
            for k in cells:
                cells[k].append(float(v[k]))
            rows_p.append(
                f"{SHORT[fm]} & {h} & ${v['mm']:+.3f}$ & ${v['mM']:+.3f}$ & "
                f"${v['Mm']:+.3f}$ [{c['medsel_over_cc_lo']:+.3f}, {c['medsel_over_cc_hi']:+.3f}] "
                f"& ${v['MM']:+.3f}$ \\\\"
            )
    rows_p.append("\\midrule")
    med = {k: float(np.median(v)) for k, v in cells.items()}
    rows_p.append(
        f"\\emph{{median over cells}} & & ${med['mm']:+.3f}$ & ${med['mM']:+.3f}$ & "
        f"${med['Mm']:+.3f}$ & ${med['MM']:+.3f}$ \\\\"
    )
    rows_p += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_policy2x2.tex").write_text("\n".join(rows_p) + "\n")
    n_pos = int(sum(v > 0 for v in cells["Mm"]))
    print(
        f"[tab] wrote tab_policy2x2: median-selected policy under the pooled mean, median "
        f"{med['Mm']:+.3f} pp, positive in {n_pos}/8",
        flush=True,
    )
    n_excl = sum(lo > 0 for h in (8760, 17520) for lo, _ in ci_e[h])
    assert n_excl == 8, f"only {n_excl}/8 long-horizon intervals exclude zero"
    print(f"[tab] wrote tab_estimands; {n_excl}/8 intervals exclude zero", flush=True)

    print(
        f"[tab] wrote tab_year over {len(published_h)} horizons (h=26280 measured but excluded)",
        flush=True,
    )
    print(f"[tab] wrote tab_controls with {len(controls)} country controls", flush=True)
    print(
        f"[tab] by-year: positive {int((yr.frozen_gain_pp > 0).sum())}/48, "
        f"negative {int((yr.frozen_gain_pp < 0).sum())}, "
        f"zero {int((yr.frozen_gain_pp == 0).sum())}, "
        f"above {MARGIN_PP} in {int((yr.frozen_gain_pp > MARGIN_PP).sum())}",
        flush=True,
    )

    # gate discrimination and margin concentration. The body cites this table
    # for the top-1% share as well as the top-5%, so both columns are emitted.
    wd = pd.read_csv(OUT / "why_diagnostics.csv")
    wc = pd.read_csv(OUT / "why_concentration.csv")
    w = wd.merge(wc, on=["fm", "horizon"], validate="one_to_one")
    assert len(w) == 8, f"why tables have {len(w)} of 8 cells"

    def why_row(r: pd.Series) -> str:
        return (
            f" & {int(r.horizon)} & {r.auc:.3f} & {r.spearman_score_vs_win:.3f}"
            f" & {r.spearman_score_vs_margin:.3f}"
            f" & {100 * r.top1pct_share_of_gain:.0f}\\% & {100 * r.top5pct_share_of_gain:.0f}\\%"
            f" & {r.panel_median_gap_closed:.3f} & {r.pooled_mean_gap_closed:.3f}"
            f" & {r.country_mean_gap_closed:.3f} \\\\"
        )

    why = "\n".join(
        [
            "\\begin{tabular}{llrrrrrrrr}",
            "\\toprule",
            "& & & \\multicolumn{2}{c}{Spearman ($\\rho$)}"
            " & \\multicolumn{2}{c}{share of gain held by}"
            " & \\multicolumn{3}{c}{gap closed under} \\\\",
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}",
            "FM & $h$ & AUC & vs.\\ win & vs.\\ margin & top 1\\% & top 5\\%"
            " & median & pooled & per-cc \\\\",
            "\\midrule",
            *_rows(w, why_row),
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (OUT / "tab_why.tex").write_text(why + "\n")
    print(
        f"[tab] concentration: top 1% holds "
        f"{100 * wc.top1pct_share_of_gain.min():.0f}-{100 * wc.top1pct_share_of_gain.max():.0f}%, "
        f"top 5% holds "
        f"{100 * wc.top5pct_share_of_gain.min():.0f}-{100 * wc.top5pct_share_of_gain.max():.0f}%",
        flush=True,
    )

    # The energy table was hand-typed and had already drifted from the JSON it
    # describes. Generated now, like every other table, and carrying the settled
    # loaded-resident baseline that replaced a between-window transient.
    erows: list[str] = []
    for model, short in (("chronos", "Chronos-2"), ("timesfm", "TimesFM-2.5")):
        ej = json.loads((OUT / f"energy_inference_{model}.json").read_text())
        # p95 latency is per-run, not aggregated, so take the median across
        # repeats here rather than printing nan from a key the aggregate lacks
        p95 = {}
        for run in ej["runs"]:
            if "latency_p95_s" in run:
                p95.setdefault((run["horizon"], run["batch_size"]), []).append(run["latency_p95_s"])
        erows.append(f"\\multirow{{8}}{{*}}{{{short}}}")
        for r in sorted(ej["aggregate"], key=lambda x: (x["horizon"], x["batch_size"])):
            q, res = r["marginal_j"], r["marginal_j_resident"]
            v = p95.get((r["horizon"], r["batch_size"]), [])
            assert v, f"no p95 latency for {model} h={r['horizon']} b={r['batch_size']}"
            erows.append(
                f" & {r['horizon']} / {r['batch_size']}"
                f" & {q['median']:.3f} [{q['q25']:.3f}, {q['q75']:.3f}]"
                f" & {res['median']:.3f} & {r['provisioned_j']['median']:.3f}"
                f" & {r['latency_p50_s']['median']:.3f}"
                f" & {sorted(v)[len(v) // 2]:.3f}"
                " \\\\"
            )
        erows.append("\\midrule")
    # the cheap side, for scale: CPU-only, and its joules are not measurable here
    erows.append(
        "structural & any & n/a & n/a & n/a & \\multicolumn{2}{c}{0.00011 (1 CPU core)} \\\\"
    )
    ener = "\n".join(
        [
            "\\begin{tabular}{llrrrrr}",
            "\\toprule",
            "& & \\multicolumn{3}{c}{J / forecast} & \\multicolumn{2}{c}{latency (s)} \\\\",
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
            "model & $h$ / batch & quiet-idle & loaded-idle & gross & p50 & p95 \\\\",
            "\\midrule",
            *erows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (OUT / "tab_energy.tex").write_text(ener + "\n")
    _ech = json.loads((OUT / "energy_inference_chronos.json").read_text())
    _etf = json.loads((OUT / "energy_inference_timesfm.json").read_text())
    allagg = _ech["aggregate"] + _etf["aggregate"]
    ratios = [a["marginal_j"]["median"] / a["marginal_j_resident"]["median"] for a in allagg]
    print(
        f"[tab] energy: quiet-idle baseline {_ech['gpu_idle_preload']['median_w']:.2f}/"
        f"{_etf['gpu_idle_preload']['median_w']:.2f} W, loaded-idle "
        f"{_ech['gpu_idle_resident']['median_w']:.2f}/"
        f"{_etf['gpu_idle_resident']['median_w']:.2f} W; quiet/loaded ratio "
        f"{min(ratios):.2f}-{max(ratios):.2f}x; quiet range "
        f"{min(a['marginal_j']['median'] for a in allagg):.3f}-"
        f"{max(a['marginal_j']['median'] for a in allagg):.3f} J, gross "
        f"{min(a['provisioned_j']['median'] for a in allagg):.3f}-"
        f"{max(a['provisioned_j']['median'] for a in allagg):.3f} J",
        flush=True,
    )

    thr = "\n".join(
        [
            "\\begin{tabular}{llrrrr}",
            "\\toprule",
            "FM & $h$ & 2024 target & threshold $\\tau$ & realised 2025 & difference \\\\",
            "\\midrule",
            *_rows(m, thr_row),
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (OUT / "tab_threshold.tex").write_text(thr + "\n")

    # One body table rather than two: the page budget does not hold both, and a
    # reader comparing "router vs static" with "router vs country-static" should
    # not have to hold two floats in mind. Levels and share move to the caption.
    # The verdicts are computed against the COUNTRY-static baseline, not the
    # horizon-only one. Which baseline the practical verdict uses changes it:
    # Chronos h=168 clears the margin against horizon-static (lower bound 0.130)
    # and does not against country-static (0.081). Reporting the verdict against
    # the weaker baseline while printing both gains was ambiguous at best.
    def main_row(r: pd.Series) -> str:
        # The c-static LEVEL, not the word "FM"/"struct": without an error level
        # every pp gain in the paper floats with no denominator, and a reader
        # cannot tell a 3% relative improvement from a 30% one.
        #
        # And delta is measured against C-STATIC, the same baseline the accuracy
        # column uses. Against horizon-static it would be the resource price of a
        # gain the paper does not report: c-static already routes some countries
        # to the FM, so at TimesFM h=24 the router saves 29 points of FM calls,
        # not the 50 it saves against an always-structural reference.
        cc_fm_share = r.cc_pick_fm_n / r.n_countries
        delta = r.router_realized_frac - cc_fm_share
        # the headroom that matters is c-static to oracle, and the fraction of it
        # the router takes; both from headroom.csv so they cannot drift from it
        hr = _hr.loc[(_hr.model == r.fm) & (_hr.horizon == int(r.horizon))].iloc[0]
        return (
            f" & {int(r.horizon)} & {r.cc_static:.2f} & {hr.cstatic_minus_oracle:.3f}"
            f" & ${r.router_over_cc_pp:+.3f}$"
            f" [{r.router_over_cc_lo:.3f}, {r.router_over_cc_hi:.3f}]"
            f" & {100 * hr.headroom_captured:.0f}"
            f" & {100 * cc_fm_share:.0f} & {100 * r.router_realized_frac:.0f}"
            f" & ${100 * delta:+.0f}$"
            f" & {_stat(r.router_over_cc_lo)}"
            f" & {_prac(r.router_over_cc_lo, r.router_over_cc_hi)} \\\\"
        )

    _hr = pd.read_csv(OUT / "headroom.csv")
    mainb = "\n".join(
        [
            "\\begin{tabular}{llrrlrrrrcc}",
            "\\toprule",
            "FM & $h$ & c-static & headroom & gain [95\\% CI] & captured"
            " & \\multicolumn{2}{c}{FM share (\\%)} & $\\Delta$"
            " & excl.\\ 0 & vs. \\\\",
            " & & & & & (\\%) & c-static & router & & & 0.10\\,pp \\\\",
            "\\midrule",
            *_rows(m, main_row),
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (OUT / "tab_main.tex").write_text(mainb + "\n")

    gap = "\n".join(
        [
            "\\begin{tabular}{llrrlcc}",
            "\\toprule",
            "FM & $h$ & hindsight & gap (identity) & "
            "router $-$ hindsight [95\\% CI] & stat & prac \\\\",
            "\\midrule",
            *_rows(m, gap_row),
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    pol = "\n".join(
        [
            "\\begin{tabular}{llrrlrr}",
            "\\toprule",
            "FM & $h$ & c-static & c-static $-$ h-static & gain over c-static [95\\% CI]"
            " & share & esc.\\% \\\\",
            "\\midrule",
            *_rows(m, pol_row),
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (OUT / "tab_gap.tex").write_text(gap + "\n")
    (OUT / "tab_policies.tex").write_text(pol + "\n")

    print(gap)
    print()
    print(pol)
    print()
    print(
        f"[tab] router over static : median {m.frozen_enr_gain_pp.median():+.4f} pp, "
        f"stat + in {int((m.frozen_enr_lo > 0).sum())}/8, "
        f"prac > in {int((m.frozen_enr_lo > MARGIN_PP).sum())}/8 vs horizon-static, "
        f"{int((m.router_over_cc_lo > MARGIN_PP).sum())}/8 vs country-static",
        flush=True,
    )
    print(
        f"[tab] router over c-static: median {m.router_over_cc_pp.median():+.4f} pp, "
        f"stat + in {int(m.router_over_cc_sig.sum())}/8",
        flush=True,
    )
    print(
        f"[tab] country config explains "
        f"{m.cc_over_horizon_pp.sum() / m.router_over_horizon_pp.sum():.1%} of the gain; "
        f"max |target - realised| escalation "
        f"{abs(m.router_budget - m.router_realized_frac).max():.4f}",
        flush=True,
    )
    # The three anchor weather features are a retrospective reanalysis proxy, so
    # what the gate is worth without them is the honest number to show.
    _sh = {"Chronos-2-Uni-ZS": "Chronos-2", "TimesFM-2.5-Uni-ZS": "TimesFM-2.5"}
    nw = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "FM & $h$ & deployed gate & no anchor weather & difference \\\\",
        "\\midrule",
    ]
    for r in m.itertuples():
        nw.append(
            f"{_sh.get(str(r.fm), r.fm)} & {int(r.horizon)} & "
            f"${r.router_over_cc_pp:+.3f}$ & ${r.noweather_over_cc_pp:+.3f}$ & "
            f"${r.noweather_over_cc_pp - r.router_over_cc_pp:+.3f}$ \\\\"
        )
    nw += ["\\bottomrule", "\\end{tabular}", ""]
    (OUT / "tab_noweather.tex").write_text("\n".join(nw))
    print(
        f"[tab] wrote tab_noweather: median {m.noweather_over_cc_pp.median():+.4f} pp "
        f"vs deployed {m.router_over_cc_pp.median():+.4f}, positive in "
        f"{int((m.noweather_over_cc_pp > 0).sum())}/8, better in "
        f"{int((m.noweather_over_cc_pp > m.router_over_cc_pp).sum())}/8",
        flush=True,
    )
    # Selection with a priced FM call (tslimits_cost_selection.py). One row per
    # cell for the cost-penalised router; the plug-in rule enters as summary
    # rows only. A dagger marks an interval that covers zero.
    cs_all = pd.read_csv(OUT / "cost_selection.csv")
    lam_cols = (0.05, 0.1, 0.2, 0.5, 1.0)
    assert set(lam_cols) <= set(cs_all["lam"]), "cost_selection.csv lacks a tabulated price"

    def _gain(r: pd.Series) -> str:
        return f"${r.cost_gain_pp:+.3f}$" + ("" if r.verdict == "pos" else "$^{\\dagger}$")

    router = cs_all[cs_all["policy"] == "router"]
    cells = router[router["lam"] == 0.0][
        ["fm", "horizon", "breakeven_router_2024", "breakeven_cc_2024"]
    ].reset_index(drop=True)
    assert len(cells) == 8, f"cost_selection has {len(cells)} of 8 cells"

    def cost_row(r: pd.Series) -> str:
        g = router[(router["fm"] == r.fm) & (router["horizon"] == r.horizon)].set_index("lam")
        return (
            f" & {int(r.horizon)} & {r.breakeven_router_2024:.2f} & {r.breakeven_cc_2024:.2f} & "
            + " & ".join(_gain(g.loc[lam]) for lam in lam_cols)
            + " \\\\"
        )

    def _summary(key: str, label: str) -> list[str]:
        p = cs_all[cs_all["policy"] == key]
        assert len(p) == 8 * len(set(cs_all["lam"])), f"{key}: incomplete price grid"
        med = [f"${p[p['lam'] == lam]['cost_gain_pp'].median():+.3f}$" for lam in lam_cols]
        pos = [f"{int((p[p['lam'] == lam]['verdict'] == 'pos').sum())} of 8" for lam in lam_cols]
        return [
            f"\\multicolumn{{4}}{{l}}{{{label}, median over cells}} & " + " & ".join(med) + " \\\\",
            f"\\multicolumn{{4}}{{l}}{{{label}, intervals above zero}} & "
            + " & ".join(pos)
            + " \\\\",
        ]

    cost = "\n".join(
        [
            "\\begin{tabular}{llrrrrrrr}",
            "\\toprule",
            " & & \\multicolumn{2}{c}{break-even price (2024)}"
            " & \\multicolumn{5}{c}{cost-adjusted gain over c-static$^{\\lambda}$"
            " at price $\\lambda$}"
            " \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-9}",
            "FM & $h$ & router & c-static & "
            + " & ".join(f"{lam:g}" for lam in lam_cols)
            + " \\\\",
            "\\midrule",
            *_rows(cells, cost_row),
            "\\midrule",
            *_summary("router", "router"),
            *_summary("plugin", "plug-in rule"),
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (OUT / "tab_costsel.tex").write_text(cost + "\n")
    print(f"[tab] wrote tab_costsel from {len(cs_all)} priced rows", flush=True)

    print(f"[tab] wrote tab_gap, tab_policies, tab_paired, tab_threshold under {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
