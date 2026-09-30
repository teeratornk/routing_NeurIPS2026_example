"""TS-LIMITS: selection-aware country bootstrap of the metric-specific deployment difference.

The paired-estimand intervals (tslimits_paired_estimand.py) condition on the two
thresholds selected on the observed 2024 panel. This script re-selects them
inside every replicate. Each draw resamples the 19 countries with replacement
ONCE, and that multiset is used for the 2024 selection and for the 2025
scoring, under both metrics, so the interval carries the variability of
threshold selection as well as of scoring.

What is and is not refit. The gate is never refit: its scores are bit-stable
across refits and 2024 never enters the fit, so a resampled 2024 panel changes
only the budget (a panel-level argmin) and the threshold (a quantile of the
resampled scores). The c-static choices are invariant under a country
resample, since each country's choice depends on its own 2024 rows alone; the
script asserts that rather than assuming it.

Under the SAME draws the conditional variant (thresholds frozen at the observed
selection) is recomputed. The random stream is the one paired_estimand uses,
one generator across cells in the same order, so the conditional per-cell
intervals reproduce the published ones to the last digit and the only change
between the two columns is selection.

Out: reports/tslimits/selection_bootstrap.csv, selection_bootstrap_aggregate.csv,
     tab_selection.tex
Run: OMP_NUM_THREADS=48 PYTHONPATH=src .venv/bin/python scripts/tslimits_selection_bootstrap.py
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
SHORT = {"Chronos-2-Uni-ZS": "Chronos-2", "TimesFM-2.5-Uni-ZS": "TimesFM-2.5"}


def _mod(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_BS = _mod("tslimits_bootstrap")
BUDGETS = tuple(_BS.BUDGETS)


def _ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


class Cell:
    """Per-country structures of one (FM, horizon) cell, scores frozen."""

    def __init__(
        self,
        v_h: pd.DataFrame,
        t_h: pd.DataFrame,
        s_val: np.ndarray,
        s_test: np.ndarray,
    ) -> None:
        cs = sorted(v_h["country"].unique())
        assert cs == sorted(t_h["country"].unique()), "country sets differ across windows"
        self.n = len(cs)
        self.vi = [np.flatnonzero((v_h["country"] == c).to_numpy()) for c in cs]
        self.ti = [np.flatnonzero((t_h["country"] == c).to_numpy()) for c in cs]
        self.s_val, self.s_test = s_val, s_test
        self.v_fm, self.v_st = v_h["ape_fm"].to_numpy(), v_h["ape_st"].to_numpy()
        self.t_fm, self.t_st = t_h["ape_fm"].to_numpy(), t_h["ape_st"].to_numpy()
        # c-static choices, one per country, from that country's own 2024 rows
        use_fm = np.array([self.v_fm[i].mean() < self.v_st[i].mean() for i in self.vi])
        use_med = np.array([np.median(self.v_fm[i]) < np.median(self.v_st[i]) for i in self.vi])
        t_use = np.zeros(len(t_h), dtype=bool)
        t_use_med = np.zeros(len(t_h), dtype=bool)
        for k in range(self.n):
            t_use[self.ti[k]] = use_fm[k]
            t_use_med[self.ti[k]] = use_med[k]
        self.cc = np.where(t_use, self.t_fm, self.t_st)
        self.cc_med = np.where(t_use_med, self.t_fm, self.t_st)
        self.med_cc = self._t_medians(self.cc)
        self.med_cc_med = self._t_medians(self.cc_med)

    def _t_medians(self, served: np.ndarray) -> np.ndarray:
        return np.array([np.median(served[i]) for i in self.ti])

    def _v_medians(self, served: np.ndarray) -> np.ndarray:
        return np.array([np.median(served[i]) for i in self.vi])

    def select(self, idx: np.ndarray) -> tuple[float, float, float, float]:
        """Budgets and thresholds re-selected on the 2024 multiset `idx`.

        The threshold is a quantile of the resampled score vector, as the
        deployed rule takes it over the observed panel. The tie-break is the
        deployed one: smallest loss, then smallest budget.
        """
        rows = np.concatenate([self.vi[k] for k in idx])
        sv = self.s_val[rows]
        best_pool: tuple[float, float] | None = None
        best_med: tuple[float, float] | None = None
        for b in BUDGETS:
            tau = frozen_threshold(sv, b)
            served = np.where(escalate_above(self.s_val, tau), self.v_fm, self.v_st)
            pooled = float(served[rows].mean())
            med = float(np.median(self._v_medians(served)[idx]))
            if best_pool is None or (pooled, b) < best_pool:
                best_pool = (pooled, b)
            if best_med is None or (med, b) < best_med:
                best_med = (med, b)
        assert best_pool is not None and best_med is not None
        b_pool, b_med = best_pool[1], best_med[1]
        return b_pool, b_med, frozen_threshold(sv, b_pool), frozen_threshold(sv, b_med)

    def score(self, idx: np.ndarray, tau_pool: float, tau_med: float) -> tuple[float, float, float]:
        """(gain_pool, gain_med_own, gain_med_sym) on the 2025 multiset `idx`."""
        rows = np.concatenate([self.ti[k] for k in idx])
        pol_pool = np.where(escalate_above(self.s_test, tau_pool), self.t_fm, self.t_st)
        pol_med = np.where(escalate_above(self.s_test, tau_med), self.t_fm, self.t_st)
        gain_pool = float(self.cc[rows].mean() - pol_pool[rows].mean())
        med_pol = self._t_medians(pol_med)[idx]
        gain_med_own = float(np.median(self.med_cc[idx]) - np.median(med_pol))
        gain_med_sym = float(np.median(self.med_cc_med[idx]) - np.median(med_pol))
        return gain_pool, gain_med_own, gain_med_sym


def main() -> int:
    rng = np.random.default_rng(SEED)
    cols = list(_BS.ENRICHED)
    pub = pd.read_csv(OUT / f"paired_estimand{_BS.GRID_SFX}.csv").set_index(["fm", "horizon"])
    rows: list[dict[str, Any]] = []
    draws_by_cell: list[dict[str, np.ndarray]] = []

    for fm_label, (test_pat, dev_pat) in _BS.FM_PATTERNS.items():
        dev = _BS._paired("dev", dev_pat)
        test = _BS._paired("2025", test_pat)
        fit = dev[dev["test_year"].isin(_BS.FIT_YEARS)]
        val = dev[dev["test_year"].isin(_BS.VAL_YEARS)]
        for h in sorted(test["horizon"].unique()):
            f_h = fit[fit["horizon"] == h].reset_index(drop=True)
            v_h = val[val["horizon"] == h].reset_index(drop=True)
            t_h = test[test["horizon"] == h].reset_index(drop=True)
            reg = _BS._reg(cols).fit(f_h[cols], f_h["ape_st"] - f_h["ape_fm"])
            cell = Cell(
                v_h,
                t_h,
                np.asarray(reg.predict(v_h[cols]), dtype=float),
                np.asarray(reg.predict(t_h[cols]), dtype=float),
            )
            ident = np.arange(cell.n)
            b_pool0, b_med0, tau_pool0, tau_med0 = cell.select(ident)
            p = pub.loc[(fm_label, int(h))]
            # determinism gate: the identity multiset must reproduce the deployed selection
            assert np.isclose(b_pool0, p.budget_pooled) and np.isclose(b_med0, p.budget_median), (
                f"{fm_label} h={h}: reselected {b_pool0}/{b_med0} vs "
                f"deployed {p.budget_pooled}/{p.budget_median}"
            )
            gp0, go0, gs0 = cell.score(ident, tau_pool0, tau_med0)
            for got, want, tag in (
                (gp0, p.gain_pooled_pp, "gain_pooled"),
                (go0, p.gain_median_ownpolicy_pp, "gain_median_own"),
                (gs0, p.gain_median_symmetric_pp, "gain_median_sym"),
            ):
                assert abs(got - want) < 1e-4 + 1e-9, f"{fm_label} h={h}: {tag} {got} vs {want}"

            cond = {k: np.empty(B) for k in ("d_total", "d_total_sym", "g_pool", "g_own")}
            sel = {k: np.empty(B) for k in ("d_total", "d_total_sym", "g_pool", "g_own")}
            bp, bm = np.empty(B), np.empty(B)
            for b in range(B):
                idx = rng.integers(0, cell.n, cell.n)
                gp, go, gs = cell.score(idx, tau_pool0, tau_med0)
                cond["g_pool"][b], cond["g_own"][b] = gp, go
                cond["d_total"][b], cond["d_total_sym"][b] = gp - go, gp - gs
                bp[b], bm[b], tp, tm = cell.select(idx)
                gp, go, gs = cell.score(idx, tp, tm)
                sel["g_pool"][b], sel["g_own"][b] = gp, go
                sel["d_total"][b], sel["d_total_sym"][b] = gp - go, gp - gs
            # the conditional column must reproduce the published interval exactly
            for tag, lo_c, hi_c in (
                ("d_total", "diff_total_lo", "diff_total_hi"),
                ("d_total_sym", "diff_total_symmetric_lo", "diff_total_symmetric_hi"),
            ):
                lo, hi = _ci(cond[tag])
                assert abs(lo - p[lo_c]) < 1e-4 + 1e-9 and abs(hi - p[hi_c]) < 1e-4 + 1e-9, (
                    f"{fm_label} h={h}: conditional {tag} [{lo:.4f}, {hi:.4f}] vs "
                    f"published [{p[lo_c]}, {p[hi_c]}]"
                )
            draws_by_cell.append({f"cond_{k}": v.copy() for k, v in cond.items()})
            draws_by_cell[-1].update({f"sel_{k}": v.copy() for k, v in sel.items()})
            r: dict[str, Any] = {
                "fm": fm_label,
                "horizon": int(h),
                "budget_pooled": b_pool0,
                "budget_median": b_med0,
                "diff_total_pp": round(gp0 - go0, 4),
                "diff_total_symmetric_pp": round(gp0 - gs0, 4),
                "gain_pooled_pp": round(gp0, 4),
                "gain_median_ownpolicy_pp": round(go0, 4),
                "resel_pooled_share_same": float(np.mean(np.isclose(bp, b_pool0))),
                "resel_pooled_q25": float(np.percentile(bp, 25)),
                "resel_pooled_q75": float(np.percentile(bp, 75)),
                "resel_median_share_same": float(np.mean(np.isclose(bm, b_med0))),
                "resel_median_q25": float(np.percentile(bm, 25)),
                "resel_median_q75": float(np.percentile(bm, 75)),
                "resel_median_share_zero": float(np.mean(np.isclose(bm, 0.0))),
            }
            for k in ("d_total", "d_total_sym", "g_pool", "g_own"):
                for kind, arr in (("cond", cond[k]), ("sel", sel[k])):
                    lo, hi = _ci(arr)
                    r[f"{k}_{kind}_lo"], r[f"{k}_{kind}_hi"] = round(lo, 4), round(hi, 4)
                    r[f"{k}_{kind}_sig"] = bool(lo > 0)
            rows.append(r)
            print(
                f"[sel] {SHORT[fm_label]:11s} h={int(h):<4} b={b_pool0:.2f}/{b_med0:.2f} "
                f"d_total {gp0 - go0:+.3f} cond [{r['d_total_cond_lo']:+.3f},"
                f"{r['d_total_cond_hi']:+.3f}] sel [{r['d_total_sel_lo']:+.3f},"
                f"{r['d_total_sel_hi']:+.3f}]{'*' if r['d_total_sel_sig'] else ' '} | "
                f"sym {gp0 - gs0:+.3f} sel [{r['d_total_sym_sel_lo']:+.3f},"
                f"{r['d_total_sym_sel_hi']:+.3f}]{'*' if r['d_total_sym_sel_sig'] else ' '} | "
                f"reselected same {r['resel_pooled_share_same']:.2f}/"
                f"{r['resel_median_share_same']:.2f}",
                flush=True,
            )

    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"selection_bootstrap{_BS.GRID_SFX}.csv", index=False)

    agg_rows = []
    agg_ci: dict[str, dict[str, tuple[float, float]]] = {}
    for tag, col in (
        ("d_total", "diff_total_pp"),
        ("d_total_sym", "diff_total_symmetric_pp"),
        ("g_pool", "gain_pooled_pp"),
        ("g_own", "gain_median_ownpolicy_pp"),
    ):
        point = float(np.median(d[col]))
        agg_ci[tag] = {}
        r = {"contrast": tag, "column": col, "median_across_cells_pp": round(point, 4)}
        for kind in ("cond", "sel"):
            mat = np.vstack([c[f"{kind}_{tag}"] for c in draws_by_cell])
            assert mat.shape == (8, B), mat.shape
            lo, hi = _ci(np.median(mat, axis=0))
            agg_ci[tag][kind] = (lo, hi)
            r[f"{kind}_lo"], r[f"{kind}_hi"], r[f"{kind}_sig"] = round(lo, 4), round(hi, 4), lo > 0
        r["n_cells"], r["n_draws"] = 8, B
        agg_rows.append(r)
        print(
            f"[sel] AGGREGATE {tag}: median across 8 cells {point:+.4f} pp; conditional "
            f"[{r['cond_lo']:+.4f}, {r['cond_hi']:+.4f}], selection-aware "
            f"[{r['sel_lo']:+.4f}, {r['sel_hi']:+.4f}]",
            flush=True,
        )
    pd.DataFrame(agg_rows).to_csv(
        OUT / f"selection_bootstrap_aggregate{_BS.GRID_SFX}.csv", index=False
    )

    def iv(lo: float, hi: float) -> str:
        return f"[{lo:+.2f}, {hi:+.2f}]"

    def iv3(lo: float, hi: float) -> str:
        return f"[{lo:+.3f}, {hi:+.3f}]"

    tex = [
        "\\begin{tabular}{llcrllrl}",
        "\\toprule",
        " & & budget & \\multicolumn{3}{c}{mean-selected baseline}"
        " & \\multicolumn{2}{c}{metric-matched baseline} \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}",
        "FM & $h$ & mean$|$median & diff. & conditional & selection-aware"
        " & diff. & selection-aware \\\\",
        "\\midrule",
    ]
    for r in d.itertuples():
        tex.append(
            f"{SHORT[r.fm]} & {r.horizon} & {r.budget_pooled:.2f}$|${r.budget_median:.2f}"
            f" & ${r.diff_total_pp:+.2f}$ & {iv(r.d_total_cond_lo, r.d_total_cond_hi)}"
            f" & {iv(r.d_total_sel_lo, r.d_total_sel_hi)}"
            f" & ${r.diff_total_symmetric_pp:+.2f}$"
            f" & {iv(r.d_total_sym_sel_lo, r.d_total_sym_sel_hi)} \\\\"
        )
    a = {r["contrast"]: r for r in agg_rows}
    tex += [
        "\\midrule",
        f"\\multicolumn{{2}}{{l}}{{median over cells}} & "
        f" & ${a['d_total']['median_across_cells_pp']:+.3f}$"
        f" & {iv3(*agg_ci['d_total']['cond'])} & {iv3(*agg_ci['d_total']['sel'])}"
        f" & ${a['d_total_sym']['median_across_cells_pp']:+.3f}$"
        f" & {iv3(*agg_ci['d_total_sym']['sel'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (OUT / "tab_selection.tex").write_text("\n".join(tex) + "\n")
    print(
        f"[sel] selection-aware d_total excludes zero in {int(d.d_total_sel_sig.sum())}/8 cells "
        f"(conditional {int(d.d_total_cond_sig.sum())}/8); symmetric "
        f"{int(d.d_total_sym_sel_sig.sum())}/8 (conditional {int(d.d_total_sym_cond_sig.sum())}/8)",
        flush=True,
    )
    print(f"[sel] wrote {OUT}/selection_bootstrap.csv and tab_selection.tex", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
