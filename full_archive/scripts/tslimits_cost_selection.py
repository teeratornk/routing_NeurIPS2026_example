"""TS-LIMITS: selecting the routing policy with a priced FM call.

Every deployed policy is selected with lambda = 0, on accuracy alone, and the
serving cost is measured afterwards. A reviewer's main concern was that this
leaves the paper's own question, whether a call is worth its cost, unanswered.
This script selects with a price instead: each FM call costs `lam` percentage
points of APE, the scalar lambda of Proposition 1, and every policy is chosen on
2024 and scored once on 2025, as the deployed ones are.

Design, fixed before any outcome was computed:

    prices           LAMBDAS = 0, 0.05, 0.1, 0.2, 0.5, 1.0 pp per call
    c-static^lam     a country takes the FM iff its 2024 mean ape_fm + lam is
                     below its 2024 mean ape_st; at lam = 0 this IS c-static
    router^lam       the deployed gate, with the budget chosen on 2024 to
                     minimise pooled-mean APE + lam * (2024 escalated share),
                     ties to the smaller budget; frozen threshold as deployed.
                     At lam = 0 this IS the deployed router
    plug-in^lam      escalate iff the gate's score, a plug-in estimate of
                     E[D | X] in pp, exceeds lam: the Bayes rule of eq:lambda
                     read literally, with no selection step
    estimand         cost-adjusted pooled mean on 2025,
                     R^lam(pi) = mean APE(pi) + lam * FM share(pi)
    contrast         R^lam(c-static^lam) - R^lam(policy), positive = policy wins
    intervals        95% country-cluster bootstrap, B = 2000, conditional on the
                     selected thresholds; one default_rng(0), the cells in the
                     order of country_static.csv and one draw of 19 countries per
                     replicate, so lam = 0 reproduces router_over_cc exactly
    break-even       the 2024 price at and above which a policy never calls the
                     FM: max_b [R(0) - R(b)] / f(b) for the router, and the
                     largest per-country 2024 mean D for c-static

The price is a constant per call. Appendix sec:energyprotocol shows the real
cost depends on batch occupancy, so this is Proposition 1's simplification, not
a claim about any operator's cost.

Two stages:

    scores   (lab only) refit each deployed gate, score its 2024 selection year,
             assert the 2025 scores equal the served record, and write the
             2024 per-request record: model, horizon, country, anchor_t, score,
             ape_st, ape_fm. Absolute percentage errors and scores only.
    select   reads that record and served_per_request.parquet and nothing else,
             so it runs unchanged on the public archive.

Out: reports/tslimits/selection_2024_per_request.parquet (stage scores)
     reports/tslimits/cost_selection.csv, cost_selection_aggregate.csv (select)
Run: PYTHONPATH=src OMP_NUM_THREADS=8 .venv/bin/python scripts/tslimits_cost_selection.py
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from load_forecast.eval.routing import escalate_above, frozen_threshold, pooled_mean
except ImportError:  # the public archive ships the same helpers as src/routing.py
    from routing import escalate_above, frozen_threshold, pooled_mean  # type: ignore[no-redef]

OUT = Path("reports/tslimits")
LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
# the deployed candidate grid, the same expression as tslimits_bootstrap.BUDGETS
BUDGETS = tuple(round(0.05 * i, 4) for i in range(21))
B = 2000
SEED = 0
KEY = ["country", "horizon", "anchor_t"]


def _mod(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_selection_record(served_path: Path, out_path: Path) -> None:
    """Refit the deployed gates, score 2024, prove the 2025 scores, write the record."""
    bs = _mod("tslimits_bootstrap")
    assert bs.GATE_LEARNER == "hgb" and bs.FAMILY_SFX == "" and bs.GRID_SFX == "", (
        "stage 'scores' describes the deployed gate family only"
    )
    assert tuple(bs.BUDGETS) == BUDGETS, "budget grid differs from the deployed one"
    served = pd.read_parquet(served_path)
    served["anchor_t"] = pd.to_datetime(served["anchor_t"])
    enr = list(bs.ENRICHED)
    frames = []
    for fm_label, (test_pat, dev_pat) in bs.FM_PATTERNS.items():
        dev = bs._paired("dev", dev_pat)
        test = bs._paired("2025", test_pat)
        fit = dev[dev["test_year"].isin(bs.FIT_YEARS)]
        val = dev[dev["test_year"].isin(bs.VAL_YEARS)]
        for h in sorted(test["horizon"].unique()):
            f = fit[fit["horizon"] == h].reset_index(drop=True)
            v = val[val["horizon"] == h].reset_index(drop=True)
            t = test[test["horizon"] == h].reset_index(drop=True)
            reg = bs._reg(enr).fit(f[enr], f["ape_st"] - f["ape_fm"])
            sv = np.asarray(reg.predict(v[enr]), dtype=float)
            stt = np.asarray(reg.predict(t[enr]), dtype=float)
            s = served[(served["model"] == fm_label) & (served["horizon"] == h)].reset_index(
                drop=True
            )
            assert s[["country", "anchor_t"]].equals(t[["country", "anchor_t"]]), (
                f"{fm_label} h={h}: served record is not in request order"
            )
            # Bit-identical in most cells; in a few the refit differs from the
            # deployed score by one ulp on scores within ~1e-17 of zero, the same
            # under 1 and 8 threads. The stage-2 asserts at lam = 0 then check
            # that the 2024 threshold and every 2025 decision are the deployed ones.
            gap = float(np.max(np.abs(s["score"].to_numpy() - stt)))
            assert gap < 1e-12, (
                f"{fm_label} h={h}: refitted gate misses the deployed 2025 scores by {gap:.3g}"
            )
            frames.append(
                pd.DataFrame(
                    {
                        "model": fm_label,
                        "horizon": int(h),
                        "country": v["country"].to_numpy(),
                        "anchor_t": v["anchor_t"].to_numpy(),
                        "score": sv,
                        "ape_st": v["ape_st"].to_numpy(),
                        "ape_fm": v["ape_fm"].to_numpy(),
                    }
                )
            )
            print(f"[scores] {fm_label} h={h}: {len(v)} selection-year requests", flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)
    print(f"[scores] wrote {out_path}", flush=True)


def select_router(v: pd.DataFrame, lam: float) -> tuple[float, float, float]:
    """Budget, frozen threshold and 2024 share minimising 2024 mean APE + lam * share."""
    sv = v["score"].to_numpy()
    st = v["ape_st"].to_numpy()
    fm = v["ape_fm"].to_numpy()
    best = min(
        (
            pooled_mean(
                pd.Series(np.where(escalate_above(sv, frozen_threshold(sv, q)), fm, st)),
                v["country"],
            )
            + lam * float(escalate_above(sv, frozen_threshold(sv, q)).mean()),
            q,
        )
        for q in BUDGETS
    )
    b = best[1]
    tau = frozen_threshold(sv, b)
    return b, tau, float(escalate_above(sv, tau).mean())


def breakeven_router(v: pd.DataFrame) -> float:
    """The 2024 price at and above which the router's selected budget is zero."""
    sv = v["score"].to_numpy()
    st = v["ape_st"].to_numpy()
    fm = v["ape_fm"].to_numpy()
    r0 = float(np.mean(st))
    worth = [0.0]
    for q in BUDGETS[1:]:
        m = escalate_above(sv, frozen_threshold(sv, q))
        if m.any():
            worth.append((r0 - float(np.mean(np.where(m, fm, st)))) / float(m.mean()))
    return max(worth)


def select_cc(v: pd.DataFrame, lam: float) -> pd.Series:
    """Per country: the FM iff its 2024 mean APE plus the price beats the structural mean."""
    m = v.groupby("country")[["ape_fm", "ape_st"]].mean()
    return (m["ape_fm"] + lam) < m["ape_st"]


def _verdict(lo: float, hi: float) -> str:
    return "pos" if lo > 0 else ("neg" if hi < 0 else "open")


def run_selection(
    selection_path: Path, served_path: Path, country_static_path: Path, out_dir: Path
) -> None:
    sel = pd.read_parquet(selection_path)
    served = pd.read_parquet(served_path)
    served["anchor_t"] = pd.to_datetime(served["anchor_t"])
    cs = pd.read_csv(country_static_path)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    for _, cell in cs[["fm", "horizon"]].iterrows():
        fm_label, h = str(cell["fm"]), int(cell["horizon"])
        dep = cs[(cs["fm"] == fm_label) & (cs["horizon"] == h)].iloc[0]
        v = sel[(sel["model"] == fm_label) & (sel["horizon"] == h)].reset_index(drop=True)
        t = served[(served["model"] == fm_label) & (served["horizon"] == h)].reset_index(drop=True)
        st = t["st"].to_numpy()
        fm = t["fm"].to_numpy()
        score = t["score"].to_numpy()
        be_router = breakeven_router(v)
        dmean = v.groupby("country")[["ape_st", "ape_fm"]].mean()
        be_cc = max(0.0, float((dmean["ape_st"] - dmean["ape_fm"]).max()))

        cols: dict[str, np.ndarray] = {}
        meta: dict[float, dict[str, Any]] = {}
        prev_share, prev_ncc = 2.0, 10**9
        for lam in LAMBDAS:
            pick = select_cc(v, lam)
            use_cc = t["country"].map(pick).to_numpy()
            assert not pd.isna(use_cc).any(), f"{fm_label} h={h}: a 2025 country has no 2024 rule"
            use_cc = use_cc.astype(bool)
            b, tau, share24 = select_router(v, lam)
            m_router = escalate_above(score, tau)
            m_plugin = escalate_above(score, lam)
            if lam == 0.0:
                # the priced selection at a zero price must be the deployed one
                assert np.array_equal(np.where(use_cc, fm, st), t["cc_static"].to_numpy())
                assert np.array_equal(np.where(m_router, fm, st), t["router_cc"].to_numpy())
                assert abs(b - float(dep["router_budget"])) < 1e-9
                assert round(tau, 6) == float(dep["router_threshold"])
            assert share24 <= prev_share + 1e-12 and int(pick.sum()) <= prev_ncc, (
                f"{fm_label} h={h}: a higher price selected more FM calls"
            )
            prev_share, prev_ncc = share24, int(pick.sum())
            for name, mask in (("cc", use_cc), ("router", m_router), ("plugin", m_plugin)):
                served_ape = np.where(mask, fm, st)
                cols[f"{name}|{lam}|cost"] = served_ape + lam * mask
                cols[f"{name}|{lam}|ape"] = served_ape
                cols[f"{name}|{lam}|share"] = mask.astype(float)
            meta[lam] = {
                "budget": b,
                "threshold": tau,
                "share_2024": share24,
                "cc_fm_countries": int(pick.sum()),
            }

        names = list(cols)
        frame = pd.DataFrame({"c": t["country"].to_numpy(), **cols})
        g = frame.groupby("c")
        sums = g[names].sum().to_numpy()
        cnt = g[names[0]].size().to_numpy().astype(float)

        def panel(ix: np.ndarray, s: np.ndarray = sums, n: np.ndarray = cnt) -> np.ndarray:
            return s[ix].sum(axis=0) / n[ix].sum()

        s0 = panel(np.arange(sums.shape[0]))
        ci = {k: names.index(k) for k in names}
        contrasts = {
            (lam, pol): (ci[f"cc|{lam}|cost"], ci[f"{pol}|{lam}|cost"])
            for lam in LAMBDAS
            for pol in ("router", "plugin")
        }
        draws = {k: np.empty(B) for k in contrasts}
        for bi in range(B):
            ix = rng.integers(0, sums.shape[0], sums.shape[0])
            q = panel(ix)
            for k, (a, b_) in contrasts.items():
                draws[k][bi] = q[a] - q[b_]

        for (lam, pol), (a, b_) in contrasts.items():
            pp = float(s0[a] - s0[b_])
            lo, hi = (
                float(np.percentile(draws[(lam, pol)], 2.5)),
                float(np.percentile(draws[(lam, pol)], 97.5)),
            )
            if lam == 0.0 and pol == "router":
                for got, col in ((pp, "pp"), (lo, "lo"), (hi, "hi")):
                    assert round(got, 4) == float(dep[f"router_over_cc_{col}"]), (
                        f"{fm_label} h={h}: lam=0 does not reproduce router_over_cc_{col}"
                    )
            is_router = pol == "router"
            rows.append(
                {
                    "fm": fm_label,
                    "horizon": h,
                    "lam": lam,
                    "policy": pol,
                    "budget": meta[lam]["budget"] if is_router else np.nan,
                    "threshold": round(meta[lam]["threshold"] if is_router else lam, 6),
                    "share_2024": round(meta[lam]["share_2024"], 6) if is_router else np.nan,
                    "share_2025": round(float(s0[ci[f"{pol}|{lam}|share"]]), 6),
                    "cc_share_2025": round(float(s0[ci[f"cc|{lam}|share"]]), 6),
                    "cc_fm_countries": meta[lam]["cc_fm_countries"],
                    "ape_gain_pp": round(
                        float(s0[ci[f"cc|{lam}|ape"]] - s0[ci[f"{pol}|{lam}|ape"]]), 6
                    ),
                    "cost_gain_pp": round(pp, 6),
                    "cost_gain_lo": round(lo, 6),
                    "cost_gain_hi": round(hi, 6),
                    "verdict": _verdict(lo, hi),
                    "breakeven_router_2024": round(be_router, 6),
                    "breakeven_cc_2024": round(be_cc, 6),
                }
            )
        r = [x for x in rows if (x["fm"], x["horizon"], x["policy"]) == (fm_label, h, "router")]
        print(
            f"[select] {fm_label} h={h}: break-even router {be_router:.3f} cc {be_cc:.3f}; "
            + " ".join(f"{x['lam']:g}:{x['cost_gain_pp']:+.3f}({x['verdict']})" for x in r),
            flush=True,
        )

    d = pd.DataFrame(rows)
    d.to_csv(out_dir / "cost_selection.csv", index=False)
    agg = (
        d.groupby(["policy", "lam"])
        .agg(
            median_cost_gain_pp=("cost_gain_pp", "median"),
            n_pos=("verdict", lambda x: int((x == "pos").sum())),
            n_neg=("verdict", lambda x: int((x == "neg").sum())),
            median_share_2025=("share_2025", "median"),
            median_cc_share_2025=("cc_share_2025", "median"),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / "cost_selection_aggregate.csv", index=False)
    print(agg.to_string(index=False), flush=True)
    print(f"[select] wrote {out_dir / 'cost_selection.csv'} and the aggregate", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=("scores", "select", "all"), default="all")
    p.add_argument("--selection", type=Path, default=OUT / "selection_2024_per_request.parquet")
    p.add_argument("--served", type=Path, default=OUT / "served_per_request.parquet")
    p.add_argument("--country-static", type=Path, default=OUT / "country_static.csv")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    if a.stage in ("scores", "all"):
        build_selection_record(a.served, a.selection)
    if a.stage in ("select", "all"):
        run_selection(a.selection, a.served, a.country_static, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
