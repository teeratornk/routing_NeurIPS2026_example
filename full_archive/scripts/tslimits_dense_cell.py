"""Dense-grid sensitivity, one cell: the frozen Chronos-2 h=24 policies on every hour of 2025.

The published grid scores one origin per country per day with a rotating
anchor hour. This scores all 8,760 hours (stride 1) for Chronos-2 at h=24 and
applies the SAME frozen objects: the deployed gate fit on the stride-24
development records, its 2024-frozen threshold, and the 2024 c-static rule.
Nothing is re-selected on the dense grid; it only asks whether the routed
gain depends on which hours the published grid happened to sample.

Out:  reports/tslimits/dense_cell.csv
Run:  PYTHONPATH=src .venv/bin/python scripts/tslimits_dense_cell.py
"""

from __future__ import annotations

import glob
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from load_forecast.eval.routing import ESTIMANDS, escalate_above, frozen_threshold

OUT = Path("reports/tslimits")
FM = "Chronos-2-Uni-ZS"
H = 24
SEED, B = 0, 2000


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


def _ci(a: np.ndarray, b: np.ndarray, cc: np.ndarray, rng) -> tuple[float, float, float]:
    frame = pd.DataFrame({"c": cc, "a": a, "b": b})
    g = frame.groupby("c")
    sums = g[["a", "b"]].sum().to_numpy()
    cnt = g["a"].size().to_numpy().astype(float)
    point = (sums[:, 0].sum() - sums[:, 1].sum()) / cnt.sum()
    d = np.empty(B)
    for bi in range(B):
        ix = rng.integers(0, len(cnt), len(cnt))
        d[bi] = (sums[ix, 0].sum() - sums[ix, 1].sum()) / cnt[ix].sum()
    return float(point), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> int:
    cols = list(_BS.ENRICHED)
    agg = ESTIMANDS["pooled_mean"]
    _test_pat, dev_pat = _BS.FM_PATTERNS[FM]
    dev = _BS._paired("dev", dev_pat)
    fit = dev[(dev["test_year"].isin(_BS.FIT_YEARS)) & (dev["horizon"] == H)].reset_index(drop=True)
    val = dev[(dev["test_year"].isin(_BS.VAL_YEARS)) & (dev["horizon"] == H)].reset_index(drop=True)
    reg = _BS._reg(cols).fit(fit[cols], fit["ape_st"] - fit["ape_fm"])
    sv = np.asarray(reg.predict(val[cols]), dtype=float)
    b = _select(sv, val, agg)
    thr = frozen_threshold(sv, b)
    cs = pd.read_csv(OUT / "country_static.csv").set_index(["fm", "horizon"])
    assert abs(b - float(cs.loc[(FM, H), "router_budget"])) < 1e-9, "budget drift"
    assert abs(thr - float(cs.loc[(FM, H), "router_threshold"])) < 1e-5, "threshold drift"
    sel = (
        val.groupby("country")[["ape_fm", "ape_st"]]
        .mean()
        .pipe(lambda x: x["ape_fm"] < x["ape_st"])
    )
    # the dense records
    st = pd.read_parquet(OUT / "per_origin_structural_2025_stride1.parquet")
    st = st[st["horizon"] == H].rename(columns={"ape": "ape_st"})
    st["anchor_t"] = pd.to_datetime(st["anchor_t"])
    files = sorted(glob.glob("reports/tslimits/fm/tsldense_chronos_h24_q_*.parquet"))
    fm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fm = fm.rename(columns={"cc": "country", "origin_ts": "anchor_t"})
    fm["anchor_t"] = pd.to_datetime(fm["anchor_t"])
    fm["ape_fm"] = 100.0 * (fm["q50"] - fm["actual"]).abs() / fm["actual"].abs()
    j = st.merge(
        fm[["country", "horizon", "anchor_t", "actual", "ape_fm"]],
        on=["country", "horizon", "anchor_t"],
        validate="one_to_one",
    )
    assert len(j) == len(st), f"joined {len(j)} of {len(st)}"
    assert float((j["y_true"] - j["actual"]).abs().max()) == 0.0
    s_dense = np.asarray(reg.predict(j[cols]), dtype=float)
    use = j["country"].map(sel).to_numpy()
    served = {
        "st": j["ape_st"].to_numpy(),
        "fm": j["ape_fm"].to_numpy(),
        "cc_static": np.where(use, j["ape_fm"], j["ape_st"]),
        "router": np.where(escalate_above(s_dense, thr), j["ape_fm"], j["ape_st"]),
        "oracle": np.minimum(j["ape_st"], j["ape_fm"]),
    }
    cc = j["country"].to_numpy()
    rng = np.random.default_rng(SEED)
    pt, lo, hi = _ci(served["cc_static"], served["router"], cc, rng)
    esc = float(escalate_above(s_dense, thr).mean())
    row = {
        "fm": FM,
        "horizon": H,
        "n_dense": len(j),
        "n_published": int(cs.loc[(FM, H), "n_countries"]) * 0 + 6934,
        "gain_dense_pp": round(pt, 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "escalated_dense": round(esc, 4),
        "gain_published_pp": float(cs.loc[(FM, H), "router_over_cc_pp"]),
        "lo_published": float(cs.loc[(FM, H), "router_over_cc_lo"]),
        "hi_published": float(cs.loc[(FM, H), "router_over_cc_hi"]),
        "escalated_published": float(cs.loc[(FM, H), "router_realized_frac"]),
        "headroom_dense_pp": round(
            float(agg(pd.Series(served["cc_static"]), cc) - agg(pd.Series(served["oracle"]), cc)), 4
        ),
    }
    pd.DataFrame([row]).to_csv(OUT / "dense_cell.csv", index=False)
    print(
        f"[dense] Chronos-2 h=24: dense grid n={len(j):,} gain over c-static {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] "
        f"esc {esc:.2f} | published grid {row['gain_published_pp']:+.3f} "
        f"[{row['lo_published']:+.3f}, {row['hi_published']:+.3f}] esc {row['escalated_published']:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
