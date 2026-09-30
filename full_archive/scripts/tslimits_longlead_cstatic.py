"""A 2024-selected country-static policy at the long leads, scored on 2025.

The routed horizons compare the router with a per-(country, horizon) static
policy frozen on 2024. At one and two years no router exists, and until now
no 2024-target structural record did either, so the long leads compared fixed
choices only. This scores the same c-static rule there: per country, the
model with the lower median APE on 2024 targets (origins in 2023 for one
year, 2022 for two), applied unchanged to 2025. The Chronos-2 side is the
frozen-cap arm on both windows (rollout_sel2024_h*_cap* on 2024,
rollout_ablation_h*_cap* on 2025), context 2048.

Out:  reports/tslimits/longlead_cstatic.csv, tab_longlead_cstatic.tex
Run:  .venv/bin/python scripts/tslimits_longlead_cstatic.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("reports/tslimits")
FROZEN_CAP = {8760: 48, 17520: 56}
SEED, B = 0, 2000
EU19 = (
    "AT",
    "BE",
    "BG",
    "CH",
    "CZ",
    "DE",
    "ES",
    "FR",
    "GR",
    "HR",
    "HU",
    "IT",
    "LU",
    "NL",
    "PL",
    "PT",
    "RO",
    "SI",
    "SK",
)


def _fm(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path).rename(columns={"cc": "country", "origin_ts": "anchor_t"})
    d["anchor_t"] = pd.to_datetime(d["anchor_t"])
    d["ape_fm"] = 100.0 * (d["q50"] - d["actual"]).abs() / d["actual"].abs()
    return d[["country", "horizon", "anchor_t", "actual", "ape_fm"]]


def _join(st: pd.DataFrame, fm: pd.DataFrame, h: int) -> pd.DataFrame:
    s = st[st["horizon"] == h].rename(columns={"ape": "ape_st"}).copy()
    s["anchor_t"] = pd.to_datetime(s["anchor_t"])
    j = s.merge(
        fm[fm["horizon"] == h],
        on=["country", "horizon", "anchor_t"],
        how="inner",
        validate="one_to_one",
    )
    assert len(j) == len(s), f"h={h}: joined {len(j)} of {len(s)}"
    assert float((j["y_true"] - j["actual"]).abs().max()) == 0.0, f"h={h}: targets disagree"
    return j[["country", "anchor_t", "ape_st", "ape_fm"]].reset_index(drop=True)


def _pm(x: pd.Series, cc: pd.Series) -> float:
    return float(x.groupby(cc).median().median())


def main() -> int:
    dev = pd.read_parquet(OUT / "per_origin_structural_devlong.parquet")
    rows = []
    for h, cap in FROZEN_CAP.items():
        sel = _join(dev, _fm(OUT / f"rollout_sel2024_h{h}_cap{cap}.parquet"), h)
        st25 = pd.read_parquet(
            OUT
            / (
                "per_origin_structural_2025_year.parquet"
                if h == 8760
                else f"per_origin_structural_2025_h{h}.parquet"
            )
        )
        tst = _join(st25, _fm(OUT / f"rollout_ablation_h{h}_cap{cap}.parquet"), h)
        assert set(sel["country"]) == set(EU19) == set(tst["country"]), f"h={h}: country set"
        # the 2024 rule: per country, the lower median APE
        med24 = sel.groupby("country")[["ape_fm", "ape_st"]].median()
        pick_fm = med24["ape_fm"] < med24["ape_st"]
        use = tst["country"].map(pick_fm).to_numpy()
        served = {
            "structural": tst["ape_st"].to_numpy(),
            "chronos2": tst["ape_fm"].to_numpy(),
            "cc_static": np.where(use, tst["ape_fm"], tst["ape_st"]),
            "oracle": np.minimum(tst["ape_st"], tst["ape_fm"]),
        }
        cc = tst["country"]
        pm = {k: _pm(pd.Series(v), cc) for k, v in served.items()}
        med = pd.DataFrame(
            {k: pd.Series(v).groupby(cc.to_numpy()).median() for k, v in served.items()}
        )
        rng = np.random.default_rng(SEED)
        arr = med[["structural", "cc_static", "chronos2", "oracle"]].to_numpy()
        d_cs, d_fm = np.empty(B), np.empty(B)
        for bi in range(B):
            q = np.median(arr[rng.integers(0, len(arr), len(arr))], axis=0)
            d_cs[bi] = q[1] - q[0]
            d_fm[bi] = q[2] - q[0]
        # where the FM wins on 2025, does the country rule catch it?
        wins = tst["ape_fm"] < tst["ape_st"]
        caught = float((wins & use).sum() / max(wins.sum(), 1))
        rows.append(
            {
                "horizon": h,
                "cap": cap,
                "n_2024": len(sel),
                "n_2025": len(tst),
                "cc_picks_fm": int(pick_fm.sum()),
                "picked": " ".join(sorted(pick_fm[pick_fm].index)),
                **{f"pm_{k}": round(v, 4) for k, v in pm.items()},
                "cstatic_minus_structural_pp": round(pm["cc_static"] - pm["structural"], 4),
                "cstatic_minus_structural_lo": round(float(np.percentile(d_cs, 2.5)), 4),
                "cstatic_minus_structural_hi": round(float(np.percentile(d_cs, 97.5)), 4),
                "chronos_minus_structural_pp": round(pm["chronos2"] - pm["structural"], 4),
                "oracle_gap_pp": round(pm["cc_static"] - pm["oracle"], 4),
                "fm_win_share_2025": round(float(wins.mean()), 4),
                "fm_wins_caught_by_rule": round(caught, 4),
            }
        )
        print(
            f"[longlead] h={h}: 2024 rule picks the FM in {int(pick_fm.sum())}/19 "
            f"({' '.join(sorted(pick_fm[pick_fm].index)) or '-'}); "
            f"2025 panel-median structural {pm['structural']:.2f} c-static {pm['cc_static']:.2f} "
            f"[{np.percentile(d_cs, 2.5):+.2f},{np.percentile(d_cs, 97.5):+.2f}] Chronos-2 "
            f"{pm['chronos2']:.2f} oracle {pm['oracle']:.2f}; "
            f"rule catches {caught:.0%} of the FM's 2025 wins",
            flush=True,
        )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "longlead_cstatic.csv", index=False)
    tex = [
        "\\begin{tabular}{lrrrrlr}",
        "\\toprule",
        "$h$ & structural & c-static (2024) & Chronos-2 & oracle & c-static $-$ structural [95\\% "
        "CI] & picks FM \\\\",
        "\\midrule",
    ]
    for r in d.itertuples():
        lab = "1\\,yr" if r.horizon == 8760 else "2\\,yr"
        tex.append(
            f"{lab} & {r.pm_structural:.2f} & {r.pm_cc_static:.2f} & {r.pm_chronos2:.2f} & "
            f"{r.pm_oracle:.2f} & "
            f"${r.cstatic_minus_structural_pp:+.2f}$ [{r.cstatic_minus_structural_lo:+.2f}, "
            f"{r.cstatic_minus_structural_hi:+.2f}] & {r.cc_picks_fm} of 19 \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_longlead_cstatic.tex").write_text("\n".join(tex) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
