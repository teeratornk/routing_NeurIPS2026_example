"""Budget-matched against capability-matched: the two policy families side by side.

The published family scores both checkpoints at a common 2048-hour context on
every window. The native family gives each its usable ceiling on BOTH windows,
so the per-(country, horizon) static rule and the router are re-selected on
records of the configuration they are scored on (TSLIMITS_FM_FAMILY=native in
tslimits_bootstrap.py and tslimits_country_static.py). This table reads the
two families' artifacts and prints them together; nothing is recomputed.

Out:  reports/tslimits/native_family.csv, tab_native_family.tex
Run:  .venv/bin/python scripts/tslimits_native_family.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("reports/tslimits")
SHORT = {"Chronos-2-Uni-ZS": "Chronos-2", "TimesFM-2.5-Uni-ZS": "TimesFM-2.5"}
FAMILIES = (("budget", ""), ("native", "_native"))


def main() -> int:
    rows = []
    for fam, sfx in FAMILIES:
        cs = pd.read_csv(OUT / f"country_static{sfx}.csv").set_index(["fm", "horizon"])
        bc = pd.read_csv(OUT / f"bootstrap_ci_pooled_mean{sfx}.csv").set_index(["fm", "horizon"])
        for (fm, h), c in cs.iterrows():
            b = bc.loc[(fm, h)]
            oracle = float(b["best_fixed"]) - float(b["orc_gain_pp"])
            headroom = float(c["cc_static"]) - oracle
            gain = float(c["router_over_cc_pp"])
            rows.append(
                {
                    "family": fam,
                    "fm": fm,
                    "horizon": int(h),
                    "best_fixed": round(float(b["best_fixed"]), 4),
                    "fixed_choice": b["fixed_choice"],
                    "cc_static": round(float(c["cc_static"]), 4),
                    "cc_picks_fm": int(c["cc_pick_fm_n"]),
                    "router": round(float(c["router_cc"]), 4),
                    "router_over_cc_pp": round(gain, 4),
                    "lo": round(float(c["router_over_cc_lo"]), 4),
                    "hi": round(float(c["router_over_cc_hi"]), 4),
                    "headroom_pp": round(headroom, 4),
                    "captured": round(gain / headroom, 4) if headroom > 0 else float("nan"),
                    "escalated": round(float(c["router_realized_frac"]), 4),
                }
            )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "native_family.csv", index=False)
    tex = [
        "\\begin{tabular}{llrrrrlr}",
        "\\toprule",
        "FM & $h$ & family & best fixed & c-static (picks FM) & router & router $-$ c-static "
        "[95\\% CI] & captured \\\\",
        "\\midrule",
    ]
    for fm in ("Chronos-2-Uni-ZS", "TimesFM-2.5-Uni-ZS"):
        for h in (1, 24, 168, 720):
            for fam, _ in FAMILIES:
                r = d[(d.fm == fm) & (d.horizon == h) & (d.family == fam)].iloc[0]
                lead = f"{SHORT[fm]} & {h}" if fam == "budget" else " & "
                tex.append(
                    f"{lead} & {'2048' if fam == 'budget' else 'native'} & {r.best_fixed:.3f} "
                    f"({'FM' if r.fixed_choice == 'expensive' else 'st'}) & "
                    f"{r.cc_static:.3f} ({r.cc_picks_fm}) & {r.router:.3f} & "
                    f"${r.router_over_cc_pp:+.3f}$ [{r.lo:+.3f}, {r.hi:+.3f}] & "
                    f"{100 * r.captured:.0f}\\% \\\\"
                )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_native_family.tex").write_text("\n".join(tex) + "\n")
    for fam, _ in FAMILIES:
        g = d[d.family == fam]
        print(
            f"[family] {fam:7s}: router over c-static median {g.router_over_cc_pp.median():+.3f} "
            f"pp, "
            f"positive {int((g.router_over_cc_pp > 0).sum())}/8, excl. zero "
            f"{int((g.lo > 0).sum())}/8, "
            f"captured median {100 * g.captured.median():.0f}%",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
