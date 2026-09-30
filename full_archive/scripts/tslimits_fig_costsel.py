"""TS-LIMITS: the cost-sensitive selection figure (appendix sec:costsel).

Top row: the cost-adjusted gain of the priced router over the priced c-static
policy, per horizon, against the price of one FM call, with 95% country-cluster
intervals. Bottom row: the share of 2025 requests each policy sends to the FM,
the router solid and c-static dashed. One column per FM.

Horizons use a fixed four-colour order (Okabe-Ito; validated for CVD, with the
deutan pair in the 6-8 band) and a distinct marker each, so identity never rests
on colour alone. tab_costsel.tex carries the numbers.

In:  reports/tslimits/cost_selection.csv (tslimits_cost_selection.py)
Out: reports/tslimits/fig_costsel.pdf
Run: .venv/bin/python scripts/tslimits_fig_costsel.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("reports/tslimits")
FMS = (("Chronos-2-Uni-ZS", "Chronos-2"), ("TimesFM-2.5-Uni-ZS", "TimesFM-2.5"))
HORIZONS = (
    (1, "1 h", "#0072B2", "o"),
    (24, "1 d", "#E69F00", "s"),
    (168, "1 w", "#009E73", "^"),
    (720, "1 mo", "#CC79A7", "D"),
)
# Type 42 (TrueType), not matplotlib's Type 3 default, as in the other figures
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def main() -> int:
    d = pd.read_csv(OUT / "cost_selection.csv")
    d = d[d["policy"] == "router"]
    lams = sorted(d["lam"].unique())
    assert len(d) == len(lams) * 8, "expected every price in every cell"
    x = np.arange(len(lams))
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.2), sharex=True, sharey="row")
    for col, (fm, name) in enumerate(FMS):
        top, bot = axes[0, col], axes[1, col]
        for k, (h, lab, color, marker) in enumerate(HORIZONS):
            g = d[(d["fm"] == fm) & (d["horizon"] == h)].sort_values("lam")
            off = (k - 1.5) * 0.09
            top.errorbar(
                x + off,
                g["cost_gain_pp"],
                yerr=[g["cost_gain_pp"] - g["cost_gain_lo"], g["cost_gain_hi"] - g["cost_gain_pp"]],
                fmt=marker + "-",
                ms=4,
                lw=1.2,
                elinewidth=0.9,
                capsize=2,
                color=color,
                label=lab,
            )
            bot.plot(x, 100 * g["share_2025"], marker + "-", ms=4, lw=1.2, color=color, label=lab)
            bot.plot(x, 100 * g["cc_share_2025"], "--", lw=1.0, color=color, alpha=0.8)
        top.axhline(0, color="0.4", lw=0.9)
        top.set_title(name, fontsize=10)
        bot.set_xticks(x)
        bot.set_xticklabels([f"{v:g}" for v in lams])
        bot.set_xlabel("price of one FM call, $\\lambda$ (pp)")
    axes[0, 0].set_ylabel("cost-adjusted gain\nover c-static$^{\\lambda}$ (pp)")
    axes[1, 0].set_ylabel("requests sent\nto the FM (%)")
    # One legend row above the panels: inside any panel it sat on the data.
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], color="0.3", lw=1.2, label="router"),
        plt.Line2D([], [], color="0.3", lw=1.0, ls="--", label="c-static$^{\\lambda}$"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_costsel.pdf", bbox_inches="tight")
    print(f"[fig] wrote {OUT}/fig_costsel.pdf", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
