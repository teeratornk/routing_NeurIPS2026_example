"""TS-LIMITS: the horizon ladder, and where the two forecasters cross.

Figure 1 plots the router's gain over the per-(country, horizon) static policy,
which exists only where a router was fitted: h in {1, 24, 168, 720}. The
long-horizon result is a different quantity entirely, the standing between the
two forecasters as fixed choices, and it had no figure at all. This draws it.

One panel, six horizons, Chronos-2 minus the structural composition in
panel-median MdAPE. Positive is the FM ahead, matching the sign of the
margin D = L_c - L_e in the paper. The crossover between a day and a
week is the whole story: it is where the better fixed choice changes hands, and
it is the same cut the conclusion's three regimes are organised around.

h=26280 is deliberately absent. It was measured (+5.99 pp, below the two-year
+7.87) and is retained in year_scale.csv, but it is not in the paper, and the
paper makes no claim that the deficit grows without bound.

Out: reports/tslimits/fig_ladder.pdf
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_fig_ladder.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

OUT = Path("reports/tslimits")
PUBLISHED_H = (1, 24, 168, 720, 8760, 17520)
LABEL = {1: "1 h", 24: "1 d", 168: "1 w", 720: "1 mo", 8760: "1 yr", 17520: "2 yr"}
plt.rcParams.update(
    {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
     "pdf.fonttype": 42, "ps.fonttype": 42}
)


def main() -> int:
    d = pd.read_csv(OUT / "year_scale.csv").set_index("horizon")
    missing = [h for h in PUBLISHED_H if h not in d.index]
    assert not missing, f"year_scale.csv lacks {missing}"
    g = d.loc[list(PUBLISHED_H)]

    x = np.arange(len(PUBLISHED_H))
    # negate: the stored contrast is FM minus structural, the paper's D runs
    # the other way, and one sign rule is better than two
    y = -g["fm_tuned_minus_structural_pp"].to_numpy()
    lo = -g["fm_tuned_minus_structural_hi"].to_numpy()
    hi = -g["fm_tuned_minus_structural_lo"].to_numpy()

    # Compact canvas: the figure is included at about 0.55 of the text width, so a
    # 6.6 in drawing rendered its 9 pt labels at 4.4 pt. At 4.4 in the same labels
    # land near 6 pt, and the 2:1 aspect is 40% taller than the earlier strip.
    fig, ax = plt.subplots(figsize=(4.4, 2.2))
    ahead = y > 0  # positive is now the FM ahead
    # Hollow where the interval covers zero: at a week the sign is not resolved,
    # and colouring it "structural ahead" would assert a standing we do not have.
    sig = (lo > 0) | (hi < 0)
    for m, c, lab in ((ahead, "C0", "FM ahead"), (~ahead, "C3", "structural ahead")):
        for keep, face in ((m & sig, c), (m & ~sig, "none")):
            if keep.any():
                ax.errorbar(x[keep], y[keep], yerr=[y[keep] - lo[keep], hi[keep] - y[keep]],
                            fmt="o", ms=5, lw=1.4, capsize=3, color=c, mfc=face,
                            label=lab if face == c else None)
    ax.axhline(0, color="0.35", lw=0.9)

    # The crossover, drawn only if one exists. argmax on an all-False array returns
    # 0, so the earlier version would have drawn this line at the axis edge and
    # labelled it "changes hands" even for data where the sign never changed.
    assert ahead[0], "ladder assumes the FM leads at the shortest horizon"
    if (~ahead).any():
        cross = int(np.argmax(~ahead))
        ax.axvline(cross - 0.5, color="0.6", lw=0.9, ls=(0, (4, 2)))
        # a point-estimate transition: the interval at a week covers zero
        ax.annotate("point-estimate\ntransition", xy=(cross - 0.5, -3.2),
                    xytext=(cross - 0.45, -4.6), fontsize=7, color="0.35", ha="left")
    else:
        cross = len(x)
        print("[ladder] no crossover in this data; line omitted", flush=True)

    # the passes the FM spends, where it spends more than one
    for xi, (h, yi) in enumerate(zip(PUBLISHED_H, y, strict=True)):
        p = int(g.loc[h, "tuned_passes"])
        if p > 1:
            ax.annotate(f"{p} passes", xy=(xi, yi), xytext=(xi, hi[xi] + 0.35),
                        fontsize=7, ha="center", color="0.3")

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[h] for h in PUBLISHED_H])
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("structural $-$ Chronos-2 (pp)")
    # the regimes, named above the axes rather than in the caption
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for (x0, x1), lab in (((0, 1), "FM default"), ((2, 3), "structural default"),
                          ((4, 5), "fixed-choice\nevaluation only")):
        ax.text((x0 + x1) / 2, 1.03, lab, transform=tr, ha="center", va="bottom",
                fontsize=7.5, color="0.25")
        ax.plot([x0 - 0.35, x1 + 0.35], [1.02, 1.02], transform=tr, color="0.6", lw=0.8,
                clip_on=False)
    h_, l_ = ax.get_legend_handles_labels()
    h_.append(Line2D([], [], marker="o", ls="none", mfc="none", mec="0.45"))
    l_.append("interval covers zero")
    ax.legend(h_, l_, frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ladder.pdf", bbox_inches="tight")

    print("[ladder] " + "  ".join(
        f"{LABEL[h]} {v:+.2f}" for h, v in zip(PUBLISHED_H, y, strict=True)), flush=True)
    print(f"[ladder] crossover between {LABEL[PUBLISHED_H[cross-1]]} and "
          f"{LABEL[PUBLISHED_H[cross]]}", flush=True)
    print(f"[ladder] wrote {OUT}/fig_ladder.pdf", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
