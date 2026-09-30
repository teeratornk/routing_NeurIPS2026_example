"""TS-LIMITS main figure: one policy, one set of requests, two risk functionals.

The paper's claim is that whether request-time routing pays depends on the
functional used to aggregate request-level errors. Isolating that means holding
EVERYTHING else fixed, so both panels score the identical frozen thresholds --
the ones the pooled mean selected -- and differ only in how the served errors
are aggregated.

The earlier version let each functional pick its own escalation budget on 2024.
That is a legitimate deployment comparison, and it is the one a practitioner
following each convention would face, but it is the wrong figure for this claim:
scoring AND policy selection both change, so the panels cannot separate the two.
It moves to the appendix figure, drawn here as a hollow marker for reference.

Out: reports/tslimits/fig_estimand.pdf
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_fig_estimand.py
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
MARGIN_PP = 0.10
# Type 42 (TrueType), not matplotlib's Type 3 default: Type 3 fonts are the only
# non-Type-1 fonts that reach the compiled PDF and many venues reject them.
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
# Panel titles name the RISK FUNCTIONAL only. The earlier right-hand title said
# "identical thresholds", which the panel then contradicted by carrying hollow
# own-threshold markers; the threshold distinction belongs to the marker legend,
# which already carries it.
PANELS = (
    ("gain_pooled", "pooled-mean APE"),
    ("gain_median_samepolicy", "panel-median MdAPE"),
)


def main() -> int:
    d = pd.read_csv(OUT / "paired_estimand.csv")
    # the fourth cell of the policy-by-metric table: the median-selected policy
    # scored under the pooled mean, with its own interval, plus both budgets
    cs = pd.read_csv(OUT / "country_static.csv")
    assert "medsel_over_cc_pp" in cs.columns, "run tslimits_country_static.py first"
    need = [f"{p}_{s}" for p, _ in PANELS for s in ("pp", "lo", "hi")]
    missing = [c for c in need if c not in d.columns]
    assert not missing, f"paired_estimand.csv lacks level CIs: {missing}"

    # Wide and short on purpose. This figure is included at full text
    # width, and 2.45 in is the tallest aspect that keeps the camera-ready body
    # on seven pages (2.55 in pushes two lines onto an eighth). Two interval
    # panels with four horizon groups each stay legible at this height.
    # the budgets are in Table tab:paired; no labels under the markers, so the
    # base font can go up a point at the same figure height
    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.45), sharey=True)
    for ax, (key, name) in zip(axes, PANELS, strict=True):
        for k, fm in enumerate(sorted(d["fm"].unique())):
            g = d[d["fm"] == fm].sort_values("horizon").reset_index(drop=True)
            x = np.arange(len(g)) + (k - 0.5) * 0.22
            ax.errorbar(
                x,
                g[f"{key}_pp"],
                yerr=[g[f"{key}_pp"] - g[f"{key}_lo"], g[f"{key}_hi"] - g[f"{key}_pp"]],
                fmt="o",
                ms=4.5,
                lw=1.3,
                capsize=2.5,
                color=f"C{k}",
                label=fm.split("-Uni")[0],
            )
            # the median-selected policy under the pooled mean: the same
            # hollow-marker convention as the right-hand panel
            if key == "gain_pooled":
                m = cs[cs["fm"] == fm].sort_values("horizon").reset_index(drop=True)
                ax.errorbar(
                    x + 0.07,
                    m["medsel_over_cc_pp"],
                    yerr=[
                        m["medsel_over_cc_pp"] - m["medsel_over_cc_lo"],
                        m["medsel_over_cc_hi"] - m["medsel_over_cc_pp"],
                    ],
                    fmt="o",
                    ms=4.5,
                    mfc="none",
                    mec=f"C{k}",
                    ecolor=f"C{k}",
                    elinewidth=0.7,
                    capsize=0,
                    alpha=0.6,
                )
            # the own-budget deployment comparison, for reference only
            if key.endswith("samepolicy"):
                ax.plot(
                    x,
                    g["gain_median_ownpolicy_pp"],
                    marker="o",
                    ms=4.5,
                    mfc="none",
                    ls="none",
                    mec=f"C{k}",
                    alpha=0.55,
                )
            ax.set_xticks(np.arange(len(g)))
            ax.set_xticklabels([f"h={int(v)}" for v in g["horizon"]])
        ax.axhline(0, color="0.4", lw=0.9)
        ax.axhline(MARGIN_PP, color="0.55", lw=0.9, ls=(0, (4, 2)))
        ax.set_title(name, fontsize=10.5)
    # The long form of this label ("router gain over country x horizon static
    # (pp)") is 205 bp of rotated text in a 160 bp figure, so 22% of it rendered
    # outside the bounding box: "rout" was clipped off the front and "(pp)" off
    # the end. The figure is deliberately short to save column height, so the
    # label is shortened to fit rather than the figure grown to accommodate it.
    axes[0].set_ylabel("gain over c-static (pp)")
    h, lab = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color="0.55", lw=0.9, ls=(0, (4, 2))))
    lab.append(f"{MARGIN_PP:.2f} pp practical margin")
    h.append(Line2D([], [], marker="o", ls="none", mfc="none", mec="0.45"))
    lab.append("each metric picks its own budget")
    # At full text width the legend's last row sat on the TimesFM-2.5 h=24 cap
    # in the right panel; 0.2 pp of headroom clears it at the same figure size.
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi + 0.2)
    axes[1].legend(h, lab, frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_estimand.pdf", bbox_inches="tight")

    same = d["gain_median_samepolicy_pp"]
    print(
        f"[fig] pooled median {d.gain_pooled_pp.median():+.4f} pp "
        f"(sig {int(d.gain_pooled_sig.sum())}/8); "
        f"same-policy median {same.median():+.4f} pp "
        f"(sig {int(d.gain_median_samepolicy_sig.sum())}/8); "
        f"own-budget median {d.gain_median_ownpolicy_pp.median():+.4f} pp "
        f"(sig {int(d.gain_median_ownpolicy_sig.sum())}/8)",
        flush=True,
    )
    print(f"[fig] wrote {OUT}/fig_estimand.pdf", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
