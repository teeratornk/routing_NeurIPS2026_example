"""Net gain over c-static against the escalation fraction, at h = 720.

The mechanism figure credits a policy for the positive margin it escalates
without charging it for the losses escalated alongside. This one charges
them: the y-axis is the realised gain over c-static (pooled mean APE, pp) of
the frozen gate at every candidate budget (from frontier.csv, thresholds
frozen on 2024), beside the same curve for the oracle ordering (requests
escalated in order of their true margin) and for a random ordering, both
computed on the 2025 requests from the served records. The star is the
deployed budget.

Out:  reports/tslimits/fig_netgain.pdf
Run:  .venv/bin/python scripts/tslimits_fig_netgain.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("reports/tslimits")
H = 720
SEED = 0
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
    fr = pd.read_csv(OUT / "frontier.csv")
    pr = pd.read_parquet(OUT / "served_per_request.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, fm in zip(axes, sorted(pr["model"].unique()), strict=True):
        cell = pr[(pr["model"] == fm) & (pr["horizon"] == H)].reset_index(drop=True)
        st, fmv, cs = cell["st"].to_numpy(), cell["fm"].to_numpy(), cell["cc_static"].to_numpy()
        base = cs.mean()
        margin = st - fmv
        fracs = np.linspace(0, 1, 101)
        # oracle ordering: escalate the largest true margins first
        order = np.argsort(-margin)
        cum = np.concatenate([[0.0], np.cumsum(margin[order])])
        n = len(margin)
        # escalating a set S serves st - margin on S, so its gain over c-static
        # is (base - mean(st)) + sum_S(margin) / n; every curve starts at
        # base - mean(st) (structural everywhere) and ends at base - mean(fm),
        # and the oracle's peak is Table 1's headroom
        offset = base - st.mean()
        oracle = [(cum[int(round(f * n))] / n) + offset for f in fracs]
        # a random ordering, averaged over draws
        rng = np.random.default_rng(SEED)
        rand = np.zeros(len(fracs))
        for _ in range(50):
            perm = rng.permutation(n)
            c = np.concatenate([[0.0], np.cumsum(margin[perm])])
            rand += np.array([(c[int(round(f * n))] / n) + offset for f in fracs])
        rand /= 50
        # headroom.csv is rounded to 4 decimals
        hd = pd.read_csv(OUT / "headroom.csv").set_index(["model", "horizon"])
        assert abs(max(oracle) - float(hd.loc[(fm, H), "cstatic_minus_oracle"])) < 5e-5, (
            f"{fm}: oracle peak {max(oracle):.6f} is not the c-static headroom"
        )
        assert abs(oracle[0] - offset) < 1e-12 and abs(oracle[-1] - (base - fmv.mean())) < 1e-9
        g = fr[(fr["fm"] == fm) & (fr["horizon"] == H)].sort_values("realized_frac")
        gate = base - g["pooled_mean_ape"].to_numpy()
        ax.plot(fracs, oracle, color="0.2", lw=1.2, label="oracle ordering")
        ax.plot(
            g["realized_frac"],
            gate,
            "-o",
            ms=2.5,
            lw=1.1,
            color="C0",
            label="frozen gate, by budget",
        )
        ax.plot(fracs, rand, ls=(0, (2, 2)), color="0.55", lw=1.0, label="random ordering")
        dep = g[g["is_deployed"]]
        ax.plot(
            dep["realized_frac"],
            base - dep["pooled_mean_ape"],
            "*",
            ms=11,
            color="k",
            zorder=5,
            label="deployed",
        )
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_title(f"{fm.split('-Uni')[0]}, $h$=720", fontsize=9)
        ax.set_xlabel("fraction of requests escalated")
    axes[0].set_ylabel("net gain over c-static (pp)")
    # one legend row above the panels: inside either panel it sat on a curve
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.5,
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig_netgain.pdf", bbox_inches="tight")
    print(f"[fig] wrote {OUT}/fig_netgain.pdf", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
