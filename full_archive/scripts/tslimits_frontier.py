"""TS-LIMITS: the accuracy-versus-energy frontier the paper maps but does not solve.

Equation (3) gives a cost-aware Bayes rule, but the deployed threshold is chosen
to minimise error over a grid of escalation fractions with the serving
measurements applied afterwards. So the paper maps a frontier rather than solving
the cost-aware objective, and this script draws the frontier so that is visible
rather than asserted.

For each (fm, horizon) and each candidate fraction f on the same grid the policy
search uses, the threshold is frozen on 2024 at f and applied to 2025. The x-axis
is average GPU joules per request -- the realised escalation fraction times the
measured per-call cost -- and the y-axis is 2025 pooled mean APE. No conversion
from joules to APE is needed or made: the operator supplies that exchange rate,
which is exactly the point of lambda.

Two envelopes are drawn per cell because energy per call depends on batch
occupancy, which the escalation fraction itself influences: batch 1 (low volume)
and a saturated batch 32. The truth for a given deployment lies between them, and
neither is a function of f alone, which is why the cost is closer to lambda(f)
than to the constant lambda that (3) assumes.

The frozen operating point actually deployed is marked, so a reader can see
whether accuracy-only selection landed somewhere reasonable on the cost axis.

Out: reports/tslimits/frontier.csv, reports/tslimits/fig_frontier.pdf
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_frontier.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_forecast.eval.routing import ESTIMANDS, escalate_above, frozen_threshold

OUT = Path("reports/tslimits")
ESTIMAND = "pooled_mean"
JKEY = {"Chronos-2-Uni-ZS": "chronos", "TimesFM-2.5-Uni-ZS": "timesfm"}
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _mod(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_BS = _mod("tslimits_bootstrap")


def _energy() -> dict[tuple[str, int, int], tuple[float, float]]:
    """Per (model, horizon, batch): (quiet-idle, loaded-idle) marginal joules.

    The two idle conventions bracket what declining a call saves: quiet-idle if
    the accelerator could otherwise be unloaded, loaded-idle on a resident
    service. Neither is chosen silently; the frontier draws the band between
    them and the tables report both.
    """
    out: dict[tuple[str, int, int], tuple[float, float]] = {}
    for m in ("chronos", "timesfm"):
        j = json.loads((OUT / f"energy_inference_{m}.json").read_text())
        for r in j["aggregate"]:
            out[(m, r["horizon"], r["batch_size"])] = (
                r["marginal_j"]["median"],
                r["marginal_j_resident"]["median"],
            )
    return out


def main() -> int:
    agg = ESTIMANDS[ESTIMAND]
    cols = list(_BS.ENRICHED)
    en = _energy()
    rows: list[dict[str, Any]] = []

    for fm_label, (test_pat, dev_pat) in _BS.FM_PATTERNS.items():
        dev = _BS._paired("dev", dev_pat)
        test = _BS._paired("2025", test_pat)
        fit = dev[dev["test_year"].isin(_BS.FIT_YEARS)]
        val = dev[dev["test_year"].isin(_BS.VAL_YEARS)]
        key = JKEY[fm_label]

        for h in sorted(test["horizon"].unique()):
            f = fit[fit["horizon"] == h].reset_index(drop=True)
            v = val[val["horizon"] == h].reset_index(drop=True)
            t = test[test["horizon"] == h].reset_index(drop=True)
            st, fmv = t["ape_st"].to_numpy(), t["ape_fm"].to_numpy()

            reg = _BS._reg(cols).fit(f[cols], f["ape_st"] - f["ape_fm"])
            sv = np.asarray(reg.predict(v[cols]), dtype=float)
            stt = np.asarray(reg.predict(t[cols]), dtype=float)

            # the fraction the deployed policy actually selected, on accuracy alone
            chosen = min(
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

            for q in _BS.BUDGETS:
                mask = escalate_above(stt, frozen_threshold(sv, q))
                realized = float(mask.mean())
                ape = agg(pd.Series(np.where(mask, fmv, st)), t["country"])
                rows.append(
                    {
                        "fm": fm_label,
                        "horizon": int(h),
                        "target_frac": q,
                        "realized_frac": round(realized, 4),
                        "pooled_mean_ape": round(ape, 4),
                        "j_per_request_b1": round(realized * en[(key, h, 1)][0], 4),
                        "j_per_request_b32": round(realized * en[(key, h, 32)][0], 4),
                        "j_per_request_b1_loaded": round(realized * en[(key, h, 1)][1], 4),
                        "j_per_request_b32_loaded": round(realized * en[(key, h, 32)][1], 4),
                        "is_deployed": bool(q == chosen),
                    }
                )
            print(f"[frontier] {fm_label:19s} h={int(h):<4} deployed at f={chosen:.2f}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "frontier.csv", index=False)

    def _panel(ax, fm_label, h, legend=False, zero_floor=None):
        """One frontier panel. `zero_floor` (J) draws the panel on a log axis with
        zero-call policies, whose GPU energy is exactly zero, shown as open markers
        at the floor; None keeps the symlog axis of the body figure."""
        g = d[(d["fm"] == fm_label) & (d["horizon"] == h)].sort_values("j_per_request_b32")
        if zero_floor is not None:
            g = g.copy()
            zero = g["j_per_request_b32"] <= 0
            for col in (
                "j_per_request_b32",
                "j_per_request_b1",
                "j_per_request_b32_loaded",
                "j_per_request_b1_loaded",
            ):
                g[col] = g[col].clip(lower=zero_floor)
        ax.plot(
            g["j_per_request_b32"],
            g["pooled_mean_ape"],
            "-o",
            ms=2.5,
            lw=1.0,
            color="C0",
            label="batch 32",
        )
        g1 = g.sort_values("j_per_request_b1")
        ax.plot(
            g1["j_per_request_b1"],
            g1["pooled_mean_ape"],
            "-o",
            ms=2.5,
            lw=1.0,
            color="C1",
            alpha=0.75,
            label="batch 1",
        )
        # the accounting band: loaded-idle (left edge) to quiet-idle (right edge)
        for gg, xq, xl, col in (
            (g, "j_per_request_b32", "j_per_request_b32_loaded", "C0"),
            (g1, "j_per_request_b1", "j_per_request_b1_loaded", "C1"),
        ):
            ax.fill_betweenx(gg["pooled_mean_ape"], gg[xl], gg[xq], color=col, alpha=0.12, lw=0)
        dep = g[g["is_deployed"]]
        ax.plot(
            dep["j_per_request_b32"],
            dep["pooled_mean_ape"],
            "*",
            ms=11,
            color="k",
            zorder=5,
            label="deployed",
        )
        ax.set_title(f"{fm_label.split('-Uni')[0]}, $h$={h}", fontsize=9)
        if zero_floor is None:
            ax.set_xscale("symlog", linthresh=0.05)
        else:
            ax.set_xscale("log")
            z = g[zero]
            if len(z):
                ax.plot(
                    z["j_per_request_b32"],
                    z["pooled_mean_ape"],
                    "o",
                    ms=4,
                    mfc="none",
                    mec="0.3",
                    zorder=6,
                )
                ax.annotate(
                    "no FM calls",
                    xy=(float(z["j_per_request_b32"].iloc[0]), float(z["pooled_mean_ape"].iloc[0])),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6.5,
                    color="0.3",
                )
        if legend:
            ax.legend(frameon=False, fontsize=7.5, loc="best")

    fms = sorted(d["fm"].unique())
    hs = sorted(d["horizon"].unique())

    # Body figure: three representative cells at a readable size. Eight panels on
    # one row of a 7-page paper are unreadable, and the comparison a reader needs
    # is across regimes rather than across every cell.
    chronos = next(f for f in fms if f.startswith("Chronos"))
    # two cells, one per regime: a day-and-below cell where routing avoids
    # calls and a week-to-month cell where it buys them; the full grid is the
    # appendix figure
    picks = [(chronos, 1), (chronos, 168)]
    figb, axb = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for i, (fm_label, h) in enumerate(picks):
        _panel(axb[i], fm_label, h, legend=(i == 0))
        axb[i].set_xlabel("GPU J / request (band: loaded-idle to quiet-idle)")
    axb[0].set_ylabel("pooled mean APE (%)")
    figb.tight_layout()
    figb.savefig(OUT / "fig_frontier_main.pdf", bbox_inches="tight")

    # Appendix figure: the full grid, unchanged.
    fig, axes = plt.subplots(2, 4, figsize=(9.4, 4.4), sharex=False)
    for r, fm_label in enumerate(fms):
        for c, h in enumerate(hs):
            _panel(axes[r, c], fm_label, h, legend=(r == 0 and c == 0), zero_floor=1e-2)
            if c == 0:
                axes[r, c].set_ylabel("pooled mean APE (%)")
            if r == 1:
                axes[r, c].set_xlabel("GPU J / request")
    fig.tight_layout()
    fig.savefig(OUT / "fig_frontier.pdf", bbox_inches="tight")
    print(f"\n[frontier] wrote {OUT}/frontier.csv and {OUT}/fig_frontier.pdf", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
