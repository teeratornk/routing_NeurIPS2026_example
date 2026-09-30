"""TS-LIMITS figures.

fig1  how much of the exploitable gain is captured as a function of the
      fraction of requests escalated, ordered by the oracle, by the learned
      gate, and at random. The oracle curve is steeply concave (a thin tail
      holds most of the gain); the gate curve sits near the random diagonal,
      which is the whole finding in one line.

fig2  the argument in two panels.
      (a) the observed oracle gap against the difficulty-matched permutation
          null. The null band and the observed interval overlap in every cell,
          so the gap is not evidence of routable structure.
      (b) the ladder that replaces it: the oracle, the conditional ceiling (the
          best an x-measurable policy could reach, cross-fitted on 2025 and so
          not deployable), and the frozen deployable policy.

Out: reports/tslimits/fig1_mechanism.pdf, reports/tslimits/fig2_intervals.pdf
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_figures.py
"""

from __future__ import annotations

import glob
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

OUT = Path("reports/tslimits")


def _emitter() -> Any:
    p = Path("scripts") / "tslimits_per_origin_structural.py"
    spec = importlib.util.spec_from_file_location("tslimits_per_origin_structural", p)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FEATURES: tuple[str, ...] = _emitter().GATE_FEATURES
FIT_YEARS = (2018, 2019, 2020, 2021, 2022, 2023)
SEED = 0
H_SHOW = 720
PANELS = [("Chronos-2-Uni-ZS", "chronos"), ("TimesFM-2.5-Uni-ZS", "timesfm")]
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


def _load_fm(pattern: str) -> pd.DataFrame:
    fm = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(pattern))], ignore_index=True)
    fm = fm.rename(columns={"cc": "country", "origin_ts": "anchor_t"})
    fm["anchor_t"] = pd.to_datetime(fm["anchor_t"])
    fm["ape_fm"] = 100.0 * (fm["q50"] - fm["actual"]).abs() / fm["actual"].abs()
    return fm[["country", "horizon", "anchor_t", "ape_fm"]]


def _paired(window: str, pattern: str) -> pd.DataFrame:
    st = pd.read_parquet(OUT / f"per_origin_structural_{window}.parquet")
    st = st.rename(columns={"ape": "ape_st"})
    st["anchor_t"] = pd.to_datetime(st["anchor_t"])
    j = st.merge(
        _load_fm(pattern), on=["country", "horizon", "anchor_t"], how="inner", validate="one_to_one"
    )
    assert len(j) == len(st)
    return j.reset_index(drop=True)


def _captured(gain: np.ndarray, order: np.ndarray, fracs: np.ndarray) -> np.ndarray:
    """Cumulative share of exploitable gain when requests are escalated in `order`.

    Module scope rather than a closure over the loop body: a nested definition
    would capture `gain` by reference and silently describe whichever series
    the loop happened to reach last.
    """
    cum = np.concatenate([[0.0], np.cumsum(gain[order])]) / max(gain.sum(), 1e-12)
    return cum[(fracs * gain.size).astype(int)]


def _panel_median(v: np.ndarray, c: pd.Series) -> float:
    return float(pd.DataFrame({"v": v, "c": np.asarray(c)}).groupby("c")["v"].median().median())


def main() -> int:
    fig1, ax1_single = plt.subplots(1, 1, figsize=(6.4, 3.5))
    ax1 = [ax1_single, None]
    fig2, ax2 = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=False)

    boot = pd.read_csv(OUT / "bootstrap_ci.csv")
    noise = pd.read_csv(OUT / "noise_floor.csv")
    fracs = np.linspace(0, 1, 41)

    for k, (label, tag) in enumerate(PANELS):
        dev = _paired("dev", f"reports/tslimits/fm/tsldev_{tag}_q_*.parquet")
        test = _paired("2025", f"reports/tslimits/fm/tsl_{tag}_q_*.parquet")
        fit = dev[dev["test_year"].isin(FIT_YEARS)]
        f_h = fit[fit["horizon"] == H_SHOW]
        t_h = test[test["horizon"] == H_SHOW].reset_index(drop=True)

        # the paper's best deployable policy: the margin regressor on the
        # enriched feature set, frozen on 2018-2023
        reg = HistGradientBoostingRegressor(max_iter=300, random_state=SEED)
        reg.fit(f_h[list(FEATURES)], f_h["ape_st"] - f_h["ape_fm"])
        p = np.asarray(reg.predict(t_h[list(FEATURES)]), dtype=float)

        st, fm = t_h["ape_st"].to_numpy(), t_h["ape_fm"].to_numpy()
        gain = np.clip(st - fm, 0.0, None)
        n = len(gain)

        # (a) share of exploitable gain captured vs fraction escalated
        rng = np.random.default_rng(SEED)
        ax1[0].plot(
            fracs,
            _captured(gain, np.argsort(-gain), fracs),
            color=f"C{k}",
            lw=1.8,
            ls="-" if k == 0 else (0, (6, 1.5)),
            label=f"{label.split('-Uni')[0]}: oracle order",
        )
        ax1[0].plot(
            fracs,
            _captured(gain, np.argsort(-p), fracs),
            color=f"C{k}",
            lw=1.8,
            ls=(0, (5, 2)) if k == 0 else (0, (1.5, 1.5)),
            label=f"{label.split('-Uni')[0]}: learned gate",
        )
        if k == 0:
            ax1[0].plot(
                fracs,
                _captured(gain, rng.permutation(n), fracs),
                color="0.6",
                lw=1.0,
                ls=":",
                label="random",
            )

    # The oracle curve must reach 1.0 at the positive-margin share, since
    # requests with negative margin contribute nothing to the clipped mass.
    # Without this marker the saturation reads as a concentration finding.
    for k2, (_lab, tag2) in enumerate(PANELS):
        t2 = _paired("2025", f"reports/tslimits/fm/tsl_{tag2}_q_*.parquet")
        t2 = t2[t2["horizon"] == H_SHOW]
        share = float((t2["ape_st"] > t2["ape_fm"]).mean())
        ax1[0].axvline(share, color=f"C{k2}", lw=0.8, ls=(0, (1, 3)), alpha=0.7)
    ax1[0].set(
        xlabel="fraction of requests escalated",
        ylabel="share of positive-margin mass captured",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax1[0].legend(frameon=False, fontsize=11, loc="lower right")
    ax1[0].set_title(f"positive-margin mass captured, h={H_SHOW}", fontsize=13)
    fig1.tight_layout()
    fig1.savefig(OUT / "fig1_mechanism.pdf", bbox_inches="tight")

    # fig2(a): the observed gap against the difficulty-matched permutation null.
    # The null band is drawn as a bar so the eye compares an interval with an
    # interval rather than a point with a point.
    a = ax2[0]
    for k, (label, _tag) in enumerate(PANELS):
        b = boot[boot["fm"] == label].sort_values("horizon").reset_index(drop=True)
        nz = noise[noise["fm"] == label].sort_values("horizon").reset_index(drop=True)
        x = np.arange(len(b)) + (k - 0.5) * 0.20
        a.bar(
            x,
            nz["matched_hi"] - nz["matched_lo"],
            bottom=nz["matched_lo"],
            width=0.17,
            color="0.78",
            edgecolor="none",
            label="difficulty-matched null (95%)" if k == 0 else None,
            zorder=1,
        )
        a.errorbar(
            x,
            b["orc_gain_pp"],
            yerr=[b["orc_gain_pp"] - b["orc_lo"], b["orc_hi"] - b["orc_gain_pp"]],
            fmt="o",
            ms=4.0,
            lw=1.2,
            capsize=2,
            color=f"C{k}",
            label=label.split("-Uni")[0],
            zorder=3,
        )
        a.set_xticks(np.arange(len(b)))
        a.set_xticklabels([f"h={int(v)}" for v in b["horizon"]])
    a.set(ylabel="oracle gap (pp)")
    a.set_title("(a) the oracle gap matches a random pairing", fontsize=11.5)
    a.legend(frameon=False, fontsize=9, loc="upper left")

    # fig2(b): the ladder. Oracle, then the conditional ceiling, then what a
    # frozen deployable policy actually gets.
    a = ax2[1]
    for k, (label, _tag) in enumerate(PANELS):
        b = boot[boot["fm"] == label].sort_values("horizon").reset_index(drop=True)
        x = np.arange(len(b)) + (k - 0.5) * 0.20
        a.plot(
            x,
            b["orc_gain_pp"],
            marker="_",
            ms=12,
            lw=0,
            color="0.5",
            label="oracle" if k == 0 else None,
        )
        a.errorbar(
            x,
            b["ceiling_enr_gain_pp"],
            yerr=[
                b["ceiling_enr_gain_pp"] - b["ceiling_enr_lo"],
                b["ceiling_enr_hi"] - b["ceiling_enr_gain_pp"],
            ],
            fmt="o",
            mfc="white",
            ms=4.5,
            lw=1.1,
            capsize=2,
            color=f"C{k}",
            label="conditional ceiling" if k == 0 else None,
        )
        a.errorbar(
            x,
            b["frozen_enr_gain_pp"],
            yerr=[
                b["frozen_enr_gain_pp"] - b["frozen_enr_lo"],
                b["frozen_enr_hi"] - b["frozen_enr_gain_pp"],
            ],
            fmt="s",
            ms=3.5,
            lw=1.1,
            capsize=2,
            color=f"C{k}",
            label="frozen policy" if k == 0 else None,
        )
        a.set_xticks(np.arange(len(b)))
        a.set_xticklabels([f"h={int(v)}" for v in b["horizon"]])
    a.axhline(0, color="0.4", lw=0.8)
    a.set(ylabel="gain over best-fixed (pp)")
    a.set_title("(b) what is reachable, and what is reached", fontsize=11.5)
    a.legend(frameon=False, fontsize=9, loc="upper left")

    fig2.tight_layout()
    fig2.savefig(OUT / "fig2_intervals.pdf", bbox_inches="tight")

    print(f"[fig] wrote {OUT}/fig1_mechanism.pdf and {OUT}/fig2_intervals.pdf", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
