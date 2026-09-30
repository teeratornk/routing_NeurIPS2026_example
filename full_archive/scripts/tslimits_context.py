"""TS-LIMITS: does the foundation model do better with its native context?

Every Chronos-2 number in the paper is produced at a 2048-hour context, chosen as
a budget both foundation models could meet. That is defensible as a controlled
comparison, but the paper then explains the long-lead result by observing that
2048 hours is 85 days and so holds no complete annual cycle. Once the context
carries an explanatory load, holding it fixed is a scientific choice rather than
a neutral control, and a reviewer was right to ask what happens at the
checkpoint's own limit of 8192.

Everything else is held: same checkpoint, same panel, same origin grid, same 19
countries, same rollout cap. Only the context changes.

Out: reports/tslimits/context_sensitivity.csv
     reports/tslimits/tab_context.tex
Run: .venv/bin/python scripts/tslimits_context.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

OUT = Path("reports/tslimits")

# (horizon, label, cap note, 2048 source, 8192 source). At h=168 and h=720 the
# whole target fits one call, so no rollout cap is involved. At the long leads it
# is, and the arms below use the FROZEN caps the paper deploys -- 48 and 56 --
# not the library default, so this table and the horizon ladder quote the same
# configuration. Earlier revisions of this script compared default-cap arms,
# which were internally valid but two points away from the ladder and invited a
# reader to think the two tables disagreed.
ARMS = (
    (
        1,
        "1 h",
        "one call",
        "fm/tsl_chronos_q_*.parquet",
        "fm/tsl_chronos_ctx8192_h1h24_q_*.parquet",
    ),
    (
        24,
        "1 d",
        "one call",
        "fm/tsl_chronos_q_*.parquet",
        "fm/tsl_chronos_ctx8192_h1h24_q_*.parquet",
    ),
    (168, "1 w", "one call", "fm/tsl_chronos_q_*.parquet", "fm/tsl_chronos_ctx8192_q_*.parquet"),
    (720, "1 mo", "one call", "fm/tsl_chronos_q_*.parquet", "fm/tsl_chronos_ctx8192_q_*.parquet"),
    (
        8760,
        "1 yr",
        "cap 48",
        "rollout_ablation_h8760_cap48.parquet",
        "ctx8192_rollout_ablation_h8760_cap48.parquet",
    ),
    (
        17520,
        "2 yr",
        "cap 56",
        "rollout_ablation_h17520_cap56.parquet",
        "ctx8192_rollout_ablation_h17520_cap56.parquet",
    ),
)

# TimesFM-2.5 at its usable native context. The card says 16384 but the library
# enforces context + horizon <= 16384 with a 128-step internal block, so 16256.
# One file per (country, horizon), so the glob carries the horizon.
# The ceiling is horizon-dependent: the library enforces context + block(h) <=
# 16384 with block(h) the horizon rounded up to the output patch, 128 at h<=24,
# 256 at h=168, 768 at h=720. Each arm runs at 16384 minus its block.
TFM_ARMS = (
    (1, "1 h", 16256, "fm/tsl_timesfm_q_*_h1.parquet", "fm/tsl_timesfm_ctx16256_q_*_h1.parquet"),
    (24, "1 d", 16256, "fm/tsl_timesfm_q_*_h24.parquet", "fm/tsl_timesfm_ctx16256_q_*_h24.parquet"),
    (
        168,
        "1 w",
        16128,
        "fm/tsl_timesfm_q_*_h168.parquet",
        "fm/tsl_timesfm_ctxnative_q_*_h168.parquet",
    ),
    (
        720,
        "1 mo",
        15616,
        "fm/tsl_timesfm_q_*_h720.parquet",
        "fm/tsl_timesfm_ctxnative_q_*_h720.parquet",
    ),
)


def _mdape(pattern: str, horizon: int) -> tuple[float | None, int, frozenset[str]]:
    """Panel-median MdAPE: median within country, then median across countries."""
    files = sorted(glob.glob(str(OUT / pattern)))
    if not files:
        return None, 0, frozenset()
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d = d[d["horizon"] == horizon]
    if d.empty:
        return None, 0, frozenset()
    ape = 100.0 * (d["q50"] - d["actual"]).abs() / d["actual"].abs()
    return (
        float(ape.groupby(d["cc"]).median().median()),
        len(d),
        frozenset(d["cc"].unique()),
    )


def main() -> int:
    rows = []
    for h, label, cap, pat_a, pat_b in ARMS:
        a, na, cs_a = _mdape(pat_a, h)
        b, nb, cs_b = _mdape(pat_b, h)
        # A panel median over a different country set is a different estimand.
        # An in-flight run will happily produce a plausible-looking number on a
        # subset, so refuse to difference the arms unless they cover the same
        # countries.
        if cs_a and cs_b and cs_a != cs_b:
            print(
                f"[ctx] h={h}: country sets differ ({len(cs_a)} vs {len(cs_b)}), "
                f"arm incomplete, not compared",
                flush=True,
            )
            b = None
        rows.append(
            {
                "horizon": h,
                "label": label,
                "cap": cap,
                "ctx2048_mdape": None if a is None else round(a, 4),
                "ctx8192_mdape": None if b is None else round(b, 4),
                "delta_pp": None if (a is None or b is None) else round(b - a, 4),
                "n_2048": na,
                "n_8192": nb,
            }
        )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "context_sensitivity.csv", index=False)

    trows = []
    for h, label, ctx, pat_a, pat_b in TFM_ARMS:
        a, na, cs_a = _mdape(pat_a, h)
        b, nb, cs_b = _mdape(pat_b, h)
        if cs_a and cs_b and cs_a != cs_b:
            print(
                f"[ctx] timesfm h={h}: country sets differ ({len(cs_a)} vs {len(cs_b)}), "
                f"arm incomplete, not compared",
                flush=True,
            )
            b = None
        trows.append(
            {
                "horizon": h,
                "label": label,
                "native_context": ctx,
                "ctx2048_mdape": None if a is None else round(a, 4),
                "ctx16256_mdape": None if b is None else round(b, 4),
                "delta_pp": None if (a is None or b is None) else round(b - a, 4),
                "n_2048": na,
                "n_16256": nb,
            }
        )
    td = pd.DataFrame(trows)
    td.to_csv(OUT / "context_sensitivity_timesfm.csv", index=False)
    have_t = td.dropna(subset=["delta_pp"])
    if not have_t.empty:
        tex = [
            "\\begin{tabular}{llrrr}",
            "\\toprule",
            "lead & $h$ & context 2048 & native context & change \\\\",
            "\\midrule",
        ]
        for r in have_t.itertuples():
            tex.append(
                f"{r.label} & {r.horizon} & {r.ctx2048_mdape:.2f} & "
                f"{r.ctx16256_mdape:.2f} ({r.native_context}) & ${r.delta_pp:+.2f}$ \\\\"
            )
        tex += ["\\bottomrule", "\\end{tabular}", ""]
        (OUT / "tab_context_timesfm.tex").write_text("\n".join(tex))
    for r in td.itertuples():
        if pd.isna(r.delta_pp):
            print(
                f"[ctx] timesfm h={r.horizon:<5} not comparable "
                f"(n2048={r.n_2048} n16256={r.n_16256})",
                flush=True,
            )
        else:
            print(
                f"[ctx] timesfm h={r.horizon:<5} 2048 {r.ctx2048_mdape:7.3f}  "
                f"16256 {r.ctx16256_mdape:7.3f}  "
                f"{r.delta_pp:+.3f} pp",
                flush=True,
            )

    have = d.dropna(subset=["delta_pp"])
    tex = [
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "lead & $h$ & rollout & context 2048 & context 8192 & change \\\\",
        "\\midrule",
    ]
    for r in have.itertuples():
        tex.append(
            f"{r.label} & {r.horizon} & {r.cap} & {r.ctx2048_mdape:.2f} & "
            f"{r.ctx8192_mdape:.2f} & ${r.delta_pp:+.2f}$ \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    (OUT / "tab_context.tex").write_text("\n".join(tex))

    print(
        "[ctx] Chronos-2 panel-median MdAPE on the 2025 holdout, context held vs native", flush=True
    )
    for r in d.itertuples():
        # pandas turns None into NaN on the round trip, so `is None` never fires
        if pd.isna(r.delta_pp):
            print(
                f"[ctx] h={r.horizon:<6} {r.label:<5} not comparable "
                f"(n2048={r.n_2048} n8192={r.n_8192})",
                flush=True,
            )
            continue
        arrow = "better" if r.delta_pp < 0 else "worse"
        print(
            f"[ctx] h={r.horizon:<6} {r.label:<5} 2048 {r.ctx2048_mdape:7.3f}   "
            f"8192 {r.ctx8192_mdape:7.3f}   {r.delta_pp:+.3f} pp  {arrow}",
            flush=True,
        )
    if not have.empty:
        print(
            f"[ctx] native context helps in {int((have.delta_pp < 0).sum())}/{len(have)} "
            f"of the horizons measured",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
