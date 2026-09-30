"""TS-LIMITS: interval variants and cluster leverage for the headline routing gain.

The paper reports 95% percentile country-cluster intervals throughout, which is
the least accurate of the standard constructions on nineteen clusters. Two
questions follow that the percentile interval cannot answer on its own:

    1 does the verdict survive a better interval, and a correction for the
      eight model-horizon comparisons the paper makes;
    2 is any single country carrying a cell, which nineteen clusters makes
      easy to hide.

Both are pure post-processing. The bootstrap depends on nothing but the
per-country matrix of summed absolute percentage errors and the per-country
request counts, which `tslimits_country_static.py` now persists to
`cluster_matrices.npz`. No model is refitted here.

The statistic is the pooled mean, a ratio of resampled sums rather than an
average of per-country means, so every replicate recomputes numerator and
denominator over the drawn countries. Contrasts are differenced inside the
replicate, never across two separately-resampled quantities.

Out: reports/tslimits/interval_variants.csv
     reports/tslimits/cluster_leverage.csv
Run: .venv/bin/python scripts/tslimits_interval_variants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

OUT = Path("reports/tslimits")
MATS = OUT / "cluster_matrices.npz"

# Matches tslimits_country_static.py so the percentile column reproduces the
# published interval rather than merely resembling it.
B_REPRO = 2000
SEED = 0

# The Bonferroni bound sits at the 0.3125th percentile, which is draw 6 of 2000.
# A far larger B is needed before that tail means anything.
B_LARGE = 200_000
N_COMPARISONS = 8
ALPHA = 0.05

# The contrast the paper's Table 1 reports: the frozen router against the
# deployable per-(country, horizon) fixed choice.
CONTRAST = ("cc_static", "router_cc")

# The margin fixed before the results were read.
MARGIN_PP = 0.10


def _panel(sums: np.ndarray, cnt: np.ndarray, ix: np.ndarray) -> np.ndarray:
    """Pooled mean over the drawn countries, as a ratio of sums."""
    return sums[ix].sum(axis=0) / cnt[ix].sum()


def _theta(sums: np.ndarray, cnt: np.ndarray, ix: np.ndarray, a: int, b: int) -> float:
    q = _panel(sums, cnt, ix)
    return float(q[a] - q[b])


def _draws(sums: np.ndarray, cnt: np.ndarray, a: int, b: int, n_boot: int, rng) -> np.ndarray:
    """`n_boot` cluster-resampled contrast values."""
    k = sums.shape[0]
    out = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, k, k)
        q = _panel(sums, cnt, ix)
        out[i] = q[a] - q[b]
    return out


def _jackknife(sums: np.ndarray, cnt: np.ndarray, a: int, b: int) -> np.ndarray:
    """Delete-one-country values. Exact here, since the statistic is smooth."""
    k = sums.shape[0]
    return np.array([_theta(sums, cnt, np.delete(np.arange(k), i), a, b) for i in range(k)])


def _bca_bounds(draws: np.ndarray, theta: float, jack: np.ndarray, alpha: float) -> tuple:
    """BCa percentiles, plus the bias and acceleration that produced them."""
    prop = float((draws < theta).mean())
    # A degenerate proportion would send z0 to infinity; guard rather than emit nan.
    prop = min(max(prop, 1.0 / len(draws)), 1.0 - 1.0 / len(draws))
    z0 = float(norm.ppf(prop))
    d = jack.mean() - jack
    denom = 6.0 * (float((d**2).sum()) ** 1.5)
    acc = float((d**3).sum()) / denom if denom > 0 else 0.0
    lo_q, hi_q = [], []
    for z in (norm.ppf(alpha / 2.0), norm.ppf(1.0 - alpha / 2.0)):
        adj = z0 + (z0 + z) / (1.0 - acc * (z0 + z))
        (lo_q if z < 0 else hi_q).append(float(norm.cdf(adj)))
    lo = float(np.percentile(draws, 100.0 * lo_q[0]))
    hi = float(np.percentile(draws, 100.0 * hi_q[0]))
    return lo, hi, z0, acc


def main() -> int:
    if not MATS.is_file():
        print(f"[iv] missing {MATS}; run scripts/tslimits_country_static.py first", flush=True)
        return 1
    z = np.load(MATS, allow_pickle=True)
    cells = sorted({k.rsplit("|", 1)[0] for k in z.files})
    # Reproduce in the order the source script wrote them, so the shared RNG
    # stream lines up and the percentile column matches the published table.
    order = sorted(cells, key=lambda c: (c.split("|")[0], int(c.split("|")[1])))

    repro_rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    lev_rows: list[dict[str, object]] = []

    for cell in order:
        sums = z[f"{cell}|sums"]
        cnt = z[f"{cell}|cnt"].astype(float)
        names = list(z[f"{cell}|names"])
        labels = list(z[f"{cell}|labels"])
        a, b = names.index(CONTRAST[0]), names.index(CONTRAST[1])
        fm, h = cell.split("|")
        k = sums.shape[0]
        allr = np.arange(k)
        theta = _theta(sums, cnt, allr, a, b)

        # 1. the published interval, same B and same stream position
        d_repro = _draws(sums, cnt, a, b, B_REPRO, repro_rng)
        p_lo, p_hi = np.percentile(d_repro, 2.5), np.percentile(d_repro, 97.5)

        # 2. a large independent run for BCa and the Bonferroni tail
        big_rng = np.random.default_rng(SEED)
        d_big = _draws(sums, cnt, a, b, B_LARGE, big_rng)
        jack = _jackknife(sums, cnt, a, b)
        bca_lo, bca_hi, z0, acc = _bca_bounds(d_big, theta, jack, ALPHA)
        bonf_lo, bonf_hi, _, _ = _bca_bounds(d_big, theta, jack, ALPHA / N_COMPARISONS)
        bp_lo = float(np.percentile(d_big, 100.0 * (ALPHA / N_COMPARISONS) / 2.0))

        rows.append(
            {
                "fm": fm,
                "horizon": int(h),
                "point_pp": round(theta, 4),
                "pct_lo": round(float(p_lo), 4),
                "pct_hi": round(float(p_hi), 4),
                "bca_lo": round(bca_lo, 4),
                "bca_hi": round(bca_hi, 4),
                "bca_z0": round(z0, 4),
                "bca_accel": round(acc, 4),
                "bonf_bca_lo": round(bonf_lo, 4),
                "bonf_bca_hi": round(bonf_hi, 4),
                "bonf_pct_lo": round(bp_lo, 4),
                "pct_sig": bool(p_lo > 0),
                "bca_sig": bool(bca_lo > 0),
                "bonf_bca_sig": bool(bonf_lo > 0),
                "bonf_pct_sig": bool(bp_lo > 0),
                "bca_above_margin": bool(bca_lo > MARGIN_PP),
            }
        )

        # 3. cluster leverage: what one country is worth to point and width
        base_rng = np.random.default_rng(SEED)
        d_base = _draws(sums, cnt, a, b, B_REPRO, base_rng)
        w_base = float(np.percentile(d_base, 97.5) - np.percentile(d_base, 2.5))
        for i, cc in enumerate(labels):
            keep = np.delete(allr, i)
            loo_rng = np.random.default_rng(SEED)
            d_loo = _draws(sums[keep], cnt[keep], a, b, B_REPRO, loo_rng)
            l_lo, l_hi = np.percentile(d_loo, 2.5), np.percentile(d_loo, 97.5)
            lev_rows.append(
                {
                    "fm": fm,
                    "horizon": int(h),
                    "country": str(cc),
                    "loo_point_pp": round(_theta(sums, cnt, keep, a, b), 4),
                    "delta_point_pp": round(_theta(sums, cnt, keep, a, b) - theta, 4),
                    "loo_lo": round(float(l_lo), 4),
                    "loo_hi": round(float(l_hi), 4),
                    "delta_width": round(float(l_hi - l_lo) - w_base, 4),
                    "loo_sig": bool(l_lo > 0),
                    "loo_above_margin": bool(l_lo > MARGIN_PP),
                }
            )

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "interval_variants.csv", index=False)

    # Emitted here rather than hand-copied, so a rerun cannot leave the paper
    # quoting a stale interval.
    short = {"Chronos-2-Uni-ZS": "Chronos-2", "TimesFM-2.5-Uni-ZS": "TimesFM-2.5"}
    lines = [
        "\\begin{tabular}{llrllll}",
        "\\toprule",
        "FM & $h$ & gain & percentile & BCa & Bonferroni & $\\widehat{a}$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{short.get(str(r['fm']), r['fm'])} & {r['horizon']} & "
            f"${r['point_pp']:+.3f}$ & "
            f"[{r['pct_lo']:.3f}, {r['pct_hi']:.3f}] & "
            f"[{r['bca_lo']:.3f}, {r['bca_hi']:.3f}] & "
            f"[{r['bonf_bca_lo']:.3f}, {r['bonf_bca_hi']:.3f}] & "
            f"${r['bca_accel']:+.3f}$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (OUT / "tab_intervals.tex").write_text("\n".join(lines))
    print(f"[iv] wrote {OUT / 'tab_intervals.tex'}", flush=True)
    lev = pd.DataFrame(lev_rows)
    lev.to_csv(OUT / "cluster_leverage.csv", index=False)

    print(
        f"[iv] B={B_REPRO} for the reproduction, B={B_LARGE:,} for BCa and Bonferroni",
        flush=True,
    )
    for r in rows:
        print(
            f"[iv] {r['fm']:<20} h={r['horizon']:<4} {r['point_pp']:+.4f} "
            f"pct[{r['pct_lo']:+.4f},{r['pct_hi']:+.4f}] "
            f"BCa[{r['bca_lo']:+.4f},{r['bca_hi']:+.4f}] a={r['bca_accel']:+.4f} "
            f"Bonf-BCa[{r['bonf_bca_lo']:+.4f},{r['bonf_bca_hi']:+.4f}] "
            f"{'sig' if r['bonf_bca_sig'] else 'NS '}",
            flush=True,
        )
    print(
        f"[iv] excludes zero: percentile {int(d.pct_sig.sum())}/8, BCa {int(d.bca_sig.sum())}/8, "
        f"Bonferroni-BCa {int(d.bonf_bca_sig.sum())}/8, Bonferroni-percentile "
        f"{int(d.bonf_pct_sig.sum())}/8",
        flush=True,
    )
    print(
        f"[iv] BCa lower bound clears {MARGIN_PP} pp in {int(d.bca_above_margin.sum())}/8",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
