"""Selective-escalation policies: when is the expensive forecaster worth calling?

Two forecasters score the same requests: a cheap one (the deployed structural
model, CPU, ~0.11 ms) and an expensive one (a time-series foundation
model on GPU). A routing policy sees only request-time information and decides
which answer to serve.

The quantities the paper reports:

    always_cheap / always_expensive   the two fixed policies
    best_fixed                        the better of the two, computed on the
                                      window it is handed, so on a test frame it
                                      is a hindsight quantity
    oracle                            per-request min; unattainable, and the
                                      gap to best_fixed is the entire budget a
                                      deployable policy can compete for
    frontier                          error as a function of the fraction of
                                      requests escalated, ordered by a score

The routing estimand is pooled mean APE, the mean over requests. The panel
median, the median over countries of each country's median APE, is the
concurrent submission's convention and is reported alongside: the two disagree,
and that disagreement is the paper's second contribution. Which one is in force
changes every number here, so it is always passed explicitly.

The frontier endpoints are exact by construction and asserted: escalating 0% is
always_cheap and escalating 100% is always_expensive. A policy that cannot
reproduce those has a ranking or indexing bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "ESTIMANDS",
    "escalation_frontier",
    "fixed_and_oracle",
    "margin_decomposition",
    "panel_median",
    "pooled_mean",
    "weighted_mape",
]


def panel_median(values: pd.Series, countries: pd.Series) -> float:
    """Median over countries of each country's median.

    The concurrent submission's convention, reported here alongside the routing
    estimand rather than in place of it.
    It is NOT additive, so the decomposition in `margin_decomposition` does not
    describe it; see that docstring.
    """
    frame = pd.DataFrame({"v": np.asarray(values, dtype=float), "c": np.asarray(countries)})
    return float(frame.groupby("c")["v"].median().median())


def pooled_mean(values: pd.Series, countries: pd.Series) -> float:
    """Mean over all requests. Additive, and the routing estimand.

    `countries` is accepted and ignored so every estimand shares one signature.
    """
    del countries
    return float(np.mean(np.asarray(values, dtype=float)))


def weighted_mape(values: pd.Series, countries: pd.Series, actual: pd.Series) -> float:
    """Load-weighted MAPE, sum|error| / sum|actual|, in percent.

    Also additive, and the operationally meaningful one: a 1% error on Germany
    and on Luxembourg are not the same event. `values` is per-request APE in
    percent, so the absolute error is recovered as APE * |actual| / 100.
    """
    del countries
    ape = np.asarray(values, dtype=float)
    a = np.abs(np.asarray(actual, dtype=float))
    return float((ape * a).sum() / a.sum())


# Estimands share the (values, countries) signature so a caller can be written
# once and parameterised. weighted_mape needs the load as well and is applied
# through a partial at the call site rather than being forced into this shape.
ESTIMANDS = {"pooled_mean": pooled_mean, "panel_median": panel_median}


def margin_decomposition(ape_cheap: pd.Series, ape_expensive: pd.Series) -> dict[str, float]:
    """Close the oracle gap in closed form, for an ADDITIVE loss.

    With D = L_cheap - L_expensive and min(a,b) = (a+b)/2 - |a-b|/2,

        G_oracle = min{E L_c, E L_e} - E min{L_c, L_e} = ( E|D| - |E D| ) / 2

    so the oracle gap is a DISPERSION statistic of the margin. It is positive
    whenever the relative winner varies across requests, whether or not that
    variation is predictable from anything. That is the whole reason an oracle
    gap is not routing headroom, and it is a fact about the arithmetic rather
    than about this dataset.

    For request-time information X the feature-measurable opportunity is

        G_X = ( E|E[D|X]| - |E D| ) / 2,   G_oracle - G_X = ( E|D| - E|E[D|X]| ) / 2 >= 0

    by Jensen, the last term being variation no policy measurable in X can
    exploit.

    None of this holds for the panel median, which is not an expectation. That
    is why the routing estimand is additive and the median is a robustness
    check.
    """
    c = np.asarray(ape_cheap, dtype=float)
    e = np.asarray(ape_expensive, dtype=float)
    d = c - e
    return {
        "mean_abs_margin": float(np.abs(d).mean()),
        "abs_mean_margin": float(abs(d.mean())),
        "oracle_gain_identity": float((np.abs(d).mean() - abs(d.mean())) / 2.0),
    }


def fixed_and_oracle(
    ape_cheap: pd.Series,
    ape_expensive: pd.Series,
    countries: pd.Series,
    estimand: str,
) -> dict[str, float]:
    """The two fixed policies, their better, and the per-request oracle.

    `estimand` is required rather than defaulted: which one is in force changes
    every number this returns, and a silent default is exactly the kind of
    knob this project has been bitten by before. Note that `best_fixed` here is
    computed on whatever window it is handed, so on a test frame it is a
    HINDSIGHT quantity, correct for oracle-gap accounting and wrong as a
    deployable baseline.
    """
    if estimand not in ESTIMANDS:
        raise ValueError(f"unknown estimand {estimand!r}; expected one of {sorted(ESTIMANDS)}")
    agg = ESTIMANDS[estimand]
    cheap = agg(ape_cheap, countries)
    expensive = agg(ape_expensive, countries)
    oracle = agg(
        pd.Series(np.minimum(np.asarray(ape_cheap), np.asarray(ape_expensive))), countries
    )
    best = min(cheap, expensive)
    out = {
        "always_cheap": cheap,
        "always_expensive": expensive,
        "best_fixed": best,
        "oracle": oracle,
        "oracle_gain_pp": best - oracle,
        "expensive_better_share": float((np.asarray(ape_expensive) < np.asarray(ape_cheap)).mean()),
    }
    if estimand == "pooled_mean":
        out.update(margin_decomposition(ape_cheap, ape_expensive))
    return out


def _escalate_mask(score: NDArray[np.float64], fraction: float) -> NDArray[np.bool_]:
    """Top-`fraction` of requests by score, resolving ties deterministically.

    Rank-based rather than threshold-based so the escalated count is exactly
    round(fraction * n) even when the score has heavy ties, which a raw
    quantile cut would not guarantee.
    """
    n = score.size
    k = round(fraction * n)  # ndarray.size is int, so round() already returns int
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    if k >= n:
        mask[:] = True
        return mask
    # argsort descending, stable, so equal scores break by original position
    order = np.argsort(-score, kind="stable")
    mask[order[:k]] = True
    return mask


def frozen_threshold(score_val: NDArray[np.float64], fraction: float) -> float:
    """The numeric score cut on the SELECTION window that targets `fraction`.

    `_escalate_mask` ranks whatever array it is handed, so applying it to the
    test window makes request i's decision depend on the other n-1 requests
    scored alongside it. That is batch-level adaptation to the test period: it
    presupposes the whole request pool is known before any of it is answered.
    An online router sees one request at a time and must compare it against a
    number decided in advance.

    So the selection window yields a NUMBER, not a fraction, and the number is
    what freezes. The realised test-window fraction is then an OUTCOME that can
    and does differ from the target, which is the honest thing to report.

    Returns +/-inf for the degenerate targets so that `escalate_above` gives
    exactly "never" and "always" rather than depending on a quantile of ties.
    """
    v = np.asarray(score_val, dtype=float)
    # A NaN score would make the quantile NaN, and `NaN > x` is False, so the
    # whole policy would collapse to always-cheap and report a gain of exactly
    # 0.00 as if it were a measurement. `escalation_frontier` already refuses to
    # let NaN scores fake a policy; the same refusal belongs on this path, which
    # now produces every deployable number in the paper.
    if not np.isfinite(v).all():
        raise ValueError(f"{int((~np.isfinite(v)).sum())} non-finite selection scores")
    if fraction <= 0.0:
        return float("inf")
    if fraction >= 1.0:
        return float("-inf")
    return float(np.quantile(v, 1.0 - fraction))


def escalate_above(score: NDArray[np.float64], threshold: float) -> NDArray[np.bool_]:
    """Escalate each request whose score clears `threshold`, independently.

    Strictly greater, matching `frozen_threshold`'s upper-tail convention: at a
    target of 0 the threshold is +inf and nothing escalates.
    """
    v = np.asarray(score, dtype=float)
    if not np.isfinite(v).all():
        raise ValueError(f"{int((~np.isfinite(v)).sum())} non-finite scores at escalation time")
    # -inf threshold means "escalate everything", which strict > would miss for a
    # score of exactly -inf; scores are finite by the guard above, so >= is only
    # needed for that sentinel.
    return v > threshold if threshold != float("-inf") else np.ones(v.shape, dtype=bool)


def escalation_frontier(
    ape_cheap: pd.Series,
    ape_expensive: pd.Series,
    countries: pd.Series,
    score: pd.Series,
    fractions: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0),
) -> pd.DataFrame:
    """Panel-median error when the top-`fraction` requests by `score` escalate.

    `score` must be computable at request time. Higher means more worth
    escalating. Its absolute scale is irrelevant; only the ordering is used.
    """
    c = np.asarray(ape_cheap, dtype=float)
    e = np.asarray(ape_expensive, dtype=float)
    s = np.asarray(score, dtype=float)
    if not (c.size == e.size == s.size == len(countries)):
        raise ValueError("ape_cheap, ape_expensive, countries and score must align")
    # NaN scores must not silently sort to one end and fake a policy
    if np.isnan(s).any():
        raise ValueError(f"score has {int(np.isnan(s).sum())} NaNs; impute or drop before ranking")

    rows = []
    for f in fractions:
        mask = _escalate_mask(s, f)
        served = np.where(mask, e, c)
        rows.append(
            {
                "escalated_fraction": f,
                "escalated_n": int(mask.sum()),
                "panel_median_ape": panel_median(pd.Series(served), countries),
            }
        )
    out = pd.DataFrame(rows)

    # Endpoints are exact by construction; a mismatch means a ranking bug.
    lo = out.loc[out["escalated_fraction"] == 0.0, "panel_median_ape"]
    hi = out.loc[out["escalated_fraction"] == 1.0, "panel_median_ape"]
    if not lo.empty:
        assert lo.iloc[0] == panel_median(pd.Series(c), countries), "0% must equal always-cheap"
    if not hi.empty:
        assert hi.iloc[0] == panel_median(pd.Series(e), countries), (
            "100% must equal always-expensive"
        )
    return out
