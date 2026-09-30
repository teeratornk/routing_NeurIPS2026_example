"""TS-LIMITS: is the pooled-mean routing gain an artifact of a few huge APEs?

A mean is tail-sensitive, and the paper's headline is a mean-scored gain from
escalating a thin tail. The obvious objection is that the whole effect is a
handful of extreme percentage errors, which are exactly what a percentage error
produces when the denominator is small.

The transform is applied to the SCORING ONLY. The policy is held at the
pooled-mean escalation budget throughout, so this measures "is the gain robust
to how the tail is scored" and not "does a differently-tuned router still gain".
Re-selecting the budget under each transform confounds the two and was how an
earlier, unreproducible figure was produced.

One definition each, fixed here so the reported number has an artifact:

All gains are measured against c-static, the per-(country, horizon) rule chosen
on 2024, so they are directly comparable with the headline numbers.

    trim      drop requests where EITHER model's APE exceeds the cell's 99th
              percentile of the pooled two-model APE distribution. Union, so
              the surviving request set is identical for both models and the
              comparison stays paired.
    winsor    clip both APE series at that same threshold. Keeps every request.
    mw_mae    mean absolute error in megawatts. No percentage denominator, so
              a small-load request cannot manufacture a large error.
    wmape     load-weighted MAPE, i.e. sum(APE * |y|) / sum(|y|).

Out: reports/tslimits/tail_robustness.csv
Run: PYTHONPATH=src .venv/bin/python scripts/tslimits_tail_robustness.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from load_forecast.eval.routing import escalate_above, frozen_threshold

OUT = Path("reports/tslimits")
PCTL = 99.0
BUDGETS = tuple(round(0.05 * i, 2) for i in range(21))


def _mod(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_BS = _mod("tslimits_bootstrap")


def _gain(st: np.ndarray, fm: np.ndarray, mask: np.ndarray, cc: np.ndarray) -> float:
    """Served-vs-c-static gain under a plain pooled mean.

    The baseline is the per-(country, horizon) rule chosen on 2024, the same one
    every other practical claim in the paper is measured against, and it is
    recomputed under whatever transform the caller has already applied to st/fm.
    An earlier version used min(st.mean(), fm.mean()), the HINDSIGHT better-fixed
    model, which differed from c-static in half the cells and made these numbers
    incomparable with the headline ones.
    """
    return float(np.where(cc, fm, st).mean() - np.where(mask, fm, st).mean())


def main() -> int:
    rows: list[dict[str, Any]] = []
    cols = list(_BS.ENRICHED)

    for fm_label, (test_pat, dev_pat) in _BS.FM_PATTERNS.items():
        dev = _BS._paired("dev", dev_pat)
        test = _BS._paired("2025", test_pat)
        fit = dev[dev["test_year"].isin(_BS.FIT_YEARS)]
        val = dev[dev["test_year"].isin(_BS.VAL_YEARS)]

        for h in sorted(test["horizon"].unique()):
            f_h = fit[fit["horizon"] == h].reset_index(drop=True)
            v_h = val[val["horizon"] == h].reset_index(drop=True)
            t_h = test[test["horizon"] == h].reset_index(drop=True)

            st = t_h["ape_st"].to_numpy()
            fm = t_h["ape_fm"].to_numpy()
            y = np.abs(t_h["y_true"].to_numpy())

            reg = _BS._reg(cols).fit(f_h[cols], f_h["ape_st"] - f_h["ape_fm"])
            s_val = np.asarray(reg.predict(v_h[cols]), dtype=float)
            s_test = np.asarray(reg.predict(t_h[cols]), dtype=float)
            # budget from the pooled mean, then HELD FIXED for every transform
            b = min(
                (
                    np.where(
                        escalate_above(s_val, frozen_threshold(s_val, q)),
                        v_h["ape_fm"].to_numpy(),
                        v_h["ape_st"].to_numpy(),
                    ).mean(),
                    q,
                )
                for q in BUDGETS
            )[1]
            mask = escalate_above(s_test, frozen_threshold(s_val, b))

            # c-static: per country, take the FM only where it won on 2024
            sel = v_h.groupby("country")[["ape_fm", "ape_st"]].mean()
            pick_fm = sel["ape_fm"] < sel["ape_st"]
            missing = set(t_h["country"]) - set(pick_fm.index)
            assert not missing, f"h={h}: no 2024 selection for {sorted(missing)}"
            cc = t_h["country"].map(pick_fm).to_numpy(dtype=bool)

            thr = float(np.percentile(np.concatenate([st, fm]), PCTL))
            keep = (st <= thr) & (fm <= thr)
            st_w, fm_w = np.minimum(st, thr), np.minimum(fm, thr)

            # megawatt MAE: APE back to MW via the request's own actual
            mw_st, mw_fm = st * y / 100.0, fm * y / 100.0
            g_mw = float(
                np.where(cc, mw_fm, mw_st).mean() - np.where(mask, mw_fm, mw_st).mean()
            )

            def wm(v: np.ndarray, y: np.ndarray = y) -> float:
                return float((v * y).sum() / y.sum())

            g_wmape = wm(np.where(cc, fm, st)) - wm(np.where(mask, fm, st))

            base = _gain(st, fm, mask, cc)
            rows.append(
                {
                    "fm": fm_label,
                    "horizon": int(h),
                    "budget": b,
                    "n": len(st),
                    "pctl_threshold_ape": round(thr, 4),
                    "n_dropped": int((~keep).sum()),
                    "gain_pp": round(base, 4),
                    "gain_trimmed_pp": round(_gain(st[keep], fm[keep], mask[keep], cc[keep]), 4),
                    "gain_winsorised_pp": round(_gain(st_w, fm_w, mask, cc), 4),
                    "gain_mw_mae": round(g_mw, 4),
                    "gain_wmape_pp": round(g_wmape, 4),
                }
            )
            r = rows[-1]
            print(
                f"[tail] {fm_label:19s} h={h:<4} b={b:.2f}  base {base:+.4f}  "
                f"trim {r['gain_trimmed_pp']:+.4f}  winsor {r['gain_winsorised_pp']:+.4f}  "
                f"MW {r['gain_mw_mae']:+.3f}  wMAPE {r['gain_wmape_pp']:+.4f}  "
                f"(drop {r['n_dropped']}/{r['n']} at APE>{thr:.1f})",
                flush=True,
            )

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "tail_robustness.csv", index=False)
    print("", flush=True)
    for c in ("gain_trimmed_pp", "gain_winsorised_pp", "gain_mw_mae", "gain_wmape_pp"):
        print(
            f"[tail] {c:20s}: positive in {int((d[c] > 0).sum())}/{len(d)}, "
            f"range [{d[c].min():+.4f}, {d[c].max():+.4f}]",
            flush=True,
        )
    worst = d.loc[(d.gain_trimmed_pp / d.gain_pp).idxmin()]
    print(
        f"[tail] largest trim shrinkage: {worst.fm} h={int(worst.horizon)} "
        f"{worst.gain_pp:.4f} -> {worst.gain_trimmed_pp:.4f} pp "
        f"({100 * worst.gain_trimmed_pp / worst.gain_pp:.0f}% retained)",
        flush=True,
    )
    print(f"[tail] wrote {OUT}/tail_robustness.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
