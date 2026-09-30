"""Reproduce the paper's headline numbers from the released per-request records.

No model is run, no gate is refitted and no GPU is needed. The example reads only
`full_archive/`: the absolute percentage error (APE) of both forecasters at every
request, the deployed gates' stored scores, and the thresholds frozen on 2024. It
re-derives the numbers the paper reports and checks each against the published
value, so a mismatch is an error rather than something you have to notice.

    python run_example.py

Why the gate is applied rather than refitted: two of its 22 request-time features,
anchor_load and recent_load_range, are load values. Like the raw load and the
forecasts, they are not redistributed, because the load provider's redistribution
terms are not stated and a forecast beside its error reveals the load. The eight
fitted gates are in `full_archive/gates/` for anyone who regenerates those features
from the source data. The structural forecaster's fitting library is not part of
this release; its per-request errors are, and they are what the results rest on.

Runs in seconds. On a many-core machine set OMP_NUM_THREADS=4.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from routing import escalate_above, panel_median, pooled_mean  # noqa: E402

ARCHIVE = Path(__file__).parent / "full_archive"
RECORDS = ARCHIVE / "records"
EXPECTED = Path(__file__).parent / "expected"
HORIZONS = (1, 24, 168, 720)
SELECTION_YEAR = 2024
KEY = ["country", "horizon", "anchor_t"]
# display name -> (record label, 2025 FM records, development FM records)
FM = {"Chronos-2": ("Chronos-2-Uni-ZS", "tsl_chronos_q", "tsldev_chronos_q"),
      "TimesFM-2.5": ("TimesFM-2.5-Uni-ZS", "tsl_timesfm_q", "tsldev_timesfm_q")}

_fails: list[str] = []


def check(label: str, got: float, want: float, tol: float = 1e-4) -> None:
    ok = abs(got - want) <= tol
    print(f"  {'OK  ' if ok else 'FAIL'} {label:<46} {got:>8.4f}  (paper {want})")
    if not ok:
        _fails.append(label)


def _fm_ape(prefix: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(RECORDS / "fm" / f"{prefix}_*.parquet")))
    if not files:
        raise FileNotFoundError(f"no records matching {prefix}_*.parquet under {RECORDS / 'fm'}")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d = d.rename(columns={"cc": "country", "origin_ts": "anchor_t", "ape": "ape_fm"})
    d["anchor_t"] = pd.to_datetime(d["anchor_t"])
    return d[[*KEY, "test_year", "ape_fm"]]


def paired(window: str, prefix: str) -> pd.DataFrame:
    """Join the structural and FM errors request by request, and prove the join."""
    st = pd.read_parquet(RECORDS / "structural" / f"per_origin_structural_{window}.parquet",
                         columns=[*KEY, "test_year", "ape"])
    st["anchor_t"] = pd.to_datetime(st["anchor_t"])
    st = st.rename(columns={"ape": "ape_st"})
    fm = _fm_ape(prefix).drop(columns="test_year")
    j = st.merge(fm, on=KEY, how="inner", validate="one_to_one")
    if len(j) != len(st):
        raise SystemExit(
            f"{window}/{prefix}: matched {len(j)} of {len(st)} requests. This is almost "
            f"always an incomplete download. Verify the archive:\n"
            f"  cd full_archive && sha256sum -c MANIFEST.sha256")
    if window == "2025" and (j["country"].nunique(), len(j)) != (19, 27735):
        raise SystemExit(f"{window}/{prefix}: {len(j)} requests over {j['country'].nunique()} "
                         f"countries; the release is 27735 over 19.")
    return j


def main() -> int:
    cs = pd.read_csv(EXPECTED / "country_static.csv")
    pe = pd.read_csv(EXPECTED / "paired_estimand.csv")
    ys = pd.read_csv(EXPECTED / "year_scale.csv").set_index("horizon")
    thr = pd.read_csv(ARCHIVE / "gates" / "thresholds.csv").set_index(["fm", "horizon"])
    served = pd.read_parquet(RECORDS / "served" / "served_per_request.parquet")
    served["anchor_t"] = pd.to_datetime(served["anchor_t"])

    print("\n1. The frozen router beats a per-(country, horizon) static policy")
    print("   deployed gate scores, threshold frozen on 2024, scored once on 2025\n")
    gains, pm_gains = [], []
    for name, (label, test_prefix, dev_prefix) in FM.items():
        test, dev = paired("2025", test_prefix), paired("dev", dev_prefix)
        sel = dev[dev["test_year"] == SELECTION_YEAR]
        for h in HORIZONS:
            t = test[test.horizon == h].merge(
                served.loc[(served.model == label) & (served.horizon == h),
                           [*KEY, "score", "st", "fm", "cc_static", "router_cc"]],
                on=KEY, how="inner", validate="one_to_one")
            if len(t) != int((test.horizon == h).sum()):
                raise SystemExit(f"{name} h={h}: stored scores cover {len(t)} requests")
            # the served record must describe the same errors as the APE records
            if not (np.allclose(t["st"], t["ape_st"], rtol=0, atol=1e-9)
                    and np.allclose(t["fm"], t["ape_fm"], rtol=0, atol=1e-9)):
                raise SystemExit(f"{name} h={h}: served record and APE records disagree")

            # c-static: per country, the FM only where it won on the 2024 selection year
            v = sel[sel.horizon == h]
            pick = v.groupby("country")[["ape_fm", "ape_st"]].mean().pipe(
                lambda x: x["ape_fm"] < x["ape_st"])
            base = np.where(t["country"].map(pick).to_numpy(), t["ape_fm"], t["ape_st"])

            # the router: escalate where the deployed gate's score clears the frozen number
            tau = float(thr.loc[(label, h), "threshold_pooled_mean"])
            routed = np.where(escalate_above(t["score"].to_numpy(), tau), t["ape_fm"], t["ape_st"])
            if not (np.array_equal(base, t["cc_static"]) and np.array_equal(routed, t["router_cc"])):
                raise SystemExit(f"{name} h={h}: recomputed decisions differ from the record")

            gain = pooled_mean(pd.Series(base), t["country"]) - pooled_mean(
                pd.Series(routed), t["country"])
            gains.append(gain)
            # The same decisions on the same requests, re-scored by the panel median.
            pm_gains.append(panel_median(pd.Series(base), t["country"])
                            - panel_median(pd.Series(routed), t["country"]))
            want = float(cs[(cs.fm == label) & (cs.horizon == h)].router_over_cc_pp.iloc[0])
            check(f"{name} h={h} gain over c-static (pp)", gain, round(want, 4))
    check("median gain across the 8 cells (pp)", float(np.median(gains)), 0.1913)

    print("\n2. The same decisions re-scored by a median gain about half as much")
    check("pooled-mean median gain (pp)", float(np.median(gains)), 0.1913)
    check("panel-median median gain (pp)", float(np.median(pm_gains)),
          round(float(pe.gain_median_samepolicy_pp.median()), 4))
    neg = sum(1 for x in pm_gains if x < 0)
    print(f"   ({neg} of 8 cells go negative under the median; the paper reports 1.)")
    if neg != 1:
        _fails.append("cells negative under the median")

    print("\n3. At one and two years the FM is not the better fixed choice")
    for h, cap in ((8760, 48), (17520, 56)):
        st = pd.read_parquet(RECORDS / "structural" / (
            "per_origin_structural_2025_year.parquet" if h == 8760
            else f"per_origin_structural_2025_h{h}.parquet"))
        fm = pd.read_parquet(RECORDS / "fm" / f"rollout_ablation_h{h}_cap{cap}.parquet")
        gap = panel_median(fm["ape"], fm["cc"]) - panel_median(st["ape"], st["country"])
        check(f"h={h} Chronos-2 minus structural (pp)", gap,
              float(ys.loc[h, "fm_tuned_minus_structural_pp"]))

    print()
    if _fails:
        print(f"FAILED: {len(_fails)} check(s) did not reproduce: {_fails}")
        return 1
    print("All checks reproduced the published values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
