# Which command regenerates which table

Run every command from the archive root after placing records/ and results/
under reports/tslimits/ (the scripts read that path), with the paper's
environment (Python 3.11, scikit-learn 1.5.2, numpy 1.26.4, pandas 2.x):

    export PYTHONPATH=src

| table | command | rests on |
|---|---|---|
| tab_main, tab_paired, tab_threshold, tab_year, tab_capselect, tab_capselect8192, tab_policy2x2, tab_noweather, tab_controls, tab_policies, tab_countrybreak, tab_why | `python scripts/tslimits_tables.py` | records (bootstrap_ci_*.csv, country_static.csv, country_breakdown.csv, paired_estimand.csv, year_scale.csv come from `tslimits_bootstrap.py`, `tslimits_country_static.py`, `tslimits_paired_estimand.py`, `tslimits_year_scale.py`; why_diagnostics.csv and why_concentration.csv from `tslimits_why.py`; tab_capselect and tab_capselect8192 read the rollout-selection records; all on records) |
| tab_estimands (panel-median and pooled-mean columns) | `python scripts/tslimits_tables.py` | records |
| tab_estimands (load-weighted MAPE and MW MAE columns) | `python scripts/tslimits_tables.py` | aggregates: needs target loads, shipped as computed values in results/ |
| tab_agg2x2, tab_longlead | `python scripts/tslimits_headroom.py` | records |
| tab_intervals | `python scripts/tslimits_interval_variants.py` | records (cluster_matrices.npz) |
| tab_dst | `python scripts/tslimits_dst_sensitivity.py` | records |
| tab_thrstab | `python scripts/tslimits_threshold_stability.py` | records |
| tab_selection | `python scripts/tslimits_selection_bootstrap.py` | records |
| tab_gate_pareto | `TSLIMITS_GATE=shallow python scripts/tslimits_country_static.py; TSLIMITS_GATE=ridge python scripts/tslimits_country_static.py; python scripts/tslimits_gate_pareto.py` | records + aggregates (router_cost*.json latency measurements) |
| tab_latency | `python scripts/tslimits_latency.py` | aggregates (router_cost.json, energy_inference*.json) |
| tab_energy | `python scripts/tslimits_tables.py`, from `energy_inference_*.json` measured by `tslimits_energy_inference.py` on an H200 | aggregates |
| tab_context, tab_context_timesfm | `python scripts/tslimits_context.py` | records |
| tab_native_family | `TSLIMITS_FM_FAMILY=native python scripts/tslimits_bootstrap.py; TSLIMITS_FM_FAMILY=native python scripts/tslimits_country_static.py; python scripts/tslimits_native_family.py` | records |
| tab_longlead_cstatic | `python scripts/tslimits_longlead_cstatic.py` | records |
| tab_nogdp | `python scripts/tslimits_nogdp.py` | records (structural no-GDP arms) |
| tab_covariates | `python scripts/tslimits_covariates.py` | records |
| tab_replication2026 | `python scripts/tslimits_replication_2026.py` | records; `TSLIMITS_START_2026=2026-01-07` and `TSLIMITS_STRUCT_2026=2026_nogdp` give the two sensitivity arms |
| tab_replication2025q1 | `TSLIMITS_REPL_WINDOW=2025q1 python scripts/tslimits_replication_2026.py` | records (the January-March 2025 seasonal control) |
| tab_costsel | `PYTHONPATH=../src python scripts/tslimits_cost_selection.py --stage select --selection records/served/selection_2024_per_request.parquet --served records/served/served_per_request.parquet --country-static results/country_static.csv --out results; python scripts/tslimits_tables.py` | records (`--stage scores`, which refits the gates to write the 2024 record, needs the two load-valued features, which are not redistributed; the record itself is shipped) |
| tab_byyear | `python scripts/tslimits_by_year.py` | records |
| tail_robustness.csv (text only) | `python scripts/tslimits_tail_robustness.py` | aggregates: MW MAE and load-weighted columns need target loads |
| baselines.csv wmape_* (text only) | `python scripts/tslimits_baselines.py` | aggregates |
| fig_ladder, fig_estimand, fig_netgain, fig_frontier* | `python scripts/tslimits_fig_ladder.py; python scripts/tslimits_fig_estimand.py; python scripts/tslimits_fig_netgain.py; python scripts/tslimits_frontier.py` | records + energy aggregates for the frontier |
