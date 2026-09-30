#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_chronos_2025
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=reports/tslimits/logs/chronos_2025_%j.log
set -euo pipefail

# TS-LIMITS Phase 0: Chronos-2 per-origin forecasts on the frozen 2025 window.
#
# Pairs with reports/tslimits/per_origin_structural_2025.parquet, which already
# holds the 19-country x 4-horizon structural side (27,735 origins). Both use
# common_protocol._origin_positions, so the origin grids are identical by
# construction and the two sides join on (country, horizon, origin_ts).
#
# No new runner: d216_gpu_chronos_quantiles.py already emits per-origin rows
# (cc, test_year, horizon, origin_ts, target_ts, actual, q025..q975). It gained
# a --panel flag, defaulting to the pre-2025 panel, so every earlier invocation
# still reproduces byte-for-byte.
#
# MK is excluded: the pre-registered D224 completeness filter drops it (77.3%).

cd $REPO
mkdir -p reports/tslimits/logs reports/tslimits/fm

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi -L || true

.venv-tier1-chronos2/bin/python scripts/d216_gpu_chronos_quantiles.py \
  --countries AT BE BG CH CZ DE ES FR GR HR HU IT LU NL PL PT RO SI SK \
  --horizons 1 24 168 720 \
  --years 2025 \
  --context 2048 \
  --batch-size 32 \
  --panel data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet \
  --out reports/tslimits/fm \
  --prefix tsl_chronos_q

echo "DONE exit=$?"
