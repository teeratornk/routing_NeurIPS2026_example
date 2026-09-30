#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_chronos_year
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=reports/tslimits/logs/chronos_year_%j.log
set -euo pipefail

# Y1: Chronos-2 per-origin forecasts at the one-year horizon on the frozen 2025
# window, same origin grid as every other horizon (one origin per country per
# day, anchors in 2024, targets in 2025, 19 countries, 6,935 forecasts).
#
# Pairs with reports/tslimits/per_origin_structural_2025_year.parquet, which
# reproduces the published 5.29 at h=8760 (gate PASS, 6,935 origins).
#
# This horizon is NOT native. Chronos-2's single-shot prediction length is
# 1024 steps (64 output patches x 16), so 8760 is reached by nine chained
# autoregressive unroll passes, and the library says so:
#   UserWarning: We recommend keeping prediction length <= 1024.
# That is the measurement, not a caveat on it: the paper's claim is about what
# reaching a year by rollout costs a model whose native resolution is hourly.
#
# Sized from the smoke run (job 172033): DE alone, 365 origins, 7m03s including
# model load, so 19 countries is on the order of two hours. The smoke run also
# recorded the calibration consequence, 80% intervals covering 28.5%.
#
# TimesFM-2.5 is deliberately absent. Its continuous quantile head is capped at
# 1024 steps and raises above it, and that head is the configuration behind
# every published TimesFM number in this study, so a year row for it would be a
# protocol change rather than a longer run.

cd $REPO
mkdir -p reports/tslimits/logs reports/tslimits/fm_smoke

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi -L || true
echo "=== start $(date -Is) ==="

time .venv-tier1-chronos2/bin/python scripts/d216_gpu_chronos_quantiles.py \
  --countries AT BE BG CH CZ DE ES FR GR HR HU IT LU NL PL PT RO SI SK \
  --horizons 8760 \
  --years 2025 \
  --context 2048 \
  --batch-size 16 \
  --panel data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet \
  --out reports/tslimits/fm \
  --prefix tsl_chronos_year_q

echo "=== end $(date -Is) exit=$? ==="
ls -la reports/tslimits/fm_smoke/ || true
