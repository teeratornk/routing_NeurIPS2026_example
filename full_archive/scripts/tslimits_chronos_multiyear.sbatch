#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_chronos_my
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=reports/tslimits/logs/chronos_my_%j.log
set -euo pipefail

# Y2/Y3: Chronos-2 per-origin forecasts at the two- and three-year horizons, on
# the same frozen 2025 window and the same origin grid as every other horizon
# (one origin per country per day, 19 countries, 6,935 forecasts). Sister of
# tslimits_chronos_year.sbatch, which covers h=8760; the horizon is passed in so
# the two leads run as separate jobs and a failure at one does not cost the other.
#
# The point of these rows is what reaching a horizon by rollout costs. Chronos-2
# emits 1024 steps in one shot (64 output patches x 16), so:
#   h= 8760 ->  9 passes   (already run, 11.32 MdAPE, +6.03 pp behind structural)
#   h=17520 -> 18 passes
#   h=26280 -> 26 passes
# against exactly one evaluation for the structural composition at every lead.
# The library warns "We recommend keeping prediction length <= 1024" on each
# call; that warning is the measurement, not a caveat on it.
#
# Cost, extrapolated from job 172051 (h=8760, real 76m57s): weighting the first
# pass at batch B and each unroll at 9B gives 1.9x and 3.1x, so about 2.5 h and
# 4 h. Both fit the 8 h limit with margin.
#
# TimesFM-2.5 is deliberately absent, for the same reason as at one year: its
# continuous quantile head is capped at 1024 steps and raises above it, and that
# head is the configuration behind every published TimesFM number in this study.
# A multi-year TimesFM column would be a protocol change, not a longer run.
#
# Usage: sbatch --export=ALL,H=26280 scripts/tslimits_chronos_multiyear.sbatch

: "${H:?set H to the horizon in hours, e.g. --export=ALL,H=26280}"

cd $REPO
mkdir -p reports/tslimits/logs

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi -L || true
echo "=== horizon H=$H  expected unroll passes: $(( (H + 1023) / 1024 )) ==="
echo "=== start $(date -Is) ==="

time .venv-tier1-chronos2/bin/python scripts/d216_gpu_chronos_quantiles.py \
  --countries AT BE BG CH CZ DE ES FR GR HR HU IT LU NL PL PT RO SI SK \
  --horizons "$H" \
  --years 2025 \
  --context 2048 \
  --batch-size 16 \
  --panel data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet \
  --out reports/tslimits/fm \
  --prefix "tsl_chronos_h${H}_q"

echo "=== end $(date -Is) exit=$? ==="
