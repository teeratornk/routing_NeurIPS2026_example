#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_year_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=reports/tslimits/logs/year_smoke_%j.log
set -euo pipefail

# Y1 smoke test: can Chronos-2 reach h=8760 at all, and how long does it take?
#
# The full year-scale run is 6,935 forecasts per model. Before spending that,
# find out on ONE country whether the library accepts a prediction length of
# 8760 from a 2048-hour context, and time it, so the full job can be sized
# rather than guessed. A 2048-context model reaching 8760 steps must extend
# autoregressively; this is the run that tells us what that costs.
#
# DE only, one origin per day, so 365 forecasts. If this takes T seconds, the
# full 19-country job is about 19T, and both models are about 19T x (the
# Chronos/TimesFM ratio measured at h=720, which was 3.6x at batch 32).
#
# Writes to a scratch prefix so a failed or slow smoke run can never be mistaken
# for the scored artifact.

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
  --countries DE \
  --horizons 8760 \
  --years 2025 \
  --context 2048 \
  --batch-size 8 \
  --panel data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet \
  --out reports/tslimits/fm_smoke \
  --prefix smoke_chronos_year

echo "=== end $(date -Is) exit=$? ==="
ls -la reports/tslimits/fm_smoke/ || true
