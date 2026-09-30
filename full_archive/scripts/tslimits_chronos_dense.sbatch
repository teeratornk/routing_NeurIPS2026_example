#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_chronos_dense
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=reports/tslimits/logs/chronos_dense_%j.log
set -euo pipefail
# Dense-grid sensitivity, ONE cell: Chronos-2 at h=24 on every hour of 2025 (stride 1).
# Not the published protocol; a check that the rotating one-origin-per-day grid does not
# carry the routed result. Own prefix, so the stride-24 records are untouched.
cd $REPO
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "=== node $(hostname) start $(date -Is) ==="
time .venv-tier1-chronos2/bin/python scripts/d216_gpu_chronos_quantiles.py \
  --countries AT BE BG CH CZ DE ES FR GR HR HU IT LU NL PL PT RO SI SK \
  --horizons 24 --years 2025 --context 2048 --batch-size 64 --stride 1 \
  --panel data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet \
  --out reports/tslimits/fm --prefix tsldense_chronos_h24_q
echo "=== end $(date -Is) exit=$? ==="
