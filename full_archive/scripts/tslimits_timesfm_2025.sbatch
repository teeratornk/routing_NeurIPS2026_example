#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_timesfm_2025
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=reports/tslimits/logs/timesfm_2025_%j.log
set -euo pipefail

# TS-LIMITS Phase 0: TimesFM-2.5 per-origin forecasts on the frozen 2025 window.
# Companion to tslimits_chronos_2025.sbatch; same origin grid, same countries.
#
# Environment is copied verbatim from revision_timesfm_zeroshot.sbatch, including
# the pinned package revision, because TimesFM compiles per horizon and the
# published zero-shot numbers were produced with exactly these pins. The only
# differences are --years 2025, --panel (ext2025) and the output location.

cd $REPO
mkdir -p reports/tslimits/logs reports/tslimits/fm

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-8}"
export UV_HTTP_RETRY_DELAY="${UV_HTTP_RETRY_DELAY:-5}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
TIMESFM_PACKAGE_REVISION="${LF_TIER1_TIMESFM_PACKAGE_REVISION:-d720daa6786539c2566a44464fbda1019c0a82c0}"

echo "=== node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi -L || true

exec uv run \
  --no-project \
  --isolated \
  --with pandas==2.3.3 \
  --with pyarrow==20.0.0 \
  --with numpy==2.3.5 \
  --with torch==2.9.1 \
  --with "timesfm[xreg] @ git+https://github.com/google-research/timesfm.git@${TIMESFM_PACKAGE_REVISION}" \
  python scripts/revision_timesfm_zeroshot_quantiles.py \
    --countries AT BE BG CH CZ DE ES FR GR HR HU IT LU NL PL PT RO SI SK \
    --horizons 1 24 168 720 \
    --years 2025 \
    --context 2048 \
    --batch-size 256 \
    --panel data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet \
    --out reports/tslimits/fm \
    --prefix tsl_timesfm_q
echo "DONE exit=$?"
