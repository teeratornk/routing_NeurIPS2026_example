#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_tfm_ctx_h720
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=reports/tslimits/logs/timesfm_ctx%j.log
set -euo pipefail

# TimesFM-2.5 at its native context. The card says 16384, but the library
# enforces context + horizon <= 16384 with a 128-step internal block even at
# h=1, so the usable ceiling is 16256. That is what this runs at,, the counterpart of the Chronos-2
# 8192 arm. Every published TimesFM number uses 2048 as a common budget with
# Chronos-2; a reviewer asked what the checkpoint does at its own limit. Same
# origin grid, same countries, same horizons; only the context and the batch
# change, the latter because 8x the context at batch 256 will not fit.
# Distinct prefix, or this overwrites the 2048 arm one file per country.
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
    --horizons 720 \
    --years 2025 \
    --context 15616 \
    --batch-size 32 \
    --panel data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet \
    --out reports/tslimits/fm \
    --prefix tsl_timesfm_ctxnative_q
echo "DONE exit=$?"
