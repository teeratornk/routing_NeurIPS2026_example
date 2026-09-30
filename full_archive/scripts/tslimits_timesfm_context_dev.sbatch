#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_tfm_ctxdev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=reports/tslimits/logs/timesfm_ctxdev_%j.log
set -euo pipefail

# TimesFM-2.5 at its native context on the DEVELOPMENT window (2018-2024).
# The 2025 native arms (tslimits_timesfm_context*.sbatch) answered what the
# checkpoint does at its own limit; they cannot re-select c-static or refit the
# router, because both are chosen on 2018-2024 records that exist only at the
# 2048 budget. These arms supply that half, so a capability-matched policy
# family can be built beside the budget-matched one.
#
# The ceiling is horizon-dependent: the library enforces context + block(h)
# <= 16384 with blocks of 128/256/768 at h<=24/168/720, so H and CTX travel
# together and PREFIX must name the family (one file per country per prefix).
#
# Usage: sbatch --export=ALL,H="1 24",CTX=16256,PREFIX=tsldev_timesfm_ctx16256_q scripts/tslimits_timesfm_context_dev.sbatch
#        sbatch --export=ALL,H=168,CTX=16128,PREFIX=tsldev_timesfm_ctxnative_q scripts/tslimits_timesfm_context_dev.sbatch
#        sbatch --export=ALL,H=720,CTX=15616,PREFIX=tsldev_timesfm_ctxnative_q scripts/tslimits_timesfm_context_dev.sbatch

: "${H:?set H}"; : "${CTX:?set CTX}"; : "${PREFIX:?set PREFIX}"
BS="${BS:-32}"
YEARS="${YEARS:-2018 2019 2020 2021 2022 2023 2024}"
PANEL="${PANEL:-data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet}"
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
echo "=== horizons=$H context=$CTX batch=$BS prefix=$PREFIX years=$YEARS panel=$PANEL ==="
nvidia-smi -L || true

# shellcheck disable=SC2086
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
    --horizons $H \
    --years $YEARS \
    --context "$CTX" \
    --batch-size "$BS" \
    --panel "$PANEL" \
    --out reports/tslimits/fm \
    --prefix "$PREFIX"
echo "DONE exit=$?"
