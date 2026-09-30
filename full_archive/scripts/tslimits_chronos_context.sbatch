#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_chronos_ctx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=reports/tslimits/logs/chronos_ctx%j.log
set -euo pipefail

# Native-context sensitivity. Every Chronos-2 number in the paper was produced at
# a 2048-hour context, chosen as a common budget across the two foundation
# models. A reviewer pointed out that this is not the checkpoint's limit: the
# model card gives 8192, and the paper leans on "2048 hours is 85 days, so the
# window holds no complete annual cycle" as the explanation for the long-lead
# result. That makes the context a scientifically load-bearing choice rather than
# a neutral control, so it has to be varied.
#
# This runs the same emitter with --context changed and a distinct --prefix, so
# the 2048 arm on disk is untouched and the two are scored side by side. Nothing
# else moves: same checkpoint, same panel, same origin grid, same 19 countries.
#
# 8192 hours is 341 days. It still holds no complete annual cycle, but it holds
# four times the seasonal information, which is the comparison the reviewer asked
# for.
#
# Usage: sbatch --export=ALL,CTX=8192,H="168 720",BS=16,PREFIX=tsl_chronos_ctx8192_q \
#          scripts/tslimits_chronos_context.sbatch
#        sbatch --export=ALL,CTX=8192,H=8760,BS=8,PREFIX=tsl_chronos_ctx8192_h8760_q \
#          scripts/tslimits_chronos_context.sbatch

: "${CTX:?set CTX to the context length in hours, e.g. --export=ALL,CTX=8192}"
: "${H:?set H to one or more horizons in hours, e.g. H=\"168 720\"}"
BS="${BS:-16}"
# Target years; default is the 2025 evaluation window. The development window
# (2018-2024) is what a capability-matched c-static/router re-selection needs.
YEARS="${YEARS:-2025}"
# The panel must hold the target years; the 2026 window needs the ext2026 panel.
PANEL="${PANEL:-data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet}"
# One file per country per prefix, so every horizon family needs its own or a
# later job silently overwrites an earlier one's rows.
: "${PREFIX:?set PREFIX, e.g. PREFIX=tsl_chronos_ctx8192_q for the paired horizons}"

cd $REPO
mkdir -p reports/tslimits/logs reports/tslimits/fm

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi -L || true
echo "=== context=$CTX  horizons=$H  batch=$BS  years=$YEARS ==="
echo "=== start $(date -Is) ==="

# shellcheck disable=SC2086
time .venv-tier1-chronos2/bin/python scripts/d216_gpu_chronos_quantiles.py \
  --countries AT BE BG CH CZ DE ES FR GR HR HU IT LU NL PL PT RO SI SK \
  --horizons $H \
  --years $YEARS \
  --context "$CTX" \
  --batch-size "$BS" \
  --panel "$PANEL" \
  --out reports/tslimits/fm \
  --prefix "$PREFIX"

echo "=== end $(date -Is) exit=$? ==="
