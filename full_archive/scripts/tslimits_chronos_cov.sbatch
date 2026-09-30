#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_chronos_cov
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=reports/tslimits/logs/chronos_cov_%j.log
set -euo pipefail
# Chronos-2 with known-future calendar covariates; see tslimits_chronos_covariates.py.
# Usage: sbatch --export=ALL,ARM=cal,H="1 24 168 720",YEARS=2025,PREFIX=tslcov_cal_q scripts/tslimits_chronos_cov.sbatch
: "${ARM:?set ARM (cal|calclim)}"; : "${H:?set H}"; : "${PREFIX:?set PREFIX}"
YEARS="${YEARS:-2025}"; BS="${BS:-32}"; CTX="${CTX:-2048}"
CAP="${CAP:-}"  # rollout cap for the long leads; empty = library default
COUNTRIES="${COUNTRIES:-AT BE BG CH CZ DE ES FR GR HR HU IT LU NL PL PT RO SI SK}"
PANEL="${PANEL:-data/feature_store/multi_resolution/hourly/clean20_hourly_panel_d214_ode_ext2025.parquet}"
cd $REPO
mkdir -p reports/tslimits/logs reports/tslimits/fm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "=== node $(hostname) arm=$ARM h=$H years=$YEARS ctx=$CTX prefix=$PREFIX start $(date -Is) ==="
# shellcheck disable=SC2086
time .venv-tier1-chronos2/bin/python scripts/tslimits_chronos_covariates.py \
  --arm "$ARM" --countries $COUNTRIES --horizons $H --years $YEARS --context "$CTX" \
  --batch-size "$BS" --panel "$PANEL" --out reports/tslimits/fm --prefix "$PREFIX" \
  ${CAP:+--max-output-patches "$CAP"}
echo "=== end $(date -Is) exit=$? ==="
