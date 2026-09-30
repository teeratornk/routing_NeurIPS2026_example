#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_abl_yr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=reports/tslimits/logs/abl_year_%j.log
set -euo pipefail

# The rollout ablation at the YEAR horizon, closing the caveat left by the
# h=720 run: that the per-pass penalty was measured at a month and not shown to
# transfer to a year.
#
# There is NO direct arm here and there cannot be. 8760 > 1024 native steps, so
# every arm is a rollout; what varies is how many. cap=64 is the DEPLOYED
# configuration (9 passes) and is the control: if it reproduces the published
# 11.3193, the other arms are interpretable.
#
#   cap 64 ->  9 passes (deployed)   cap 32 -> 18 passes
#   cap 48 -> 12 passes              cap 16 -> 35 passes
#
# One arm per job, because 62 pass-units in a single job would exceed the 8 h
# limit. Each writes its own tag; they are merged afterwards.
#
# Usage: sbatch --export=ALL,CAP=32 scripts/tslimits_rollout_ablation_year.sbatch

: "${CAP:?set CAP, e.g. --export=ALL,CAP=32}"
cd $REPO
mkdir -p reports/tslimits/logs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "=== node $(hostname) cap=$CAP start $(date -Is) ==="
time .venv-tier1-chronos2/bin/python scripts/tslimits_rollout_ablation.py \
  --horizon 8760 --context 2048 --batch-size 16 \
  --caps "$CAP" --tag "rollout_ablation_h8760_cap${CAP}"
echo "=== end $(date -Is) ==="
