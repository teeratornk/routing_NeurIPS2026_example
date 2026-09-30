#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_ablation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=reports/tslimits/logs/ablation_%j.log
set -euo pipefail

# The direct-versus-recursive ablation the paper names as missing. Horizon is
# HELD at 720, which is inside Chronos-2's native window (ceil(720/16) = 45
# patches), so the 45-cap arm is a genuine one-pass DIRECT forecast. Lowering
# max_output_patches forces the same model to the same target in 2, 4, 6, 9 and
# 15 passes through the library's own quantile-trajectory rollout.
#
# The 9-pass arm is the decisive one: nine is what the model spends to reach
# h=8760. If nine passes cost little at h=720, pass count does not explain the
# year-scale gap and the paper's hypothesis is refuted.

cd $REPO
mkdir -p reports/tslimits/logs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "=== node $(hostname) start $(date -Is) ==="
time .venv-tier1-chronos2/bin/python scripts/tslimits_rollout_ablation.py \
  --horizon 720 --context 2048 --batch-size 16 --tag rollout_ablation_h720
echo "=== end $(date -Is) ==="
