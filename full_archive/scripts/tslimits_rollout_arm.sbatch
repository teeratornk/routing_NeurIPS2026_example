#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_abl_arm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=reports/tslimits/logs/abl_arm_%j.log
set -euo pipefail

# One rollout-ablation arm, fully parameterised. Supersedes the two earlier
# sbatch wrappers, which each hardcoded a horizon or a year and so could not
# express "h=17520, the cap frozen on 2024, scored on 2025" -- the arm the
# selection protocol actually calls for.
#
# CTX defaults to 2048, the common budget every published arm uses. Setting it
# names the context in the tag, so a native-context arm cannot overwrite the
# 2048 one -- the failure mode that cost a run earlier in this revision.
#
# Usage: sbatch --export=ALL,H=17520,CAP=56,YEAR=2025 scripts/tslimits_rollout_arm.sbatch
#        sbatch --export=ALL,H=17520,CAP=56,YEAR=2025,CTX=8192 scripts/tslimits_rollout_arm.sbatch

: "${H:?set H}"; : "${CAP:?set CAP}"; : "${YEAR:?set YEAR}"
CTX="${CTX:-2048}"
cd $REPO
mkdir -p reports/tslimits/logs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
BASE="rollout_$([ "$YEAR" = 2024 ] && echo sel2024 || echo ablation)_h${H}_cap${CAP}"
TAG="$([ "$CTX" = 2048 ] && echo "$BASE" || echo "ctx${CTX}_${BASE}")"
echo "=== node $(hostname) h=$H cap=$CAP year=$YEAR ctx=$CTX tag=$TAG start $(date -Is) ==="
time .venv-tier1-chronos2/bin/python scripts/tslimits_rollout_ablation.py \
  --horizon "$H" --year "$YEAR" --context "$CTX" --batch-size 16 \
  --caps "$CAP" --tag "$TAG"
echo "=== end $(date -Is) ==="
