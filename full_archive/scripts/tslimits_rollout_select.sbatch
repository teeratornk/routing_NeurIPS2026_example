#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_abl_sel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=reports/tslimits/logs/abl_sel_%j.log
set -euo pipefail

# Cap SELECTION on the 2024 window. The h=8760 sweep showed the library default
# is not the FM's best chunk size, but that best was read off 2025, which is the
# scored year. Choosing it there would be test-set selection, and the rest of
# this study freezes everything on 2024 precisely to avoid that. So the cap is
# picked here, on 2024, and then applied frozen to 2025.
#
# Usage: sbatch --export=ALL,H=8760,CAP=48 scripts/tslimits_rollout_select.sbatch

: "${H:?set H}"; : "${CAP:?set CAP}"
cd $REPO
mkdir -p reports/tslimits/logs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "=== node $(hostname) h=$H cap=$CAP year=2024 start $(date -Is) ==="
time .venv-tier1-chronos2/bin/python scripts/tslimits_rollout_ablation.py \
  --horizon "$H" --year 2024 --context 2048 --batch-size 16 \
  --caps "$CAP" --tag "rollout_sel2024_h${H}_cap${CAP}"
echo "=== end $(date -Is) ==="
