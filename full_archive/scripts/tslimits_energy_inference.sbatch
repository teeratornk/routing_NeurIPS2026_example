#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_energy_inf
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=reports/tslimits/logs/energy_inference_%j.log
set -euo pipefail

# TS-LIMITS Phase 3: joules per forecast for real Chronos-2 and TimesFM-2.5
# inference, replacing Phase 0's synthetic matmul.
#
# --exclusive allocates a whole node, so the QOS you choose must allow a job to
# hold all of that node's CPUs; a QOS with a per-user CPU cap below the node's
# core count leaves the job pending indefinitely.
#
# gres=gpu:1 with --exclusive is deliberate: we reserve the whole node so it
# stays quiet, but expose ONE GPU to the job. Asking for gpu:8 made
# Chronos2Pipeline's device_map="auto" shard the model across cuda:0 and
# cuda:1, which failed every configuration with a device-mismatch error. Even
# had it run, a model spread over 8 GPUs while the sampler reads one of them
# would understate joules per forecast, so single-GPU placement is required for
# the measurement to mean anything, not merely to avoid the crash.
#
# --exclusive is mandatory, not a nicety. The idle baseline is subtracted from
# every marginal energy figure, and Phase 0 measured it at sd 12.17 W with the
# two FM jobs co-resident against 0.056 W on a quiet node. A shared node does
# not produce a slower measurement, it produces a wrong one. The script records
# the baseline sd and flags it, so a contaminated run is visible rather than
# silently folded into the table.
#
# Reviewer item 7: 5 repeats per configuration across ALL FOUR horizons, not a
# single run at two of them. Each repeat re-takes its own idle baseline, and the
# script now records torch/driver/CUDA versions, GPU name, power cap and clocks,
# and the EFFECTIVE power-sampling interval -- nvidia-smi is forked per reading,
# so the realised interval is roughly 0.22 s against a nominal 0.1 s, and each
# joules figure rests on order 85 samples. That belongs in the paper.
#
# Both models run on the SAME node in one allocation so they share hardware and
# a common idle baseline; comparing joules measured on different GPUs would not
# be a comparison. They need different environments (Chronos has a venv,
# TimesFM is pinned to a git revision via uv), hence the two invocations.

cd $REPO
mkdir -p reports/tslimits/logs

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
# --exclusive hands over EVERY GPU on the node no matter what --gres asks
# for, so CUDA_VISIBLE_DEVICES arrives as 0,1,...,7 and device_map="auto"
# shards the model across cards. Pin it to one card: we still hold the whole
# node (quiet idle baseline) but the model, and the power sampler, address a
# single GPU. Removing this silently reintroduces the sharding failure.
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
TIMESFM_PACKAGE_REVISION="${LF_TIER1_TIMESFM_PACKAGE_REVISION:-d720daa6786539c2566a44464fbda1019c0a82c0}"

echo "=== node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi -L || true
echo "=== co-resident jobs on this node (should be only this one) ==="
squeue -w "$(hostname)" -h -o "%i %j" || true

echo "=== [1/2] Chronos-2 ==="
.venv-tier1-chronos2/bin/python scripts/tslimits_energy_inference.py --model chronos

echo "=== [2/2] TimesFM-2.5 ==="
uv run \
  --no-project \
  --isolated \
  --with pandas==2.3.3 \
  --with pyarrow==20.0.0 \
  --with numpy==2.3.5 \
  --with torch==2.9.1 \
  --with "timesfm[xreg] @ git+https://github.com/google-research/timesfm.git@${TIMESFM_PACKAGE_REVISION}" \
  python scripts/tslimits_energy_inference.py --model timesfm

rc=$?
echo "DONE exit=${rc}"
exit "${rc}"
