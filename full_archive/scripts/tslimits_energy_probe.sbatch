#!/usr/bin/env bash
# Set your cluster's partition, QOS and account with #SBATCH lines or sbatch flags.
#SBATCH --job-name=tsl_energy_probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=reports/tslimits/logs/energy_probe_%j.log
set -euo pipefail

# TS-LIMITS Phase 0: can serving energy be measured directly on a compute node?
#
# Cheap and independent of the two FM jobs, so run it first. The answer decides
# whether Phase 1 reports measured joules or estimated ones, which changes what
# the paper is allowed to claim. On the login node intel-rapl exists but
# energy_uj is root-only, so the CPU side is unresolved; NVML should cover the
# GPU side, which is how foundation models actually get served anyway.

cd $REPO
mkdir -p reports/tslimits/logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi -L || true

# Prefer the Chronos env (it has torch, so the synthetic GPU load runs);
# fall back to the plain venv, where the probe still answers the RAPL and
# NVML-availability questions without applying load.
if [ -x .venv-tier1-chronos2/bin/python ]; then
  PY=.venv-tier1-chronos2/bin/python
else
  PY=.venv/bin/python
fi
echo "=== interpreter: $PY ==="

"$PY" scripts/tslimits_energy_probe.py
echo "DONE exit=$?"
