#!/bin/bash
#SBATCH --job-name=scaling-sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-42
# One measurement per array task.  Submit with scripts/submit_sweep.sh, which
# regenerates the config file and sizes --array to match it.
#
# Timing is only meaningful on an uncontended GPU.  If the cluster packs several
# jobs per node, add `#SBATCH --exclusive` above (or request the whole node) --
# a co-tenant on the same device inflates every number here.

set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-configs/sweep.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-results/raw}"
PYTHON="${PYTHON:-python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${RESULTS_DIR}" logs

# One file per array task: concurrent appends to a single JSONL from several
# nodes interleave and corrupt lines, especially on NFS.  scaling.aggregate
# merges them afterwards.
OUT="${RESULTS_DIR}/${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.jsonl"

echo "=== task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -Is) ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

srun "${PYTHON}" -m scaling.benchmark \
    --config-file "${CONFIG_FILE}" \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --device cuda \
    --out "${OUT}" \
    ${EXTRA_ARGS}

echo "=== done at $(date -Is) -> ${OUT} ==="
