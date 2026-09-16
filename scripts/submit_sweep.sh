#!/bin/bash
# Regenerate the sweep config, size the job array to it, and submit.
#SBATCH --time=06:00:00
#SBATCH --job-name=RTCNN_Cost_GPU
#SBATCH --output=logs/%j/output_%a.out
#SBATCH --error=logs/%j/error_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=Brain_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-1
# Arguments are forwarded to `python -m scaling.sweep`.  Throttle concurrency
# with MAX_CONCURRENT (default 4) and pass extra sbatch flags via SBATCH_ARGS.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_FILE="${CONFIG_FILE:-configs/sweep.jsonl}"
PYTHON="${PYTHON:-python}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

"${PYTHON}" -m scaling.sweep --out "${CONFIG_FILE}" "$@"
N=$(wc -l < "${CONFIG_FILE}")

if [ "${N}" -eq 0 ]; then
    echo "error: ${CONFIG_FILE} is empty" >&2
    exit 1
fi

mkdir -p logs results/raw
echo "submitting array 0-$((N - 1))%${MAX_CONCURRENT}"
sbatch --array="0-$((N - 1))%${MAX_CONCURRENT}" \
       --export=ALL,CONFIG_FILE="${CONFIG_FILE}",PYTHON="${PYTHON}" \
       ${SBATCH_ARGS:-} \
       scripts/sweep.slurm
