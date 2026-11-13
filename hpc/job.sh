#!/bin/bash
# Run one command on an A100 (rdg hosts, the paper's): sbatch hpc/job.sh <command> [arguments]
#SBATCH --partition=sae --account=pilot_sae_gpu --gres=gpu:1 --constraint=rdg
#SBATCH --cpus-per-task=8 --mem=96G --time=12:00:00
#SBATCH --output=/data/scratch/acw794/kc-string-gpu/logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/environment.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$@"
