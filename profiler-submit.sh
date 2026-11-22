#!/bin/bash
# Submit from the repository directory: sbatch profiler-submit.sh
#SBATCH --job-name=transformer-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=artifacts/profiler_trace/profiler-%j.out
#SBATCH --error=artifacts/profiler_trace/profiler-%j.out
#SBATCH --mail-user=m4sedagh@uwaterloo.ca
#SBATCH --mail-type=ALL

# Stop if environment setup or a command fails.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"

# Initialize Conda explicitly for the non-interactive batch shell.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mlp

exec python -u scripts/profiler.py
