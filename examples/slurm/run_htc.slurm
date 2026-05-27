#!/bin/bash
#SBATCH -p polariton
#SBATCH --job-name=htc_heom
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH -t 2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs Data

export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1
unset OMP_NESTED

htc-heom-run \
  --Nmol 20 \
  --L 10 \
  --K-matsubara 0 \
  --lambda-cminv 50 \
  --gamma-cminv 18 \
  --temperature-K 300 \
  --Omega-R-mev 100 \
  --dt-fs 1.0 \
  --tmax-fs 1000 \
  --save-every 1 \
  --progress-every 10 \
  --initial-state UP \
  --store obs \
  --max-ram-gb 200 \
  --output Data/htc_N20_K0_L10_UP_obs.npz
