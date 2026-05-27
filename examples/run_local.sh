#!/usr/bin/env bash
set -euo pipefail

mkdir -p Data Plots

# Keep BLAS libraries from oversubscribing your CPU while Numba is parallel.
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

htc-heom-run \
  --Nmol 5 \
  --L 4 \
  --K-matsubara 0 \
  --lambda-cminv 50 \
  --gamma-cminv 18 \
  --temperature-K 300 \
  --Omega-R-mev 100 \
  --dt-fs 0.25 \
  --tmax-fs 100 \
  --save-every 1 \
  --progress-every 20 \
  --initial-state UP \
  --store obs \
  --output Data/htc_N5_K0_L4_UP_obs.npz
