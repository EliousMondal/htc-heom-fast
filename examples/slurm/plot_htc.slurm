#!/bin/bash
#SBATCH -p polariton
#SBATCH --job-name=htc_plot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH -t 01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs Plots

htc-heom-plot Data/htc_N20_K0_L10_UP_obs.npz \
  --time-unit ps \
  --output Plots/htc_N20_K0_L10_UP_obs.png
