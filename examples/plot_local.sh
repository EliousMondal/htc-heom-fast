#!/usr/bin/env bash
set -euo pipefail

mkdir -p Plots

htc-heom-plot Data/htc_N5_K0_L4_UP_obs.npz \
  --time-unit ps \
  --output Plots/htc_N5_K0_L4_UP_obs.png
