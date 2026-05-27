# ===============================================
# Simple constants and default parameters for the
# HTC-HEOM code.
#
# Internal units:
#     energy : atomic units / Hartree
#     time   : atomic units
#     hbar   : 1
# ===============================================


# ===============================================
# Unit conversions to atomic units
# ===============================================
fs2au    = 41.341374575751
ps2au    = 1000.0 * fs2au
cminv2au = 4.55633e-6
eV2au    = 0.036749405469679
meV2au   = 1.0e-3 * eV2au
K2au     = 0.00000316678

au2fs    = 1.0 / fs2au


# ===============================================
# HTC system parameters in au
# ===============================================
Nmol = 3

ε_x = 0.0 * meV2au
ω_c = 0.0 * meV2au
Δ_c = ω_c - ε_x

Ω_R = 100.0 * meV2au
g   = Ω_R / (2.0 * (Nmol ** 0.5))


# ===============================================
# Bath parameters in au
#
# Drude-Lorentz spectral density:
#     J(ω) = 2 λ γ ω / (ω² + γ²)
# ===============================================
γ = 18.0 * cminv2au
λ = 50.0 * cminv2au

T_K = 300.0
T   = T_K * K2au
β   = 1.0 / T


# ===============================================
# Default HEOM parameters
# ===============================================
K_matsubara = 0
L_heom      = 10


# ===============================================
# Default propagation parameters
# ===============================================
dt_fs      = 0.25
tmax_fs    = 1000.0
save_every = 1

dt_au      = dt_fs * fs2au
tmax_au    = tmax_fs * fs2au


# ===============================================
# Optional ASCII aliases
# ===============================================
eps_x         = ε_x
omega_c       = ω_c
Delta_c       = Δ_c
Omega_R       = Ω_R
gamma         = γ
lambda_reorg  = λ
temperature_K = T_K
temperature   = T
beta          = β