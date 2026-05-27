# ===============================================
# bath_drude.py
#
# Drude-Lorentz bath correlation coefficients and
# bath-channel construction for site-local HEOM.
#
# Internal units:
#     energy : atomic units / Hartree
#     time   : atomic units
#     hbar   : 1
# ===============================================

import numpy as np


# ===============================================
# Small helper functions
# ===============================================

def coth(x):
    """
    Hyperbolic cotangent.
    coth(x) = 1 / tanh(x)
    """
    return 1.0 / np.tanh(x)


def cot(x):
    """
    Ordinary cotangent.
    cot(x) = 1 / tan(x)
    """
    return 1.0 / np.tan(x)


# ===============================================
# Drude-Lorentz Matsubara decomposition
# ===============================================

def drude_matsubara_coefficients(λ, γ, β, K):
    """
    Return the Matsubara decomposition of the Drude-Lorentz
    bath correlation function
        C(t) = ∑ₖ cₖ exp(-νₖ t).

    The Drude-Lorentz spectral density convention is
        J(ω) = 2 λ γ ω / (ω² + γ²).

    Parameters
    ----------
    λ → float
        Reorganization energy in atomic units.
    γ → float
        Drude decay rate / bath cutoff frequency in atomic units.
    β → float
        Inverse temperature, β = 1 / (k_B T), in inverse atomic units.
    K → int
        Number of Matsubara terms retained beyond the Drude pole.
        K = 0 means only the Drude pole is retained.

    Returns
    -------
    ν → ndarray, shape (K+1,), float64
        Exponential decay rates.
    c → ndarray, shape (K+1,), complex128
        Complex correlation-function coefficients.

    Notes
    -----
    ν_0 = γ
    c_0 = λ γ [cot(β γ / 2) - i]

    For k ≥ 1:
        νₖ = 2πk / β
        cₖ = (4 λ γ / β) * νₖ / (νₖ² - γ²)
    """
    if λ < 0.0:
        raise ValueError("λ must be non-negative.")
    if γ <= 0.0:
        raise ValueError("γ must be positive.")
    if β <= 0.0:
        raise ValueError("β must be positive.")
    if K < 0:
        raise ValueError("K must be non-negative.")

    K = int(K)
    ν = np.empty(K + 1, dtype=np.float64)
    c = np.empty(K + 1, dtype=np.complex128)

    ν[0] = γ
    c[0] = λ * γ * (cot(0.5 * β * γ) - 1.0j)

    for k in range(1, K + 1):
        ν_k = 2.0 * np.pi * k / β
        denom = ν_k * ν_k - γ * γ

        if abs(denom) < 1.0e-14 * max(ν_k * ν_k, γ * γ, 1.0):
            raise ValueError(
                "Matsubara frequency is too close to γ. "
                "The coefficient would be numerically singular."
            )

        ν[k] = ν_k
        c[k] = (4.0 * λ * γ / β) * ν_k / denom

    return ν, c


def drude_terminator_delta(λ, γ, β, ν, c):
    """
    Return the standard Drude-Lorentz Matsubara terminator coefficient.

    If only a finite set of coefficients is kept explicitly,
        C(t) ≈ ∑ₖ cₖ exp(-νₖ t) + Δ_LT δ(t),

    then the usual Ishizaki-Tanimura / Tanimura terminator coefficient is
        Δ_LT = 2λ/(βγ) - iλ - ∑ₖ cₖ/νₖ.

    For the Drude-Lorentz Matsubara expansion, this should be real up to
    roundoff error. In the HEOM RHS this contributes a Markovian pure
    dephasing term of the form

        - Δ_LT [Q, [Q, ρ]].

    Parameters
    ----------
    λ, γ, β : float
        Same physical parameters used to generate ν and c.
    ν : ndarray
        Retained decay rates.
    c : ndarray
        Retained coefficients.

    Returns
    -------
    Δ_LT : float
        Real terminator coefficient in atomic units.
    """
    if λ < 0.0:
        raise ValueError("λ must be non-negative.")
    if γ <= 0.0:
        raise ValueError("γ must be positive.")
    if β <= 0.0:
        raise ValueError("β must be positive.")

    Δ_LT_complex = 2.0 * λ / (β * γ) - 1.0j * λ - np.sum(c / ν)

    if abs(Δ_LT_complex.imag) > 1.0e-10 * max(abs(Δ_LT_complex.real), 1.0):
        print("Warning: Drude terminator has a non-negligible imaginary part:", Δ_LT_complex)

    return float(Δ_LT_complex.real)


# ===============================================
# Site-local independent bath channel construction
# ===============================================

def make_site_bath_channels(Nsite, ν_site, c_site):
    """
    Expand one site's bath decomposition to Nsite independent,
    identical site-local baths.

    This function is generic for site-local HEOM. For the HTC
    first-exciton manifold, the system basis is usually
        {|C⟩, |1⟩, |2⟩, ..., |Nsite⟩},
    so the molecular site |s⟩ has system index s+1. This is
    returned as sys_α.

    Parameters
    ----------
    Nsite  → int
        Number of molecular sites / independent baths.
    ν_site → ndarray, shape (K+1,)
        Decay rates for one site bath.
    c_site → ndarray, shape (K+1,)
        Coefficients for one site bath.

    Returns
    -------
    ν_α     → ndarray, shape (Nsite*(K+1),), float64
        Decay rate for each HEOM channel α.
    c_α     → ndarray, shape (Nsite*(K+1),), complex128
        Correlation coefficient for each HEOM channel α.
    abs_c_α → ndarray, shape (Nsite*(K+1),), float64
        Absolute value |c_α| for scaled HEOM prefactors.
    site_α  → ndarray, shape (Nsite*(K+1),), int64
        Molecular site index for each channel, using 0-based site indexing.
    k_α     → ndarray, shape (Nsite*(K+1),), int64
        Exponential index for each channel.
    sys_α   → ndarray, shape (Nsite*(K+1),), int64
        System-basis index of the local projector Q_α.
        For HTC: sys_α = site_α + 1 because index 0 is the cavity.
    """
    if Nsite <= 0:
        raise ValueError("Nsite must be positive.")

    ν_site = np.asarray(ν_site, dtype=np.float64)
    c_site = np.asarray(c_site, dtype=np.complex128)

    if ν_site.ndim != 1:
        raise ValueError("ν_site must be a one-dimensional array.")
    if c_site.ndim != 1:
        raise ValueError("c_site must be a one-dimensional array.")
    if ν_site.shape[0] != c_site.shape[0]:
        raise ValueError("ν_site and c_site must have the same length.")

    Nsite   = int(Nsite)
    Kp1     = ν_site.shape[0]
    M       = Nsite * Kp1

    ν_α     = np.empty(M, dtype=np.float64)
    c_α     = np.empty(M, dtype=np.complex128)
    abs_c_α = np.empty(M, dtype=np.float64)
    site_α  = np.empty(M, dtype=np.int64)
    k_α     = np.empty(M, dtype=np.int64)
    sys_α   = np.empty(M, dtype=np.int64)

    α = 0
    for site in range(Nsite):
        for k in range(Kp1):
            ν_α[α]     = ν_site[k]
            c_α[α]     = c_site[k]
            abs_c_α[α] = abs(c_site[k])
            site_α[α]  = site
            k_α[α]     = k
            sys_α[α]   = site + 1
            α         += 1

    return ν_α, c_α, abs_c_α, site_α, k_α, sys_α


def make_htc_qdiag_dense(Nmol, sys_α):
    """
    Build a dense qdiag array for testing/reference RHS routines.

    For optimized HTC propagation we should NOT use this dense array.
    It is only useful for small-N debugging, where we compare the optimized
    projector-based RHS against a generic diagonal-coupling RHS.

    Basis:
        {|C⟩, |1⟩, |2⟩, ..., |Nmol⟩}

    For channel α,
        Q_α = |sys_α⟩⟨sys_α|.

    Therefore qdiag[α, i] = 1 if i == sys_α[α], otherwise 0.

    Parameters
    ----------
    Nmol  → int
        Number of molecules.
    sys_α → ndarray
        System index of each bath-channel projector.

    Returns
    -------
    qdiag → ndarray, shape (M, Nmol+1), float64
        Dense diagonal representation of all Q_α.
    """
    if Nmol <= 0:
        raise ValueError("Nmol must be positive.")

    sys_α = np.asarray(sys_α, dtype=np.int64)
    M     = sys_α.shape[0]
    d     = int(Nmol) + 1

    qdiag = np.zeros((M, d), dtype=np.float64)
    for α in range(M):
        s = sys_α[α]
        if s < 0 or s >= d:
            raise ValueError("sys_α contains an invalid system index.")
        qdiag[α, s] = 1.0

    return qdiag


# ===============================================
# Convenience wrapper for HTC with identical baths
# ===============================================

def build_drude_htc_bath(Nmol, λ, γ, β, K):
    """
    Build all HEOM bath-channel arrays for an HTC model with Nmol
    independent identical Drude-Lorentz baths.

    Parameters
    ----------
    Nmol → int
        Number of molecules.
    λ    → float
        Reorganization energy in atomic units.
    γ    → float
        Drude decay rate in atomic units.
    β    → float
        Inverse temperature in inverse atomic units.
    K    → int
        Number of Matsubara terms beyond the Drude pole.

    Returns
    -------
    ν_site, c_site : arrays for one bath.
    Δ_LT : float
        Standard low-temperature / Matsubara terminator coefficient per site.
    ν_α, c_α, abs_c_α, site_α, k_α, sys_α : arrays for all HEOM channels.
    """
    ν_site, c_site = drude_matsubara_coefficients(λ, γ, β, K)
    Δ_LT = drude_terminator_delta(λ, γ, β, ν_site, c_site)

    ν_α, c_α, abs_c_α, site_α, k_α, sys_α = make_site_bath_channels(Nmol, ν_site, c_site)

    return ν_site, c_site, Δ_LT, ν_α, c_α, abs_c_α, site_α, k_α, sys_α
