# ============================================================
# Bath-channel bookkeeping for naive matrix-free HTC-HEOM.
#
# Basis convention used everywhere:
#     index 0      : |C⟩      = one cavity photon
#     index n >= 1 : |n⟩      = molecule n excited
#
# Therefore each local bath coupling operator is
#
#     Q_n = |n⟩⟨n|,
# and the system-basis index of molecular site n is n.
#
# Molecule/site indexing convention inside this file:
#
#     site = 0, 1, ..., Nmol-1      Python/array index
#     sys  = 1, 2, ..., Nmol        HTC basis index
#
# Channel ordering convention:
#
#     α = site * Kp1 + k,
#
# where Kp1 = K + 1 is the number of exponentials per site bath.
# ============================================================

import numpy as np
from numba import njit

from .bath_drude import (
    drude_matsubara_coefficients,
    drude_terminator_delta,
)


# ============================================================
# Small checks
# ============================================================

def _check_Nmol(Nmol):
    Nmol = int(Nmol)
    if Nmol <= 0:
        raise ValueError("Nmol must be positive.")
    return Nmol


def _as_1d_float_array(x, name):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(name + " must be one-dimensional.")
    return x


def _as_1d_complex_array(x, name):
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1:
        raise ValueError(name + " must be one-dimensional.")
    return x


# ============================================================
# Channel indexing utilities
# ============================================================

def channel_id(site, k, Kp1):
    """
    Return the channel index
        α = site * Kp1 + k.
    site and k are 0-based. 
    Kp1 is the number of exponentials per site bath,
        kp1 = K + 1, 
    where K is the number of Matsubara terms beyond the Drude pole.
    
    For eg:
    For a bath with only the Drude pole (K=0, Kp1=1), 
    the channels are ordered as
        1) site=0, k=0, Kp1=1  ⇒ α = 0
        2) site=1, k=0, Kp1=1  ⇒ α = 1
    For a bath with the Drude pole and 3 Matsubara terms (K=3, Kp1=4), 
    the channels are ordered as
        3) site=0, k=0, Kp1=4  ⇒ α = 0
        4) site=0, k=1, Kp1=4  ⇒ α = 1
        5) site=1, k=0, Kp1=4  ⇒ α = 4
        6) site=1, k=3, Kp1=4  ⇒ α = 7
    """
    site = int(site)
    k    = int(k)
    Kp1  = int(Kp1)

    if site < 0:
        raise ValueError("site must be non-negative.")
    if Kp1 <= 0:
        raise ValueError("Kp1 must be positive.")
    if k < 0 or k >= Kp1:
        raise ValueError("k must satisfy 0 <= k < Kp1.")

    return site * Kp1 + k


def make_site_channel_index(Nmol, Kp1, dtype=np.int32):
    """
    Return site_channel_index[site, k] = α.

    This is useful when we later want all channels belonging to a given
    molecule/site.
    
    For eg:
    1) For Nmol=4, K=0, Kp1=1, it gives,
        site_channel_index = [[0], [1], [2], [3]]
    2) For Nmol=4, K=1, Kp1=2, it gives,
        site_channel_index = [[0, 1], [2, 3], [4, 5], [6, 7]]
        
    Note:
    This is mostly a convenience table. It lets us quickly find all 
    channels belonging to a given molecule. For example, with K=1, 
    site 2 has channels
            α = site_channel_index[2, :] = [4, 5].
    which correspnond to the two exponentials k=0,1 of molecule 3 in 
    physical one-based indexing. In the optimized HTC RHS, we will usually 
    want to loop over channels α directly, but this table can be useful 
    for diagnostics and for building dense reference RHS routines.
    """
    Nmol = _check_Nmol(Nmol)
    Kp1  = int(Kp1)
    if Kp1 <= 0:
        raise ValueError("Kp1 must be positive.")

    site_channel_index = np.empty((Nmol, Kp1), dtype=dtype)
    for site in range(Nmol):
        for k in range(Kp1):
            site_channel_index[site, k] = channel_id(site, k, Kp1)

    return site_channel_index


def make_sys_site_indices(Nmol, dtype=np.int32):
    """
    Return the HTC system indices of the molecular sites:
        [1, 2, ..., Nmol].
    """
    Nmol = _check_Nmol(Nmol)
    return np.arange(1, Nmol + 1, dtype=dtype)


# ============================================================
# Generic HTC site-bath channel construction
# ============================================================

def make_htc_site_channels(Nmol, ν_site, c_site, id_dtype=np.int32):
    """
    Expand one site's exponential bath decomposition into all Nmol
    independent identical site-local baths.

    Parameters
    ----------
    Nmol : int
        Number of molecules / independent site baths.
    ν_site : ndarray, shape (K+1,)
        Decay rates for one site's bath correlation expansion.
    c_site : ndarray, shape (K+1,)
        Complex coefficients for one site's bath correlation expansion.
    id_dtype : dtype
        Integer dtype for channel/site/system index arrays.

    Returns
    -------
    channels : dict
        Dictionary containing the channel metadata. The most important
        arrays are:

            ν_α[α]
            c_α[α]
            abs_c_α[α]
            sqrt_abs_c_α[α]
            inv_sqrt_abs_c_α[α]
            site_α[α]
            k_α[α]
            sys_α[α]

        with α = site * (K+1) + k.
        
    For eg:
    Suppose Nmol=3, K=0, then Kp1 = 1 and M=3. If 
        ν_site = [ν_0], 
        c_site = [c_0], 
    then the output arrays are
        ν_α              = [ν_0, ν_0, ν_0]
        c_α              = [c_0, c_0, c_0]
        abs_c_α          = [|c_0|, |c_0|, |c_0|]
        sqrt_abs_c_α     = [sqrt(|c_0|), sqrt(|c_0|), sqrt(|c_0|)]
        inv_sqrt_abs_c_α = [1/sqrt(|c_0|), 1/sqrt(|c_0|), 1/sqrt(|c_0|)]
        site_α           = [0, 1, 2]
        k_α              = [0, 0, 0]
        sys_α            = [1, 2, 3]
    """
    Nmol   = _check_Nmol(Nmol)
    ν_site = _as_1d_float_array(ν_site, "ν_site")
    c_site = _as_1d_complex_array(c_site, "c_site")

    if ν_site.shape[0] != c_site.shape[0]:
        raise ValueError("ν_site and c_site must have the same length.")
    if ν_site.shape[0] == 0:
        raise ValueError("At least one exponential is required.")
    if np.any(ν_site <= 0.0):
        raise ValueError("All ν_site entries must be positive.")

    Kp1 = int(ν_site.shape[0])
    K   = Kp1 - 1
    M   = Nmol * Kp1

    ν_α     = np.empty(M, dtype=np.float64)
    c_α     = np.empty(M, dtype=np.complex128)
    abs_c_α = np.empty(M, dtype=np.float64)
    site_α  = np.empty(M, dtype=id_dtype)
    k_α     = np.empty(M, dtype=id_dtype)
    sys_α   = np.empty(M, dtype=id_dtype)

    for site in range(Nmol):
        for k in range(Kp1):
            α          = channel_id(site, k, Kp1)
            ν_α[α]     = ν_site[k]
            c_α[α]     = c_site[k]
            abs_c_α[α] = abs(c_site[k])
            site_α[α]  = site
            k_α[α]     = k
            sys_α[α]   = site + 1

    sqrt_abs_c_α     = np.sqrt(abs_c_α)
    inv_sqrt_abs_c_α = np.zeros(M, dtype=np.float64)

    for α in range(M):
        if abs_c_α[α] > 0.0:
            inv_sqrt_abs_c_α[α] = 1.0 / sqrt_abs_c_α[α]
        else:
            # For λ=0, all c_α vanish. In that case the channel should
            # usually be omitted entirely. We keep this finite to avoid NaNs
            # in diagnostics, but the scaled RHS should not use zero-coupling
            # channels.
            inv_sqrt_abs_c_α[α] = 0.0

    site_channel_index = make_site_channel_index(Nmol, Kp1, dtype=id_dtype)
    sys_site = make_sys_site_indices(Nmol, dtype=id_dtype)

    channels = {
        "Nmol": Nmol,
        "d": Nmol + 1,
        "K": K,
        "Kp1": Kp1,
        "M": M,
        "ν_site": ν_site.copy(),
        "c_site": c_site.copy(),
        "ν_α": ν_α,
        "c_α": c_α,
        "abs_c_α": abs_c_α,
        "sqrt_abs_c_α": sqrt_abs_c_α,
        "inv_sqrt_abs_c_α": inv_sqrt_abs_c_α,
        "site_α": site_α,
        "k_α": k_α,
        "sys_α": sys_α,
        "site_channel_index": site_channel_index,
        "sys_site": sys_site,
    }

    return channels


# ============================================================
# Drude-Lorentz HTC channels
# ============================================================

def build_drude_htc_channels(Nmol, λ, γ, β, K, id_dtype=np.int32,
                             include_terminator=True,
                             build_qdiag=False):
    """
    Build HEOM channel metadata for an HTC system where each molecule has
    an independent identical Drude-Lorentz bath. This is the high-level
    function used in run_htc.py.

    The bath correlation function for one site is approximated as
        C(t) = ∑_{k=0}^{K} cₖ exp(-νₖ t),
    where k=0 is the Drude pole and k>=1 are Matsubara terms.

    Parameters
    ----------
    Nmol     → int
        Number of molecules.
    λ        → float
        Reorganization energy in atomic units.
    γ        → float
        Drude decay rate in atomic units.
    β        → float
        Inverse temperature in inverse atomic units.
    K        → int
        Number of Matsubara terms beyond the Drude pole.
        K=0 means only the Drude pole is explicit.
    id_dtype → dtype
        Integer dtype for metadata arrays.

    include_terminator → bool
        If True, compute the Matsubara terminator coefficient Δ_LT and
        attach the per-site array Δ_site. In the RHS this enters as

            - Δ_LT [Q_n, [Q_n, ρ]].

    build_qdiag        → bool
        If True, also build dense qdiag[α, i]. This is only for debugging
        dense reference RHS routines. The optimized HTC RHS should use sys_α.

    Returns
    -------
    channels : dict
        Channel metadata dictionary.
        
    For eg:
        For Nmol=2, K=0, it returns a dictionary containing:
            channels["Nmol"] = 2
            channels["d"]    = 3
            channels["K"]    = 0
            channels["Kp1"]  = 1
            channels["M"]    = 2
            
            channels["ν_site"] = [γ]
            channels["c_site"] = [c0]
            
            channels["ν_α"] = [γ, γ]
            channels["c_α"] = [c0, c0]
            
            channels["abs_c_α"]          = [|c0|, |c0|]
            channels["sqrt_abs_c_α"]     = [sqrt(|c0|), sqrt(|c0|)]
            channels["inv_sqrt_abs_c_α"] = [1/sqrt(|c0|), 1/sqrt(|c0|)]
            
            channels["site_α"] = [0, 1]
            channels["k_α"]    = [0, 0]
            channels["sys_α"]  = [1, 2]
            
            channels["site_channel_index"] = [[0], [1]]
            channels["sys_site"] = [1, 2]
            
        It also stores the input bath parameters,
            channels["λ"] = λ
            channels["γ"] = γ
            channels["β"] = β
            
        and the terminator coefficient if requested,
            channels["Δ_LT"] = Δ_LT
            channels["Δ_site"] = [Δ_LT, Δ_LT]
    """
    
    # Generate the Drude-Matsubara coefficients for one molecule.
    ν_site, c_site = drude_matsubara_coefficients(λ, γ, β, K)
    
    # Replicate the above coefficients across all molecules.
    channels = make_htc_site_channels(
        Nmol=Nmol,
        ν_site=ν_site,
        c_site=c_site,
        id_dtype=id_dtype,
    )

    channels["λ"] = float(λ)
    channels["γ"] = float(γ)
    channels["β"] = float(β)

    if include_terminator:
        Δ_LT = drude_terminator_delta(λ, γ, β, ν_site, c_site)
    else:
        Δ_LT = 0.0

    channels["Δ_LT"] = float(Δ_LT)
    channels["Δ_site"] = np.full(channels["Nmol"], Δ_LT, dtype=np.float64)

    if build_qdiag:
        channels["qdiag"] = make_qdiag_dense(channels["Nmol"], channels["sys_α"])

    return channels


# ============================================================
# Dense debug representations of Q_α and Q_site
# ============================================================

def make_qdiag_dense(Nmol, sys_α):
    """
    Build dense diagonal entries for each channel projector Q_α.
    qdiag[α, i] = 1 if i == sys_α[α], otherwise 0.

    This is only for debugging/testing. For the optimized RHS, use sys_α
    directly because each Q_α is a one-hot projector (|i⟩⟨i| for i^th site).
    """
    Nmol = _check_Nmol(Nmol)
    d = Nmol + 1
    sys_α = np.asarray(sys_α)

    if sys_α.ndim != 1:
        raise ValueError("sys_α must be one-dimensional.")

    M = sys_α.shape[0]
    qdiag = np.zeros((M, d), dtype=np.float64)

    for α in range(M):
        p = int(sys_α[α])
        if p < 1 or p > Nmol:
            raise ValueError("sys_α contains an invalid molecular system index.")
        qdiag[α, p] = 1.0

    return qdiag


def make_site_qdiag_dense(Nmol):
    """
    Build dense diagonal entries for one local projector per molecule.
    site_qdiag[site, i] corresponds to Q_site = |site+1⟩⟨site+1|.
    """
    Nmol = _check_Nmol(Nmol)
    d = Nmol + 1
    site_qdiag = np.zeros((Nmol, d), dtype=np.float64)

    for site in range(Nmol):
        site_qdiag[site, site + 1] = 1.0

    return site_qdiag


def make_dense_Q_list_from_sys(Nmol, sys_α):
    """
    Build dense Q_α matrices for small-system tests.

    Returns
    -------
    Q_list : ndarray, shape (M, d, d), complex128
    """
    Nmol  = _check_Nmol(Nmol)
    d     = Nmol + 1
    sys_α = np.asarray(sys_α)
    M     = sys_α.shape[0]

    Q_list = np.zeros((M, d, d), dtype=np.complex128)
    for α in range(M):
        p  = int(sys_α[α])
        if p < 1 or p > Nmol:
            raise ValueError("sys_α contains an invalid molecular system index.")
        Q_list[α, p, p] = 1.0

    return Q_list


# ============================================================
# Hierarchy-dependent channel helper arrays
# ============================================================

@njit(cache=True)
def _build_ado_decay_numba(ado_indices, ν_α, ado_decay):
    """
    Compute 
        ado_decay[I] = sum_α n_{Iα} ν_α.
    
    Look "build_ado_decay" below for more example.
    """
    N_ado = ado_indices.shape[0]
    M     = ado_indices.shape[1]

    for I in range(N_ado):
        total = 0.0
        for α in range(M):
            total += float(ado_indices[I, α]) * ν_α[α]
        ado_decay[I] = total


@njit(cache=True)
def _build_scaled_down_prefactor_numba(ado_indices, inv_sqrt_abs_c_α,
                                       down_prefactor):
    """
    Compute the scalar part of the scaled downward prefactor
        sqrt(n_{Iα} / |c_α|)
    for all I and α.

    This array can become large, so it is optional. The RHS can also compute
    this value on the fly from ado_indices and inv_sqrt_abs_c_α.
    """
    N_ado = ado_indices.shape[0]
    M = ado_indices.shape[1]

    for I in range(N_ado):
        for α in range(M):
            n = float(ado_indices[I, α])
            if n > 0.0:
                down_prefactor[I, α] = np.sqrt(n) * inv_sqrt_abs_c_α[α]
            else:
                down_prefactor[I, α] = 0.0


@njit(cache=True)
def _build_scaled_up_prefactor_numba(ado_indices, sqrt_abs_c_α,
                                     up_prefactor):
    """
    Compute the scalar part of the scaled upward prefactor
        sqrt((n_{Iα}+1) |c_α|)
    for all I and α.

    This array can become large, so it is optional. The RHS can also compute
    this value on the fly from ado_indices and sqrt_abs_c_α.
    """
    N_ado = ado_indices.shape[0]
    M   = ado_indices.shape[1]

    for I in range(N_ado):
        for α in range(M):
            n = float(ado_indices[I, α])
            up_prefactor[I, α] = np.sqrt(n + 1.0) * sqrt_abs_c_α[α]


def build_ado_decay(ado_indices, ν_α):
    """
    Build the hierarchy damping array
        Γ_I = sum_α n_{Iα} ν_α.

    This is used in the RHS as
        -Γ_I ρ_I.
        
    For eg:
       For Nmol=4, K=0, L=3, since all ν_α = γ
       Γ = 
           [0,
            γ, γ, γ, γ,
            2γ, 2γ, 2γ, 2γ, 2γ, 2γ, 2γ, 2γ, 2γ, 2γ,
            3γ, 3γ, 3γ, 3γ, 3γ, 3γ, 3γ, 3γ, 3γ, 3γ,
            3γ, 3γ, 3γ, 3γ, 3γ, 3γ, 3γ, 3γ, 3γ, 3γ]   
    
    Note:
    This array is precomputed once before propropagation.
    """
    ado_indices = np.asarray(ado_indices)
    ν_α = _as_1d_float_array(ν_α, "ν_α")

    if ado_indices.ndim != 2:
        raise ValueError("ado_indices must be a two-dimensional array.")
    if ado_indices.shape[1] != ν_α.shape[0]:
        raise ValueError("ado_indices.shape[1] must equal len(ν_α).")

    ado_decay = np.empty(ado_indices.shape[0], dtype=np.float64)
    _build_ado_decay_numba(ado_indices, ν_α, ado_decay)
    return ado_decay


def build_scaled_prefactors(ado_indices, channels):
    """
    Optionally precompute the scaled HEOM square-root prefactors.

    Returns
    -------
    up_prefactor : ndarray, shape (N_ado, M)
        up_prefactor[I, α] = sqrt((n_{Iα}+1)|c_α|)

    down_prefactor : ndarray, shape (N_ado, M)
        down_prefactor[I, α] = sqrt(n_{Iα}/|c_α|)

    Warning
    -------
    These arrays can be very large. For production runs at large N and L,
    computing these prefactors on the fly inside the RHS may be more memory
    efficient. This function is mainly useful for profiling and debugging.
    """
    ado_indices = np.asarray(ado_indices)
    M = int(channels["M"])

    if ado_indices.ndim != 2:
        raise ValueError("ado_indices must be a two-dimensional array.")
    if ado_indices.shape[1] != M:
        raise ValueError("ado_indices.shape[1] must match channels['M'].")

    up_prefactor   = np.empty(ado_indices.shape, dtype=np.float64)
    down_prefactor = np.empty(ado_indices.shape, dtype=np.float64)

    _build_scaled_up_prefactor_numba(
        ado_indices,
        channels["sqrt_abs_c_α"],
        up_prefactor,
    )
    _build_scaled_down_prefactor_numba(
        ado_indices,
        channels["inv_sqrt_abs_c_α"],
        down_prefactor,
    )

    return up_prefactor, down_prefactor


# ============================================================
# Diagnostics and labels
# ============================================================

def make_channel_labels(channels):
    """
    Return readable labels for all HEOM channels. This is useful 
    for diagnostics and debugging.

    For eg: Nmol=2, K=1, returns
        [
            "site=0,k=0,sys=1",
            "site=0,k=1,sys=1",
            "site=1,k=0,sys=2",
            "site=1,k=1,sys=2",
        ]
    """
    M = int(channels["M"])
    labels = []
    for α in range(M):
        site = int(channels["site_α"][α])
        k    = int(channels["k_α"][α])
        sys  = int(channels["sys_α"][α])
        labels.append(f"site={site},k={k},sys={sys}")
    return labels


def estimate_channel_metadata_gb(Nmol, K):
    """
    Rough memory estimate for the channel metadata arrays only.
    This is tiny compared with the ADO density arrays, but useful 
    for sanity checking.
    """
    Nmol = _check_Nmol(Nmol)
    K = int(K)
    if K < 0:
        raise ValueError("K must be non-negative.")

    M = Nmol * (K + 1)

    # ν_α, abs_c_α, sqrt_abs_c_α, inv_sqrt_abs_c_α are float64: 4 arrays.
    # c_α is complex128: equivalent to 2 float64 arrays.
    # site_α, k_α, sys_α are int32 in intended use: 3 arrays * 4 bytes.
    bytes_total = M * (4 * 8 + 16 + 3 * 4)

    # site_channel_index and sys_site.
    bytes_total += Nmol * (K + 1) * 4
    bytes_total += Nmol * 4

    return bytes_total / 1.0e9


def check_channel_consistency(channels, atol=1.0e-12):
    """
    Check the internal consistency of a channel dictionary.

    Returns True if all checks pass; raises AssertionError otherwise.
    """
    Nmol = int(channels["Nmol"])
    d    = int(channels["d"])
    K    = int(channels["K"])
    Kp1  = int(channels["Kp1"])
    M    = int(channels["M"])

    assert d   == Nmol + 1
    assert Kp1 == K + 1
    assert M   == Nmol * Kp1

    ν_site  = channels["ν_site"]
    c_site  = channels["c_site"]
    ν_α     = channels["ν_α"]
    c_α     = channels["c_α"]
    abs_c_α = channels["abs_c_α"]
    site_α  = channels["site_α"]
    k_α     = channels["k_α"]
    sys_α   = channels["sys_α"]

    assert ν_site.shape  == (Kp1,)
    assert c_site.shape  == (Kp1,)
    assert ν_α.shape     == (M,)
    assert c_α.shape     == (M,)
    assert abs_c_α.shape == (M,)
    assert site_α.shape  == (M,)
    assert k_α.shape     == (M,)
    assert sys_α.shape   == (M,)

    for site in range(Nmol):
        for k in range(Kp1):
            α = channel_id(site, k, Kp1)
            assert int(site_α[α]) == site
            assert int(k_α[α])    == k
            assert int(sys_α[α])  == site + 1
            assert abs(ν_α[α] - ν_site[k]) < atol * max(abs(ν_site[k]), 1.0)
            assert abs(c_α[α] - c_site[k]) < atol * max(abs(c_site[k]), 1.0)
            assert abs(abs_c_α[α] - abs(c_site[k])) < atol * max(abs(c_site[k]), 1.0)
            assert int(channels["site_channel_index"][site, k]) == α

    assert np.all(channels["sys_site"] == np.arange(1, Nmol + 1))

    if "qdiag" in channels:
        qdiag = channels["qdiag"]
        assert qdiag.shape == (M, d)
        assert np.allclose(np.sum(qdiag, axis=1), 1.0, atol=atol)
        for α in range(M):
            assert qdiag[α, int(sys_α[α])] == 1.0

    if "Δ_site" in channels:
        assert channels["Δ_site"].shape == (Nmol,)

    return True


if __name__ == "__main__":
    # Lightweight self-test.
    from math import comb
    from .hierarchy import generate_ado_indices

    λ = 50.0 * 4.55633e-6
    γ = 18.0 * 4.55633e-6
    T = 300.0 * 0.00000316678
    β = 1.0 / T

    for Nmol in (1, 2, 5, 10, 25, 30):
        for K in (0, 1, 3):
            channels = build_drude_htc_channels(
                Nmol=Nmol,
                λ=λ,
                γ=γ,
                β=β,
                K=K,
                build_qdiag=True,
            )
            check_channel_consistency(channels)

            M = channels["M"]
            L = 2
            ado_indices = generate_ado_indices(M, L)
            assert ado_indices.shape[0] == comb(M + L, L)

            Γ     = build_ado_decay(ado_indices, channels["ν_α"])
            Γ_ref = ado_indices.astype(np.float64) @ channels["ν_α"]
            assert np.allclose(Γ, Γ_ref)

    print("htc_channels.py self-test passed.")
