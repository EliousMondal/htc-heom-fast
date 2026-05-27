# ============================================================
# Observable utilities for the first-excitation HTC / TC model.
#
# Basis convention:
#     index 0      : |C>      = one cavity photon
#     index n >= 1 : |n>      = molecule n excited
#
# The most useful collective observables are
#
#     P_C  = <C|ρ|C>
#     P_X  = sum_n <n|ρ|n>
#     P_B  = <B|ρ|B>,  |B> = (1/sqrt(N)) sum_n |n>
#     P_D  = P_X - P_B
#     P_LP = <LP|ρ|LP>
#     P_UP = <UP|ρ|UP>
#
# All functions here assume the physical reduced density matrix ρ_0,
# i.e. the zeroth ADO.  The HEOM state array has ρ[0] = ρ_0.
# ============================================================

import numpy as np
from numba import njit


# ============================================================
# Names and allocation helpers
# ============================================================

OBSERVABLE_NAMES = (
    "trace_re",
    "trace_im",
    "hermiticity_max_abs",
    "P_C",
    "P_X",
    "P_B",
    "P_D",
    "P_LP",
    "P_UP",
    "abs_rho_CB",
    "Re_rho_CB",
    "Im_rho_CB",
)


def observable_names():
    """
    Return the names corresponding to the columns produced by
    compute_observables_inplace and the propagation routines.
    """
    return OBSERVABLE_NAMES


def allocate_observable_array(Nsave):
    """
    Allocate an observable array with shape

        (Nsave, len(OBSERVABLE_NAMES)).
    """
    Nsave = int(Nsave)
    if Nsave <= 0:
        raise ValueError("Nsave must be positive.")
    return np.zeros((Nsave, len(OBSERVABLE_NAMES)), dtype=np.float64)


# ============================================================
# Low-level collective matrix elements
# ============================================================

@njit(cache=True)
def trace_complex(ρ0):
    """
    Return Tr[ρ0].
    """
    d = ρ0.shape[0]
    tr = 0.0 + 0.0j
    for i in range(d):
        tr += ρ0[i, i]
    return tr


@njit(cache=True)
def hermiticity_max_abs(ρ0):
    """
    Return max_ij |ρ_ij - ρ_ji^*|.
    """
    d = ρ0.shape[0]
    err = 0.0
    for i in range(d):
        for j in range(d):
            val = abs(ρ0[i, j] - np.conjugate(ρ0[j, i]))
            if val > err:
                err = val
    return err


@njit(cache=True)
def cavity_population(ρ0):
    """
    Return P_C = <C|ρ0|C>.
    """
    return ρ0[0, 0].real


@njit(cache=True)
def exciton_population(ρ0):
    """
    Return P_X = sum_n <n|ρ0|n>.
    """
    d = ρ0.shape[0]
    P_X = 0.0
    for n in range(1, d):
        P_X += ρ0[n, n].real
    return P_X


@njit(cache=True)
def bright_population(ρ0):
    """
    Return P_B = <B|ρ0|B>, where

        |B> = (1/sqrt(Nmol)) sum_n |n>.

    In the site basis,

        P_B = (1/Nmol) sum_{m,n=1}^{Nmol} ρ_{mn}.
    """
    d = ρ0.shape[0]
    Nmol = d - 1

    if Nmol <= 0:
        return 0.0

    s = 0.0 + 0.0j
    for m in range(1, d):
        for n in range(1, d):
            s += ρ0[m, n]

    return (s / float(Nmol)).real


@njit(cache=True)
def cavity_bright_coherence(ρ0):
    """
    Return ρ_CB = <C|ρ0|B>.

    With |B> = (1/sqrt(Nmol)) sum_n |n>,

        ρ_CB = (1/sqrt(Nmol)) sum_n ρ_{0n}.
    """
    d = ρ0.shape[0]
    Nmol = d - 1

    if Nmol <= 0:
        return 0.0 + 0.0j

    s = 0.0 + 0.0j
    for n in range(1, d):
        s += ρ0[0, n]

    return s / np.sqrt(float(Nmol))


@njit(cache=True)
def bright_cavity_coherence(ρ0):
    """
    Return ρ_BC = <B|ρ0|C>.
    """
    d = ρ0.shape[0]
    Nmol = d - 1

    if Nmol <= 0:
        return 0.0 + 0.0j

    s = 0.0 + 0.0j
    for n in range(1, d):
        s += ρ0[n, 0]

    return s / np.sqrt(float(Nmol))


@njit(cache=True)
def expectation_ket(ρ0, ψ):
    """
    Return <ψ|ρ0|ψ> for a dense ket ψ.

    This is general and costs O(d^2).  For LP/UP in the HTC model,
    d is small enough that this is perfectly fine, and it provides a
    robust check against the specialized cavity-bright formula.
    """
    d = ρ0.shape[0]
    val = 0.0 + 0.0j
    for i in range(d):
        ψi_conj = np.conjugate(ψ[i])
        for j in range(d):
            val += ψi_conj * ρ0[i, j] * ψ[j]
    return val


@njit(cache=True)
def site_populations_inplace(ρ0, P_site):
    """
    Fill P_site[n] = <n+1|ρ0|n+1> for all molecular sites.

    P_site must have shape (Nmol,). Site indexing is 0-based here.
    """
    Nmol = ρ0.shape[0] - 1
    for site in range(Nmol):
        p = site + 1
        P_site[site] = ρ0[p, p].real


# ============================================================
# Main observable calculators
# ============================================================

@njit(cache=True)
def compute_observables_inplace(ρ0, ψ_LP, ψ_UP, obs):
    """
    Fill one row of observable output.

    obs must have length len(OBSERVABLE_NAMES) = 12.
    """
    tr = trace_complex(ρ0)
    P_C = cavity_population(ρ0)
    P_X = exciton_population(ρ0)
    P_B = bright_population(ρ0)
    P_D = P_X - P_B
    P_LP = expectation_ket(ρ0, ψ_LP).real
    P_UP = expectation_ket(ρ0, ψ_UP).real
    ρ_CB = cavity_bright_coherence(ρ0)

    obs[0] = tr.real
    obs[1] = tr.imag
    obs[2] = hermiticity_max_abs(ρ0)
    obs[3] = P_C
    obs[4] = P_X
    obs[5] = P_B
    obs[6] = P_D
    obs[7] = P_LP
    obs[8] = P_UP
    obs[9] = abs(ρ_CB)
    obs[10] = ρ_CB.real
    obs[11] = ρ_CB.imag


@njit(cache=True)
def compute_observables(ρ0, ψ_LP, ψ_UP):
    """
    Allocate and return the standard observable vector.
    """
    obs = np.empty(12, dtype=np.float64)
    compute_observables_inplace(ρ0, ψ_LP, ψ_UP, obs)
    return obs


def compute_observables_dict(ρ0, ψ_LP, ψ_UP):
    """
    Python convenience wrapper returning a dictionary.
    """
    obs = compute_observables(
        np.asarray(ρ0, dtype=np.complex128),
        np.asarray(ψ_LP, dtype=np.complex128),
        np.asarray(ψ_UP, dtype=np.complex128),
    )
    names = observable_names()
    return {names[i]: float(obs[i]) for i in range(len(names))}


# ============================================================
# Small validation helper
# ============================================================


def check_observables_against_projectors(ρ0, system, atol=1.0e-12):
    """
    Compare the fast observable formulas against dense projector traces.

    This is only for small/debug tests.  It uses the projectors stored in
    the system dictionary from htc_system_builder.build_htc_system.
    """
    ρ0 = np.asarray(ρ0, dtype=np.complex128)
    ψ_LP = np.asarray(system["ψ_LP"], dtype=np.complex128)
    ψ_UP = np.asarray(system["ψ_UP"], dtype=np.complex128)

    obs = compute_observables(ρ0, ψ_LP, ψ_UP)
    names = observable_names()
    odict = {names[i]: obs[i] for i in range(len(names))}

    def tr_proj(P):
        return np.trace(P @ ρ0).real

    assert abs(odict["P_C"] - tr_proj(system["P_C"])) < atol
    assert abs(odict["P_X"] - tr_proj(system["P_X"])) < atol
    assert abs(odict["P_B"] - tr_proj(system["P_B"])) < atol
    assert abs(odict["P_D"] - tr_proj(system["P_D"])) < atol
    assert abs(odict["P_LP"] - tr_proj(system["P_LP"])) < atol
    assert abs(odict["P_UP"] - tr_proj(system["P_UP"])) < atol

    return True


if __name__ == "__main__":
    from .htc_system_builder import build_htc_system

    rng = np.random.default_rng(123)

    for Nmol in (1, 2, 5, 10, 25, 30):
        system = build_htc_system(
            Nmol=Nmol,
            ε_x=0.0,
            Δ_c=0.0,
            Ω_R=0.1,
            use_detuning_frame=True,
        )
        d = Nmol + 1

        # Hermitian positive-ish test density matrix.
        A = rng.normal(size=(d, d)) + 1.0j * rng.normal(size=(d, d))
        ρ0 = A @ A.conjugate().T
        ρ0 = ρ0 / np.trace(ρ0)

        check_observables_against_projectors(ρ0, system)

    print("observables_htc.py self-test passed.")
