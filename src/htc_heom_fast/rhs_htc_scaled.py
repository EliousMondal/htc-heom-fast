# ============================================================
# rhs_htc_scaled.py
#
# Matrix-free scaled HEOM right-hand side for the first-excitation
# HTC / Tavis-Cummings Hamiltonian.
#
# Basis convention:
#     index 0      : |C⟩      = one cavity photon
#     index n >= 1 : |n⟩      = molecule n excited
#
# Local bath projectors:
#     Q_n = |n⟩⟨n|.
#
# Scaled HEOM convention:
#
#     d/dt ρ̃_I = -i[H, ρ̃_I] - Γ_I ρ̃_I
#
#              - i sum_α sqrt((n_{Iα}+1)|c_α|)
#                    [Q_α, ρ̃_{I+e_α}]
#
#              - i sum_α sqrt(n_{Iα}/|c_α|)
#                    ( c_α Q_α ρ̃_{I-e_α}
#                    - c_α* ρ̃_{I-e_α} Q_α )
#
#              - sum_site Δ_site [Q_site, [Q_site, ρ̃_I]].
#
# The final Δ_site term is the optional Drude Matsubara terminator.
# Passing Δ_site as an all-zero array disables it.
#
# The optimized HTC RHS exploits:
#     H = diag(E_i) + g sum_n (|C⟩⟨n| + |n⟩⟨C|),
# and
#     Q_α = |sys_α><sys_α|,
#
# so the Hamiltonian commutator costs O(d^2) per ADO and each
# projector bath coupling costs O(d), rather than using dense
# O(d^3) matrix multiplications.
# ============================================================

import numpy as np
from numba import njit, prange


# ============================================================
# Small Python-side validation helpers
# ============================================================

def check_rhs_shapes(ρ, dρ, E_diag, ado_indices, up, down, Γ, c_α,
                     sqrt_abs_c_α, inv_sqrt_abs_c_α, sys_α, Δ_site):
    """
    Lightweight shape checks for the optimized HTC RHS inputs.

    This is intentionally a Python function. Do not call it inside tight
    propagation loops.
    """
    ρ                = np.asarray(ρ)
    dρ               = np.asarray(dρ)
    E_diag           = np.asarray(E_diag)
    ado_indices      = np.asarray(ado_indices)
    up               = np.asarray(up)
    down             = np.asarray(down)
    Γ                = np.asarray(Γ)
    c_α              = np.asarray(c_α)
    sqrt_abs_c_α     = np.asarray(sqrt_abs_c_α)
    inv_sqrt_abs_c_α = np.asarray(inv_sqrt_abs_c_α)
    sys_α            = np.asarray(sys_α)
    Δ_site           = np.asarray(Δ_site)

    if ρ.ndim != 3:
        raise ValueError("ρ must have shape (N_ado, d, d).")
    if dρ.shape != ρ.shape:
        raise ValueError("dρ must have the same shape as ρ.")

    N_ado, d1, d2 = ρ.shape
    if d1 != d2:
        raise ValueError("ρ must contain square density matrices.")
    d = d1

    if E_diag.shape != (d,):
        raise ValueError("E_diag must have shape (d,).")
    if Γ.shape != (N_ado,):
        raise ValueError("Γ must have shape (N_ado,).")
    if ado_indices.ndim != 2:
        raise ValueError("ado_indices must have shape (N_ado, M).")

    M = ado_indices.shape[1]
    if ado_indices.shape[0] != N_ado:
        raise ValueError("ado_indices.shape[0] must match N_ado.")
    if up.shape != (N_ado, M):
        raise ValueError("up must have shape (N_ado, M).")
    if down.shape != (N_ado, M):
        raise ValueError("down must have shape (N_ado, M).")

    if c_α.shape != (M,):
        raise ValueError("c_α must have shape (M,).")
    if sqrt_abs_c_α.shape != (M,):
        raise ValueError("sqrt_abs_c_α must have shape (M,).")
    if inv_sqrt_abs_c_α.shape != (M,):
        raise ValueError("inv_sqrt_abs_c_α must have shape (M,).")
    if sys_α.shape != (M,):
        raise ValueError("sys_α must have shape (M,).")

    Nmol = d - 1
    if Δ_site.shape != (Nmol,):
        raise ValueError("Δ_site must have shape (Nmol,), where d=Nmol+1.")

    for α in range(M):
        p = int(sys_α[α])
        if p < 1 or p > Nmol:
            raise ValueError("sys_α must contain molecular system indices 1..Nmol.")

    return True


# ============================================================
# Optimized HTC Hamiltonian commutator
# ============================================================

@njit(cache=True)
def _add_htc_star_liouvillian_one_ado(ρI, dρI, E_diag, g, ΓI):
    """
    Add
        -i[H_HTC, ρI] - ΓI ρI
    to dρI for one ADO.

    H_HTC has the star form in the basis {|C⟩, |1⟩, ..., |N⟩}:
        H_ii = E_i,
        H_0n = H_n0 = g.

    This implementation avoids dense matrix products.
    """
    d       = ρI.shape[0]
    Nmol    = d - 1
    minus_i = -1.0j

    # Cavity-cavity element.
    row_sum_0 = 0.0 + 0.0j
    col_sum_0 = 0.0 + 0.0j
    for m in range(1, d):
        row_sum_0 += ρI[0, m]
        col_sum_0 += ρI[m, 0]

    comm = g * (col_sum_0 - row_sum_0)
    dρI[0, 0] = minus_i * comm - ΓI * ρI[0, 0]

    # Cavity row: i=0, j>=1.
    E0 = E_diag[0]
    ρ00 = ρI[0, 0]
    for j in range(1, d):
        col_sum_j = 0.0 + 0.0j
        for m in range(1, d):
            col_sum_j += ρI[m, j]

        comm  = (E0 - E_diag[j]) * ρI[0, j]
        comm += g * col_sum_j
        comm -= g * ρ00
        dρI[0, j] = minus_i * comm - ΓI * ρI[0, j]

    # Cavity column: i>=1, j=0.
    for i in range(1, d):
        row_sum_i = 0.0 + 0.0j
        for m in range(1, d):
            row_sum_i += ρI[i, m]

        comm  = (E_diag[i] - E0) * ρI[i, 0]
        comm += g * ρ00
        comm -= g * row_sum_i
        dρI[i, 0] = minus_i * comm - ΓI * ρI[i, 0]

    # Exciton-exciton block: i>=1, j>=1.
    for i in range(1, d):
        ρi0 = ρI[i, 0]
        Ei = E_diag[i]
        for j in range(1, d):
            comm = (Ei - E_diag[j]) * ρI[i, j]
            comm += g * ρI[0, j]
            comm -= g * ρi0
            dρI[i, j] = minus_i * comm - ΓI * ρI[i, j]


@njit(cache=True)
def _add_projector_terminator_one_ado(ρI, dρI, Δ_site):
    """
    Add the Drude/Matsubara terminator
        - sum_site Δ_site[site] [Q_site, [Q_site, ρI]]
    for projectors Q_site = |site+1⟩⟨site+1|.

    For one projector p, the double commutator only affects row p and
    column p, except the diagonal element (p,p), where it vanishes.
    """
    d = ρI.shape[0]
    Nmol = d - 1

    for site in range(Nmol):
        Δ = Δ_site[site]
        if Δ != 0.0:
            p = site + 1

            # Row p, excluding (p,p).
            for j in range(d):
                if j != p:
                    dρI[p, j] -= Δ * ρI[p, j]

            # Column p, excluding (p,p).
            for i in range(d):
                if i != p:
                    dρI[i, p] -= Δ * ρI[i, p]


@njit(cache=True)
def _add_scaled_upward_projector_one_channel(ρJ, dρI, A, p):
    """
    Add
        -i A [Q_p, ρJ]
    where 
        Q_p = |p⟩⟨p| 
    and 
        A = sqrt((n_{Iα}+1)|c_α|) 
    is the upward coupling prefactor for this channel.
    """
    d = ρJ.shape[0]

    # Row p: [Q,ρ]_{p,j} = ρ_{p,j}, j != p.
    for j in range(d):
        if j != p:
            dρI[p, j] += (-1.0j) * A * ρJ[p, j]

    # Column p: [Q,ρ]_{i,p} = -ρ_{i,p}, i != p.
    for i in range(d):
        if i != p:
            dρI[i, p] += (1.0j) * A * ρJ[i, p]


@njit(cache=True)
def _add_scaled_downward_projector_one_channel(ρJ, dρI, B, c, p):
    """
    Add
        -i B ( c Q_p ρJ - c* ρJ Q_p )
    where 
        Q_p = |p⟩⟨p|,
    and
        B = sqrt(n_{Iα}/|c_α|)
    is the downward coupling prefactor for this channel.

    Row p and column p are both included. At (p,p), both pieces
    contribute, giving the correct diagonal term.
    """
    d      = ρJ.shape[0]
    c_conj = np.conjugate(c)

    # Row p: c Qρ.
    for j in range(d):
        dρI[p, j] += (-1.0j) * B * c * ρJ[p, j]

    # Column p: -c* ρQ, multiplied by -i -> +i c*.
    for i in range(d):
        dρI[i, p] += (1.0j) * B * c_conj * ρJ[i, p]


# ============================================================
# Main optimized scaled HTC RHS
# ============================================================

@njit(cache=True, parallel=True)
def rhs_htc_scaled_inplace(ρ, dρ, E_diag, g,
                           ado_indices, up, down, Γ,
                           c_α, sqrt_abs_c_α, inv_sqrt_abs_c_α,
                           sys_α, Δ_site):
    """
    Compute the optimized matrix-free HTC scaled HEOM RHS in place.

    Parameters
    ----------
    ρ : complex128 ndarray, shape (N_ado, d, d)
        Current scaled ADOs.

    dρ : complex128 ndarray, shape (N_ado, d, d)
        Output RHS. It is overwritten.

    E_diag : float64 ndarray, shape (d,)
        Diagonal energies of the HTC star Hamiltonian.

    g : float
        Single-molecule cavity coupling.

    ado_indices : integer ndarray, shape (N_ado, M)
        ADO occupation indices n_{Iα}.

    up, down : integer ndarray, shape (N_ado, M)
        Neighbor maps. Missing neighbor is -1.

    Γ : float64 ndarray, shape (N_ado,)
        Γ[I] = sum_α n_{Iα} ν_α.

    c_α : complex128 ndarray, shape (M,)
        Bath correlation coefficients.

    sqrt_abs_c_α : float64 ndarray, shape (M,)
        sqrt(|c_α|).

    inv_sqrt_abs_c_α : float64 ndarray, shape (M,)
        1/sqrt(|c_α|). Should be zero for zero-coupling channels.

    sys_α : integer ndarray, shape (M,)
        System index p for Q_α = |p><p|.

    Δ_site : float64 ndarray, shape (Nmol,)
        Per-site terminator coefficient. Use zeros to disable.
    """
    N_ado = ρ.shape[0]
    M = ado_indices.shape[1]

    for I in prange(N_ado):
        ρI = ρ[I]
        dρI = dρ[I]

        # Base Liouvillian and hierarchy damping. This overwrites dρI.
        _add_htc_star_liouvillian_one_ado(
            ρI,
            dρI,
            E_diag,
            g,
            Γ[I],
        )

        # Optional low-temperature / Matsubara terminator.
        _add_projector_terminator_one_ado(ρI, dρI, Δ_site)

        # Couplings to neighboring ADOs.
        for α in range(M):
            p = int(sys_α[α])

            J_up = up[I, α]
            if J_up >= 0:
                n_plus_1 = float(ado_indices[I, α] + 1)
                A = np.sqrt(n_plus_1) * sqrt_abs_c_α[α]
                _add_scaled_upward_projector_one_channel(
                    ρ[J_up],
                    dρI,
                    A,
                    p,
                )

            J_down = down[I, α]
            if J_down >= 0:
                n = float(ado_indices[I, α])
                if n > 0.0:
                    B = np.sqrt(n) * inv_sqrt_abs_c_α[α]
                    if B != 0.0:
                        _add_scaled_downward_projector_one_channel(
                            ρ[J_down],
                            dρI,
                            B,
                            c_α[α],
                            p,
                        )


# ============================================================
# Dense reference RHS for small-system tests
# ============================================================

@njit(cache=True)
def _add_dense_hamiltonian_liouvillian_one_ado(ρI, dρI, H, ΓI):
    """
    Dense reference implementation of -i[H,ρI] - ΓI ρI.
    """
    d = ρI.shape[0]

    for i in range(d):
        for j in range(d):
            Hρ = 0.0 + 0.0j
            ρH = 0.0 + 0.0j
            for k in range(d):
                Hρ += H[i, k] * ρI[k, j]
                ρH += ρI[i, k] * H[k, j]
            dρI[i, j] = (-1.0j) * (Hρ - ρH) - ΓI * ρI[i, j]


@njit(cache=True)
def rhs_dense_scaled_reference_inplace(ρ, dρ, H, qdiag,
                                       site_qdiag, Δ_site,
                                       ado_indices, up, down, Γ,
                                       c_α, sqrt_abs_c_α,
                                       inv_sqrt_abs_c_α):
    """
    Dense/debug scaled HEOM RHS.

    This function is deliberately general and slow. It is only meant for
    small-system tests against rhs_htc_scaled_inplace.

    qdiag[α, i] stores the diagonal of Q_α.
    site_qdiag[site, i] stores the diagonal of Q_site for the terminator.
    """
    N_ado = ρ.shape[0]
    d = ρ.shape[1]
    M = ado_indices.shape[1]
    Nsite = Δ_site.shape[0]

    for I in range(N_ado):
        ρI = ρ[I]
        dρI = dρ[I]

        _add_dense_hamiltonian_liouvillian_one_ado(ρI, dρI, H, Γ[I])

        # Dense terminator: -Δ (q_i - q_j)^2 ρ_{ij}.
        for site in range(Nsite):
            Δ = Δ_site[site]
            if Δ != 0.0:
                for i in range(d):
                    qi = site_qdiag[site, i]
                    for j in range(d):
                        qj = site_qdiag[site, j]
                        diff = qi - qj
                        dρI[i, j] -= Δ * diff * diff * ρI[i, j]

        for α in range(M):
            # Upward coupling.
            J_up = up[I, α]
            if J_up >= 0:
                A = np.sqrt(float(ado_indices[I, α] + 1)) * sqrt_abs_c_α[α]
                ρJ = ρ[J_up]
                for i in range(d):
                    qi = qdiag[α, i]
                    for j in range(d):
                        qj = qdiag[α, j]
                        dρI[i, j] += (-1.0j) * A * (qi - qj) * ρJ[i, j]

            # Downward coupling.
            J_down = down[I, α]
            if J_down >= 0:
                n = float(ado_indices[I, α])
                if n > 0.0:
                    B = np.sqrt(n) * inv_sqrt_abs_c_α[α]
                    if B != 0.0:
                        c = c_α[α]
                        c_conj = np.conjugate(c)
                        ρJ = ρ[J_down]
                        for i in range(d):
                            qi = qdiag[α, i]
                            for j in range(d):
                                qj = qdiag[α, j]
                                factor = c * qi - c_conj * qj
                                dρI[i, j] += (-1.0j) * B * factor * ρJ[i, j]


# ============================================================
# Convenience allocators for debugging and tests
# ============================================================

def rhs_htc_scaled(ρ, E_diag, g, ado_indices, up, down, Γ,
                   c_α, sqrt_abs_c_α, inv_sqrt_abs_c_α, sys_α, Δ_site,
                   check_shapes=True):
    """
    Allocate and return the optimized HTC scaled RHS.

    For production propagation, prefer rhs_htc_scaled_inplace to avoid
    repeated allocations.
    """
    ρ = np.asarray(ρ, dtype=np.complex128)
    dρ = np.empty_like(ρ)

    if check_shapes:
        check_rhs_shapes(
            ρ,
            dρ,
            E_diag,
            ado_indices,
            up,
            down,
            Γ,
            c_α,
            sqrt_abs_c_α,
            inv_sqrt_abs_c_α,
            sys_α,
            Δ_site,
        )

    rhs_htc_scaled_inplace(
        ρ,
        dρ,
        np.asarray(E_diag, dtype=np.float64),
        float(g),
        np.asarray(ado_indices),
        np.asarray(up),
        np.asarray(down),
        np.asarray(Γ, dtype=np.float64),
        np.asarray(c_α, dtype=np.complex128),
        np.asarray(sqrt_abs_c_α, dtype=np.float64),
        np.asarray(inv_sqrt_abs_c_α, dtype=np.float64),
        np.asarray(sys_α),
        np.asarray(Δ_site, dtype=np.float64),
    )

    return dρ


def rhs_dense_scaled_reference(ρ, H, qdiag, site_qdiag, Δ_site,
                               ado_indices, up, down, Γ,
                               c_α, sqrt_abs_c_α, inv_sqrt_abs_c_α):
    """
    Allocate and return the dense reference scaled HEOM RHS.
    """
    ρ = np.asarray(ρ, dtype=np.complex128)
    dρ = np.empty_like(ρ)

    rhs_dense_scaled_reference_inplace(
        ρ,
        dρ,
        np.asarray(H, dtype=np.complex128),
        np.asarray(qdiag, dtype=np.float64),
        np.asarray(site_qdiag, dtype=np.float64),
        np.asarray(Δ_site, dtype=np.float64),
        np.asarray(ado_indices),
        np.asarray(up),
        np.asarray(down),
        np.asarray(Γ, dtype=np.float64),
        np.asarray(c_α, dtype=np.complex128),
        np.asarray(sqrt_abs_c_α, dtype=np.float64),
        np.asarray(inv_sqrt_abs_c_α, dtype=np.float64),
    )

    return dρ


# ============================================================
# Simple self-test
# ============================================================

if __name__ == "__main__":
    from .constants import meV2au, cminv2au, K2au
    from .htc_system_builder import build_htc_system
    from .htc_channels import (
        build_drude_htc_channels,
        build_ado_decay,
        make_site_qdiag_dense,
    )
    from .hierarchy import build_hierarchy

    rng = np.random.default_rng(1234)

    for Nmol in (1, 2, 3, 5):
        for K in (0, 1):
            L = 2
            system = build_htc_system(
                Nmol=Nmol,
                ε_x=0.0,
                Δ_c=15.0 * meV2au,
                Ω_R=100.0 * meV2au,
                use_detuning_frame=True,
            )

            λ = 30.0 * cminv2au
            γ = 50.0 * cminv2au
            T = 300.0 * K2au
            β = 1.0 / T

            channels = build_drude_htc_channels(
                Nmol=Nmol,
                λ=λ,
                γ=γ,
                β=β,
                K=K,
                include_terminator=True,
                build_qdiag=True,
            )

            M = int(channels["M"])
            ado_indices, up, down, tier, tier_offsets = build_hierarchy(
                M=M,
                L=L,
                validate=True,
            )
            Γ = build_ado_decay(ado_indices, channels["ν_α"])

            N_ado = ado_indices.shape[0]
            d = Nmol + 1
            ρ = rng.normal(size=(N_ado, d, d)) + 1.0j * rng.normal(size=(N_ado, d, d))
            ρ = ρ.astype(np.complex128)

            dρ_opt = rhs_htc_scaled(
                ρ,
                system["E_diag"],
                system["g"],
                ado_indices,
                up,
                down,
                Γ,
                channels["c_α"],
                channels["sqrt_abs_c_α"],
                channels["inv_sqrt_abs_c_α"],
                channels["sys_α"],
                channels["Δ_site"],
            )

            site_qdiag = make_site_qdiag_dense(Nmol)
            dρ_ref = rhs_dense_scaled_reference(
                ρ,
                system["H"],
                channels["qdiag"],
                site_qdiag,
                channels["Δ_site"],
                ado_indices,
                up,
                down,
                Γ,
                channels["c_α"],
                channels["sqrt_abs_c_α"],
                channels["inv_sqrt_abs_c_α"],
            )

            err = np.max(np.abs(dρ_opt - dρ_ref))
            scale = max(1.0, np.max(np.abs(dρ_ref)))
            rel = err / scale
            assert rel < 1.0e-12, (Nmol, K, err, rel)

    print("rhs_htc_scaled.py self-test passed.")
