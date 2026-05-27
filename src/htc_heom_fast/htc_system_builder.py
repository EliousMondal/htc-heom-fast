# ============================================================
# System-basis construction for the first-excitation-manifold
# Holstein-Tavis-Cummings / Tavis-Cummings Hamiltonian used in
# the HTC-HEOM code.
#
# Basis convention:
#     index 0      : |C⟩     = one cavity photon, all molecules in |g⟩
#     index n >= 1 : |n⟩     = molecule n excited, no cavity photon
#
# Therefore the ordered basis is
#     {|C⟩, |1⟩, |2⟩, ..., |Nmol⟩}.
#
# In this basis, the HTC first-manifold Hamiltonian is
#     H = ω_c |C⟩⟨C| + ε_x ∑ₙ |n⟩⟨n|
#       + g ∑ₙ ( |C⟩⟨n| + |n⟩⟨C| ).
#
# Equivalently, after subtracting ε_x I,
#     H' = Δ_c |C⟩⟨C|
#        + g ∑ₙ ( |C⟩⟨n| + |n⟩⟨C| ),
# where Δ_c = ω_c - ε_x.
#
# Local bath coupling projectors are
#     Q_n = |n⟩⟨n|,
# so the molecular site n has system-basis index n.
# ============================================================

import numpy as np


# ============================================================
# Validation and small helpers
# ============================================================

def _check_Nmol(Nmol):
    """
    Checks whether Nmol > 0 and returns Nmol as an int.
    """
    Nmol = int(Nmol)
    if Nmol <= 0:
        raise ValueError("Nmol must be positive.")
    return Nmol


def compute_detuning(ε_x=0.0, ω_c=None, Δ_c=None):
    """
    Return (ω_c, Δ_c) consistently.

    Parameters
    ----------
    ε_x : float
        Molecular excitation energy.
    ω_c : float or None
        Cavity frequency / energy. If None, it is inferred from Δ_c.
    Δ_c : float or None
        Detuning Δ_c = ω_c - ε_x. If None, it is inferred from ω_c.

    Returns
    -------
    ω_c : float
        Absolute cavity energy.
    Δ_c : float
        Cavity-exciton detuning.
    """
    ε_x = float(ε_x)

    if ω_c is None and Δ_c is None:
        Δ_c = 0.0
        ω_c = ε_x
    elif ω_c is None:
        Δ_c = float(Δ_c)
        ω_c = ε_x + Δ_c
    elif Δ_c is None:
        ω_c = float(ω_c)
        Δ_c = ω_c - ε_x
    else:
        ω_c = float(ω_c)
        Δ_c = float(Δ_c)
        Δ_from_ω = ω_c - ε_x
        scale = max(abs(ω_c), abs(ε_x), abs(Δ_c), 1.0)
        if abs(Δ_from_ω - Δ_c) > 1.0e-12 * scale:
            raise ValueError("Inconsistent ε_x, ω_c, and Δ_c: expected Δ_c = ω_c - ε_x.")

    return float(ω_c), float(Δ_c)


def compute_single_molecule_g(Nmol, g=None, Ω_R=None):
    """
    Return the single-molecule cavity coupling g.
    The collective Rabi splitting convention is
        Ω_R = 2 g sqrt(Nmol).
    Give either g or Ω_R. If both are supplied, they must agree.
    """
    Nmol = _check_Nmol(Nmol)

    if g is None and Ω_R is None:
        raise ValueError("Either g or Ω_R must be supplied.")

    if g is None:
        Ω_R = float(Ω_R)
        g = Ω_R / (2.0 * np.sqrt(float(Nmol)))
    elif Ω_R is None:
        g = float(g)
        Ω_R = 2.0 * g * np.sqrt(float(Nmol))
    else:
        g = float(g)
        Ω_R = float(Ω_R)
        Ω_from_g = 2.0 * g * np.sqrt(float(Nmol))
        scale = max(abs(Ω_R), abs(Ω_from_g), 1.0)
        if abs(Ω_R - Ω_from_g) > 1.0e-12 * scale:
            raise ValueError("Inconsistent g and Ω_R: expected Ω_R = 2 g sqrt(Nmol).")

    return float(g), float(Ω_R)


# ============================================================
# Basis vectors and labels
# ============================================================

def make_htc_basis_labels(Nmol):
    """
    Return basis labels for
        {|C⟩, |1⟩, |2⟩, ..., |Nmol⟩}.
    """
    Nmol = _check_Nmol(Nmol)
    labels = ["|C⟩"]
    for n in range(1, Nmol + 1):
        labels.append(f"|{n}⟩")
    return labels


def ket(d, i, dtype=np.complex128):
    """
    Return the basis ket |i⟩ as a dense vector of length d.
    """
    if d <= 0:
        raise ValueError("d must be positive.")
    if i < 0 or i >= d:
        raise ValueError("basis index i is out of range.")

    ψ    = np.zeros(d, dtype=dtype)
    ψ[i] = 1.0
    return ψ


def ket_cavity(Nmol, dtype=np.complex128):
    """
    Return |C⟩.
    """
    Nmol = _check_Nmol(Nmol)
    return ket(Nmol + 1, 0, dtype=dtype)


def ket_site(Nmol, n, dtype=np.complex128):
    """
    Return molecular site ket |n⟩, where n uses 1-based molecule indexing.
    """
    Nmol = _check_Nmol(Nmol)
    n    = int(n)
    if n < 1 or n > Nmol:
        raise ValueError("site index n must satisfy 1 <= n <= Nmol.")
    return ket(Nmol + 1, n, dtype=dtype)


def ket_bright(Nmol, dtype=np.complex128):
    """
    Return the normalized bright exciton state
        |B⟩ = (1/sqrt(Nmol)) ∑ₙ |n⟩.
    """
    Nmol    = _check_Nmol(Nmol)
    d       = Nmol + 1
    ψ_B     = np.zeros(d, dtype=dtype)
    ψ_B[1:] = 1.0 / np.sqrt(float(Nmol))
    return ψ_B


def site_sys_indices(Nmol, dtype=np.int64):
    """
    Return system-basis indices for the molecular sites.
    Since the basis is {|C⟩, |1⟩, ..., |Nmol⟩}, this is simply
        [1, 2, ..., Nmol].
    """
    Nmol = _check_Nmol(Nmol)
    return np.arange(1, Nmol + 1, dtype=dtype)


# ============================================================
# Projectors and density matrices
# ============================================================

def projector(ψ):
    """
    Return |ψ⟩⟨ψ|.
    """
    ψ = np.asarray(ψ, dtype=np.complex128)
    if ψ.ndim != 1:
        raise ValueError("ψ must be a one-dimensional ket vector.")
    return np.outer(ψ, np.conjugate(ψ))


def density_from_ket(ψ):
    """
    Alias for projector(ψ). Useful for initial density matrices.
    """
    return projector(ψ)


def projector_cavity(Nmol):
    """
    Return |C⟩⟨C|.
    """
    return projector(ket_cavity(Nmol))


def projector_site(Nmol, n):
    """
    Return |n⟩⟨n| for molecule n, using 1-based molecule indexing.
    """
    return projector(ket_site(Nmol, n))


def projector_bright(Nmol):
    """
    Return |B⟩⟨B|.
    """
    return projector(ket_bright(Nmol))


def projector_exciton_manifold(Nmol):
    """
    Return the molecular single-excitation identity
        P_X = ∑ₙ |n⟩⟨n|.
    In the HTC basis this is diag(0, 1, 1, ..., 1).
    """
    Nmol = _check_Nmol(Nmol)
    d    = Nmol + 1
    P_X  = np.zeros((d, d), dtype=np.complex128)
    for n in range(1, Nmol + 1):
        P_X[n, n] = 1.0
    return P_X


def projector_dark_manifold(Nmol):
    """
    Return the total dark-manifold projector
        P_D = P_X - |B⟩⟨B|.

    We do not construct an explicit dark-state basis here, because for
    dynamics and observables the projector is enough.
    """
    return projector_exciton_manifold(Nmol) - projector_bright(Nmol)


# ============================================================
# HTC Hamiltonian construction
# ============================================================

def htc_diagonal_energies(Nmol, ε_x=0.0, ω_c=None, Δ_c=None, use_detuning_frame=True):
    """
    Return the diagonal energy array E_diag for the HTC first manifold.

    If use_detuning_frame is True, we subtract ε_x I and use
        E_C = Δ_c,
        E_n = 0.

    If use_detuning_frame is False, we use the absolute energies
        E_C = ω_c,
        E_n = ε_x.
    """
    Nmol     = _check_Nmol(Nmol)
    d        = Nmol + 1
    ε_x      = float(ε_x)
    ω_c, Δ_c = compute_detuning(ε_x=ε_x, ω_c=ω_c, Δ_c=Δ_c)

    E_diag   = np.empty(d, dtype=np.float64)

    if use_detuning_frame:
        E_diag[0] = Δ_c
        E_diag[1:] = 0.0
    else:
        E_diag[0] = ω_c
        E_diag[1:] = ε_x

    return E_diag


def dense_htc_hamiltonian(Nmol, ε_x=0.0, ω_c=None, Δ_c=None, g=None, Ω_R=None, use_detuning_frame=True):
    """
    Build the dense HTC Hamiltonian matrix in the basis
        {|C⟩, |1⟩, |2⟩, ..., |Nmol⟩}.
    This dense matrix is useful for testing and for small systems.
    The optimized RHS should use E_diag and g directly rather than
    performing dense matrix multiplications.
    """
    Nmol   = _check_Nmol(Nmol)
    d      = Nmol + 1
    g, Ω_R = compute_single_molecule_g(Nmol, g=g, Ω_R=Ω_R)

    E_diag = htc_diagonal_energies(Nmol, ε_x=ε_x, ω_c=ω_c, Δ_c=Δ_c, use_detuning_frame=use_detuning_frame)

    H = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        H[i, i] = E_diag[i]

    for n in range(1, Nmol + 1):
        H[0, n] = g
        H[n, 0] = g

    return H


def cavity_bright_block(Nmol, ε_x=0.0, ω_c=None, Δ_c=None, g=None, Ω_R=None, use_detuning_frame=True):
    """
    Return the 2x2 cavity-bright Hamiltonian block in basis {|C⟩, |B⟩}.

    In the detuning frame this is
        [[Δ_c, g sqrt(Nmol)],
         [g sqrt(Nmol), 0]].
    """
    Nmol  = _check_Nmol(Nmol)
    g, Ω_R = compute_single_molecule_g(Nmol, g=g, Ω_R=Ω_R)

    E_diag = htc_diagonal_energies(Nmol, ε_x=ε_x, ω_c=ω_c, Δ_c=Δ_c, use_detuning_frame=use_detuning_frame)

    E_C = E_diag[0]
    E_B = E_diag[1]
    G   = g * np.sqrt(float(Nmol))

    H_CB = np.array(
        [[E_C, G],
         [G, E_B]],
        dtype=np.complex128,
    )
    return H_CB


def polariton_states(Nmol, ε_x=0.0, ω_c=None, Δ_c=None,
                     g=None, Ω_R=None, use_detuning_frame=True):
    """
    Diagonalize the cavity-bright 2x2 block and return LP/UP states.

    Returns
    -------
    E_LP : float
        Lower polariton energy.
    E_UP : float
        Upper polariton energy.
    ψ_LP : ndarray, shape (Nmol+1,), complex128
        Lower polariton ket in the full HTC basis.
    ψ_UP : ndarray, shape (Nmol+1,), complex128
        Upper polariton ket in the full HTC basis.
    U_CB : ndarray, shape (2, 2), complex128
        Eigenvectors in the {|C⟩, |B⟩} basis. Columns are LP and UP.
    """
    Nmol = _check_Nmol(Nmol)

    H_CB = cavity_bright_block(Nmol, ε_x=ε_x, ω_c=ω_c, Δ_c=Δ_c, g=g, Ω_R=Ω_R, use_detuning_frame=use_detuning_frame)

    E, U = np.linalg.eigh(H_CB)

    E_LP = float(E[0].real)
    E_UP = float(E[1].real)

    ψ_C  = ket_cavity(Nmol)
    ψ_B  = ket_bright(Nmol)

    # Column 0 = LP in {|C⟩, |B⟩}, column 1 = UP.
    ψ_LP = U[0, 0] * ψ_C + U[1, 0] * ψ_B
    ψ_UP = U[0, 1] * ψ_C + U[1, 1] * ψ_B

    # Normalize again to remove tiny roundoff errors.
    ψ_LP = ψ_LP / np.sqrt(np.vdot(ψ_LP, ψ_LP).real)
    ψ_UP = ψ_UP / np.sqrt(np.vdot(ψ_UP, ψ_UP).real)

    return E_LP, E_UP, ψ_LP, ψ_UP, U


# ============================================================
# Initial density matrices
# ============================================================

def initial_density(Nmol, state="UP", ε_x=0.0, ω_c=None, Δ_c=None,
                    g=None, Ω_R=None, site=1, use_detuning_frame=True):
    """
    Build a common initial density matrix ρ(0).

    Parameters
    ----------
    state : str
        One of:
            "cavity" / "C"  → |C⟩⟨C|
            "bright" / "B"  → |B⟩⟨B|
            "site"          → |site⟩⟨site|
            "LP"            → |LP⟩⟨LP|
            "UP"            → |UP⟩⟨UP|

    site : int
        Site index used only when state == "site". Uses 1-based indexing.
    """
    s = str(state).strip().lower()

    if s in ("cavity", "c", "photon", "cav"):
        return projector_cavity(Nmol)

    if s in ("bright", "b"):
        return projector_bright(Nmol)

    if s in ("site", "local", "molecule"):
        return projector_site(Nmol, site)

    if s in ("lp", "lower", "lower_polariton"):
        E_LP, E_UP, ψ_LP, ψ_UP, U = polariton_states(
            Nmol,
            ε_x=ε_x,
            ω_c=ω_c,
            Δ_c=Δ_c,
            g=g,
            Ω_R=Ω_R,
            use_detuning_frame=use_detuning_frame,
        )
        return projector(ψ_LP)

    if s in ("up", "upper", "upper_polariton"):
        E_LP, E_UP, ψ_LP, ψ_UP, U = polariton_states(
            Nmol,
            ε_x=ε_x,
            ω_c=ω_c,
            Δ_c=Δ_c,
            g=g,
            Ω_R=Ω_R,
            use_detuning_frame=use_detuning_frame,
        )
        return projector(ψ_UP)

    raise ValueError(f"Unknown initial state: {state!r}.")


# ============================================================
# One-stop builder
# ============================================================

def build_htc_system(Nmol, ε_x=0.0, ω_c=None, Δ_c=None,
                     g=None, Ω_R=None, use_detuning_frame=True,
                     build_projectors=True):
    """
    Build all small HTC system arrays needed by later code.

    This returns a dictionary rather than a class, matching the simple style
    of the dimer HEOM code.

    Most important entries for the optimized RHS are

        system["Nmol"]
        system["d"]
        system["E_diag"]
        system["g"]
        system["site_sys_index"]

    The dense Hamiltonian H is included for testing/reference.
    """
    Nmol = _check_Nmol(Nmol)
    d = Nmol + 1

    ε_x = float(ε_x)
    ω_c, Δ_c = compute_detuning(ε_x=ε_x, ω_c=ω_c, Δ_c=Δ_c)
    g, Ω_R = compute_single_molecule_g(Nmol, g=g, Ω_R=Ω_R)

    E_diag = htc_diagonal_energies(
        Nmol,
        ε_x=ε_x,
        ω_c=ω_c,
        Δ_c=Δ_c,
        use_detuning_frame=use_detuning_frame,
    )

    H = dense_htc_hamiltonian(
        Nmol,
        ε_x=ε_x,
        ω_c=ω_c,
        Δ_c=Δ_c,
        g=g,
        use_detuning_frame=use_detuning_frame,
    )

    H_CB = cavity_bright_block(
        Nmol,
        ε_x=ε_x,
        ω_c=ω_c,
        Δ_c=Δ_c,
        g=g,
        use_detuning_frame=use_detuning_frame,
    )

    E_LP, E_UP, ψ_LP, ψ_UP, U_CB = polariton_states(
        Nmol,
        ε_x=ε_x,
        ω_c=ω_c,
        Δ_c=Δ_c,
        g=g,
        use_detuning_frame=use_detuning_frame,
    )

    system = {
        "Nmol": Nmol,
        "d": d,
        "basis_labels": make_htc_basis_labels(Nmol),
        "ε_x": ε_x,
        "ω_c": ω_c,
        "Δ_c": Δ_c,
        "g": g,
        "Ω_R": Ω_R,
        "use_detuning_frame": bool(use_detuning_frame),
        "E_diag": E_diag,
        "H": H,
        "H_CB": H_CB,
        "E_LP": E_LP,
        "E_UP": E_UP,
        "U_CB": U_CB,
        "ψ_C": ket_cavity(Nmol),
        "ψ_B": ket_bright(Nmol),
        "ψ_LP": ψ_LP,
        "ψ_UP": ψ_UP,
        "site_sys_index": site_sys_indices(Nmol),
    }

    if build_projectors:
        system["P_C"] = projector_cavity(Nmol)
        system["P_X"] = projector_exciton_manifold(Nmol)
        system["P_B"] = projector_bright(Nmol)
        system["P_D"] = projector_dark_manifold(Nmol)
        system["P_LP"] = projector(ψ_LP)
        system["P_UP"] = projector(ψ_UP)

    return system


# ============================================================
# Diagnostics / consistency checks
# ============================================================

def check_system_consistency(system, atol=1.0e-12):
    """
    Run simple consistency checks on a system dictionary.
    Returns True if all checks pass; raises AssertionError otherwise.
    """
    Nmol = int(system["Nmol"])
    d = int(system["d"])

    assert d == Nmol + 1
    assert system["H"].shape == (d, d)
    assert np.allclose(system["H"], system["H"].conjugate().T, atol=atol)

    ψ_C = system["ψ_C"]
    ψ_B = system["ψ_B"]
    ψ_LP = system["ψ_LP"]
    ψ_UP = system["ψ_UP"]

    assert abs(np.vdot(ψ_C, ψ_C).real - 1.0) < atol
    assert abs(np.vdot(ψ_B, ψ_B).real - 1.0) < atol
    assert abs(np.vdot(ψ_LP, ψ_LP).real - 1.0) < atol
    assert abs(np.vdot(ψ_UP, ψ_UP).real - 1.0) < atol

    assert abs(np.vdot(ψ_C, ψ_B)) < atol
    assert abs(np.vdot(ψ_LP, ψ_UP)) < atol

    if "P_C" in system:
        P_C = system["P_C"]
        P_X = system["P_X"]
        P_B = system["P_B"]
        P_D = system["P_D"]

        assert np.allclose(P_C @ P_C, P_C, atol=atol)
        assert np.allclose(P_X @ P_X, P_X, atol=atol)
        assert np.allclose(P_B @ P_B, P_B, atol=atol)
        assert np.allclose(P_D @ P_D, P_D, atol=atol)
        assert np.allclose(P_X, P_B + P_D, atol=atol)
        assert abs(np.trace(P_D).real - (Nmol - 1)) < 1.0e-10

    return True


if __name__ == "__main__":
    # Simple self-test.
    for Nmol in (1, 2, 5, 10, 25, 30):
        system = build_htc_system(
            Nmol=Nmol,
            ε_x=0.0,
            Δ_c=0.0,
            Ω_R=0.1,
            use_detuning_frame=True,
        )
        check_system_consistency(system)

    print("htc_system_builder.py self-test passed.")
