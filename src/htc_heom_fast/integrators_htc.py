# ============================================================
# integrators_htc.py
#
# Explicit RK4 integrators for the matrix-free scaled HTC-HEOM.
# State-array convention:
#
#     ρ[I, i, j] = scaled ADO I as a d x d matrix.
#     ρ[0]       = physical reduced density matrix.
#
# Basis convention:
#
#     index 0      : |C⟩
#     index n >= 1 : |n⟩
#
# Low-memory RK4 version:
#     old storage: ρ, k1, k2, k3, k4, ρ_tmp
#     new storage: ρ, k, acc, ρ_tmp
#
# where
#     acc = k1 + 2 k2 + 2 k3
#
# and k is reused as k1, k2, k3, k4.
# ============================================================

import time

import numpy as np
from numba import njit, prange

from .rhs_htc_scaled import rhs_htc_scaled_inplace
from .observables_htc import (
    observable_names,
    allocate_observable_array,
    compute_observables_inplace,
    site_populations_inplace,
)


# ============================================================
# Allocation and initialization helpers
# ============================================================

def allocate_ado_state(N_ado, d, dtype=np.complex128):
    N_ado = int(N_ado)
    d = int(d)

    if N_ado <= 0:
        raise ValueError("N_ado must be positive.")
    if d <= 0:
        raise ValueError("d must be positive.")

    return np.zeros((N_ado, d, d), dtype=dtype)


def initialize_ado_state(ρ0, N_ado):
    ρ0 = np.ascontiguousarray(ρ0, dtype=np.complex128)

    if ρ0.ndim != 2 or ρ0.shape[0] != ρ0.shape[1]:
        raise ValueError("ρ0 must be a square matrix.")

    ρ = allocate_ado_state(N_ado, ρ0.shape[0], dtype=np.complex128)
    ρ[0, :, :] = ρ0
    return ρ


def allocate_rk4_work_arrays(ρ):
    """
    Standard RK4 work arrays, kept for reference/backward compatibility.
    """
    ρ = np.asarray(ρ)
    return (
        np.empty_like(ρ),
        np.empty_like(ρ),
        np.empty_like(ρ),
        np.empty_like(ρ),
        np.empty_like(ρ),
    )


def allocate_rk4_lowmem_work_arrays(ρ):
    """
    Low-memory RK4 work arrays:
        k, acc, ρ_tmp.
    """
    ρ = np.asarray(ρ)
    return (
        np.empty_like(ρ),
        np.empty_like(ρ),
        np.empty_like(ρ),
    )


def number_of_saves(Nstep, save_every, save_initial=True):
    Nstep = int(Nstep)
    save_every = int(save_every)

    if Nstep < 0:
        raise ValueError("Nstep must be non-negative.")
    if save_every <= 0:
        raise ValueError("save_every must be positive.")

    count = Nstep // save_every
    if save_initial:
        count += 1
    return count


# ============================================================
# Low-level array kernels
# ============================================================

@njit(cache=True, parallel=True)
def _linear_combination_midpoint(ρ, k, dt, ρ_tmp):
    """
    ρ_tmp = ρ + 0.5 dt k.
    """
    N_ado = ρ.shape[0]
    d = ρ.shape[1]
    half_dt = 0.5 * dt

    for I in prange(N_ado):
        for i in range(d):
            for j in range(d):
                ρ_tmp[I, i, j] = ρ[I, i, j] + half_dt * k[I, i, j]


@njit(cache=True, parallel=True)
def _linear_combination_endpoint(ρ, k, dt, ρ_tmp):
    """
    ρ_tmp = ρ + dt k.
    """
    N_ado = ρ.shape[0]
    d = ρ.shape[1]

    for I in prange(N_ado):
        for i in range(d):
            for j in range(d):
                ρ_tmp[I, i, j] = ρ[I, i, j] + dt * k[I, i, j]


@njit(cache=True, parallel=True)
def _copy_array(src, dst):
    """
    dst = src.
    """
    N_ado = src.shape[0]
    d = src.shape[1]

    for I in prange(N_ado):
        for i in range(d):
            for j in range(d):
                dst[I, i, j] = src[I, i, j]


@njit(cache=True, parallel=True)
def _accumulate_scaled_inplace(acc, k, scale):
    """
    acc ← acc + scale * k.
    """
    N_ado = acc.shape[0]
    d = acc.shape[1]

    for I in prange(N_ado):
        for i in range(d):
            for j in range(d):
                acc[I, i, j] += scale * k[I, i, j]


@njit(cache=True, parallel=True)
def _rk4_finalize_inplace(ρ, k1, k2, k3, k4, dt):
    """
    Standard RK4 finalization:

        ρ ← ρ + dt/6 * (k1 + 2 k2 + 2 k3 + k4).
    """
    N_ado = ρ.shape[0]
    d = ρ.shape[1]
    fac = dt / 6.0

    for I in prange(N_ado):
        for i in range(d):
            for j in range(d):
                ρ[I, i, j] += fac * (
                    k1[I, i, j]
                    + 2.0 * k2[I, i, j]
                    + 2.0 * k3[I, i, j]
                    + k4[I, i, j]
                )


@njit(cache=True, parallel=True)
def _rk4_lowmem_finalize_inplace(ρ, acc, k4, dt):
    """
    Low-memory RK4 finalization.

    On entry:
        acc = k1 + 2 k2 + 2 k3
        k4  = f(ρ + dt k3)

    Then:
        ρ ← ρ + dt/6 * (acc + k4).
    """
    N_ado = ρ.shape[0]
    d     = ρ.shape[1]
    fac   = dt / 6.0

    for I in prange(N_ado):
        for i in range(d):
            for j in range(d):
                ρ[I, i, j] += fac * (acc[I, i, j] + k4[I, i, j])


# ============================================================
# One-step RK4 routines
# ============================================================

@njit(cache=True)
def rk4_step_htc_scaled_inplace(
    ρ,
    dt,
    k1,
    k2,
    k3,
    k4,
    ρ_tmp,
    E_diag,
    g,
    ado_indices,
    up,
    down,
    Γ,
    c_α,
    sqrt_abs_c_α,
    inv_sqrt_abs_c_α,
    sys_α,
    Δ_site,
):
    """
    Standard RK4 step using five auxiliary arrays.

    Kept as a reference implementation.
    """
    rhs_htc_scaled_inplace(
        ρ, k1, E_diag, g,
        ado_indices, up, down, Γ,
        c_α, sqrt_abs_c_α, inv_sqrt_abs_c_α, sys_α, Δ_site,
    )

    _linear_combination_midpoint(ρ, k1, dt, ρ_tmp)
    rhs_htc_scaled_inplace(
        ρ_tmp, k2, E_diag, g,
        ado_indices, up, down, Γ,
        c_α, sqrt_abs_c_α, inv_sqrt_abs_c_α, sys_α, Δ_site,
    )

    _linear_combination_midpoint(ρ, k2, dt, ρ_tmp)
    rhs_htc_scaled_inplace(
        ρ_tmp, k3, E_diag, g,
        ado_indices, up, down, Γ,
        c_α, sqrt_abs_c_α, inv_sqrt_abs_c_α, sys_α, Δ_site,
    )

    _linear_combination_endpoint(ρ, k3, dt, ρ_tmp)
    rhs_htc_scaled_inplace(
        ρ_tmp, k4, E_diag, g,
        ado_indices, up, down, Γ,
        c_α, sqrt_abs_c_α, inv_sqrt_abs_c_α, sys_α, Δ_site,
    )

    _rk4_finalize_inplace(ρ, k1, k2, k3, k4, dt)


@njit(cache=True)
def rk4_step_htc_scaled_lowmem_inplace(
    ρ,
    dt,
    k,
    acc,
    ρ_tmp,
    E_diag,
    g,
    ado_indices,
    up,
    down,
    Γ,
    c_α,
    sqrt_abs_c_α,
    inv_sqrt_abs_c_α,
    sys_α,
    Δ_site,
):
    """
    Low-memory RK4 step.
    Uses only three auxiliary hierarchy-sized arrays:
        k, acc, ρ_tmp.
    """
    # k = k1 = f(ρ)
    rhs_htc_scaled_inplace(
        ρ, k, E_diag, g,
        ado_indices, up, down, Γ,
        c_α, sqrt_abs_c_α, inv_sqrt_abs_c_α, sys_α, Δ_site,
    )

    # acc = k1
    _copy_array(k, acc)

    # k = k2 = f(ρ + 0.5 dt k1)
    _linear_combination_midpoint(ρ, k, dt, ρ_tmp)
    rhs_htc_scaled_inplace(
        ρ_tmp, k, E_diag, g,
        ado_indices, up, down, Γ,
        c_α, sqrt_abs_c_α, inv_sqrt_abs_c_α, sys_α, Δ_site,
    )

    # acc = k1 + 2 k2
    _accumulate_scaled_inplace(acc, k, 2.0)

    # k = k3 = f(ρ + 0.5 dt k2)
    _linear_combination_midpoint(ρ, k, dt, ρ_tmp)
    rhs_htc_scaled_inplace(
        ρ_tmp, k, E_diag, g,
        ado_indices, up, down, Γ,
        c_α, sqrt_abs_c_α, inv_sqrt_abs_c_α, sys_α, Δ_site,
    )

    # acc = k1 + 2 k2 + 2 k3
    _accumulate_scaled_inplace(acc, k, 2.0)

    # k = k4 = f(ρ + dt k3)
    _linear_combination_endpoint(ρ, k, dt, ρ_tmp)
    rhs_htc_scaled_inplace(
        ρ_tmp, k, E_diag, g,
        ado_indices, up, down, Γ,
        c_α, sqrt_abs_c_α, inv_sqrt_abs_c_α, sys_α, Δ_site,
    )

    # ρ ← ρ + dt/6 * (acc + k4)
    _rk4_lowmem_finalize_inplace(ρ, acc, k, dt)


# ============================================================
# Python helpers
# ============================================================

def _check_save_inputs(Nstep, save_every):
    Nstep = int(Nstep)
    save_every = int(save_every)

    if Nstep < 0:
        raise ValueError("Nstep must be non-negative.")
    if save_every <= 0:
        raise ValueError("save_every must be positive.")
    if Nstep % save_every != 0:
        raise ValueError("For now, Nstep must be exactly divisible by save_every.")

    return Nstep, save_every


def _as_runtime_arrays(
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
):
    return (
        np.ascontiguousarray(E_diag, dtype=np.float64),
        np.ascontiguousarray(ado_indices),
        np.ascontiguousarray(up),
        np.ascontiguousarray(down),
        np.ascontiguousarray(Γ, dtype=np.float64),
        np.ascontiguousarray(c_α, dtype=np.complex128),
        np.ascontiguousarray(sqrt_abs_c_α, dtype=np.float64),
        np.ascontiguousarray(inv_sqrt_abs_c_α, dtype=np.float64),
        np.ascontiguousarray(sys_α),
        np.ascontiguousarray(Δ_site, dtype=np.float64),
    )


def _format_time(seconds):
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:7.1f}s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:7.1f}m"
    return f"{seconds / 3600.0:7.1f}h"


def _resolve_progress_every(Nstep, progress_every):
    Nstep = int(Nstep)
    if progress_every is None:
        return max(1, Nstep // 100)
    progress_every = int(progress_every)
    if progress_every <= 0:
        return max(1, Nstep // 100)
    return progress_every


def _print_progress(step, Nstep, start_time, width=32, prefix="RK4"):
    Nstep = int(Nstep)
    step = int(step)
    width = max(4, int(width))

    if Nstep <= 0:
        frac = 1.0
    else:
        frac = min(1.0, max(0.0, step / float(Nstep)))

    filled = int(round(width * frac))
    bar = "=" * filled + "-" * (width - filled)
    percent = 100.0 * frac
    elapsed = time.perf_counter() - start_time

    if step > 0 and frac > 0.0:
        remaining = elapsed * (1.0 / frac - 1.0)
    else:
        remaining = 0.0

    msg = (
        f"\r{prefix}: [{bar}] {percent:6.2f}%  "
        f"step {step}/{Nstep}  "
        f"elapsed {_format_time(elapsed)}  "
        f"remaining {_format_time(remaining)}"
    )

    print(msg, end="", flush=True)
    if step >= Nstep:
        print("", flush=True)


def _should_print_progress(step, Nstep, progress_every):
    return step == 1 or step == Nstep or (step % progress_every == 0)


# ============================================================
# Propagation wrappers
# ============================================================

def propagate_rk4_htc_scaled_store_observables(
    ρ,
    dt,
    Nstep,
    save_every,
    E_diag,
    g,
    ado_indices,
    up,
    down,
    Γ,
    c_α,
    sqrt_abs_c_α,
    inv_sqrt_abs_c_α,
    sys_α,
    Δ_site,
    ψ_LP,
    ψ_UP,
    return_final_state=True,
    progress=False,
    progress_every=100,
    progress_width=32,
):
    """
    Propagate with low-memory RK4 and store collective observables.
    """
    Nstep, save_every = _check_save_inputs(Nstep, save_every)

    ρ    = np.ascontiguousarray(ρ, dtype=np.complex128)
    ψ_LP = np.ascontiguousarray(ψ_LP, dtype=np.complex128)
    ψ_UP = np.ascontiguousarray(ψ_UP, dtype=np.complex128)

    (
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
    ) = _as_runtime_arrays(
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

    k, acc, ρ_tmp = allocate_rk4_lowmem_work_arrays(ρ)

    Nsave = Nstep // save_every + 1
    t_save = np.empty(Nsave, dtype=np.float64)
    obs_save = allocate_observable_array(Nsave)

    save_id = 0
    t_save[save_id] = 0.0
    compute_observables_inplace(ρ[0], ψ_LP, ψ_UP, obs_save[save_id])
    save_id += 1

    progress_every = _resolve_progress_every(Nstep, progress_every)
    start_time = time.perf_counter()
    if progress:
        _print_progress(0, Nstep, start_time, width=progress_width, prefix="RK4")

    for step in range(1, Nstep + 1):
        rk4_step_htc_scaled_lowmem_inplace(
            ρ,
            float(dt),
            k,
            acc,
            ρ_tmp,
            E_diag,
            float(g),
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

        if step % save_every == 0:
            t_save[save_id] = float(dt) * float(step)
            compute_observables_inplace(ρ[0], ψ_LP, ψ_UP, obs_save[save_id])
            save_id += 1

        if progress and _should_print_progress(step, Nstep, progress_every):
            _print_progress(step, Nstep, start_time, width=progress_width, prefix="RK4")

    result = {
        "t": t_save,
        "obs": obs_save,
        "observable_names": observable_names(),
        "rk4_storage": "lowmem_4_arrays_total",
    }
    if return_final_state:
        result["ρ"] = ρ
    return result


def propagate_rk4_htc_scaled_store_rho0(
    ρ,
    dt,
    Nstep,
    save_every,
    E_diag,
    g,
    ado_indices,
    up,
    down,
    Γ,
    c_α,
    sqrt_abs_c_α,
    inv_sqrt_abs_c_α,
    sys_α,
    Δ_site,
    return_final_state=True,
    progress=False,
    progress_every=100,
    progress_width=32,
):
    """
    Propagate with low-memory RK4 and store the full physical density matrix
    ρ0(t) = ρ[0](t).
    """
    Nstep, save_every = _check_save_inputs(Nstep, save_every)

    ρ = np.ascontiguousarray(ρ, dtype=np.complex128)
    d = ρ.shape[1]

    (
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
    ) = _as_runtime_arrays(
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

    k, acc, ρ_tmp = allocate_rk4_lowmem_work_arrays(ρ)

    Nsave = Nstep // save_every + 1
    t_save = np.empty(Nsave, dtype=np.float64)
    ρ0_save = np.empty((Nsave, d, d), dtype=np.complex128)

    save_id = 0
    t_save[save_id] = 0.0
    ρ0_save[save_id, :, :] = ρ[0, :, :]
    save_id += 1

    progress_every = _resolve_progress_every(Nstep, progress_every)
    start_time = time.perf_counter()
    if progress:
        _print_progress(0, Nstep, start_time, width=progress_width, prefix="RK4")

    for step in range(1, Nstep + 1):
        rk4_step_htc_scaled_lowmem_inplace(
            ρ,
            float(dt),
            k,
            acc,
            ρ_tmp,
            E_diag,
            float(g),
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

        if step % save_every == 0:
            t_save[save_id] = float(dt) * float(step)
            ρ0_save[save_id, :, :] = ρ[0, :, :]
            save_id += 1

        if progress and _should_print_progress(step, Nstep, progress_every):
            _print_progress(step, Nstep, start_time, width=progress_width, prefix="RK4")

    result = {
        "t": t_save,
        "ρ0": ρ0_save,
        "rk4_storage": "lowmem_4_arrays_total",
    }
    if return_final_state:
        result["ρ"] = ρ
    return result


def propagate_rk4_htc_scaled_store_observables_and_sites(
    ρ,
    dt,
    Nstep,
    save_every,
    E_diag,
    g,
    ado_indices,
    up,
    down,
    Γ,
    c_α,
    sqrt_abs_c_α,
    inv_sqrt_abs_c_α,
    sys_α,
    Δ_site,
    ψ_LP,
    ψ_UP,
    return_final_state=True,
    progress=False,
    progress_every=100,
    progress_width=32,
):
    """
    Propagate with low-memory RK4 and store collective observables plus
    individual molecular site populations.
    """
    Nstep, save_every = _check_save_inputs(Nstep, save_every)

    ρ = np.ascontiguousarray(ρ, dtype=np.complex128)
    ψ_LP = np.ascontiguousarray(ψ_LP, dtype=np.complex128)
    ψ_UP = np.ascontiguousarray(ψ_UP, dtype=np.complex128)

    d = ρ.shape[1]
    Nmol = d - 1

    (
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
    ) = _as_runtime_arrays(
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

    k, acc, ρ_tmp = allocate_rk4_lowmem_work_arrays(ρ)

    Nsave = Nstep // save_every + 1
    t_save = np.empty(Nsave, dtype=np.float64)
    obs_save = allocate_observable_array(Nsave)
    P_site_save = np.empty((Nsave, Nmol), dtype=np.float64)

    save_id = 0
    t_save[save_id] = 0.0
    compute_observables_inplace(ρ[0], ψ_LP, ψ_UP, obs_save[save_id])
    site_populations_inplace(ρ[0], P_site_save[save_id])
    save_id += 1

    progress_every = _resolve_progress_every(Nstep, progress_every)
    start_time = time.perf_counter()
    if progress:
        _print_progress(0, Nstep, start_time, width=progress_width, prefix="RK4")

    for step in range(1, Nstep + 1):
        rk4_step_htc_scaled_lowmem_inplace(
            ρ,
            float(dt),
            k,
            acc,
            ρ_tmp,
            E_diag,
            float(g),
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

        if step % save_every == 0:
            t_save[save_id] = float(dt) * float(step)
            compute_observables_inplace(ρ[0], ψ_LP, ψ_UP, obs_save[save_id])
            site_populations_inplace(ρ[0], P_site_save[save_id])
            save_id += 1

        if progress and _should_print_progress(step, Nstep, progress_every):
            _print_progress(step, Nstep, start_time, width=progress_width, prefix="RK4")

    result = {
        "t": t_save,
        "obs": obs_save,
        "P_site": P_site_save,
        "observable_names": observable_names(),
        "rk4_storage": "lowmem_4_arrays_total",
    }
    if return_final_state:
        result["ρ"] = ρ
    return result


# ============================================================
# Small debugging helper
# ============================================================

def rk4_standard_vs_lowmem_one_step_error(
    ρ,
    dt,
    E_diag,
    g,
    ado_indices,
    up,
    down,
    Γ,
    c_α,
    sqrt_abs_c_α,
    inv_sqrt_abs_c_α,
    sys_α,
    Δ_site,
):
    """
    Compare one standard RK4 step and one low-memory RK4 step.

    Use only for small debugging tests.
    """
    ρ_std = np.ascontiguousarray(ρ.copy(), dtype=np.complex128)
    ρ_low = np.ascontiguousarray(ρ.copy(), dtype=np.complex128)

    (
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
    ) = _as_runtime_arrays(
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

    k1, k2, k3, k4, ρ_tmp_std = allocate_rk4_work_arrays(ρ_std)
    k, acc, ρ_tmp_low = allocate_rk4_lowmem_work_arrays(ρ_low)

    rk4_step_htc_scaled_inplace(
        ρ_std,
        float(dt),
        k1,
        k2,
        k3,
        k4,
        ρ_tmp_std,
        E_diag,
        float(g),
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

    rk4_step_htc_scaled_lowmem_inplace(
        ρ_low,
        float(dt),
        k,
        acc,
        ρ_tmp_low,
        E_diag,
        float(g),
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

    return float(np.max(np.abs(ρ_std - ρ_low)))


# ============================================================
# Lightweight self-test
# ============================================================

if __name__ == "__main__":
    from .constants import meV2au, cminv2au, K2au, fs2au
    from .htc_system_builder import build_htc_system, initial_density
    from .htc_channels import build_drude_htc_channels, build_ado_decay
    from .hierarchy import build_hierarchy

    Nmol = 3
    L = 2
    K = 0

    system = build_htc_system(
        Nmol=Nmol,
        ε_x=0.0,
        Δ_c=0.0,
        Ω_R=100.0 * meV2au,
        use_detuning_frame=True,
    )

    λ = 5.0 * cminv2au
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
    )

    M = channels["M"]
    ado_indices, up, down, tier, tier_offsets = build_hierarchy(M, L, validate=True)
    Γ = build_ado_decay(ado_indices, channels["ν_α"])

    ρ0 = initial_density(
        Nmol=Nmol,
        state="UP",
        Δ_c=0.0,
        Ω_R=100.0 * meV2au,
        use_detuning_frame=True,
    )
    ρ = initialize_ado_state(ρ0, ado_indices.shape[0])

    err = rk4_standard_vs_lowmem_one_step_error(
        ρ=ρ,
        dt=0.05 * fs2au,
        E_diag=system["E_diag"],
        g=system["g"],
        ado_indices=ado_indices,
        up=up,
        down=down,
        Γ=Γ,
        c_α=channels["c_α"],
        sqrt_abs_c_α=channels["sqrt_abs_c_α"],
        inv_sqrt_abs_c_α=channels["inv_sqrt_abs_c_α"],
        sys_α=channels["sys_α"],
        Δ_site=channels["Δ_site"],
    )
    print("standard vs low-memory one-step max abs error =", err, flush=True)

    result = propagate_rk4_htc_scaled_store_observables(
        ρ=ρ,
        dt=0.05 * fs2au,
        Nstep=4,
        save_every=1,
        E_diag=system["E_diag"],
        g=system["g"],
        ado_indices=ado_indices,
        up=up,
        down=down,
        Γ=Γ,
        c_α=channels["c_α"],
        sqrt_abs_c_α=channels["sqrt_abs_c_α"],
        inv_sqrt_abs_c_α=channels["inv_sqrt_abs_c_α"],
        sys_α=channels["sys_α"],
        Δ_site=channels["Δ_site"],
        ψ_LP=system["ψ_LP"],
        ψ_UP=system["ψ_UP"],
        return_final_state=False,
        progress=True,
        progress_every=1,
    )

    print("t =", result["t"], flush=True)
    print("obs shape =", result["obs"].shape, flush=True)
    print("final obs =", result["obs"][-1], flush=True)
    print("integrators_htc.py low-memory self-test passed.", flush=True)