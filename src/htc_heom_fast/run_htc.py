# ============================================================
# run_htc.py
#
# Main driver script for the naive matrix-free scaled HTC-HEOM
# code in the first-excitation manifold.
#
# Basis convention:
#
#     index 0      : |C>      = one cavity photon
#     index n >= 1 : |n>      = molecule n excited
#
# The physical reduced density matrix is the zeroth ADO:
#
#     ρ[0] = ρ_0.
#
# The default output stores only collective observables:
#
#     P_C, P_X, P_B, P_D, P_LP, P_UP, trace, etc.
#
# This script intentionally does not build the HEOM Liouvillian.
# It propagates the ADO tensor directly with a matrix-free RHS.
# ============================================================

import argparse
import os
import platform
import time
import builtins
from functools import partial

import numpy as np

# Make all print statements flush immediately in SLURM output files.
print = partial(builtins.print, flush=True)

from .constants import fs2au, au2fs, cminv2au, meV2au, K2au

from .htc_system_builder import (
    build_htc_system,
    check_system_consistency,
    initial_density,
)

from .htc_channels import build_drude_htc_channels, build_ado_decay

from .hierarchy import (
    number_of_ados,
    count_by_tier,
    estimate_hierarchy_metadata_gb,
    estimate_rk4_state_gb,
    build_hierarchy,
)

from .integrators_htc import (
    initialize_ado_state,
    propagate_rk4_htc_scaled_store_observables,
    propagate_rk4_htc_scaled_store_observables_and_sites,
    propagate_rk4_htc_scaled_store_rho0,
)

try:
    import numba
except Exception:
    numba = None


# ============================================================
# Command-line interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run matrix-free scaled HEOM dynamics for the first-manifold "
            "HTC/Tavis-Cummings model with independent identical "
            "site-local Drude-Lorentz baths."
        )
    )

    # ------------------------------
    # System parameters
    # ------------------------------
    parser.add_argument("--Nmol", type=int, default=5,
                        help="Number of molecules/sites. Default: 5")

    parser.add_argument("--eps-x-mev", type=float, default=0.0,
                        help="Molecular excitation energy ε_x in meV. Default: 0")

    parser.add_argument("--detuning-mev", type=float, default=0.0,
                        help="Cavity-exciton detuning Δ_c = ω_c - ε_x in meV. Default: 0")

    parser.add_argument("--omega-c-mev", type=float, default=None,
                        help="Absolute cavity energy ω_c in meV. If supplied, this overrides --detuning-mev.")

    parser.add_argument("--Omega-R-mev", type=float, default=100.0,
                        help="Collective Rabi splitting Ω_R = 2 g sqrt(Nmol), in meV. Default: 100")

    parser.add_argument("--g-mev", type=float, default=None,
                        help="Single-molecule coupling g in meV. If supplied, this overrides --Omega-R-mev.")

    parser.add_argument("--absolute-frame", action="store_true",
                        help="Use absolute energies instead of subtracting ε_x I. Default uses detuning frame.")

    # ------------------------------
    # Bath and HEOM parameters
    # ------------------------------
    parser.add_argument("--lambda-cminv", type=float, default=50.0,
                        help="Drude reorganization energy λ in cm^-1. Default: 50")

    parser.add_argument("--gamma-cminv", type=float, default=18.0,
                        help="Drude decay rate γ in cm^-1. Default: 18")

    parser.add_argument("--temperature-K", type=float, default=300.0,
                        help="Temperature in Kelvin. Default: 300")

    parser.add_argument("--K-matsubara", type=int, default=0,
                        help="Number of Matsubara terms beyond the Drude pole. Default: 0")

    parser.add_argument("--L", type=int, default=4,
                        help="Maximum HEOM hierarchy depth. Default: 4")

    parser.add_argument("--no-terminator", action="store_true",
                        help="Disable the Drude Matsubara terminator Δ_LT.")

    parser.add_argument("--validate-hierarchy", action="store_true",
                        help="Run full hierarchy validation. Use only for small/debug cases.")

    # ------------------------------
    # Initial condition and propagation
    # ------------------------------
    parser.add_argument("--initial-state", type=str, default="UP",
                        choices=("UP", "LP", "bright", "B", "cavity", "C", "site"),
                        help="Initial system state. Default: UP")

    parser.add_argument("--site", type=int, default=1,
                        help="1-based site index used when --initial-state site. Default: 1")

    parser.add_argument("--dt-fs", type=float, default=0.25,
                        help="RK4 time step in fs. Default: 0.25")

    parser.add_argument("--tmax-fs", type=float, default=100.0,
                        help="Maximum simulation time in fs. Default: 100")

    parser.add_argument("--save-every", type=int, default=1,
                        help="Save every this many RK4 steps. Default: 1")

    parser.add_argument("--no-progress", action="store_true",
                        help="Disable the live RK4 progress bar. Default: progress bar enabled.")

    parser.add_argument("--progress-every", type=int, default=0,
                        help="Print progress every this many RK4 steps. Use 0 for about 100 updates. Default: 0")

    parser.add_argument("--progress-width", type=int, default=32,
                        help="Character width of the progress bar. Default: 32")

    # ------------------------------
    # Output and safety controls
    # ------------------------------
    parser.add_argument("--store", type=str, default="obs",
                        choices=("obs", "obs_sites", "rho0"),
                        help="Output mode: collective observables, observables+site populations, or full rho0(t). Default: obs")

    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz filename. If omitted, an automatic name is used.")

    parser.add_argument("--estimate-only", action="store_true",
                        help="Only print ADO/memory estimates; do not allocate or propagate.")

    parser.add_argument("--max-ram-gb", type=float, default=1000.0,
                        help="Soft RAM limit for safety checks in GB. Default: 1000")

    parser.add_argument("--force", action="store_true",
                        help="Run even if the estimated memory exceeds --max-ram-gb.")

    parser.add_argument("--save-final-state", action="store_true",
                        help="Save final full HEOM ADO tensor ρ. Dangerous for large runs.")

    parser.add_argument("--save-hierarchy", action="store_true",
                        help="Save ado_indices/up/down/tier arrays. Dangerous for large runs.")

    return parser.parse_args()


# ============================================================
# Setup helpers
# ============================================================

def _safe_num_tag(x):
    """
    Return a filesystem-friendly compact numerical tag.
    """
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


def make_output_filename(args):
    """
    Make output filename. If args.output contains a folder, that folder is used.
    """
    if args.output is not None:
        return args.output

    dt_tag = _safe_num_tag(args.dt_fs)
    t_tag = _safe_num_tag(args.tmax_fs)

    return (
        f"htc_heom_N{args.Nmol}_K{args.K_matsubara}_L{args.L}_"
        f"{args.initial_state}_dt{dt_tag}fs_t{t_tag}fs_"
        f"{args.store}.npz"
    )


def format_seconds(seconds):
    seconds = float(seconds)

    if seconds < 60.0:
        return f"{seconds:.3f} s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.3f} min"

    return f"{seconds / 3600.0:.3f} h"


def build_parameter_dict(
    args,
    system,
    channels,
    M,
    N_ado,
    Nstep,
    Nsave,
    dt_au,
    tmax_au,
    metadata_gb,
    rk4_state_gb,
    estimated_total_gb,
):
    """
    Return a plain dictionary of scalar/string metadata to save in the npz.
    """
    return {
        "Nmol": int(args.Nmol),
        "d": int(args.Nmol + 1),
        "M": int(M),
        "N_ado": int(N_ado),
        "L": int(args.L),
        "K_matsubara": int(args.K_matsubara),
        "Nstep": int(Nstep),
        "Nsave": int(Nsave),
        "save_every": int(args.save_every),
        "dt_fs": float(args.dt_fs),
        "dt_au": float(dt_au),
        "tmax_fs_requested": float(args.tmax_fs),
        "tmax_fs_actual": float(tmax_au * au2fs),
        "tmax_au_actual": float(tmax_au),
        "eps_x_mev": float(args.eps_x_mev),
        "detuning_mev": float(system["Δ_c"] / meV2au),
        "omega_c_mev": float(system["ω_c"] / meV2au),
        "Omega_R_mev": float(system["Ω_R"] / meV2au),
        "g_mev": float(system["g"] / meV2au),
        "lambda_cminv": float(args.lambda_cminv),
        "gamma_cminv": float(args.gamma_cminv),
        "temperature_K": float(args.temperature_K),
        "beta_au": float(channels["β"]),
        "Delta_LT_au": float(channels["Δ_LT"]),
        "Delta_LT_cminv": float(channels["Δ_LT"] / cminv2au),
        "initial_state": str(args.initial_state),
        "store": str(args.store),
        "use_detuning_frame": bool(not args.absolute_frame),
        "include_terminator": bool(not args.no_terminator),
        "metadata_gb": float(metadata_gb),
        "rk4_state_gb": float(rk4_state_gb),
        "estimated_total_gb": float(estimated_total_gb),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numba_threads": int(numba.get_num_threads()) if numba is not None else -1,
    }


def print_run_summary(params, tier_counts):
    print("\n============================================================", flush=True)
    print("HTC-HEOM run summary", flush=True)
    print("============================================================", flush=True)
    print(f"Nmol                    = {params['Nmol']}", flush=True)
    print(f"d = Nmol + 1            = {params['d']}", flush=True)
    print(f"K_matsubara             = {params['K_matsubara']}", flush=True)
    print(f"M = Nmol*(K+1)          = {params['M']}", flush=True)
    print(f"L                       = {params['L']}", flush=True)
    print(f"N_ado                   = {params['N_ado']}", flush=True)
    print("ADO counts by tier:", flush=True)

    for ℓ, count in enumerate(tier_counts):
        print(f"  tier {ℓ:2d}: {int(count)}", flush=True)

    print("------------------------------------------------------------", flush=True)
    print(f"ε_x                     = {params['eps_x_mev']:.8g} meV", flush=True)
    print(f"ω_c                     = {params['omega_c_mev']:.8g} meV", flush=True)
    print(f"Δ_c                     = {params['detuning_mev']:.8g} meV", flush=True)
    print(f"Ω_R                     = {params['Omega_R_mev']:.8g} meV", flush=True)
    print(f"g                       = {params['g_mev']:.8g} meV", flush=True)
    print("------------------------------------------------------------", flush=True)
    print(f"λ                       = {params['lambda_cminv']:.8g} cm^-1", flush=True)
    print(f"γ                       = {params['gamma_cminv']:.8g} cm^-1", flush=True)
    print(f"T                       = {params['temperature_K']:.8g} K", flush=True)
    print(f"Δ_LT                   = {params['Delta_LT_cminv']:.8g} cm^-1", flush=True)
    print("------------------------------------------------------------", flush=True)
    print(f"initial state           = {params['initial_state']}", flush=True)
    print(f"dt                      = {params['dt_fs']:.8g} fs", flush=True)
    print(f"Nstep                   = {params['Nstep']}", flush=True)
    print(f"tmax actual             = {params['tmax_fs_actual']:.8g} fs", flush=True)
    print(f"save_every              = {params['save_every']}", flush=True)
    print(f"Nsave                   = {params['Nsave']}", flush=True)
    print(f"store                   = {params['store']}", flush=True)
    print("------------------------------------------------------------", flush=True)
    print(f"estimated hierarchy metadata = {params['metadata_gb']:.3f} GB", flush=True)
    print(f"estimated RK4 state arrays   = {params['rk4_state_gb']:.3f} GB", flush=True)
    print(f"estimated total core memory  = {params['estimated_total_gb']:.3f} GB", flush=True)
    print("============================================================\n", flush=True)


# ============================================================
# Main driver
# ============================================================

def main():
    args = parse_args()

    if args.Nmol <= 0:
        raise ValueError("--Nmol must be positive.")
    if args.L < 0:
        raise ValueError("--L must be non-negative.")
    if args.K_matsubara < 0:
        raise ValueError("--K-matsubara must be non-negative.")
    if args.dt_fs <= 0.0:
        raise ValueError("--dt-fs must be positive.")
    if args.tmax_fs < 0.0:
        raise ValueError("--tmax-fs must be non-negative.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive.")

    use_detuning_frame = not args.absolute_frame

    ε_x = args.eps_x_mev * meV2au

    if args.omega_c_mev is None:
        ω_c = None
        Δ_c = args.detuning_mev * meV2au
    else:
        ω_c = args.omega_c_mev * meV2au
        Δ_c = None

    if args.g_mev is None:
        g = None
        Ω_R = args.Omega_R_mev * meV2au
    else:
        g = args.g_mev * meV2au
        Ω_R = None

    λ = args.lambda_cminv * cminv2au
    γ = args.gamma_cminv * cminv2au
    T = args.temperature_K * K2au

    if T <= 0.0:
        raise ValueError("Temperature must be positive.")

    β = 1.0 / T

    dt_au = args.dt_fs * fs2au
    Nstep = int(round(args.tmax_fs / args.dt_fs))

    if Nstep % args.save_every != 0:
        raise ValueError(
            "Nstep must be divisible by save_every for the current integrator. "
            f"Got Nstep={Nstep}, save_every={args.save_every}."
        )

    Nsave = Nstep // args.save_every + 1
    tmax_au = Nstep * dt_au

    M = args.Nmol * (args.K_matsubara + 1)
    N_ado = number_of_ados(M, args.L)

    metadata_gb = estimate_hierarchy_metadata_gb(M, args.L)

    rk4_state_gb = estimate_rk4_state_gb(
        Nmol=args.Nmol,
        L=args.L,
        K_matsubara=args.K_matsubara,
        n_work_arrays=4,   # set it to 6 for original RK4
    )

    estimated_total_gb = metadata_gb + rk4_state_gb

    system = build_htc_system(
        Nmol=args.Nmol,
        ε_x=ε_x,
        ω_c=ω_c,
        Δ_c=Δ_c,
        g=g,
        Ω_R=Ω_R,
        use_detuning_frame=use_detuning_frame,
        build_projectors=False,
    )

    check_system_consistency(system)

    channels = build_drude_htc_channels(
        Nmol=args.Nmol,
        λ=λ,
        γ=γ,
        β=β,
        K=args.K_matsubara,
        include_terminator=(not args.no_terminator),
        build_qdiag=False,
    )

    tier_counts_est = np.array(
        [
            number_of_ados(M, ℓ) - number_of_ados(M, ℓ - 1)
            if ℓ > 0 else 1
            for ℓ in range(args.L + 1)
        ],
        dtype=np.int64,
    )

    params_for_print = build_parameter_dict(
        args=args,
        system=system,
        channels=channels,
        M=M,
        N_ado=N_ado,
        Nstep=Nstep,
        Nsave=Nsave,
        dt_au=dt_au,
        tmax_au=tmax_au,
        metadata_gb=metadata_gb,
        rk4_state_gb=rk4_state_gb,
        estimated_total_gb=estimated_total_gb,
    )

    print_run_summary(params_for_print, tier_counts_est)

    if estimated_total_gb > args.max_ram_gb and not args.force:
        raise MemoryError(
            f"Estimated core memory {estimated_total_gb:.3f} GB exceeds "
            f"--max-ram-gb={args.max_ram_gb:.3f}. Use --force to run anyway."
        )

    if estimated_total_gb > 0.8 * args.max_ram_gb:
        print(
            "Warning: estimated memory is above 80% of the requested RAM limit. "
            "Avoid saving full hierarchy arrays or final ρ unless necessary.\n",
            flush=True,
        )

    if args.estimate_only:
        print("Estimate-only mode: stopping before hierarchy allocation and propagation.", flush=True)
        return

    # ------------------------------
    # Build hierarchy and initial state
    # ------------------------------
    t0 = time.perf_counter()
    print("Building hierarchy...", flush=True)

    ado_indices, up, down, tier, tier_offsets = build_hierarchy(
        M=M,
        L=args.L,
        validate=args.validate_hierarchy,
    )

    tier_counts = count_by_tier(tier, args.L)

    print(f"Hierarchy built in {format_seconds(time.perf_counter() - t0)}", flush=True)

    t0 = time.perf_counter()
    print("Building ADO decay Γ_I...", flush=True)

    Γ = build_ado_decay(ado_indices, channels["ν_α"])

    print(f"Γ_I built in {format_seconds(time.perf_counter() - t0)}", flush=True)

    ρ0 = initial_density(
        Nmol=args.Nmol,
        state=args.initial_state,
        ε_x=ε_x,
        ω_c=ω_c,
        Δ_c=Δ_c,
        g=g,
        Ω_R=Ω_R,
        site=args.site,
        use_detuning_frame=use_detuning_frame,
    )

    ρ = initialize_ado_state(ρ0, N_ado)

    # ------------------------------
    # Propagate
    # ------------------------------
    print("Starting RK4 propagation...", flush=True)

    t0 = time.perf_counter()

    if args.store == "obs":
        result = propagate_rk4_htc_scaled_store_observables(
            ρ=ρ,
            dt=dt_au,
            Nstep=Nstep,
            save_every=args.save_every,
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
            return_final_state=args.save_final_state,
            progress=(not args.no_progress),
            progress_every=args.progress_every,
            progress_width=args.progress_width,
        )

    elif args.store == "obs_sites":
        result = propagate_rk4_htc_scaled_store_observables_and_sites(
            ρ=ρ,
            dt=dt_au,
            Nstep=Nstep,
            save_every=args.save_every,
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
            return_final_state=args.save_final_state,
            progress=(not args.no_progress),
            progress_every=args.progress_every,
            progress_width=args.progress_width,
        )

    elif args.store == "rho0":
        result = propagate_rk4_htc_scaled_store_rho0(
            ρ=ρ,
            dt=dt_au,
            Nstep=Nstep,
            save_every=args.save_every,
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
            return_final_state=args.save_final_state,
            progress=(not args.no_progress),
            progress_every=args.progress_every,
            progress_width=args.progress_width,
        )

    else:
        raise ValueError(f"Unknown store mode: {args.store!r}")

    runtime = time.perf_counter() - t0

    print(f"Propagation finished in {format_seconds(runtime)}", flush=True)

    # ------------------------------
    # Save output
    # ------------------------------
    output = make_output_filename(args)
    output_dir = os.path.dirname(output)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving output to {output!r}...", flush=True)

    params = build_parameter_dict(
        args=args,
        system=system,
        channels=channels,
        M=M,
        N_ado=N_ado,
        Nstep=Nstep,
        Nsave=Nsave,
        dt_au=dt_au,
        tmax_au=tmax_au,
        metadata_gb=metadata_gb,
        rk4_state_gb=rk4_state_gb,
        estimated_total_gb=estimated_total_gb,
    )

    params["runtime_seconds"] = float(runtime)

    save_dict = {
        "t_au": result["t"],
        "t_fs": result["t"] * au2fs,
        "params": np.array(params, dtype=object),
        "tier_counts": tier_counts,
        "E_diag": system["E_diag"],
        "ψ_LP": system["ψ_LP"],
        "ψ_UP": system["ψ_UP"],
        "ν_site": channels["ν_site"],
        "c_site": channels["c_site"],
        "ν_α": channels["ν_α"],
        "c_α": channels["c_α"],
        "sys_α": channels["sys_α"],
        "site_α": channels["site_α"],
        "k_α": channels["k_α"],
        "Δ_site": channels["Δ_site"],
    }
    
    if "rk4_storage" in result:
        save_dict["rk4_storage"] = np.array(result["rk4_storage"])

    if "obs" in result:
        save_dict["obs"] = result["obs"]
        save_dict["observable_names"] = np.array(result["observable_names"])

    if "P_site" in result:
        save_dict["P_site"] = result["P_site"]

    if "ρ0" in result:
        save_dict["rho0"] = result["ρ0"]

    if args.save_final_state and "ρ" in result:
        save_dict["rho_final"] = result["ρ"]

    if args.save_hierarchy:
        save_dict["ado_indices"] = ado_indices
        save_dict["up"] = up
        save_dict["down"] = down
        save_dict["tier"] = tier
        save_dict["tier_offsets"] = tier_offsets

    np.savez_compressed(output, **save_dict)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()