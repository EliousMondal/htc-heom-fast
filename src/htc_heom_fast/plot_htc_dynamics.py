# ============================================================
# plot_htc_dynamics.py
#
# Plot polariton-state populations from the output of run_htc.py.
#
# Expected run_htc.py output keys:
#
#     t_fs
#     obs
#     observable_names
#
# The main plotted quantities are
#
#     P_LP(t)   = <LP|ρ_0(t)|LP>
#     P_UP(t)   = <UP|ρ_0(t)|UP>
#     P_D(t)    = total dark population = P_X(t) - P_B(t)
#
# Optional quantities:
#
#     P_C(t)    = cavity population
#     P_B(t)    = bright exciton population
#     P_X(t)    = total exciton population
#
# Usage examples:
#
#     python plot_htc_dynamics.py htc_heom_N25_K0_L7_UP_obs.npz
#
#     python plot_htc_dynamics.py htc_heom_N25_K0_L7_UP_obs.npz \
#         --output htc_polariton_populations.png
#
#     python plot_htc_dynamics.py htc_heom_N25_K0_L7_UP_obs.npz \
#         --time-unit ps --include-cavity-bright --show
# ============================================================

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Small utilities
# ============================================================

def _decode_name(x):
    """Convert a saved numpy string/bytes/object entry to a Python string."""
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def load_observable_table(filename):
    """
    Load time, observable array, and observable-name map from a run_htc.py npz file.

    Returns
    -------
    t_fs : ndarray, shape (Nt,)
        Time grid in femtoseconds.
    obs : ndarray, shape (Nt, Nobs)
        Observable table.
    name_to_col : dict[str, int]
        Dictionary mapping observable name to column index.
    data : numpy.lib.npyio.NpzFile
        Open npz data object, returned mostly for optional metadata access.
    """
    data = np.load(filename, allow_pickle=True)

    required = ("t_fs", "obs", "observable_names")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(
            f"File {filename!r} is missing required keys: {missing}. "
            "This script expects an output file produced by run_htc.py."
        )

    t_fs = np.asarray(data["t_fs"], dtype=np.float64)
    obs = np.asarray(data["obs"], dtype=np.float64)
    observable_names = [_decode_name(x) for x in data["observable_names"]]

    if obs.ndim != 2:
        raise ValueError(f"Expected obs to be 2D, got shape {obs.shape}.")
    if t_fs.ndim != 1:
        raise ValueError(f"Expected t_fs to be 1D, got shape {t_fs.shape}.")
    if obs.shape[0] != t_fs.size:
        raise ValueError(
            f"Time and observable lengths do not match: "
            f"len(t_fs)={t_fs.size}, obs.shape[0]={obs.shape[0]}."
        )
    if obs.shape[1] != len(observable_names):
        raise ValueError(
            f"Observable column mismatch: obs.shape[1]={obs.shape[1]}, "
            f"len(observable_names)={len(observable_names)}."
        )

    name_to_col = {name: i for i, name in enumerate(observable_names)}
    return t_fs, obs, name_to_col, data


def get_column(obs, name_to_col, name):
    """Return one observable column by name with a clear error if absent."""
    if name not in name_to_col:
        available = ", ".join(name_to_col.keys())
        raise KeyError(f"Observable {name!r} was not found. Available columns: {available}")
    return obs[:, name_to_col[name]]


def default_output_name(input_file, time_unit, extension="png"):
    """Build a default output filename from the input filename."""
    base, _ = os.path.splitext(os.path.basename(input_file))
    return f"{base}_polariton_populations_{time_unit}.{extension}"


def _metadata_title(data):
    """
    Try to build a compact title from the saved params dictionary.

    If anything is missing, return an empty string instead of failing.
    """
    if "params" not in data.files:
        return ""

    try:
        params = data["params"].item()
    except Exception:
        return ""

    try:
        Nmol = params.get("Nmol", None)
        L = params.get("L", None)
        K = params.get("K_matsubara", None)
        init = params.get("initial_state", None)
        Ω_R = params.get("Omega_R_mev", None)

        pieces = []
        if Nmol is not None:
            pieces.append(f"N={Nmol}")
        if L is not None:
            pieces.append(f"L={L}")
        if K is not None:
            pieces.append(f"K={K}")
        if Ω_R is not None:
            pieces.append(rf"$\Omega_R$={float(Ω_R):g} meV")
        if init is not None:
            pieces.append(f"init={init}")

        return ", ".join(pieces)
    except Exception:
        return ""


# ============================================================
# Plotting
# ============================================================

def plot_polariton_populations(
    input_file,
    output_file=None,
    time_unit="fs",
    include_cavity_bright=False,
    include_exciton=False,
    include_trace=False,
    title=None,
    xlim=None,
    ylim=None,
    dpi=300,
    show=False,
):
    """
    Plot polariton populations from a run_htc.py npz output file.
    """
    t_fs, obs, name_to_col, data = load_observable_table(input_file)

    if time_unit == "fs":
        t = t_fs
        xlabel = "time / fs"
    elif time_unit == "ps":
        t = t_fs / 1000.0
        xlabel = "time / ps"
    else:
        raise ValueError("time_unit must be either 'fs' or 'ps'.")

    P_LP = get_column(obs, name_to_col, "P_LP")
    P_UP = get_column(obs, name_to_col, "P_UP")
    P_D = get_column(obs, name_to_col, "P_D")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    ax.plot(t, P_UP, label=r"$P_{UP}$")
    ax.plot(t, P_LP, label=r"$P_{LP}$")
    ax.plot(t, P_D, label=r"$P_D$")

    if include_cavity_bright:
        P_C = get_column(obs, name_to_col, "P_C")
        P_B = get_column(obs, name_to_col, "P_B")
        ax.plot(t, P_C, linestyle="--", label=r"$P_C$")
        ax.plot(t, P_B, linestyle="--", label=r"$P_B$")

    if include_exciton:
        P_X = get_column(obs, name_to_col, "P_X")
        ax.plot(t, P_X, linestyle=":", label=r"$P_X$")

    if include_trace:
        trace_re = get_column(obs, name_to_col, "trace_re")
        ax.plot(t, trace_re, linestyle=":", label=r"$\mathrm{Tr}\rho_0$")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("population")

    if title is None:
        title = _metadata_title(data)
    if title:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(bottom=-0.02)

    ax.legend(frameon=False)
    ax.tick_params(direction="in")
    fig.tight_layout()

    if output_file is None:
        output_file = default_output_name(input_file, time_unit, extension="png")

    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_file


# ============================================================
# Command-line interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot HTC polariton populations from a run_htc.py output npz file."
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input .npz file produced by run_htc.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image filename. Default: derived from input filename.",
    )
    parser.add_argument(
        "--time-unit",
        type=str,
        default="fs",
        choices=("fs", "ps"),
        help="Time axis unit. Default: fs.",
    )
    parser.add_argument(
        "--include-cavity-bright",
        action="store_true",
        help="Also plot P_C and P_B using dashed curves.",
    )
    parser.add_argument(
        "--include-exciton",
        action="store_true",
        help="Also plot total exciton population P_X.",
    )
    parser.add_argument(
        "--include-trace",
        action="store_true",
        help="Also plot Re Tr[rho_0] as a trace-conservation check.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Manual plot title. Default: use metadata from the npz file if available.",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Optional x-axis limits in the selected time unit.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-axis limits.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for saved raster images. Default: 300.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_file = plot_polariton_populations(
        input_file=args.input,
        output_file=args.output,
        time_unit=args.time_unit,
        include_cavity_bright=args.include_cavity_bright,
        include_exciton=args.include_exciton,
        include_trace=args.include_trace,
        title=args.title,
        xlim=args.xlim,
        ylim=args.ylim,
        dpi=args.dpi,
        show=args.show,
    )

    print(f"Saved plot to: {output_file}")


if __name__ == "__main__":
    main()

# python plot_htc_dynamics.py htc_heom_N25_K0_L7_UP_obs.npz \
#    --output populations_N25_L7.png