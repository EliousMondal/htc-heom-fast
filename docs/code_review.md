# Code review notes

## Current strengths

- The code is already separated into physically meaningful modules: constants, bath decomposition, hierarchy generation, HTC system construction, channel bookkeeping, RHS evaluation, integration, observables, plotting, and a CLI driver.
- The HEOM RHS is matrix-free and avoids explicitly constructing the HEOM Liouvillian.
- The HTC Hamiltonian action exploits the star structure of the first-excitation HTC/Tavis-Cummings Hamiltonian.
- The hierarchy code uses compact integer dtypes and precomputed neighbor maps.
- The integrator includes a low-memory RK4 implementation.
- The driver already includes memory estimates, RAM safety checks, progress output, and cluster-friendly thread-control examples.
- The existing tests pass and cover constants, bath coefficients, bath-channel expansion, hierarchy counting, hierarchy construction, and memory estimates.

## Main issues to fix before public release

1. Keep generated data out of git.  The original archive included `Data/`, `Plots/`, `logs/`, `.DS_Store`, and `__MACOSX/` files.  These should not be committed.
2. Use a package layout.  Top-level scripts with sibling imports are convenient locally but fragile for users.  The generated repository uses a `src/htc_heom_fast/` layout and console commands.
3. Add more physics-level regression tests.  The current tests do not yet strongly protect the RHS, integrator, or observable code.
4. Avoid object-array metadata as the only metadata store.  Saving `params` as an object array requires `allow_pickle=True` when loading.  A JSON/TOML sidecar or a stringified JSON field inside the `.npz` would be safer and easier for other users.
5. Add restart/checkpoint support for long cluster jobs.
6. Decide on public API naming.  Unicode variable names are readable for theory work, but ASCII aliases are safer for a broader user base and shell/editor compatibility.
7. Add a license and citation information before making the repository public.

## Highest-priority tests to add

- Dense-vs-matrix-free RHS comparison for tiny systems, e.g. `Nmol=2`, `K=0`, `L=1 or 2`.
- Scaled-vs-unscaled prefactor checks for individual upward and downward couplings.
- Bathless TC analytic dynamics: starting from `|UP>` or `|C>`, compare to the exact two-level bright-cavity block.
- Trace conservation of the physical reduced density matrix `rho[0]`.
- Hermiticity preservation of `rho[0]`.
- RK4 timestep convergence: compare `dt` and `dt/2` for a small case.
- Observable identities: `P_D = P_X - P_B`, and `P_C + P_X = trace` in the first-excitation manifold.

## Performance and scalability improvements

- Add a benchmark script that reports `Nmol`, `K`, `L`, number of ADOs, RAM, number of threads, wall time per RK4 step, and output mode.
- Add a checkpoint mode that periodically saves the full ADO tensor to a restart file only every large interval.
- For very large outputs, consider uncompressed `.npz` or HDF5.  `np.savez_compressed` can be slow and CPU-heavy on clusters.
- Add a `--storage-format` option: `compressed_npz`, `npz`, or `hdf5`.
- Add a `--dry-run-json` option that writes the parsed parameters and memory estimates without compiling Numba kernels.
- Add optional symmetry-reduced hierarchy backends later, separate from this naive matrix-free backend.

## Usability improvements

- Add dataclasses such as `SystemParams`, `BathParams`, `HierarchyParams`, and `RunParams`.
- Add a high-level Python API, for example `run_simulation(params)`, so users do not have to call the CLI from notebooks.
- Replace most public `print` calls by the `logging` module while keeping progress output optional.
- Add clearer error messages for unsupported regimes, e.g. `K>0`, very large `L`, non-divisible `Nstep/save_every`, or unsupported initial states.
- Add examples as notebooks or small scripts, but keep output data out of git.
