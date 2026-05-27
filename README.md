# htc-heom-fast

`htc-heom-fast` is a small research-oriented Python/Numba package for propagating the scaled hierarchical equations of motion (HEOM) for a symmetric Holstein-Tavis-Cummings / Tavis-Cummings model in the first-excitation manifold.

The main design goal is not to be a fully general HEOM library.  The goal is to provide a fast, readable, and modifiable code path for one important family of polaritonic models:

- one cavity mode,
- `Nmol` two-level molecular excitations,
- first-excitation manifold only,
- site-local independent bosonic baths,
- Drude-Lorentz bath correlation functions represented by Matsubara exponential terms,
- matrix-free HEOM right-hand side using the sparse star structure of the HTC Hamiltonian.

In other words, this package is meant for calculations where a general HEOM package is too expensive because it would build large dense system superoperators, but where the model structure is simple enough that the action of the Liouvillian can be written explicitly.



## Why this package exists

A direct HEOM calculation stores many auxiliary density operators (ADOs).  Each ADO is a system density matrix.  If the system dimension is

$$
d = N_{\mathrm{mol}} + 1.
$$

then one ADO stores a `d x d` complex matrix.  If there are `N_ado` ADOs, one full HEOM state stores

$$
N_{\mathrm{ADO}} d^2
$$

complex numbers.  The hierarchy count grows combinatorially with the number of bath exponential channels and the hierarchy depth.  For one exponential per site,

$$
M = N_{\mathrm{mol}}, \qquad
N_{\mathrm{ADO}} = \binom{M+L}{L}.
$$

where `L` is the hierarchy depth.  For `K_matsubara + 1` exponentials per site,

$$
M = N_{\mathrm{mol}}(K_{\mathrm{matsubara}}+1), \qquad
N_{\mathrm{ADO}} = \binom{M+L}{L}.
$$

The memory cost is therefore dominated by the product

$$
N_{\mathrm{ADO}}(N_{\mathrm{mol}}+1)^2.
$$

This package reduces the practical cost in three ways.

First, it uses scaled HEOM variables.  The scaling improves numerical conditioning by absorbing bath-coefficient magnitudes into the definition of the ADOs.  This is especially helpful when different hierarchy tiers differ by many orders of magnitude.

Second, it avoids constructing the full HEOM Liouvillian matrix.  Instead of forming a giant matrix `M_HEOM` and multiplying it by a vectorized HEOM state, the code evaluates the right-hand side directly:

$$
\frac{\partial \rho}{\partial t} = f(\rho).
$$

Third, it uses the special star structure of the HTC Hamiltonian in the basis

$$
|C\rangle, |1\rangle, |2\rangle, \ldots, |N_{\mathrm{mol}}\rangle.
$$

where only the cavity state couples directly to each molecular excitation.  This allows the Hamiltonian commutator to be applied without dense `d x d` matrix multiplications for every ADO.



## Model and basis convention

The package works in the first-excitation basis

$$
|C\rangle \leftrightarrow \text{index }0,
\qquad
|n\rangle \leftrightarrow \text{index }n \ge 1.
$$

In bra-ket notation, the ordered basis is

$$
\{|C\rangle, |1\rangle, |2\rangle, \ldots, |N_{\mathrm{mol}}\rangle\}.
$$

The system Hamiltonian is represented as

$$
\hat{H}_s = E_C |C\rangle\langle C| + \sum_{n=1}^{N_{\mathrm{mol}}} E_n  n\rangle\langle n| + \sum_{n=1}^{N_{\mathrm{mol}}} g_n \left( |C\rangle\langle n| + |n\rangle\langle C| \right).
$$

In matrix form in the basis above,

$$
\hat{H}_s =
\begin{pmatrix}
E_C & g_1 & g_2 & \cdots & g_N \\
g_1 & E_1 & 0 & \cdots & 0 \\
g_2 & 0 & E_2 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
g_N & 0 & 0 & \cdots & E_N
\end{pmatrix}.
$$

The current fast path assumes the symmetric HTC/Tavis-Cummings structure, usually with

$$
E_1 = E_2 = \cdots = E_N,
\qquad
g_1 = g_2 = \cdots = g_N = g.
$$

The collective Rabi splitting convention used by the command-line driver is

$$
\Omega_R = 2g\sqrt{N_{\mathrm{mol}}}.
$$

Each site bath couples to the molecular excitation projector

$$
\hat{Q}_n = |n\rangle\langle n|.
$$

In the code this is represented by the integer `sys_alpha`, because each bath channel only needs to know which molecular basis index is selected by the diagonal projector.



## Scaled HEOM convention used by the code

For each site-local bath, the bath correlation function is represented as an exponential sum

$$
C(t) = \sum_{k=0}^{K} c_k e^{-\nu_k t} + C_{\mathrm{res}}(t).
$$

Here:

- `k = 0` is the Drude pole,
- `k >= 1` are Matsubara terms,
- `K` is controlled by `--K-matsubara`,
- the optional residual is handled through a simple Drude/Matsubara terminator unless `--no-terminator` is used.

For `Nmol` molecules and `K + 1` exponential terms per site, the total number of hierarchy channels is

$$
M = N_{\mathrm{mol}}(K+1).
$$

An ADO is indexed by a non-negative integer vector

$$
\mathbf{n} = (n_0,n_1,\ldots,n_{M-1}).
$$

The hierarchy tier is

$$
\mathrm{tier}(\mathbf{n}) = n_0+n_1+\cdots+n_{M-1}.
$$

The physical reduced density matrix is the zeroth ADO:

$$
\texttt{rho}[0] = \hat{\rho}_0(t).
$$

All other ADOs encode system-bath memory and system-bath correlation information.

The code propagates a scaled hierarchy.  Schematically, the scaled equation is

<!-- $$
\begin{aligned}
\frac{d}{dt}\tilde{\rho}_{\mathbf{n}}
=& -i[\hat{H}_s,\tilde{\rho}_{\mathbf{n}}]
- \left(\sum_\alpha n_\alpha\nu_\alpha\right)
\tilde{\rho}_{\mathbf{n}} \\
&+ \text{upward couplings to }\tilde{\rho}_{\mathbf{n}+\mathbf{e}_\alpha}
+ \text{downward couplings to }\tilde{\rho}_{\mathbf{n}-\mathbf{e}_\alpha}
+ \text{optional terminator}.
\end{aligned}
$$ -->

```math
\(\begin{aligned} \frac{d}{dt}\tilde{\rho}_{\mathbf{n}} =& -i[\hat{H}_s,\tilde{\rho}_{\mathbf{n}}] - \left(\sum_\alpha n_\alpha\nu_\alpha\right) \tilde{\rho}_{\mathbf{n}} \\ &+ \text{upward couplings to }\tilde{\rho}_{\mathbf{n}+\mathbf{e}_{\alpha}} \\ &+ \text{downward couplings to }\tilde{\rho}_{\mathbf{n}-\mathbf{e}_{\alpha}} \\ &+ \text{optional terminator}. \end{aligned} \%\%\)MAGIT_PARSER_PROTECT%%```



The exact prefactors are implemented in `rhs_htc_scaled.py`.  The important implementation arrays are:

```text
ado_indices[I, alpha]         n_alpha for ADO I
up[I, alpha]                  index of ADO n + e_alpha, or -1 if outside truncation
down[I, alpha]                index of ADO n - e_alpha, or -1 if absent
Gamma[I]                      sum_alpha n_alpha nu_alpha
c_alpha[alpha]                bath coefficient c_alpha
sqrt_abs_c_alpha[alpha]       sqrt(|c_alpha|)
inv_sqrt_abs_c_alpha[alpha]   1 / sqrt(|c_alpha|)
sys_alpha[alpha]              molecular basis index selected by Q_alpha
```



## GitHub metadata files

For a plain-language explanation of `pyproject.toml`, `LICENSE.md`, `CITATION.cff`, `.gitignore`, and related repository files, see [`docs/github_metadata.md`](docs/github_metadata.md).

## Repository layout

```text
src/htc_heom_fast/
  constants.py              unit conversions and small helper functions
  bath_drude.py             Drude-Lorentz Matsubara coefficients and terminator
  hierarchy.py              ADO counting, ADO generation, tier offsets, neighbor maps
  htc_system_builder.py     HTC basis, Hamiltonian, LP/UP states, initial states
  htc_channels.py           site-local bath channel bookkeeping
  rhs_htc_scaled.py         matrix-free scaled HEOM right-hand side
  integrators_htc.py        RK4 integrators and storage routines
  observables_htc.py        cavity, exciton, bright, dark, LP, UP observables
  run_htc.py                command-line simulation driver
  plot_htc_dynamics.py      plotting command

tests/
  test_core_modules.py      tests for constants, bath coefficients, hierarchy, memory estimates
  test_cli_smoke.py         quick test that the command-line parser loads and has sane defaults

docs/
  theory.md                 expanded theory notes for the implemented equations
  manual.md                 installation, local runs, SLURM runs, and output-file guide
  testing.md                plain-English explanation of every current test
  rk4_memory.md             discussion of RK4 work-array memory and lower-memory alternatives

examples/
  run_local.sh              laptop/workstation example
  plot_local.sh             plot a saved run
  slurm/run_htc.slurm       cluster job example
  slurm/plot_htc.slurm      cluster plotting example
```



## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/<your-username>/htc-heom-fast.git
cd htc-heom-fast

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

The editable install means that changes you make inside `src/htc_heom_fast/` are immediately visible when you run the command-line tools again.

Run the tests:

```bash
pytest
```

The first run can be slower because Numba may compile kernels.  Later runs should be faster because cache files are reused.



## Quick local run

Start with an estimate-only run.  This does not propagate the HEOM.  It only builds the model metadata and reports memory estimates.

```bash
htc-heom-run \
  --Nmol 5 \
  --L 4 \
  --K-matsubara 0 \
  --dt-fs 0.25 \
  --tmax-fs 100 \
  --estimate-only
```

A short actual propagation is:

```bash
mkdir -p Data Plots

htc-heom-run \
  --Nmol 5 \
  --L 4 \
  --K-matsubara 0 \
  --lambda-cminv 50 \
  --gamma-cminv 18 \
  --temperature-K 300 \
  --Omega-R-mev 100 \
  --dt-fs 0.25 \
  --tmax-fs 100 \
  --save-every 1 \
  --initial-state UP \
  --store obs \
  --output Data/htc_N5_L4_UP_obs.npz
```

Plot the output:

```bash
htc-heom-plot Data/htc_N5_L4_UP_obs.npz \
  --time-unit ps \
  --output Plots/htc_N5_L4_UP_obs.png
```



## Choosing `--store`

The `--store` option controls how much trajectory information is saved.

### `--store obs`

This is the recommended default for large calculations.  It saves only collective observables, such as:

```text
P_cavity(t)
P_exciton_total(t)
P_bright(t)
P_dark_total(t)
P_LP(t)
P_UP(t)
trace(rho0)
```

Use this when you mainly want LP/UP/dark/cavity/exciton population dynamics.

### `--store obs_sites`

This saves collective observables and individual site populations.  It costs more output memory than `obs`, but is still much cheaper than saving full density matrices.

### `--store rho0`

This saves the full physical reduced density matrix trajectory,

```text
rho0[t_index, i, j].
```

This is useful for debugging, basis transformations, or custom observables, but it should not be used casually for very large `Nmol` or very frequent saving.



## Cluster usage

For the current implementation, use one MPI task and many CPU threads on a node.  The code parallelizes the RHS over ADOs with Numba `prange`; it is not currently a distributed-memory MPI code.

A typical SLURM header is:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
```

Set thread-related environment variables before running Python:

```bash
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1
```

Then call `htc-heom-run` with the desired parameters.  Always run an estimate first:

```bash
htc-heom-run \
  --Nmol 20 \
  --L 10 \
  --K-matsubara 0 \
  --dt-fs 0.25 \
  --tmax-fs 1000 \
  --max-ram-gb 1000 \
  --estimate-only
```

The `--max-ram-gb` option is a soft guard.  If the estimate exceeds this value, the driver stops unless `--force` is passed.



## Practical workflow for a new calculation

A safe workflow is:

1. Run with `--estimate-only`.
2. Start with a small hierarchy depth `L`.
3. Run a short propagation with `--store obs`.
4. Halve `dt-fs` and check whether the observables change significantly.
5. Increase `L` and check convergence.
6. Increase `K-matsubara` only when needed.
7. Use `--store rho0` only for smaller validation runs.
8. Save the command used for every production run.

For production-quality figures, convergence should be checked with respect to at least:

```text
hierarchy depth L,
time step dt-fs,
number of Matsubara terms K,
terminator on/off or terminator choice,
simulation time window,
observable save interval.
```



## Output files

Simulation outputs are compressed `.npz` files.  Depending on `--store`, they contain:

```text
t                   saved times in atomic units
obs                 collective observables, if requested
observable_names    names corresponding to columns of obs
P_site              site populations, if requested
rho0                full physical reduced density matrix trajectory, if requested
params              metadata dictionary saved as a NumPy object array
rho_final           final full HEOM state, only if --save-final-state is used
ado_indices         hierarchy indices, only if --save-hierarchy is used
up, down, tier      hierarchy maps, only if --save-hierarchy is used
```

For very large calculations, avoid `--save-final-state` and `--save-hierarchy` unless you need restart/debugging data.  These arrays can be very large.

A future improvement should replace the current object-array `params` storage with JSON metadata.  That would make the output easier to inspect without `allow_pickle=True`.



## What the tests currently check

The current tests are deliberately small and fast.  They are not a complete physics validation suite yet.  They check that:

- unit conversions are internally consistent,
- Drude-Lorentz Matsubara coefficients have the expected shapes and values,
- site-local bath channels map correctly to molecular projectors,
- hierarchy counts match the combinatorial formula,
- up/down ADO neighbor maps point to the correct multi-indices,
- memory estimates match direct byte counting,
- the command-line parser loads and returns expected defaults.

See `docs/testing.md` for a detailed explanation of every test.

Important tests still missing before a formal public release:

- matrix-free RHS versus dense reference RHS for multiple small HTC systems,
- trace preservation of `rho[0]`,
- Hermiticity of `rho[0]`,
- bathless Tavis-Cummings analytic Rabi oscillations,
- convergence regression tests for a known HEOM benchmark,
- output-file read/write tests.



## RK4 memory status

The current low-memory classical RK4 implementation stores:

```text
rho      current HEOM state
k        reused as k1, k2, k3, k4
acc      k1 + 2 k2 + 2 k3 accumulator
rho_tmp  temporary stage state
```

This means four hierarchy-sized arrays total, or three hierarchy-sized work arrays in addition to the state.

For exact classical RK4 with the current RHS interface `rhs(rho_in, rho_out)`, this is close to minimal.  You need one array for the current derivative, one array for the accumulated RK4 increment, and one array for the intermediate stage state.  More aggressive reduction is possible only by changing the integrator or changing the RHS interface.

The most promising option is not a smarter rearrangement of classical RK4, but a low-storage fourth-order Runge-Kutta scheme, for example a 5-stage 4th-order 2N-storage RK method.  That would use approximately

```text
rho      current HEOM state
res      low-storage residual / derivative accumulator
```

but it would require a new RHS kernel of the form

```text
res <- a_s res + dt f(rho)
rho <- rho + b_s res
```

so that `f(rho)` can be accumulated into `res` without allocating a separate derivative array.  See `docs/rk4_memory.md` for details.



## Current limitations

- First-excitation manifold only.
- Symmetric HTC/Tavis-Cummings star Hamiltonian is assumed by the fast RHS.
- Independent site-local Drude-Lorentz baths are the main supported bath model.
- Time integration currently uses explicit classical RK4; stability must be checked by decreasing `dt-fs`.
- No restart/checkpoint workflow is included yet.
- No GPU implementation is included.
- No distributed-memory MPI decomposition is included.
- No permutation-symmetry ADO reduction is included yet.
- No unique-variable reduction inside symmetry-adapted ADOs is included yet.



## Suggested citation text

If you use this code before a formal paper or DOI exists, cite the GitHub repository and the exact commit hash used for the calculations.

