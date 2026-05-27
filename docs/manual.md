# User manual

This manual is written for a new user who has the repository and wants to run, modify, and validate HTC-HEOM calculations.



## 1. Installation from a fresh clone

```bash
git clone https://github.com/<your-username>/htc-heom-fast.git
cd htc-heom-fast

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

The `-e` flag means editable mode.  If you edit a Python file under `src/htc_heom_fast/`, the installed command-line tools immediately use the edited version.

Check that the command-line tools are visible:

```bash
which htc-heom-run
which htc-heom-plot
```

Run tests:

```bash
pytest
```



## 2. Important command-line options

The main executable is

```bash
htc-heom-run
```

### System-size options

```bash
--Nmol 5
```

Number of molecules / two-level systems.  The first-excitation Hilbert-space dimension is `Nmol + 1`.

```bash
--Omega-R-mev 100
```

Collective Rabi splitting in meV.  If `--g-mev` is not supplied, the code uses

```text
g = Omega_R / (2 sqrt(Nmol)).
```

```bash
--g-mev 10
```

Single-molecule cavity coupling in meV.  If this is supplied, it overrides the value inferred from `--Omega-R-mev`.

```bash
--detuning-mev 0
```

Cavity-exciton detuning.  The exact convention is defined in `build_htc_system` in `htc_system_builder.py`.

### Bath options

```bash
--lambda-cminv 50
```

Reorganization energy in cm^-1.

```bash
--gamma-cminv 18
```

Drude bath decay rate in cm^-1.

```bash
--temperature-K 300
```

Temperature in Kelvin.

```bash
--K-matsubara 0
```

Number of Matsubara terms beyond the Drude pole.  `K=0` means one exponential per site.

```bash
--no-terminator
```

Disable the approximate residual Matsubara/low-temperature terminator.

### Hierarchy options

```bash
--L 4
```

Hierarchy depth.  This is the maximum tier included in the ADO basis.

```bash
--validate-hierarchy
```

Run additional validation checks when constructing the hierarchy.  Useful for debugging, but unnecessary for large production runs.

### Time-step options

```bash
--dt-fs 0.25
```

RK4 time step in femtoseconds.

```bash
--tmax-fs 1000
```

Total propagation time in femtoseconds.

```bash
--save-every 10
```

Save every 10 RK4 steps.  For example, if `dt = 0.25 fs`, this saves every `2.5 fs`.

### Initial-state options

```bash
--initial-state UP
```

Typical choices are `UP`, `LP`, `cavity`, `bright`, or `site` depending on what is implemented in `htc_system_builder.py`.

```bash
--site 1
```

Site index used when `--initial-state site` is selected.

### Output options

```bash
--store obs
```

Save collective observables only.  This is recommended for large calculations.

```bash
--store obs_sites
```

Save collective observables and site populations.

```bash
--store rho0
```

Save the full physical reduced density matrix `rho[0](t)`.

```bash
--save-final-state
```

Also save the full final HEOM state.  This can be huge.

```bash
--save-hierarchy
```

Also save `ado_indices`, `up`, `down`, and tier arrays.  Useful for debugging and restarting, but large.

```bash
--output Data/my_run.npz
```

Path to output `.npz` file.

### Safety options

```bash
--estimate-only
```

Build the model and print memory estimates without running dynamics.

```bash
--max-ram-gb 1000
```

Stop if the estimated memory is above this limit.

```bash
--force
```

Run even if the memory guard complains.



## 3. Recommended local workflow

Start with a very small run:

```bash
mkdir -p Data Plots

htc-heom-run \
  --Nmol 5 \
  --L 3 \
  --K-matsubara 0 \
  --lambda-cminv 50 \
  --gamma-cminv 18 \
  --temperature-K 300 \
  --Omega-R-mev 100 \
  --dt-fs 0.25 \
  --tmax-fs 50 \
  --save-every 1 \
  --initial-state UP \
  --store obs \
  --output Data/test_N5_L3.npz
```

Plot it:

```bash
htc-heom-plot Data/test_N5_L3.npz \
  --time-unit ps \
  --output Plots/test_N5_L3.png
```

Then increase only one parameter at a time:

```text
L = 3, 4, 5, ...
dt = 0.5 fs, 0.25 fs, 0.125 fs, ...
K = 0, 1, 2, ...
```

This makes it much easier to know which approximation is responsible for a change in the dynamics.



## 4. Recommended cluster workflow

Before submitting a long job, run estimate-only on the login node or in a short interactive job:

```bash
htc-heom-run \
  --Nmol 20 \
  --L 10 \
  --K-matsubara 0 \
  --dt-fs 0.25 \
  --tmax-fs 1000 \
  --save-every 10 \
  --store obs \
  --max-ram-gb 1000 \
  --estimate-only
```

Then create a SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=htc_heom_N20_L10
#SBATCH --partition=polariton
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1

mkdir -p Data logs

htc-heom-run \
  --Nmol 20 \
  --L 10 \
  --K-matsubara 0 \
  --lambda-cminv 50 \
  --gamma-cminv 18 \
  --temperature-K 300 \
  --Omega-R-mev 100 \
  --dt-fs 0.25 \
  --tmax-fs 1000 \
  --save-every 10 \
  --initial-state UP \
  --store obs \
  --max-ram-gb 1000 \
  --output Data/htc_N20_L10_UP_obs.npz
```

Submit with:

```bash
sbatch run_htc.slurm
```



## 5. Understanding memory estimates

The dominant memory cost is the HEOM state and RK4 work arrays.

A single hierarchy-sized state stores

```text
N_ado * d * d
```

complex128 numbers.  One complex128 number uses 16 bytes.

The current low-memory RK4 uses four hierarchy-sized arrays total:

```text
rho, k, acc, rho_tmp.
```

So the main state memory estimate is roughly

```text
4 * N_ado * d * d * 16 bytes.
```

There is also hierarchy metadata memory, such as:

```text
ado_indices, up, down, tier, Gamma, bath channels.
```

For large `Nmol` and `L`, the state arrays usually dominate, but the hierarchy maps can also be significant.



## 6. Reading an output file in Python

```python
import numpy as np

data = np.load("Data/htc_N5_L4_UP_obs.npz", allow_pickle=True)

print(data.files)

t_au = data["t"]
obs = data["obs"]
names = list(data["observable_names"])

for i, name in enumerate(names):
    print(i, name)
```

To convert time from atomic units to picoseconds:

```python
from htc_heom_fast.constants import ps2au

t_ps = t_au / ps2au
```

To access a named observable:

```python
idx = names.index("P_UP")
P_UP = obs[:, idx]
```

The exact observable names are defined in `observables_htc.py`.



## 7. Plotting manually

```python
import numpy as np
import matplotlib.pyplot as plt
from htc_heom_fast.constants import ps2au

data = np.load("Data/htc_N5_L4_UP_obs.npz", allow_pickle=True)
t_ps = data["t"] / ps2au
obs = data["obs"]
names = list(data["observable_names"])

for target in ["P_LP", "P_UP", "P_dark"]:
    if target in names:
        plt.plot(t_ps, obs[:, names.index(target)], label=target)

plt.xlabel("time / ps")
plt.ylabel("population")
plt.legend()
plt.tight_layout()
plt.show()
```



## 8. Convergence checklist

For any result you want to trust, check at least the following.

### Time step

Run with two time steps:

```text
dt
and
dt / 2
```

The dynamics should not change appreciably.

### Hierarchy depth

Run with increasing `L`:

```text
L, L+1, L+2
```

The reduced density dynamics should converge.

### Matsubara terms

Run with increasing `K_matsubara`:

```text
K = 0, 1, 2, ...
```

For high temperature and/or fast bath, `K=0` plus terminator may be enough.  For low temperature or slow bath, more terms may be required.

### Terminator

Compare terminator on and off:

```bash
# default: terminator on

# terminator off
--no-terminator
```

Large differences mean the residual correlation function is important.

### Storage mode

For a small case, compare `--store obs` with `--store rho0` and verify that observables computed manually from `rho0` match the saved observables.



## 9. Common mistakes

### Mistake: saving too much data

For large systems, avoid:

```bash
--store rho0
--save-final-state
--save-hierarchy
```

unless you have checked the output size.

### Mistake: using many BLAS threads and many Numba threads

Set BLAS thread counts to 1:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

and control parallelism through

```bash
export NUMBA_NUM_THREADS=<number of CPU cores>
```

### Mistake: trusting one hierarchy depth

HEOM is only useful after convergence checks.  A single `L` value is not enough.

### Mistake: assuming `K=0` is always enough

`K=0` is one exponential per site.  It can be a good starting point, but it is not automatically converged at low temperature or for difficult slow baths.



## 10. Suggested development workflow

When modifying the code:

```bash
pytest
```

Then run a tiny physical smoke test:

```bash
htc-heom-run \
  --Nmol 2 \
  --L 2 \
  --K-matsubara 0 \
  --dt-fs 0.5 \
  --tmax-fs 5 \
  --store obs \
  --output Data/smoke.npz
```

Then run a larger but still cheap validation case.

For changes to `rhs_htc_scaled.py`, add or run dense-reference RHS comparisons.  The dense reference code is already present in `rhs_htc_scaled.py`, but more pytest tests should be added before public release.

