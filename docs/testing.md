# Testing guide

This document explains the current test files in plain language.  The tests are intentionally simple because this is still a research code.  They are meant to catch broken indexing, broken units, broken hierarchy construction, and broken command-line loading.

The current tests are not yet a complete physics validation suite.



## 1. How to run the tests

From the repository root:

```bash
pytest
```

You can also run the main core test file directly:

```bash
python tests/test_core_modules.py
```

Using `pytest` is preferred because it discovers all test files automatically.



## 2. `tests/test_core_modules.py`

This file tests the helper modules that everything else depends on.

It currently covers:

```text
constants.py
bath_drude.py
hierarchy.py
```

The file uses ordinary Python `assert` statements and small helper functions:

```python
def assert_close(x, y, rtol=1e-12, atol=1e-14):
    ...

def assert_array_equal(x, y):
    ...
```

These helpers make failures easier to read.



## 3. Test: `test_constants_basic_conversions`

Purpose: make sure the unit conversion constants are positive and internally consistent.

It checks things like:

```text
ps2au = 1000 * fs2au
meV2au = 1e-3 * eV2au
```

This matters because most command-line inputs are in human units such as femtoseconds, cm^-1, meV, and Kelvin, but the propagation is done in atomic units.

If this test fails, the code may still run, but all physical time and energy scales may be wrong.



## 4. Test: `test_drude_coefficients_for_multiple_K`

Purpose: check the Drude-Lorentz Matsubara coefficient generator.

The function under test is:

```python
drude_matsubara_coefficients(lambda, gamma, beta, K)
```

For each `K`, the test checks:

```text
nu.shape == (K + 1,)
c.shape  == (K + 1,)
nu[0]    == gamma
Im(c[0]) == -lambda * gamma
nu[k]    == 2 pi k / beta for k >= 1
```

It also checks that the low-temperature / Matsubara residual terminator coefficient is finite.

This test is not a full validation of bath physics.  It is a check that the formulas are being evaluated as intended and that array shapes are correct.



## 5. Test: `test_site_bath_channels_for_many_Nmol`

Purpose: check that site-local bath channels are expanded correctly.

For `Nmol` molecules and `K + 1` bath exponentials per molecule, the total number of channels should be

```text
M = Nmol * (K + 1).
```

The test checks arrays such as:

```text
nu_alpha
c_alpha
abs_c_alpha
site_alpha
k_alpha
sys_alpha
```

The important mapping is:

```text
alpha = site * (K + 1) + k
site_alpha[alpha] = site
k_alpha[alpha]    = k
sys_alpha[alpha]  = site + 1
```

The `+1` in `sys_alpha = site + 1` is important because system basis index 0 is the cavity state.

The test also constructs dense `qdiag` projectors for debugging and checks that each bath channel selects exactly one molecular state and never the cavity state.



## 6. Test: `test_hierarchy_counting_formulas`

Purpose: check the combinatorial ADO counting formulas.

For `M` hierarchy channels and maximum depth `L`, the total number of ADOs is

```text
N_ado = binomial(M + L, L).
```

The number at exactly tier `ell` is

```text
N_tier(ell) = binomial(M + ell - 1, ell).
```

The test checks these formulas for many small and medium values of `M` and `L`.

If this test fails, memory estimates and hierarchy allocation are unreliable.



## 7. Test: `test_hierarchy_known_small_ordering`

Purpose: check one explicit hierarchy ordering by hand.

For `M = 3` and `L = 2`, the expected ADO labels are:

```text
(0,0,0)
(1,0,0)
(0,1,0)
(0,0,1)
(2,0,0)
(1,1,0)
(1,0,1)
(0,2,0)
(0,1,1)
(0,0,2)
```

The exact ordering is not physically meaningful, but once chosen, it must be stable because neighbor maps `up` and `down` refer to row indices in this ordering.



## 8. Test: `test_hierarchy_build_for_different_Nmol`

Purpose: check hierarchy construction for several representative HTC sizes.

The test builds hierarchies for cases such as:

```text
Nmol=1,  K=0, L=6
Nmol=5,  K=0, L=5
Nmol=30, K=0, L=2
Nmol=10, K=1, L=3
```

For each case it checks:

1. `ado_indices` has the expected shape.
2. `up` and `down` have the expected shape.
3. `tier[I]` equals the sum of `ado_indices[I, :]`.
4. Tier counts match the combinatorial formulas.
5. A sample of `up` and `down` neighbors point to the correct multi-index.

This is one of the most important tests because HEOM dynamics depend heavily on correct ADO neighbor indexing.



## 9. Test: `test_memory_estimates`

Purpose: check that memory estimates are just direct byte counts.

For a given `Nmol`, `K`, and `L`, the test computes:

```text
M = Nmol * (K + 1)
d = Nmol + 1
N_ado = binomial(M + L, L)
```

Then it checks that the RK4 memory estimate agrees with

```text
n_work_arrays * N_ado * d * d * sizeof(complex128).
```

This test does not say the memory estimate is complete.  It only checks that the state-array part is computed correctly.



## 10. `tests/test_cli_smoke.py`

This is a very small smoke test.

It imports the command-line parser:

```python
from htc_heom_fast.run_htc import parse_args
```

Then it pretends the user typed:

```bash
htc-heom-run
```

with no extra arguments, and checks that default values are sane:

```text
Nmol == 5
K_matsubara == 0
store == "obs"
```

This catches simple mistakes such as:

- `run_htc.py` no longer imports,
- the parser crashes,
- default option names changed accidentally,
- console-script setup is broken indirectly.



## 11. What tests should be added next

The current tests mostly check infrastructure.  Before a serious public release, add physics and numerical tests.

### Dense RHS comparison

For tiny systems, compare the optimized matrix-free RHS to the dense reference RHS already present in `rhs_htc_scaled.py`.

A good test would be:

```text
Nmol = 2 or 3
K = 0 or 1
L = 2
random complex HEOM state
```

Then compute:

```python
drho_fast = rhs_htc_scaled(...)
drho_dense = rhs_dense_scaled_reference(...)
```

and check:

```text
max(abs(drho_fast - drho_dense)) < tolerance.
```

This is the most important missing test for code correctness.

### Hermiticity of `rho[0]`

For a physical initial condition,

```text
rho0 = rho0^dagger,
```

and exact HEOM dynamics should preserve Hermiticity of the physical reduced density matrix.  Numerically, check:

```text
||rho0 - rho0^dagger||
```

stays small.

### Trace preservation

For closed system dynamics plus dephasing-type bath coupling, the physical reduced density matrix should preserve trace:

```text
Tr rho0(t) = 1.
```

A useful regression test should check that `abs(trace - 1)` stays small for a short run.

### Bathless TC analytic dynamics

Set bath coupling to zero and propagate the symmetric Tavis-Cummings system.  The dynamics should match analytic Rabi oscillations in the `|C>`, `|B>` subspace.

This is a clean test of:

- Hamiltonian construction,
- RK4 propagation,
- LP/UP/bright/cavity observables.

### Output-file tests

Run a tiny simulation, save an `.npz`, reload it, and check that required fields exist.

### CLI estimate-only test

Run:

```bash
htc-heom-run --estimate-only
```

inside a test and make sure it exits successfully.



## 12. How to interpret failures

### Failure in constants tests

Likely cause: unit conversion constants changed or helper functions changed.

### Failure in bath tests

Likely cause: Drude coefficient formula changed, array dtype changed, or channel mapping changed.

### Failure in hierarchy tests

Likely cause: ADO ordering, tier generation, or neighbor-map generation changed.

### Failure in memory tests

Likely cause: changed memory-estimate convention.  Update the test only if the new convention is intentional.

### Failure in CLI smoke test

Likely cause: parser default changed or `run_htc.py` imports something that now fails.

