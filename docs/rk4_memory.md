# RK4 memory and possible lower-storage integrators

This note explains the current RK4 memory usage and what can and cannot be reduced.



## 1. Current classical RK4 implementation

The current low-memory classical RK4 step is implemented in `integrators_htc.py` as

```python
rk4_step_htc_scaled_lowmem_inplace(...)
```

It uses the standard RK4 formula for an ODE

```text
d rho / dt = f(rho).
```

The four stages are

```text
k1 = f(rho)
k2 = f(rho + dt k1 / 2)
k3 = f(rho + dt k2 / 2)
k4 = f(rho + dt k3)

rho_new = rho + dt (k1 + 2 k2 + 2 k3 + k4) / 6.
```

The old standard implementation stored

```text
rho, k1, k2, k3, k4, rho_tmp.
```

That is one state plus five work arrays.

The current low-memory implementation stores

```text
rho, k, acc, rho_tmp.
```

where:

```text
k       is reused as k1, k2, k3, and k4,
acc     stores k1 + 2 k2 + 2 k3,
rho_tmp stores intermediate stage states.
```

So the current implementation uses one state plus three work arrays.



## 2. Why exact classical RK4 is hard to reduce further

For exact classical RK4 with an RHS interface

```python
rhs(rho_input, rhs_output)
```

we need three different kinds of information during the step.

### 1. The original state

All intermediate stages are built from the original `rho`:

```text
rho + dt k1 / 2,
rho + dt k2 / 2,
rho + dt k3.
```

So the original state must remain available.

### 2. The current derivative

At each stage, we need the current derivative:

```text
k1, then k2, then k3, then k4.
```

The current code reuses one array `k` for this.

### 3. The accumulated final increment

By the time `k4` is computed, `k1`, `k2`, and `k3` have been overwritten.  Their weighted sum must have been stored somewhere:

```text
acc = k1 + 2 k2 + 2 k3.
```

The current code stores this in `acc`.

### 4. The intermediate stage state

The RHS must be evaluated at intermediate states such as

```text
rho + dt k2 / 2.
```

Because `rhs(rho_input, rhs_output)` reads the full `rho_input`, the stage state must exist as an array.  The current code stores it in `rho_tmp`.

Therefore, with the present RHS interface, the natural minimum for exact classical RK4 is:

```text
rho      original/current state
k        current derivative
acc      accumulated weighted derivative
rho_tmp  stage state
```

That is exactly what the current low-memory RK4 uses.



## 3. Can we reduce exact RK4 by recomputing stages?

In principle, one can trade memory for additional RHS evaluations.  For example, one could recompute earlier stages rather than store an accumulator.

For this HEOM problem, this is usually not attractive because each RHS evaluation is expensive.  A single RHS pass loops over all ADOs and all hierarchy channels.  Recomputing stages would reduce memory but could make the wall time much worse.

It also does not completely remove the need for temporary storage, because one still needs both an original state and a stage state during RHS evaluation.

So, for this code, recomputation is not the recommended path.



## 4. Can we overwrite `rho` during classical RK4?

One might try to save memory by temporarily changing `rho` into the intermediate stage state.

That is dangerous for classical RK4 because the final answer needs the original `rho`, and future stages also need to be built from the original `rho`, not from the previous stage state.

If we overwrite `rho`, we must store the original state somewhere else.  That backup array costs essentially the same memory as `rho_tmp`.  Therefore this does not give a real memory reduction.



## 5. The best route: low-storage fourth-order RK

The better approach is to change the integrator, not to rearrange classical RK4.

There are fourth-order low-storage Runge-Kutta schemes that use more stages but fewer arrays.  A common form is a 5-stage 4th-order low-storage method:

```text
r <- a_s r + dt f(rho)
rho <- rho + b_s r
```

for stages `s = 1, ..., 5`.

This uses:

```text
rho    current state
r      residual / derivative accumulator
```

So the hierarchy-sized memory can be reduced from four arrays total to two arrays total.

The tradeoff is:

```text
classical RK4:        4 RHS evaluations per step, 4 arrays total
low-storage RK(5,4):  5 RHS evaluations per step, 2 arrays total
```

For memory-limited calculations, this can be worth it.



## 6. What code change is needed for low-storage RK?

The current RHS function has the form

```python
rhs_htc_scaled_inplace(rho, drho, ...)
```

and overwrites `drho` with `f(rho)`.

For a true two-array low-storage RK method, the RHS should be fused with the residual update:

```python
residual = a_s * residual + dt * f(rho)
```

This means adding a new kernel such as

```python
rhs_htc_scaled_accumulate_inplace(
    rho,
    residual,
    a_s,
    dt,
    ...
)
```

which performs the following operation without allocating another hierarchy-sized array:

```text
residual[I,i,j] <- a_s * residual[I,i,j] + dt * f(rho)[I,i,j].
```

Then the stage update is simply

```text
rho[I,i,j] <- rho[I,i,j] + b_s * residual[I,i,j].
```



## 7. Why this requires modifying `rhs_htc_scaled.py`

The current RHS internally overwrites the output array when adding the Hamiltonian part:

```text
drho = -i[H, rho] - Gamma rho
```

Then it adds terminator and hierarchy-coupling terms.

For low-storage RK, we do not want:

```text
residual = f(rho)
```

We want:

```text
residual = a_s residual + dt f(rho).
```

If we call the current RHS directly, the previous residual is destroyed.  Therefore the RHS must be rewritten or wrapped so that the previous residual is scaled first and the RHS contributions are added with a factor `dt`.

This is straightforward but must be done carefully.



## 8. Memory comparison

Let

```text
S = N_ado * d * d * 16 bytes
```

be the memory for one complex128 HEOM state.

Then:

```text
standard RK4 old:      6S total  (rho + k1 + k2 + k3 + k4 + tmp)
current lowmem RK4:    4S total  (rho + k + acc + tmp)
low-storage RK(5,4):   2S total  (rho + residual)
```

The current code already reduced the state-array memory by about

```text
6S -> 4S,
```

which is a 33% reduction relative to the old six-array implementation.

A low-storage RK method could reduce

```text
4S -> 2S,
```

which is another 50% reduction relative to the current implementation, or 67% relative to the old six-array implementation.



## 9. Should low-storage RK replace classical RK4?

Not immediately.

Recommended approach:

1. Keep classical RK4 as the reference integrator.
2. Add low-storage RK as an optional integrator.
3. Compare both for small systems.
4. Check convergence with respect to time step.
5. Use low-storage RK for memory-limited large production runs only after validation.

The reason is that a different RK scheme has different stability and error constants.  It can still be fourth order, but it is not bitwise equivalent to classical RK4.



## 10. Practical recommendation for this package

For the next version, add an option such as:

```bash
--integrator rk4
--integrator lsrk54
```

where:

```text
rk4      = current classical low-memory RK4
lsrk54   = 5-stage 4th-order low-storage RK
```

Then report memory estimates separately:

```text
rk4 state arrays:      4
lsrk54 state arrays:   2
```

This would make it possible to run larger `Nmol` and `L` values under the same RAM limit, at the cost of roughly 25% more RHS evaluations per step.

