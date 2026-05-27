# Theory notes for `htc-heom-fast`

These notes explain the equations implemented in the package from the point of view of someone who wants to read or modify the code. The purpose is not to give the most general HEOM derivation. The purpose is to connect the physics notation directly to the arrays used in this repository.

## 1. Physical model

The code targets a first-excitation Holstein-Tavis-Cummings or Tavis-Cummings type system. The system basis is

$$
\{|C\rangle, |1\rangle, |2\rangle, \ldots, |N\rangle\}.
$$

Here, $|C\rangle$ is the state with one cavity photon and all molecules in their electronic ground state. The state $|n\rangle$ is the state with molecule $n$ electronically excited and no cavity photon. The number of molecules is $N = \texttt{Nmol}$, so the system Hilbert-space dimension is

$$
d = N + 1.
$$

In operator notation, the implemented system Hamiltonian is

$$
\hat{H}_s
= E_C |C\rangle\langle C|
+ \sum_{n=1}^{N} E_n |n\rangle\langle n|
+ \sum_{n=1}^{N} g_n
\left(|C\rangle\langle n| + |n\rangle\langle C|\right).
$$

In the ordered basis

$$
\left(|C\rangle, |1\rangle, |2\rangle, \ldots, |N\rangle\right),
$$

the matrix representation is

$$
H_s =
\begin{pmatrix}
E_C & g_1 & g_2 & \cdots & g_N \\
g_1 & E_1 & 0 & \cdots & 0 \\
g_2 & 0 & E_2 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
g_N & 0 & 0 & \cdots & E_N
\end{pmatrix}.
$$

The current fast RHS assumes this star structure. The cavity row and cavity column couple to all molecular states, but the molecular states do not directly couple to each other through the Hamiltonian. This is the main reason the Hamiltonian action can be evaluated without dense matrix multiplication.

## 2. Symmetric Tavis-Cummings limit

The simplest default is the symmetric Tavis-Cummings limit:

$$
E_n = E_x, \qquad g_n = g, \qquad n = 1, \ldots, N.
$$

Define the normalized bright state

$$
|B\rangle = \frac{1}{\sqrt{N}} \sum_{n=1}^{N} |n\rangle.
$$

In the restricted basis

$$
\left(|C\rangle, |B\rangle\right),
$$

the Hamiltonian block is

$$
H_{CB} =
\begin{pmatrix}
E_C & g\sqrt{N} \\
g\sqrt{N} & E_x
\end{pmatrix}.
$$

When $E_C = E_x$, the two polariton eigenstates are approximately

$$
|UP\rangle = \frac{|C\rangle + |B\rangle}{\sqrt{2}},
\qquad
|LP\rangle = \frac{|C\rangle - |B\rangle}{\sqrt{2}},
$$

up to phase conventions. The collective Rabi splitting is

$$
\Omega_R = 2g\sqrt{N}.
$$

This is why the command-line driver accepts either `--Omega-R-mev` or `--g-mev`. If `--g-mev` is not provided, the code computes

$$
g = \frac{\Omega_R}{2\sqrt{N}}.
$$

## 3. Site-local bath coupling operators

Each molecule has its own local bath. The bath associated with site $n$ couples through the projector

$$
\hat{Q}_n = |n\rangle\langle n|.
$$

The system-bath coupling for site $n$ has the form

$$
\hat{H}_{sb,n} = \hat{Q}_n \hat{B}_n,
$$

where $\hat{B}_n$ is a bath displacement operator. For independent identical baths, all sites use the same spectral density, but the bath operators are independent.

In the full first-excitation basis, $\hat{Q}_n$ is a diagonal matrix with one nonzero entry:

$$
(Q_n)_{ij} = \delta_{i,n}\delta_{j,n}.
$$

Because $\hat{Q}_n$ is diagonal and has only one nonzero diagonal element, the code never needs to store dense projector matrices in production. Instead it stores the selected system index $p=n$ in the array `sys_alpha`.

## 4. Drude-Lorentz bath correlation function

For one site, the Drude-Lorentz spectral density is parameterized by a reorganization energy $\lambda$ and a bath decay rate $\gamma$. The code uses a Matsubara expansion of the corresponding bath correlation function:

$$
C(t) = \sum_{k=0}^{K} c_k e^{-\nu_k t} + C_{\mathrm{res}}(t).
$$

The zeroth term is the Drude pole:

$$
\nu_0 = \gamma.
$$

For $k \ge 1$, the Matsubara frequencies are

$$
\nu_k = \frac{2\pi k}{\beta},
$$

where

$$
\beta = \frac{1}{k_B T}.
$$

The code stores the one-site coefficients in arrays whose entries are conceptually

$$
\texttt{nu\_site}[k] = \nu_k, \qquad
\texttt{c\_site}[k] = c_k.
$$

For an $N$-site system, these one-site coefficients are expanded into channel coefficients. If the Matsubara index is $k$ and the site index is $s$, then

$$
\alpha = s(K+1) + k.
$$

Thus each channel $\alpha$ identifies both a molecular site and a bath exponential term.

## 5. Channel indexing

The total number of explicit bath exponential channels is

$$
M = N_{\mathrm{mol}}(K_{\mathrm{matsubara}} + 1).
$$

The code uses the following channel arrays:

```text
nu_alpha[alpha]          decay rate for channel alpha
c_alpha[alpha]           complex coefficient for channel alpha
abs_c_alpha[alpha]       absolute value of c_alpha
sqrt_abs_c_alpha[alpha]  square root of abs_c_alpha
site_alpha[alpha]        site index in 0-based Python convention
k_alpha[alpha]           Drude/Matsubara index k
sys_alpha[alpha]         system basis index selected by Q_alpha
```

Be careful with indexing. In Python convention,

$$
\texttt{site\_alpha} \in \{0,1,\ldots,N_{\mathrm{mol}}-1\},
$$

while the corresponding system Hilbert-space index is

$$
\texttt{sys\_alpha} \in \{1,2,\ldots,N_{\mathrm{mol}}\}.
$$

The offset exists because basis index $0$ is the cavity state $|C\rangle$, while molecular site $0$ corresponds to basis state $|1\rangle$.

## 6. ADO multi-indices

Each ADO is labeled by a non-negative integer vector

$$
\mathbf{n} = (n_0,n_1,\ldots,n_{M-1}).
$$

The entry $n_\alpha$ says how many times channel $\alpha$ appears in that ADO. The hierarchy tier is

$$
\ell = \sum_{\alpha=0}^{M-1} n_\alpha.
$$

The zeroth ADO is

$$
\mathbf{n} = (0,0,\ldots,0),
$$

and stores the physical reduced density matrix:

$$
\hat{\rho}_0(t) = \texttt{rho}[0].
$$

The total number of ADOs through tier $L$ is

$$
N_{\mathrm{ADO}} = \binom{M+L}{L}.
$$

The number of ADOs at exactly tier $\ell$ is

$$
N_{\mathrm{ADO}}(\ell) = \binom{M+\ell-1}{\ell}.
$$

These formulas are tested in `tests/test_core_modules.py`.

## 7. Hierarchy arrays in the code

The hierarchy is represented by several arrays.

### `ado_indices`

The array `ado_indices` stores the multi-index of each ADO:

$$
\texttt{ado\_indices}[I,\alpha] = n_\alpha
$$

for ADO index $I$.

For example, with $M=3$ and $L=2$, the first few ADO labels are

```text
I    n
0    (0,0,0)
1    (1,0,0)
2    (0,1,0)
3    (0,0,1)
4    (2,0,0)
5    (1,1,0)
6    (1,0,1)
7    (0,2,0)
8    (0,1,1)
9    (0,0,2)
```

This ordering is not physically important, but consistency is essential because the neighbor maps depend on it.

### `up`

The upward neighbor map is defined by

$$
\texttt{up}[I,\alpha] = J
$$

when ADO $J$ has multi-index

$$
\mathbf{n}_J = \mathbf{n}_I + \mathbf{e}_\alpha.
$$

If $\mathbf{n}_I + \mathbf{e}_\alpha$ lies outside the depth truncation, then

$$
\texttt{up}[I,\alpha] = -1.
$$

### `down`

The downward neighbor map is defined by

$$
\texttt{down}[I,\alpha] = J
$$

when ADO $J$ has multi-index

$$
\mathbf{n}_J = \mathbf{n}_I - \mathbf{e}_\alpha.
$$

If $n_\alpha = 0$, then the downward neighbor does not exist and

$$
\texttt{down}[I,\alpha] = -1.
$$

### `Gamma`

The hierarchy damping for ADO $I$ is

$$
\Gamma_I = \sum_{\alpha=0}^{M-1} n_{I\alpha}\nu_\alpha.
$$

This term appears in the HEOM RHS as

$$
-\Gamma_I \tilde{\rho}_I.
$$

## 8. Unscaled versus scaled HEOM

A schematic unscaled HEOM equation has the structure

$$
\begin{aligned}
\frac{d}{dt}\hat{\rho}_{\mathbf{n}}
=& -i[\hat{H}_s,\hat{\rho}_{\mathbf{n}}]
- \sum_\alpha n_\alpha \nu_\alpha \hat{\rho}_{\mathbf{n}} \\
& - i\sum_\alpha [\hat{Q}_\alpha,\hat{\rho}_{\mathbf{n}+\mathbf{e}_\alpha}] \\
& - i\sum_\alpha n_\alpha
\left(c_\alpha \hat{Q}_\alpha \hat{\rho}_{\mathbf{n}-\mathbf{e}_\alpha}
- c_\alpha^{*}\hat{\rho}_{\mathbf{n}-\mathbf{e}_\alpha}\hat{Q}_\alpha\right).
\end{aligned}
$$

The scaled hierarchy absorbs products of $\sqrt{n_\alpha!|c_\alpha|^{n_\alpha}}$ into the definition of the ADO:

$$
\tilde{\rho}_{\mathbf{n}}
= \left(\prod_\alpha n_\alpha! |c_\alpha|^{n_\alpha}\right)^{-1/2}
\hat{\rho}_{\mathbf{n}}.
$$

In the scaled convention implemented here, the upward coupling prefactor is

$$
\sqrt{(n_\alpha+1)|c_\alpha|},
$$

and the downward coupling prefactor is

$$
\sqrt{\frac{n_\alpha}{|c_\alpha|}}.
$$

The implemented RHS is therefore

$$
\begin{aligned}
\frac{d}{dt}\tilde{\rho}_{\mathbf{n}}
=& -i[\hat{H}_s,\tilde{\rho}_{\mathbf{n}}]
- \Gamma_{\mathbf{n}}\tilde{\rho}_{\mathbf{n}} \\
& - i\sum_\alpha \sqrt{(n_\alpha+1)|c_\alpha|}
[\hat{Q}_\alpha,\tilde{\rho}_{\mathbf{n}+\mathbf{e}_\alpha}] \\
& - i\sum_\alpha \sqrt{\frac{n_\alpha}{|c_\alpha|}}
\left(c_\alpha\hat{Q}_\alpha\tilde{\rho}_{\mathbf{n}-\mathbf{e}_\alpha}
- c_\alpha^{*}\tilde{\rho}_{\mathbf{n}-\mathbf{e}_\alpha}\hat{Q}_\alpha\right) \\
& - \hbox{terminator}.
\end{aligned}
$$

The scaling is numerically useful because high-tier ADOs can be extremely small in the unscaled convention. Scaling prevents the hierarchy from spanning unnecessarily extreme magnitudes.

## 9. Matrix-free Hamiltonian action

A dense implementation of the Hamiltonian part would compute

$$
-i(\hat{H}_s\hat{\rho} - \hat{\rho}\hat{H}_s)
$$

for every ADO. Dense matrix multiplication would cost approximately $O(d^3)$ per ADO.

The HTC Hamiltonian has the star form

$$
(H_s)_{00}=E_C, \qquad (H_s)_{nn}=E_n, \qquad (H_s)_{0n}=(H_s)_{n0}=g_n,
$$

with all other off-diagonal elements equal to zero. Using this structure, the code evaluates the commutator element by element using sums over the cavity row and cavity column. This reduces the practical cost and avoids constructing a dense Liouvillian.

In `rhs_htc_scaled.py`, this is implemented by

```text
_add_htc_star_liouvillian_one_ado(...)
```

which fills the RHS contribution for one ADO.

## 10. Matrix-free projector bath coupling

For each bath channel,

$$
\hat{Q}_\alpha = |p\rangle\langle p|,
$$

where $p = \texttt{sys\_alpha}[\alpha]$.

For any matrix $\rho$, the commutator has elements

$$
\left[\hat{Q}_p,\rho\right]_{ij}
= (\delta_{i,p}-\delta_{j,p})\rho_{ij}.
$$

Therefore only row $p$ and column $p$ are affected. The code exploits this directly. This is why bath coupling through projectors is much cheaper than dense matrix products.

The upward coupling uses

```text
_add_scaled_upward_projector_one_channel(...)
```

and the downward coupling uses

```text
_add_scaled_downward_projector_one_channel(...)
```

Both update only the relevant row and column.

## 11. Low-temperature Matsubara terminator in this code

When only finitely many Matsubara terms are included, the unresolved residual can be approximately treated as a Markovian double-commutator correction:

$$
-\sum_{s=1}^{N} \Delta_s [\hat{Q}_s,[\hat{Q}_s,\rho]].
$$

For a projector $\hat{Q}_p = |p\rangle\langle p|$,

$$
[\hat{Q}_p,[\hat{Q}_p,\rho]]_{ij}
= (\delta_{i,p}-\delta_{j,p})^2\rho_{ij}.
$$

So the terminator damps coherences involving site $p$, while leaving the population $\rho_{pp}$ unchanged.

In code, the terminator is represented by

```text
Delta_site[site]
```

Passing an all-zero `Delta_site` array disables it. The command-line option `--no-terminator` turns it off.

## 12. Observable definitions

The physical reduced density matrix is always

$$
\rho_0(t) = \texttt{rho}[0].
$$

The cavity population is

$$
P_{\mathrm{cavity}}(t) = \langle C|\rho_0(t)|C\rangle.
$$

The total exciton population is

$$
P_{\mathrm{exciton}}(t) = \sum_{n=1}^{N} \langle n|\rho_0(t)|n\rangle.
$$

The bright-state population is

$$
P_{\mathrm{bright}}(t) = \langle B|\rho_0(t)|B\rangle,
$$

with

$$
|B\rangle = \frac{1}{\sqrt{N}}\sum_{n=1}^{N}|n\rangle.
$$

The total dark population is computed as

$$
P_{\mathrm{dark}}(t) = P_{\mathrm{exciton}}(t) - P_{\mathrm{bright}}(t).
$$

The polariton populations are

$$
P_{LP}(t) = \langle LP|\rho_0(t)|LP\rangle,
\qquad
P_{UP}(t) = \langle UP|\rho_0(t)|UP\rangle.
$$

The $|LP\rangle$ and $|UP\rangle$ vectors are obtained by diagonalizing the single-excitation HTC Hamiltonian and identifying the two bright or cavity-like eigenstates.

## 13. What is currently exact, and what is approximate?

Within the implemented model assumptions, the HEOM equations are formally exact only in the limit

$$
L \rightarrow \infty, \qquad K_{\mathrm{matsubara}} \rightarrow \infty, \qquad \Delta t \rightarrow 0.
$$

In practice, calculations use finite values of $L$, $K_{\mathrm{matsubara}}$, and $\Delta t$. The user must check convergence with respect to these parameters.

The package makes additional model and implementation restrictions:

- first-excitation manifold only,
- star-shaped HTC Hamiltonian,
- independent site-local baths,
- identical Drude-Lorentz bath parameters for each site in the current driver,
- explicit RK4 time stepping,
- finite hierarchy depth truncation.

These are not flaws. They are the assumptions that make the code simple and fast.

## 14. Where to modify the theory in code

Use this map when changing the physics:

```text
Change bath coefficients:
    bath_drude.py

Change how channels map to sites:
    htc_channels.py

Change hierarchy generation or truncation:
    hierarchy.py

Change Hamiltonian structure:
    htc_system_builder.py
    rhs_htc_scaled.py

Change the HEOM equation itself:
    rhs_htc_scaled.py

Change time integration:
    integrators_htc.py

Change observables:
    observables_htc.py

Change command-line parameters:
    run_htc.py
```

The most delicate file is `rhs_htc_scaled.py`, because it encodes the mathematical equation being propagated.
