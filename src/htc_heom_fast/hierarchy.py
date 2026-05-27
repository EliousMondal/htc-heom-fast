# ============================================================
# hierarchy.py
#
# ADO bookkeeping for the naive_HTC HEOM code.
#
# For M HEOM channels and hierarchy depth L, this file builds
# all multi-indices
#     |n⟩ = (n₀, n₁, ..., n_{M-1})
# satisfying
#     ∑ₐ nₐ ≤ L.
#
# It also builds the nearest-neighbor maps
#     up[I, α]   = ID of |n_I + e_α⟩, or -1 if absent
#     down[I, α] = ID of |n_I - e_α⟩, or -1 if absent
#
# The hierarchy is ordered tier by tier. Within each tier, the
# ordering is reverse lexicographic in the first coordinate:
#     M=3, tier=2:
#     (2,0,0), (1,1,0), (1,0,1),
#     (0,2,0), (0,1,1), (0,0,2)
#
# This is the same ordering used in the dimer HEOM code.

# From this code, the following arrays are generated:
#     ado_indices[I, α] = n_{Iα}
#     up[I, α]          = ID of |n_I + e_α
#     down[I, α]        = ID of |n_I - e_α⟩
#     tier[I]           = sum_α n_{Iα}
#     tier_offsets[l]   = first ID of tier l
#
# Important implementation choice:
# -------------------------------
# We do NOT build a huge Python dictionary from multi-index tuples
# to ADO IDs. For N=25, depth=8, such a dictionary would be very
# memory expensive. Instead, we use the known tier ordering and a
# combinatorial rank formula to construct neighbor maps directly.
# ============================================================

from math import comb

import numpy as np
from numba import njit


# ============================================================
# Counting formulas
# ============================================================

def number_of_ados(M, L):
    """
    Total number of ADOs up to depth L for M hierarchy directions.
        N_ado = binomial(M + L, L) = binomial(M + L, M)
    """
    if M <= 0:
        raise ValueError("M must be positive.")
    if L < 0:
        raise ValueError("L must be non-negative.")
    return comb(M + L, L)


def number_of_ados_at_tier(M, l):
    """
    Number of ADOs exactly at tier l.
        N_tier(l) = binomial(M + l - 1, l)
                  = binomial(M + l - 1, M - 1)
    """
    if M <= 0:
        raise ValueError("M must be positive.")
    if l < 0:
        raise ValueError("l must be non-negative.")
    return comb(M + l - 1, l)


def make_tier_offsets(M, L):
    """
    Return tier_offsets, where tier_offsets[l]
    is the first global ADO ID at tier l.

    The number of ADOs before tier l is
        ∑_{r=0}^{l-1} binomial(M+r-1, r)
                = binomial(M+l-1, l-1)
                = binomial(M+l-1, M)
    with tier_offsets[0] = 0.
    
    For example:
        For M=3 and L=2,
        tier 0: (0,0,0)  
            IDs: 0                       tier_offsets[0] = 0
        tier 1: (1,0,0), (0,1,0),           
            (0,0,1) 
            IDs: 1, 2, 3                 tier_offsets[1] = 1
        tier 2: (2,0,0), (1,1,0), (1,0,1),  
            (0,2,0), (0,1,1), (0,0,2) 
            IDs: 4, 5, 6, 7, 8, 9        tier_offsets[2] = 4
            
        Thus, make_tier_offsets(M=3, L=2) gives
            tier_offsets = [0, 1, 4, 10]
        The extra final entry, tier_offsets[L+1], gives the total number of ADOs,
            tier_offsets[L+1] = N_{ADO}.
    """
    if M <= 0:
        raise ValueError("M must be positive.")
    if L < 0:
        raise ValueError("L must be non-negative.")

    tier_offsets    = np.empty(L + 2, dtype=np.int64)
    tier_offsets[0] = 0

    offset = 0
    for l in range(L + 1):
        tier_offsets[l] = offset
        offset += number_of_ados_at_tier(M, l)

    tier_offsets[L + 1] = offset

    expected = number_of_ados(M, L)
    if offset != expected:
        raise RuntimeError(
            f"Tier offsets inconsistent: got total {offset}, expected {expected}."
        )

    return tier_offsets


def make_binomial_table(nmax):
    """
    Precompute binomial coefficients C(n,k) for 0 ≤ n,k ≤ nmax.
    The table is stored as int64 and is used inside Numba kernels.
    
    For eg: choose = make_binomial_table(5) gives
            [[ 1  0  0  0  0  0]
             [ 1  1  0  0  0  0]
             [ 1  2  1  0  0  0]
             [ 1  3  3  1  0  0]
             [ 1  4  6  4  1  0]
             [ 1  5 10 10  5  1]]
             Thus choose[5,2] = C(5,2) = 10.
             
    Note:
    We need this because the neighbor-map builder uses a combinatorial 
    rank formula to find ADO IDs without building a giant Python dictionary.
    The old way is to have:
        index_to_id[(n0, n1, ..., nM)] = I
    but for millions of ADOs this dictionary becomes memory-expensive. 
    Instead, this code calculates the ID directly from binomial coefficients.
    """
    if nmax < 0:
        raise ValueError("nmax must be non-negative.")

    choose = np.zeros((nmax + 1, nmax + 1), dtype=np.int64)

    for n in range(nmax + 1):
        choose[n, 0] = 1
        choose[n, n] = 1
        for k in range(1, n):
            value = comb(n, k)
            if value > np.iinfo(np.int64).max:
                raise OverflowError("Binomial coefficient does not fit in int64.")
            choose[n, k] = value

    return choose


def smallest_ado_index_dtype(L):
    """
    Choose a compact unsigned dtype for ADO occupation numbers.
    
    For eg: print(smallest_ado_index_dtype(10)) → <class 'numpy.uint8'>
            print(smallest_ado_index_dtype(300)) → <class 'numpy.uint16'>
    
    Note:
    For L ≤ 255, uint8 is sufficient then every occupation number satisfies,
                0 ≤ n_{Iα} ≤ L ≤ 255,
    so uint8 is enough. For our current depths L ≤ 15, uint8 is sufficient. 
    This saves memory for ado_indices. As an example, for N=25, L=8, K=0, 
    using unit8 rather than int64 for ado_indices saves a lot of metadata memory.
    """
    if L <= np.iinfo(np.uint8).max:
        return np.uint8
    if L <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.int64


def smallest_id_dtype(N_ado):
    """
    Choose a compact signed dtype for neighbor IDs.
    We need a signed dtype because missing neighbors are stored as -1.
    
    For eg: print(smallest_id_dtype(100000)) → <class 'numpy.int32'>
    
    Note:
    If the total number of ADOs is less than 2^31, int32 is sufficient 
    to store all IDs and the -1 missing neighbor marker. For our current 
    hierarchies with millions of ADOs, int32 is sufficient. This saves 
    memory for up/down/tier arrays. If the total number of ADOs exceeds 
    2^31, int64 is used.
    """
    if N_ado <= np.iinfo(np.int32).max:
        return np.int32
    return np.int64


# ============================================================
# Numba kernels: hierarchy generation
# ============================================================

@njit(cache=True)
def _fill_fixed_tier_iterative_numba(ado_indices, offset, M, l):
    """
    Fill one fixed tier using an explicit backtracking state machine.
    This avoids recursive Numba calls and reproduces the desired ordering:
        M=3, l=2:
        (2,0,0), (1,1,0), (1,0,1),
        (0,2,0), (0,1,1), (0,0,2)
    
    Note:
    The ordering is important. The code uses reverse lexicographic ordering in 
    the first coordinate: it starts with the largest possible n₀, then decreases 
    n₀, and so on. A conceptual version is:
    
        for n0 in range(l, -1, -1):
            for n1 in range(l - n0, -1, -1):
                n2 = l - n0 - n1
    
    For M=3, l=2, this gives:
        n0=2: (2,0,0)
        n0=1: (1,1,0), (1,0,1)
        n0=0: (0,2,0), (0,1,1), (0,0,2)
    """
    current      = np.zeros(M, dtype=np.int64)
    remaining_at = np.zeros(M, dtype=np.int64)
    next_value   = np.zeros(M, dtype=np.int64)

    row = offset
    pos = 0
    remaining_at[0] = l
    next_value[0]   = l

    while pos >= 0:
        if pos == M - 1:
            current[pos] = remaining_at[pos]
            for α in range(M):
                ado_indices[row, α] = current[α]
            row += 1
            pos -= 1

        elif next_value[pos] >= 0:
            value = next_value[pos]
            next_value[pos] -= 1

            current[pos] = value
            pos += 1
            remaining_at[pos] = remaining_at[pos - 1] - value
            next_value[pos] = remaining_at[pos]

        else:
            pos -= 1

    return row - offset


@njit(cache=True)
def _generate_ado_indices_numba(ado_indices, tier_offsets, M, L):
    """
    Fill ado_indices with all tiers from l=0 to l=L.
    """
    for l in range(L + 1):
        _fill_fixed_tier_iterative_numba(ado_indices, tier_offsets[l], M, l)


def generate_ado_indices(M, L, index_dtype=None):
    """
    Generate all ADO multi-indices up to hierarchy depth L.

    Parameters
    ----------
    M : int
        Number of HEOM hierarchy directions.
        For HTC with one exponential per molecule, M = Nmol.
        For HTC with K+1 exponentials per molecule, M = Nmol*(K+1).
    L : int
        Maximum hierarchy depth.
    index_dtype : dtype or None
        dtype for occupation numbers. If None, a compact dtype is chosen.
        For L <= 255, this is uint8.

    Returns
    -------
    ado_indices : ndarray, shape (N_ado, M)
        ado_indices[I, α] = n_{Iα}.
        
    For eg: 
        ado_indices = generate_ado_indices(M=3, L=2)
        for I, n in enumerate(ado_indices):
            print(I, tuple(n))    
    gives,
                0 (0, 0, 0)
                1 (1, 0, 0)
                2 (0, 1, 0)
                3 (0, 0, 1)
                4 (2, 0, 0)
                5 (1, 1, 0)
                6 (1, 0, 1)
                7 (0, 2, 0)
                8 (0, 1, 1)
                9 (0, 0, 2)
    """
    if M <= 0:
        raise ValueError("M must be positive.")
    if L < 0:
        raise ValueError("L must be non-negative.")

    if index_dtype is None:
        index_dtype = smallest_ado_index_dtype(L)

    N_ado = number_of_ados(M, L)
    tier_offsets = make_tier_offsets(M, L)

    ado_indices = np.empty((N_ado, M), dtype=index_dtype)
    _generate_ado_indices_numba(ado_indices, tier_offsets, M, L)

    return ado_indices


# ============================================================
# Numba kernels: combinatorial rank and neighbor maps
# ============================================================

@njit(cache=True)
def _rank_within_tier(n, M, l, choose):
    """
    Return the 0-based rank of multi-index n within its fixed tier l.

    The rank corresponds to the reverse-lexicographic tier ordering
    generated by _fill_fixed_tier_recursive_numba.
    
    For eg:
    For M=3, l=2, within the tier-2 the multi-indices are ordered as
        (2,0,0), (1,1,0), (1,0,1),
        (0,2,0), (0,1,1), (0,0,2)
    The ranks and global IDs are:
        multi-index    rank    global ID
           (2,0,0)       0         4       
           (1,1,0)       1         5
           (1,0,1)       2         6
           (0,2,0)       3         7
           (0,1,1)       4         8
           (0,0,2)       5         9
    
    Note:       
    The rank can be calculated from the multi-index using a combinatorial formula.
    The idea is to count how many multi-indices come before n in the ordering.
    The number of ways to  distribute K remaining quanta over r remaining dimensions is
        C(K + r - 1, r - 1)     
    The code used a summed version of this identity to calculate the rank using the
    binomial table precomputed by make_binomial_table. The code iterates over the M 
    dimensions, and for each dimension α, it counts how many multi-indices come before n 
    due to having a larger value at dimension α, while keeping the previous dimensions 
    fixed. The remaining quanta and dimensions are updated accordingly. The formula used is:
            ∑_{s=0}^{K-1} C(s + r - 1, r - 1) = C(K + r - 1, r)
    where K is the remaining quanta after fixing the previous dimensions, and r is the 
    number of remaining dimensions after α. The final rank is the sum of contributions from 
    all dimensions.
    """
    rank = 0
    remaining = l

    for α in range(M - 1):
        value = int(n[α])
        rest_dims = M - α - 1
        K = remaining - value

        if K > 0:
            # Sum_{s=0}^{K-1} C(s + rest_dims - 1, rest_dims - 1)
            # = C(K + rest_dims - 1, rest_dims)
            rank += choose[K + rest_dims - 1, rest_dims]

        remaining -= value

    return rank


@njit(cache=True)
def _build_tier_array_numba(tier, tier_offsets, L):
    """
    Fill tier[I] from tier_offsets.
    
    For example, for M=3, L=2, the tier offsets are
        tier_offsets = [0, 1, 4, 10]
    so the tier array is
        tier = [0, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    """
    for l in range(L + 1):
        start = tier_offsets[l]
        stop = tier_offsets[l + 1]
        for I in range(start, stop):
            tier[I] = l


@njit(cache=True)
def _build_neighbor_maps_rank_numba(ado_indices, up, down, tier, tier_offsets, choose, M, L):
    """
    Build up/down neighbor maps without a Python dictionary.

    The important optimization is that, for a given ADO, the ranks of
    all |n ± e_α⟩ neighbors can be obtained in O(M), not O(M^2), by
    using prefix corrections to the current rank.
    """
    N_ado = ado_indices.shape[0]

    for I in range(N_ado):
        l = int(tier[I])
        rank_current = I - tier_offsets[l]

        # First initialize missing-neighbor markers.
        for α in range(M):
            up[I, α] = -1
            down[I, α] = -1

        # prefix_up_delta[α] is the correction to rank_current for
        # |n + e_α⟩ at tier l+1.
        # prefix_down_delta[α] is the correction to rank_current for
        # |n - e_α⟩ at tier l-1.
        prefix_up_delta = 0
        prefix_down_delta = 0
        remaining = l

        for α in range(M):
            # Use the deltas accumulated from positions p < α.
            if l < L:
                up[I, α] = tier_offsets[l + 1] + rank_current + prefix_up_delta

            if int(ado_indices[I, α]) > 0:
                down[I, α] = tier_offsets[l - 1] + rank_current - prefix_down_delta

            # Update prefix deltas for later α values.
            # There is no rank contribution from the final coordinate.
            if α < M - 1:
                value = int(ado_indices[I, α])
                rest_dims = M - α - 1
                K = remaining - value

                # For raising a later coordinate, total tier increases by 1.
                # The contribution change at this coordinate is
                # C(K + rest_dims - 1, rest_dims - 1).
                prefix_up_delta += choose[K + rest_dims - 1, rest_dims - 1]

                # For lowering a later coordinate, total tier decreases by 1.
                # The contribution change is
                # C(K + rest_dims - 2, rest_dims - 1), provided K > 0.
                # If K == 0, no later coordinate can be lowered anyway.
                if K > 0:
                    prefix_down_delta += choose[K + rest_dims - 2, rest_dims - 1]

                remaining -= value


def build_tier_array(M, L, id_dtype=None):
    """
    Build tier[I] for all ADOs.
    """
    N_ado = number_of_ados(M, L)
    if id_dtype is None:
        id_dtype = smallest_id_dtype(N_ado)

    tier_offsets = make_tier_offsets(M, L)
    tier = np.empty(N_ado, dtype=id_dtype)
    _build_tier_array_numba(tier, tier_offsets, L)
    return tier


def build_neighbor_maps(ado_indices, L, id_dtype=None):
    """
    Build upward and downward neighbor maps.

    Parameters
    ----------
    ado_indices : ndarray, shape (N_ado, M)
        ADO multi-indices generated by generate_ado_indices.
    L : int
        Maximum hierarchy depth.
    id_dtype : dtype or None
        dtype for neighbor IDs. If None, int32 is used when possible.

    Returns
    -------
    up : ndarray, shape (N_ado, M)
        up[I, α] is ID of |n_I + e_α⟩, or -1.
    down : ndarray, shape (N_ado, M)
        down[I, α] is ID of |n_I - e_α⟩, or -1.
    tier : ndarray, shape (N_ado,)
        tier[I] = sum_α n_{Iα}.
    tier_offsets : ndarray, shape (L+2,)
        First ID of each tier.
    """
    ado_indices = np.asarray(ado_indices)

    if ado_indices.ndim != 2:
        raise ValueError("ado_indices must be a 2D array.")
    if L < 0:
        raise ValueError("L must be non-negative.")

    N_ado, M = ado_indices.shape
    expected = number_of_ados(M, L)
    if N_ado != expected:
        raise ValueError(
            f"ado_indices has {N_ado} rows, but M={M}, L={L} require {expected}."
        )

    if id_dtype is None:
        id_dtype = smallest_id_dtype(N_ado)

    tier_offsets = make_tier_offsets(M, L)
    choose = make_binomial_table(M + L + 2)

    up = np.empty((N_ado, M), dtype=id_dtype)
    down = np.empty((N_ado, M), dtype=id_dtype)
    tier = np.empty(N_ado, dtype=id_dtype)

    _build_tier_array_numba(tier, tier_offsets, L)
    _build_neighbor_maps_rank_numba(ado_indices, up, down, tier, tier_offsets, choose, M, L)

    return up, down, tier, tier_offsets


def build_hierarchy(M, L, index_dtype=None, id_dtype=None, validate=True):
    """
    Convenience wrapper for the complete hierarchy.

    Returns
    -------
    ado_indices, up, down, tier, tier_offsets
    """
    ado_indices = generate_ado_indices(M, L, index_dtype=index_dtype)
    up, down, tier, tier_offsets = build_neighbor_maps(ado_indices=ado_indices, L=L, id_dtype=id_dtype)

    if validate:
        validate_hierarchy(ado_indices=ado_indices, up=up, down=down, tier=tier, tier_offsets=tier_offsets, M=M, L=L)

    return ado_indices, up, down, tier, tier_offsets


# ============================================================
# Validation and summary utilities
# ============================================================

def count_by_tier(tier, L):
    """
    Count number of ADOs in each tier.
    """
    counts = np.zeros(L + 1, dtype=np.int64)
    for x in tier:
        l = int(x)
        if 0 <= l <= L:
            counts[l] += 1
    return counts


def validate_hierarchy(ado_indices, up, down, tier, tier_offsets, M, L):
    """
    Validate hierarchy arrays.

    This is a setup/debugging function. It is intentionally not optimized
    for very large hierarchies. For production runs with millions of ADOs,
    set validate=False after testing small cases.
    """
    N_ado_expected = number_of_ados(M, L)

    if ado_indices.shape != (N_ado_expected, M):
        raise RuntimeError(
            f"ado_indices has shape {ado_indices.shape}, expected "
            f"({N_ado_expected}, {M})."
        )

    if up.shape != (N_ado_expected, M):
        raise RuntimeError(f"up has shape {up.shape}, expected ({N_ado_expected}, {M}).")

    if down.shape != (N_ado_expected, M):
        raise RuntimeError(f"down has shape {down.shape}, expected ({N_ado_expected}, {M}).")

    if tier.shape != (N_ado_expected,):
        raise RuntimeError(f"tier has shape {tier.shape}, expected ({N_ado_expected},).")

    # Check tier_offsets.
    for l in range(L + 1):
        if tier_offsets[l + 1] - tier_offsets[l] != number_of_ados_at_tier(M, l):
            raise RuntimeError(f"tier_offsets inconsistent at tier {l}.")

    # Check ADO tiers and neighbor inverse consistency.
    for I in range(N_ado_expected):
        actual_tier = int(np.sum(ado_indices[I]))
        if int(tier[I]) != actual_tier:
            raise RuntimeError(
                f"Tier mismatch at ADO {I}: tier[I]={tier[I]}, sum={actual_tier}."
            )
        if actual_tier > L:
            raise RuntimeError(f"ADO {I} has tier {actual_tier}, exceeding L={L}.")

        for α in range(M):
            J_up = int(up[I, α])
            if J_up != -1:
                if int(down[J_up, α]) != I:
                    raise RuntimeError(
                        f"Neighbor mismatch: up[{I},{α}]={J_up}, "
                        f"but down[{J_up},{α}]={down[J_up, α]}."
                    )

            J_down = int(down[I, α])
            if J_down != -1:
                if int(up[J_down, α]) != I:
                    raise RuntimeError(
                        f"Neighbor mismatch: down[{I},{α}]={J_down}, "
                        f"but up[{J_down},{α}]={up[J_down, α]}."
                    )

    counts = count_by_tier(tier, L)
    for l in range(L + 1):
        expected = number_of_ados_at_tier(M, l)
        if int(counts[l]) != expected:
            raise RuntimeError(
                f"Tier count mismatch at tier {l}: got {counts[l]}, expected {expected}."
            )

    return True


def print_hierarchy_summary(ado_indices, up, down, tier, tier_offsets, L, max_rows=50):
    """
    Print a readable hierarchy summary.
    For large hierarchies only the first max_rows rows are printed.
    """
    N_ado, M = ado_indices.shape
    counts = count_by_tier(tier, L)

    print("===============================================")
    print("HEOM hierarchy summary")
    print("===============================================")
    print(f"M      = {M}")
    print(f"L      = {L}")
    print(f"N_ado  = {N_ado}")
    print(f"index dtype = {ado_indices.dtype}")
    print(f"id dtype    = {up.dtype}")
    print("-----------------------------------------------")
    print("Counts by tier:")
    for l in range(L + 1):
        print(f"  tier {l}: {counts[l]}")

    print("-----------------------------------------------")
    print("Tier offsets:")
    for l in range(L + 1):
        print(f"  tier {l}: start={tier_offsets[l]}, stop={tier_offsets[l + 1]}")

    print("-----------------------------------------------")
    print(f"First {min(max_rows, N_ado)} ADO rows:")
    print("I     tier    index                 up                  down")
    for I in range(min(max_rows, N_ado)):
        index_tuple = tuple(int(x) for x in ado_indices[I])
        up_tuple = tuple(int(x) for x in up[I])
        down_tuple = tuple(int(x) for x in down[I])
        print(
            f"{I:<5d} {int(tier[I]):<7d} "
            f"{index_tuple!s:<21s} "
            f"{up_tuple!s:<19s} "
            f"{down_tuple!s}"
        )

    if N_ado > max_rows:
        print(f"... {N_ado - max_rows} more rows not printed ...")

    print("===============================================")


# ============================================================
# Memory estimate utility
# ============================================================

def estimate_hierarchy_metadata_gb(M, L, index_dtype=None, id_dtype=None):
    """
    Estimate memory for hierarchy metadata only:
        ado_indices + up + down + tier
    This does not include the ADO density matrices.
    """
    N_ado = number_of_ados(M, L)

    if index_dtype is None:
        index_dtype = smallest_ado_index_dtype(L)
    if id_dtype is None:
        id_dtype = smallest_id_dtype(N_ado)

    bytes_indices = N_ado * M * np.dtype(index_dtype).itemsize
    bytes_up_down = 2 * N_ado * M * np.dtype(id_dtype).itemsize
    bytes_tier = N_ado * np.dtype(id_dtype).itemsize
    total = bytes_indices + bytes_up_down + bytes_tier

    return total / 1.0e9


def estimate_rk4_state_gb(Nmol, L, K_matsubara=0, n_work_arrays=6):
    """
    Estimate memory for RK4 ADO density arrays only.
    Assumes HTC first-excitation dimension
        d = Nmol + 1
    and
        M = Nmol * (K_matsubara + 1).
    The estimate is
        n_work_arrays * N_ado * d^2 * 16 bytes
    for complex128 arrays.
    """
    M = Nmol * (K_matsubara + 1)
    d = Nmol + 1
    N_ado = number_of_ados(M, L)
    total = n_work_arrays * N_ado * d * d * np.dtype(np.complex128).itemsize
    return total / 1.0e9


# ============================================================
# Script mode sanity check
# ============================================================

if __name__ == "__main__":
    M = 3
    L = 3

    ado_indices, up, down, tier, tier_offsets = build_hierarchy(M=M, L=L, validate=True)
    print_hierarchy_summary(ado_indices=ado_indices, up=up, down=down, tier=tier, tier_offsets=tier_offsets, L=L, max_rows=100)

    print("Metadata memory estimate for HTC N=25, L=8, K=0:")
    print(estimate_hierarchy_metadata_gb(M=25, L=8), "GB")

    print("RK4 state memory estimate for HTC N=25, L=8, K=0:")
    print(estimate_rk4_state_gb(Nmol=25, L=8, K_matsubara=0), "GB")


# ============================================================
# From bath_drude.py / htc_channels.py
# 
# M = N_mol * (K+1) 
#
# ν_site       # shape (1,)
# c_site       # shape (1,)
# Δ_LT         # scalar
# ν_α          # shape (M,)
# c_α          # shape (M,)
# abs_c_α      # shape (M,)
# sqrt_abs_c_α # shape (M,)
# inv_sqrt_abs_c_α # shape (M,)
# site_α       # shape (M,)
# k_α          # shape (M,)
# sys_α        # shape (M,)
# Δ_site       # shape (M,)

# ============================================================
# From hierarchy.py
# ado_indices  # shape (N_ADO, M)
# up           # shape (N_ADO, M)
# down         # shape (N_ADO, M)
# tier         # shape (N_ADO,)
# tier_offsets # shape (M+1,)

# # From combining bath + hierarchy
# Γ            # shape (N_ADO,)