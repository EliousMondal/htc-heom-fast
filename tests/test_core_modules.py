# ===============================================
# test_core_modules.py
#
# Standalone tests for:
#     constants.py
#     bath_drude.py
#     hierarchy.py
#
# Usage:
#     python test_core_modules.py
#
# The tests are written as plain Python assertions so that you do not
# need pytest. If something is wrong, Python will stop with an
# AssertionError or with the underlying exception.
# ===============================================

from math import comb
import numpy as np

from htc_heom_fast import constants as const
from htc_heom_fast.bath_drude import (
    drude_matsubara_coefficients,
    drude_terminator_delta,
    make_site_bath_channels,
    make_htc_qdiag_dense,
    build_drude_htc_bath,
)
from htc_heom_fast.hierarchy import (
    number_of_ados,
    number_of_ados_at_tier,
    make_tier_offsets,
    generate_ado_indices,
    build_hierarchy,
    count_by_tier,
    estimate_hierarchy_metadata_gb,
    estimate_rk4_state_gb,
)


# ===============================================
# Small assertion helpers
# ===============================================

def assert_close(x, y, rtol=1.0e-12, atol=1.0e-14, msg=""):
    if not np.allclose(x, y, rtol=rtol, atol=atol):
        raise AssertionError(f"Not close: {x!r} vs {y!r}. {msg}")


def assert_array_equal(x, y, msg=""):
    if not np.array_equal(x, y):
        raise AssertionError(f"Arrays not equal.\n{x}\n!=\n{y}\n{msg}")


# ===============================================
# constants.py tests
# ===============================================

def test_constants_basic_conversions():
    print("Testing constants.py conversions...")

    assert const.fs2au > 0.0
    assert const.ps2au > 0.0
    assert const.cminv2au > 0.0
    assert const.eV2au > 0.0
    assert const.meV2au > 0.0
    assert const.K2au > 0.0

    assert_close(const.ps2au, 1000.0 * const.fs2au)
    assert_close(const.meV2au, 1.0e-3 * const.eV2au)

    # Optional helpers exist in the longer constants.py version.  These
    # checks are skipped automatically if you use the very simple dimer-style
    # constants.py with only numbers.
    if hasattr(const, "energy_to_au"):
        assert_close(const.energy_to_au(1.0, "cm-1"), const.cminv2au)
        assert_close(const.energy_to_au(1.0, "meV"), const.meV2au)
        assert_close(const.energy_to_au(300.0, "K"), 300.0 * const.K2au)

    if hasattr(const, "time_to_au"):
        assert_close(const.time_to_au(1.0, "fs"), const.fs2au)
        assert_close(const.time_to_au(1.0, "ps"), const.ps2au)

    if hasattr(const, "beta_from_temperature_au"):
        β_300 = const.beta_from_temperature_au(300.0)
        assert_close(β_300, 1.0 / (300.0 * const.K2au))

    print("  constants.py OK")


# ===============================================
# bath_drude.py tests
# ===============================================

def test_drude_coefficients_for_multiple_K():
    print("Testing Drude-Lorentz Matsubara coefficients...")

    λ = 50.0 * const.cminv2au
    γ = 18.0 * const.cminv2au
    T = 300.0 * const.K2au
    β = 1.0 / T

    for K in (0, 1, 2, 5):
        ν, c = drude_matsubara_coefficients(λ, γ, β, K)

        assert ν.shape == (K + 1,)
        assert c.shape == (K + 1,)
        assert ν.dtype == np.float64
        assert c.dtype == np.complex128

        assert_close(ν[0], γ)
        assert_close(c[0].imag, -λ * γ)
        assert np.isfinite(c[0].real)

        for k in range(1, K + 1):
            ν_expected = 2.0 * np.pi * k / β
            c_expected = (4.0 * λ * γ / β) * ν_expected / (ν_expected * ν_expected - γ * γ)
            assert_close(ν[k], ν_expected)
            assert_close(c[k].real, c_expected)
            assert_close(c[k].imag, 0.0)
            assert ν[k] > 0.0

        Δ_LT = drude_terminator_delta(λ, γ, β, ν, c)
        assert np.isfinite(Δ_LT)
        # In normal Drude-Matsubara use this residual is non-negative,
        # modulo tiny floating-point noise.
        assert Δ_LT > -1.0e-14

    print("  Drude coefficients OK")



def test_site_bath_channels_for_many_Nmol():
    print("Testing site-local bath channel expansion...")

    λ = 50.0 * const.cminv2au
    γ = 18.0 * const.cminv2au
    β = 1.0 / (300.0 * const.K2au)

    for Nmol in (1, 2, 5, 10, 15, 25, 30):
        for K in (0, 1, 2):
            ν_site, c_site = drude_matsubara_coefficients(λ, γ, β, K)
            ν_α, c_α, abs_c_α, site_α, k_α, sys_α = make_site_bath_channels(
                Nmol, ν_site, c_site
            )

            Kp1 = K + 1
            M = Nmol * Kp1
            d = Nmol + 1

            assert ν_α.shape == (M,)
            assert c_α.shape == (M,)
            assert abs_c_α.shape == (M,)
            assert site_α.shape == (M,)
            assert k_α.shape == (M,)
            assert sys_α.shape == (M,)

            # Check channel mapping α -> (site, k).
            for α in range(M):
                site_expected = α // Kp1
                k_expected = α % Kp1
                assert site_α[α] == site_expected
                assert k_α[α] == k_expected
                assert sys_α[α] == site_expected + 1
                assert_close(ν_α[α], ν_site[k_expected])
                assert_close(c_α[α], c_site[k_expected])
                assert_close(abs_c_α[α], abs(c_site[k_expected]))

            # Dense qdiag is only for tests/debugging; every row should be
            # a one-hot projector onto molecular index sys_α[α], never cavity 0.
            qdiag = make_htc_qdiag_dense(Nmol, sys_α)
            assert qdiag.shape == (M, d)
            assert_array_equal(qdiag[:, 0], np.zeros(M))
            assert_array_equal(np.sum(qdiag, axis=1), np.ones(M))
            for α in range(M):
                assert qdiag[α, sys_α[α]] == 1.0

            # Check the convenience wrapper gives the same arrays.
            out = build_drude_htc_bath(Nmol, λ, γ, β, K)
            ν_site2, c_site2, Δ_LT, ν_α2, c_α2, abs_c_α2, site_α2, k_α2, sys_α2 = out
            assert_close(ν_site2, ν_site)
            assert_close(c_site2, c_site)
            assert np.isfinite(Δ_LT)
            assert_close(ν_α2, ν_α)
            assert_close(c_α2, c_α)
            assert_close(abs_c_α2, abs_c_α)
            assert_array_equal(site_α2, site_α)
            assert_array_equal(k_α2, k_α)
            assert_array_equal(sys_α2, sys_α)

    print("  bath channel expansion OK")


# ===============================================
# hierarchy.py tests
# ===============================================

def test_hierarchy_counting_formulas():
    print("Testing hierarchy counting formulas...")

    for M in (1, 2, 3, 5, 10, 25, 30):
        for L in (0, 1, 2, 3, 5, 8, 10):
            assert number_of_ados(M, L) == comb(M + L, L)
            offsets = make_tier_offsets(M, L)
            assert offsets.shape == (L + 2,)
            assert offsets[0] == 0
            assert offsets[-1] == number_of_ados(M, L)
            for ℓ in range(L + 1):
                assert number_of_ados_at_tier(M, ℓ) == comb(M + ℓ - 1, ℓ)
                assert offsets[ℓ + 1] - offsets[ℓ] == number_of_ados_at_tier(M, ℓ)

    print("  counting formulas OK")



def test_hierarchy_known_small_ordering():
    print("Testing known small hierarchy ordering...")

    M = 3
    L = 2
    ado_indices = generate_ado_indices(M, L)

    expected = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 2, 0],
            [0, 1, 1],
            [0, 0, 2],
        ],
        dtype=ado_indices.dtype,
    )

    assert_array_equal(ado_indices, expected)
    print("  small ordering OK")



def test_hierarchy_build_for_different_Nmol():
    print("Testing hierarchy construction for different Nmol...")

    # These cases are intentionally modest so that the test file is fast.
    # They still cover the relevant M = Nmol*(K+1) logic used in HTC-HEOM.
    cases = [
        # Nmol, K_matsubara, L_heom
        (1, 0, 6),
        (2, 0, 6),
        (5, 0, 5),
        (10, 0, 4),
        (15, 0, 3),
        (25, 0, 2),
        (30, 0, 2),
        (5, 1, 4),
        (10, 1, 3),
        (5, 2, 3),
    ]

    for Nmol, K, L in cases:
        M = Nmol * (K + 1)
        N_ado_expected = comb(M + L, L)

        ado_indices, up, down, tier, tier_offsets = build_hierarchy(
            M=M,
            L=L,
            validate=True,
        )

        assert ado_indices.shape == (N_ado_expected, M)
        assert up.shape == (N_ado_expected, M)
        assert down.shape == (N_ado_expected, M)
        assert tier.shape == (N_ado_expected,)
        assert tier_offsets.shape == (L + 2,)

        # Check tiers directly.
        assert_array_equal(np.sum(ado_indices, axis=1).astype(tier.dtype), tier)

        # Check tier counts.
        counts = count_by_tier(tier, L)
        for ℓ in range(L + 1):
            assert counts[ℓ] == comb(M + ℓ - 1, ℓ)

        # Check valid up/down maps by direct coordinate comparison for a
        # sparse subset of rows. Full inverse consistency has already been
        # checked by validate=True.
        sample_rows = np.unique(
            np.linspace(0, N_ado_expected - 1, min(20, N_ado_expected), dtype=np.int64)
        )
        for I in sample_rows:
            nI = ado_indices[I].astype(np.int64)
            ℓI = int(tier[I])
            for α in range(M):
                J_up = int(up[I, α])
                if J_up != -1:
                    assert ℓI < L
                    n_expected = nI.copy()
                    n_expected[α] += 1
                    assert_array_equal(ado_indices[J_up].astype(np.int64), n_expected)

                J_down = int(down[I, α])
                if J_down != -1:
                    assert nI[α] > 0
                    n_expected = nI.copy()
                    n_expected[α] -= 1
                    assert_array_equal(ado_indices[J_down].astype(np.int64), n_expected)

        print(
            f"  Nmol={Nmol:2d}, K={K}, M={M:2d}, L={L}: "
            f"N_ado={N_ado_expected} OK"
        )

    print("  hierarchy construction OK")



def test_memory_estimates():
    print("Testing memory estimate utilities...")

    for Nmol, K, L in ((5, 0, 6), (10, 0, 6), (25, 0, 7), (30, 0, 7)):
        M = Nmol * (K + 1)
        d = Nmol + 1
        N_ado = comb(M + L, L)

        metadata_gb = estimate_hierarchy_metadata_gb(M, L)
        rk4_gb = estimate_rk4_state_gb(Nmol, L, K_matsubara=K, n_work_arrays=6)

        assert metadata_gb > 0.0
        assert rk4_gb > 0.0

        rk4_expected = 6 * N_ado * d * d * np.dtype(np.complex128).itemsize / 1.0e9
        assert_close(rk4_gb, rk4_expected)

        print(
            f"  Nmol={Nmol:2d}, K={K}, L={L}: "
            f"metadata={metadata_gb:.3f} GB, RK4-state={rk4_gb:.3f} GB"
        )

    print("  memory estimates OK")


# ===============================================
# Main test runner
# ===============================================

def main():
    print("===============================================")
    print("Running HTC-HEOM core module tests")
    print("===============================================")

    test_constants_basic_conversions()
    test_drude_coefficients_for_multiple_K()
    test_site_bath_channels_for_many_Nmol()
    test_hierarchy_counting_formulas()
    test_hierarchy_known_small_ordering()
    test_hierarchy_build_for_different_Nmol()
    test_memory_estimates()

    print("===============================================")
    print("All core module tests passed.")
    print("===============================================")


if __name__ == "__main__":
    main()
