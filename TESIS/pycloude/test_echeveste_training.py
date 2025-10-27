"""
Test script for Echeveste2020 training functionality.

This script tests the basic training implementation including:
- JAX-based ADF moment evolution
- Cost function computation
- Integration with scipy L-BFGS-B optimizer
"""

import sys
import numpy as np

sys.path.insert(0, '../scikit-neuromsi')

from skneuromsi.neural import Echeveste2020  # noqa: E402


# Mock GSM model for testing
class MockGSM:
    """Simple mock GSM model for testing."""
    def __init__(self, N_E=25):
        self.N_E = N_E
        # A: Gabor filter matrix (placeholder)
        self.A = np.random.randn(100, N_E)
        # C: Prior covariance (placeholder)
        self.C = np.eye(N_E)


def test_adf_moment_evolution():
    """Test básico de evolución de momentos con ADF."""
    print("=" * 70)
    print("TEST 1: ADF Moment Evolution")
    print("=" * 70)

    from skneuromsi.neural._echeveste_training import evolve_moments

    # Parámetros simples
    N_E = 10
    N_I = 10
    n = N_E + N_I

    # Matrices de prueba
    W = 0.01 * np.random.randn(n, n)
    h = np.ones(n) * 0.1
    sigma_eta = 0.1 * np.eye(n)
    inv_taus = np.concatenate([
        np.full(N_E, 1.0 / 0.02),  # tau_e = 20ms
        np.full(N_I, 1.0 / 0.01)   # tau_i = 10ms
    ])

    dt = 0.2e-3  # 0.2ms
    tau_eta = 20e-3  # 20ms
    t_max = 0.01  # 10ms (muy corto para prueba rápida)

    print(f"Network size: {N_E}E + {N_I}I = {n} total")
    print(f"Time parameters: dt={dt*1000}ms, t_max={t_max*1000}ms")
    print(f"Number of steps: {int(t_max/dt)}")

    try:
        mu_final, sigma_final, sigma_star_final = evolve_moments(
            W, h, sigma_eta, inv_taus, dt, tau_eta, t_max
        )

        print("\n✓ Evolution completed successfully")
        print(f"Final mean norm: {np.linalg.norm(mu_final):.6f}")
        print(f"Final sigma trace: {np.trace(sigma_final):.6f}")
        print(f"Final sigma_star trace: {np.trace(sigma_star_final):.6f}")
        return True

    except Exception as e:
        print(f"\n✗ Evolution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cost_function():
    """Test cálculo de función de costo."""
    print("\n" + "=" * 70)
    print("TEST 2: Cost Function Computation")
    print("=" * 70)

    from skneuromsi.neural._echeveste_training import (
        compute_evolution_costs
    )

    # Parámetros simples
    N_E = 10
    N_I = 10
    n = N_E + N_I
    m = N_E  # Solo excitatorias

    # Matrices de prueba
    W = 0.01 * np.random.randn(n, n)
    h = np.ones(n) * 0.1
    sigma_eta = 0.1 * np.eye(n)
    inv_taus = np.concatenate([
        np.full(N_E, 1.0 / 0.02),
        np.full(N_I, 1.0 / 0.01)
    ])

    # Targets dummy
    target_mu = np.zeros(m)
    target_sigma = np.eye(m)

    dt = 0.2e-3
    tau_eta = 20e-3
    t_max = 0.01
    t_subsamp = 5e-3

    print(f"Target statistics: μ shape {target_mu.shape}, "
          f"Σ shape {target_sigma.shape}")

    try:
        cost, mu_final, sigma_final = compute_evolution_costs(
            W, h, sigma_eta, inv_taus,
            dt, tau_eta, t_max, t_subsamp,
            target_mu, target_sigma,
            k=0.3,
            lambda_mean=1.0,
            lambda_var=1.0,
            lambda_cov=1.0,
            min_time=0.005
        )

        print("\n✓ Cost computation successful")
        print(f"Total cost: {cost:.6f}")
        print(f"Final mean (E neurons): {mu_final[:m][:5]}")
        return True

    except Exception as e:
        print(f"\n✗ Cost computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stage1_optimization():
    """Test optimización Stage 1 (muy pocas iteraciones)."""
    print("\n" + "=" * 70)
    print("TEST 3: Stage 1 Optimization (Quick Test)")
    print("=" * 70)

    # Crear modelo SSN pequeño para testing rápido
    N_E = 10
    N_I = 10

    print(f"Creating Echeveste2020 model: {N_E}E + {N_I}I neurons")

    try:
        ssn = Echeveste2020(
            N_E=N_E,
            N_I=N_I,
            seed=42
        )
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

    # Crear mock GSM
    gsm = MockGSM(N_E=N_E)
    print(f"✓ Mock GSM created (A shape: {gsm.A.shape})")

    # Parámetros de entrenamiento muy limitados para testing rápido
    stage1_params = {
        'max_iter': 2,  # Solo 2 iteraciones para test rápido
        'dt': 1e-3,  # 1ms (más grande para menos pasos)
        't_max': 0.01,  # 10ms (muy corto)
        't_subsamp': 5e-3,  # 5ms
        'lambda_mean': 1.0,
        'lambda_var': 1.0,
        'lambda_cov': 1.0,
    }

    print("\nStarting Stage 1 optimization (limited to 2 iterations)...")
    print(f"Time parameters: dt={stage1_params['dt']*1000}ms, "
          f"t_max={stage1_params['t_max']*1000}ms")

    try:
        results = ssn._optimize_stage1(gsm, stage1_params)

        print("\n✓ Stage 1 optimization completed")
        print(f"Success: {results['success']}")
        print(f"Final cost: {results.get('final_cost', 'N/A')}")
        print(f"Iterations: {results.get('n_iterations', 'N/A')}")
        print(f"Message: {results.get('message', 'N/A')}")

        print("\nOptimized parameters:")
        for key, val in results['optimized_params'].items():
            initial_val = results['initial_params'][key]
            print(f"  {key}: {initial_val:.6f} → {val:.6f}")

        return True

    except Exception as e:
        print(f"\n✗ Stage 1 optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("ECHEVESTE2020 TRAINING TESTS")
    print("=" * 70)

    results = []

    # Test 1: ADF evolution
    results.append(("ADF Moment Evolution", test_adf_moment_evolution()))

    # Test 2: Cost function
    results.append(("Cost Function", test_cost_function()))

    # Test 3: Stage 1 optimization (quick test)
    results.append(("Stage 1 Optimization", test_stage1_optimization()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
