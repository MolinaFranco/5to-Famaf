"""
Test script for 15-parameter training implementation.

This script verifies that the complete training implementation with
all 15 parameters works correctly. It tests:
1. Pack/unpack parameter consistency
2. Objective function evaluation
3. Gradient computation
4. All parameter transformations

Author: Claude Code
Date: 2025-10-26
"""

import sys
import numpy as np

# Add scikit-neuromsi to path
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

from skneuromsi.neural._echeveste_training import (  # noqa: E402
    create_objective_function
)


def test_pack_unpack_consistency():
    """
    Test that pack and unpack are inverse operations.

    This verifies that:
    unpack(pack(params)) == params (within numerical tolerance)
    """
    print("\n" + "="*70)
    print("TEST 1: Pack/Unpack Consistency")
    print("="*70)

    # Create dummy GSM model (not used in this test but required)
    class DummyGSM:
        pass

    gsm = DummyGSM()

    # Create objective function to get pack/unpack
    N_E, N_I = 25, 25
    tau_e, tau_i, tau_eta = 20e-3, 10e-3, 50e-3
    k = 0.3
    dt, t_max, t_subsamp = 0.2e-3, 0.1, 10e-3

    obj, grad_fn, pack, unpack = create_objective_function(
        gsm, N_E, N_I, tau_e, tau_i, tau_eta, k,
        dt, t_max, t_subsamp
    )

    # Define test parameters (following train.ml lines 89-101)
    test_params = {
        # Input transformation
        'input_baseline': 1.1,
        'input_scaling': 1.0,
        'input_nl_pow': 1.0,
        # Weight heights
        'a_EE': 0.02,
        'a_EI': 0.02,
        'a_IE': 0.02,
        'a_II': 0.02,
        # Weight widths
        'd_EE': 0.8,
        'd_EI': 0.8,
        'd_IE': 0.8,
        'd_II': 0.8,
        # Noise covariance
        'sigma_eta_width': 0.8,
        'sigma_eta_std_e': 2.0,
        'sigma_eta_std_i': 2.0,
        'sigma_eta_rho': 0.8
    }

    print("\nOriginal parameters:")
    for key, val in test_params.items():
        print(f"  {key:20s} = {val:.6f}")

    # Pack parameters
    x = pack(test_params)
    print(f"\nPacked vector shape: {x.shape}")
    print("Expected shape: (15,)")
    assert x.shape == (15,), f"Expected shape (15,), got {x.shape}"

    print("\nPacked values:")
    for i in range(15):
        print(f"  x[{i:2d}] = {x[i]:.6f}")

    # Unpack parameters
    recovered_params = unpack(x)
    print("\nRecovered parameters:")
    for key, val in recovered_params.items():
        print(f"  {key:20s} = {val:.6f}")

    # Check consistency
    print("\nParameter comparison:")
    all_close = True
    for key in test_params.keys():
        orig = test_params[key]
        recov = recovered_params[key]
        diff = abs(orig - recov)
        match = "✓" if diff < 1e-6 else "✗"
        print(f"  {match} {key:20s}: orig={orig:.6f}, "
              f"recov={recov:.6f}, diff={diff:.2e}")
        if diff >= 1e-6:
            all_close = False

    if all_close:
        print("\n✓ PASS: Pack/unpack are consistent")
    else:
        print("\n✗ FAIL: Pack/unpack inconsistency detected")
        sys.exit(1)


def test_objective_evaluation():
    """
    Test that objective function can be evaluated without errors.
    """
    print("\n" + "="*70)
    print("TEST 2: Objective Function Evaluation")
    print("="*70)

    # Create dummy GSM model
    class DummyGSM:
        pass

    gsm = DummyGSM()

    # Create objective function
    N_E, N_I = 25, 25
    tau_e, tau_i, tau_eta = 20e-3, 10e-3, 50e-3
    k = 0.3
    dt, t_max, t_subsamp = 0.2e-3, 0.1, 10e-3

    obj, grad_fn, pack, unpack = create_objective_function(
        gsm, N_E, N_I, tau_e, tau_i, tau_eta, k,
        dt, t_max, t_subsamp
    )

    # Create initial parameter vector
    init_params = {
        'input_baseline': 1.1,
        'input_scaling': 1.0,
        'input_nl_pow': 1.0,
        'a_EE': 0.02,
        'a_EI': 0.02,
        'a_IE': 0.02,
        'a_II': 0.02,
        'd_EE': 0.8,
        'd_EI': 0.8,
        'd_IE': 0.8,
        'd_II': 0.8,
        'sigma_eta_width': 0.8,
        'sigma_eta_std_e': 2.0,
        'sigma_eta_std_i': 2.0,
        'sigma_eta_rho': 0.8
    }

    x = pack(init_params)

    print(f"\nEvaluating objective with x.shape = {x.shape}")

    try:
        cost = obj(x)
        print("✓ Objective evaluated successfully")
        print(f"  Cost value: {cost:.6f}")
        print(f"  Cost type: {type(cost)}")

        # Check that cost is a scalar
        assert np.isscalar(cost) or cost.shape == (), \
            f"Cost should be scalar, got shape {cost.shape}"
        print("✓ Cost is scalar")

        # Check that cost is finite
        assert np.isfinite(cost), "Cost is not finite"
        print("✓ Cost is finite")

        print("\n" + "✓ PASS: Objective function evaluation successful")

    except Exception as e:
        print("\n" + "✗ FAIL: Objective evaluation failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_gradient_computation():
    """
    Test that gradients can be computed via JAX autodiff.
    """
    print("\n" + "="*70)
    print("TEST 3: Gradient Computation")
    print("="*70)

    # Create dummy GSM model
    class DummyGSM:
        pass

    gsm = DummyGSM()

    # Create objective function
    N_E, N_I = 25, 25
    tau_e, tau_i, tau_eta = 20e-3, 10e-3, 50e-3
    k = 0.3
    dt, t_max, t_subsamp = 0.2e-3, 0.1, 10e-3

    obj, grad_fn, pack, unpack = create_objective_function(
        gsm, N_E, N_I, tau_e, tau_i, tau_eta, k,
        dt, t_max, t_subsamp
    )

    # Create initial parameter vector
    init_params = {
        'input_baseline': 1.1,
        'input_scaling': 1.0,
        'input_nl_pow': 1.0,
        'a_EE': 0.02,
        'a_EI': 0.02,
        'a_IE': 0.02,
        'a_II': 0.02,
        'd_EE': 0.8,
        'd_EI': 0.8,
        'd_IE': 0.8,
        'd_II': 0.8,
        'sigma_eta_width': 0.8,
        'sigma_eta_std_e': 2.0,
        'sigma_eta_std_i': 2.0,
        'sigma_eta_rho': 0.8
    }

    x = pack(init_params)

    print(f"\nComputing gradient for x.shape = {x.shape}")

    try:
        g = grad_fn(x)
        print("✓ Gradient computed successfully")
        print(f"  Gradient shape: {g.shape}")
        print("  Expected shape: (15,)")

        assert g.shape == (15,), \
            f"Expected gradient shape (15,), got {g.shape}"
        print("✓ Gradient has correct shape")

        # Check that all gradient components are finite
        all_finite = np.all(np.isfinite(g))
        print(f"  All components finite: {all_finite}")
        if not all_finite:
            print("  Non-finite components:")
            for i in range(15):
                if not np.isfinite(g[i]):
                    print(f"    g[{i}] = {g[i]}")

        assert all_finite, "Gradient contains non-finite values"
        print("✓ All gradient components are finite")

        # Print gradient values
        print("\nGradient components:")
        param_names = [
            'input_baseline', 'input_scaling', 'input_nl_pow',
            'a_EE', 'a_EI', 'a_IE', 'a_II',
            'd_EE', 'd_EI', 'd_IE', 'd_II',
            'sigma_eta_width', 'sigma_eta_std_e',
            'sigma_eta_std_i', 'sigma_eta_rho'
        ]
        for i, name in enumerate(param_names):
            print(f"  ∂L/∂{name:20s} = {g[i]:12.6e}")

        print("\n" + "✓ PASS: Gradient computation successful")

    except Exception as e:
        print("\n" + "✗ FAIL: Gradient computation failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_parameter_bounds():
    """
    Test that parameter transformations enforce proper bounds.
    """
    print("\n" + "="*70)
    print("TEST 4: Parameter Bounds and Transformations")
    print("="*70)

    # Create dummy GSM model
    class DummyGSM:
        pass

    gsm = DummyGSM()

    # Create objective function
    N_E, N_I = 25, 25
    tau_e, tau_i, tau_eta = 20e-3, 10e-3, 50e-3
    k = 0.3
    dt, t_max, t_subsamp = 0.2e-3, 0.1, 10e-3

    obj, grad_fn, pack, unpack = create_objective_function(
        gsm, N_E, N_I, tau_e, tau_i, tau_eta, k,
        dt, t_max, t_subsamp
    )

    print("\nTesting parameter bounds:")

    # Test 1: Weight heights must be > 0.01
    print("\n1. Testing weight height constraint (a > 0.01):")
    for a_val in [0.012, 0.02, 0.05, 0.1]:
        params = {
            'input_baseline': 1.0,
            'input_scaling': 1.0,
            'input_nl_pow': 1.0,
            'a_EE': a_val, 'a_EI': a_val,
            'a_IE': a_val, 'a_II': a_val,
            'd_EE': 0.8, 'd_EI': 0.8,
            'd_IE': 0.8, 'd_II': 0.8,
            'sigma_eta_width': 0.8,
            'sigma_eta_std_e': 2.0,
            'sigma_eta_std_i': 2.0,
            'sigma_eta_rho': 0.5
        }
        x = pack(params)
        recovered = unpack(x)
        assert all(recovered[key] >= 0.01
                   for key in ['a_EE', 'a_EI', 'a_IE', 'a_II']), \
            "Weight heights should be >= 0.01"
        print(f"  ✓ a = {a_val:.3f} OK")

    # Test 2: rho should be constrained to [0, 1]
    print("\n2. Testing rho constraint (0 <= rho <= 1):")
    for rho_val in [0.0, 0.25, 0.5, 0.75, 0.99]:
        params = {
            'input_baseline': 1.0,
            'input_scaling': 1.0,
            'input_nl_pow': 1.0,
            'a_EE': 0.02, 'a_EI': 0.02,
            'a_IE': 0.02, 'a_II': 0.02,
            'd_EE': 0.8, 'd_EI': 0.8,
            'd_IE': 0.8, 'd_II': 0.8,
            'sigma_eta_width': 0.8,
            'sigma_eta_std_e': 2.0,
            'sigma_eta_std_i': 2.0,
            'sigma_eta_rho': rho_val
        }
        x = pack(params)
        recovered = unpack(x)
        rho_recovered = recovered['sigma_eta_rho']
        assert 0.0 <= rho_recovered <= 1.0, \
            f"rho should be in [0, 1], got {rho_recovered}"
        print(f"  ✓ rho = {rho_val:.2f} -> {rho_recovered:.6f} OK")

    # Test 3: Positive parameters should remain positive
    print("\n3. Testing positivity constraints:")
    positive_params = ['input_scaling', 'input_nl_pow',
                       'sigma_eta_std_e', 'sigma_eta_std_i']
    for param_name in positive_params:
        params = {
            'input_baseline': 1.0,
            'input_scaling': 1.0,
            'input_nl_pow': 1.0,
            'a_EE': 0.02, 'a_EI': 0.02,
            'a_IE': 0.02, 'a_II': 0.02,
            'd_EE': 0.8, 'd_EI': 0.8,
            'd_IE': 0.8, 'd_II': 0.8,
            'sigma_eta_width': 0.8,
            'sigma_eta_std_e': 2.0,
            'sigma_eta_std_i': 2.0,
            'sigma_eta_rho': 0.5
        }
        params[param_name] = 0.5  # Test with positive value
        x = pack(params)
        recovered = unpack(x)
        assert recovered[param_name] > 0, \
            f"{param_name} should be positive"
        print(f"  ✓ {param_name:20s} > 0")

    print("\n✓ PASS: All parameter bounds respected")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TESTING 15-PARAMETER TRAINING IMPLEMENTATION")
    print("="*70)
    print("\nThis test suite verifies that the complete 15-parameter")
    print("training implementation works correctly, including:")
    print("  - 3 input transformation parameters (α_h, β_h, γ_h)")
    print("  - 8 connectivity parameters (a_EE, a_EI, a_IE, a_II,")
    print("                                d_EE, d_EI, d_IE, d_II)")
    print("  - 4 noise covariance parameters (width, σ_E, σ_I, ρ)")
    print("\nReference: Echeveste et al. (2020) Nature Neuroscience")

    # Run all tests
    test_pack_unpack_consistency()
    test_objective_evaluation()
    test_gradient_computation()
    test_parameter_bounds()

    # Summary
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print("\nThe 15-parameter training implementation is working correctly!")
    print("You can now use this for training SSN models with the full")
    print("parameter set described in Echeveste et al. 2020.")
    print()


if __name__ == "__main__":
    main()
