#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Echeveste2020 train() method implementation.

This script tests the newly implemented train() method to ensure it:
1. Accepts a GSM model and optional parameters
2. Runs Stage 1 optimization (connectivity parameters)
3. Runs Stage 2 optimization (noise covariance)
4. Returns proper results dictionary
"""

import sys
sys.path.insert(0, 'scikit-neuromsi')

import numpy as np
from skneuromsi.neural import Echeveste2020
from skneuromsi.generative import GSM


def test_train_method():
    """Test the train() method implementation."""
    print("=" * 70)
    print("Testing Echeveste2020.train() method")
    print("=" * 70)

    # Initialize SSN model
    print("\n1. Initializing SSN model...")
    ssn = Echeveste2020(N_E=25, N_I=25, seed=42)
    print(f"   Created SSN with {ssn.N_E} E neurons and {ssn.N_I} I neurons")

    # Initialize GSM model
    print("\n2. Initializing GSM model...")
    gsm = GSM(
        n_orientations=25,
        patch_size=16,
        use_pretrained=True,
        random_seed=42
    )
    print(f"   Created GSM with {gsm.n_orientations} orientations")

    # Test Stage 1 parameters
    print("\n3. Setting up Stage 1 parameters...")
    stage1_params = {
        'a_EE': 0.02,
        'a_EI': 0.02,
        'a_IE': 0.02,
        'a_II': 0.02,
        'd_EE': 0.8,
        'd_EI': 0.8,
        'd_IE': 0.8,
        'd_II': 0.8,
    }
    print(f"   Stage 1 params: {stage1_params}")

    # Test Stage 2 parameters
    print("\n4. Setting up Stage 2 parameters...")
    stage2_params = {
        'noise_width': 0.8,
        'noise_std_e': 2.0,
        'noise_std_i': 2.0,
        'noise_rho': 0.8,
    }
    print(f"   Stage 2 params: {stage2_params}")

    # Run training
    print("\n5. Running train() method...")
    print("-" * 70)
    results = ssn.train(
        gsm_model=gsm,
        stage1_params=stage1_params,
        stage2_params=stage2_params
    )
    print("-" * 70)

    # Verify results
    print("\n6. Verifying results...")
    assert results is not None, "train() returned None"
    assert 'stage1' in results, "Missing 'stage1' in results"
    assert 'stage2' in results, "Missing 'stage2' in results"
    assert 'connectivity_params' in results, "Missing connectivity params"
    assert 'convergence_info' in results, "Missing convergence info"
    print("   ✓ All expected keys present in results")

    # Check Stage 1 results
    print("\n7. Checking Stage 1 results...")
    s1 = results['stage1']
    print(f"   Success: {s1['success']}")
    print(f"   Message: {s1['message']}")
    print(f"   Optimized params: {s1['optimized_params']}")

    # Check Stage 2 results
    print("\n8. Checking Stage 2 results...")
    s2 = results['stage2']
    print(f"   Success: {s2['success']}")
    print(f"   Message: {s2['message']}")
    print(f"   Sigma_eta shape: {s2['sigma_eta_shape']}")

    # Check connectivity parameters
    print("\n9. Checking connectivity parameters...")
    conn_params = results['connectivity_params']
    for key, value in conn_params.items():
        print(f"   {key}: {value:.4f}")

    # Check convergence info
    print("\n10. Checking convergence info...")
    conv_info = results['convergence_info']
    print(f"   Stage 1 completed: {conv_info['stage1_completed']}")
    print(f"   Stage 2 completed: {conv_info['stage2_completed']}")
    print(f"   Is trained: {conv_info['is_trained']}")

    # Verify training flags are set correctly
    assert conv_info['stage1_completed'], "Stage 1 not marked as completed"
    assert conv_info['stage2_completed'], "Stage 2 not marked as completed"
    assert conv_info['is_trained'], "Model not marked as trained"
    print("   ✓ Training flags correctly set")

    # Verify internal parameters were updated
    print("\n11. Verifying internal parameters...")
    assert ssn._a_EE > 0, "a_EE not set"
    assert ssn._a_EI > 0, "a_EI not set"
    assert ssn._a_IE > 0, "a_IE not set"
    assert ssn._a_II > 0, "a_II not set"
    print(f"   ✓ Connectivity parameters: a_EE={ssn._a_EE:.4f}, "
          f"a_EI={ssn._a_EI:.4f}")

    assert ssn._Sigma_eta is not None, "Sigma_eta not set"
    assert ssn._Sigma_eta.shape == (50, 50), f"Wrong Sigma_eta shape"
    print(f"   ✓ Noise covariance matrix: shape {ssn._Sigma_eta.shape}")

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)

    return results


if __name__ == '__main__':
    try:
        results = test_train_method()
        print("\nTest completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
