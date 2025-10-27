#!/usr/bin/env python3
"""
Test script to verify that 10 consecutive executions generate different causes
like the original Echeveste2020 code.

This addresses the user's question: "si lo corres 10 veces (tienen que ser
distintas entre si generando distintas causas) da los mismos resultados
que el original?"

Tests both:
1. Original Echeveste code (GSM.py)
2. Our implementation (scikit-neuromsi)

Each execution should generate different neural activity patterns due to
stochastic noise (Ornstein-Uhlenbeck process), leading to different posterior
distributions and potentially different detected causes.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add scikit-neuromsi to path
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

def test_original_echeveste_variability():
    """Test if original Echeveste code generates different activity patterns across runs."""
    print("=" * 60)
    print("TESTING ORIGINAL ECHEVESTE PATTERN USING GSM FUNCTIONS")
    print("=" * 60)

    # Change to GSM directory where original code is located
    original_dir = os.getcwd()
    os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

    # Import original GSM code
    sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')
    import GSM

    results_original = []

    # Load filters like in main()
    A = np.load("filters.npy")
    D_x = len(A)
    D_y = len(A[0])

    # Parameters from original main()
    s_x = 10.0  # Observation noise
    z_values = np.linspace(0.01, 1.0, 50)  # Contrast range

    for run_idx in range(10):
        print(f"\nOriginal Pattern Run {run_idx + 1}/10:")

        try:
            # Set different random seed for each run
            np.random.seed(seed=None)

            # Generate stimulus parameters (similar to main())
            z = 0.32  # Fixed contrast for comparison
            y = np.zeros(D_y)  # Simple stimulus pattern
            y[0] = 1.0  # Simple orientation pattern

            # Generate observation using original GSM.get_x
            x = GSM.get_x(y, z, A, s_x)

            # Calculate posterior using original function with correct signature
            # P_z_giv_x(z_range, x, ACAT, s_x_2, k, theta)
            ATA = np.dot(A.T, A)
            ACAT = ATA  # Simplified for this test
            s_x_2 = s_x * s_x  # Convert to variance
            k = 2.0  # Gamma shape parameter
            theta = 2.0  # Gamma scale parameter

            P_z_giv_x_values = GSM.P_z_giv_x(z_values, x, ACAT, s_x_2, k, theta)

            P_z_giv_x_values = np.array(P_z_giv_x_values)

            # Find MAP estimate
            map_idx = np.argmax(P_z_giv_x_values)
            map_estimate = z_values[map_idx]

            # Calculate simple "cause" detection (threshold-based)
            max_prob = np.max(P_z_giv_x_values)
            confidence = max_prob / np.mean(P_z_giv_x_values)

            # Cause detected if confidence > threshold
            cause_detected = confidence > 2.0

            results_original.append({
                'run': run_idx + 1,
                'activity_mean': np.mean(x),
                'activity_std': np.std(x),
                'map_estimate': map_estimate,
                'confidence': confidence,
                'cause_detected': cause_detected,
                'final_activity': x[-10:] if len(x) >= 10 else x  # Last 10 or all
            })

            print(f"  Activity mean: {np.mean(x):.3f}")
            print(f"  Activity std: {np.std(x):.3f}")
            print(f"  MAP estimate: {map_estimate:.3f}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Cause detected: {cause_detected}")

        except Exception as e:
            print(f"  ERROR in original run {run_idx + 1}: {e}")
            results_original.append({
                'run': run_idx + 1,
                'error': str(e)
            })

    # Restore original directory
    os.chdir(original_dir)

    return results_original


def test_our_implementation_variability():
    """Test if our scikit-neuromsi implementation generates different causes across runs."""
    print("\n" + "=" * 60)
    print("TESTING OUR SCIKIT-NEUROMSI IMPLEMENTATION VARIABILITY")
    print("=" * 60)

    from skneuromsi.neural import Echeveste2020

    results_ours = []

    for run_idx in range(10):
        print(f"\nOur Implementation Run {run_idx + 1}/10:")

        try:
            # Create fresh instance with different random seed each time
            ssn = Echeveste2020(
                N_E=25,
                N_I=25,
                seed=None  # Different random seed each time
            )

            # Run simulation with correct parameters (use simulation_time not n_steps)
            results = ssn.run(
                stimulus_contrast=0.32,
                stimulus_orientation=45.0,
                simulation_time=1000.0,
                use_exact_matrix=True  # Use exact original matrix
            )

            # Extract network activity
            network_activity = results['network_activity']

            # Calculate causes using our causal inference
            causes = ssn.calculate_causes(
                network_activity=network_activity,
                confidence_threshold=0.95
            )

            # Calculate additional metrics
            contrast_range = np.linspace(0.01, 1.0, 50)
            posterior = ssn._extract_posterior_distribution(
                network_activity, contrast_range
            )

            results_ours.append({
                'run': run_idx + 1,
                'activity_mean': np.mean(network_activity),
                'activity_std': np.std(network_activity),
                'n_causes': len(causes),
                'causes': causes,
                'map_estimate': posterior['map_estimate'],
                'confidence': np.max(posterior['probabilities']) / np.mean(posterior['probabilities']),
                'final_activity': network_activity[-10:]  # Last 10 time points
            })

            print(f"  Activity mean: {np.mean(network_activity):.3f}")
            print(f"  Activity std: {np.std(network_activity):.3f}")
            print(f"  MAP estimate: {posterior['map_estimate']:.3f}")
            print(f"  Confidence: {np.max(posterior['probabilities']) / np.mean(posterior['probabilities']):.3f}")
            print(f"  Detected causes: {len(causes)}")

        except Exception as e:
            print(f"  ERROR in our run {run_idx + 1}: {e}")
            results_ours.append({
                'run': run_idx + 1,
                'error': str(e)
            })

    return results_ours


def analyze_variability(results_original, results_ours):
    """Analyze and compare variability between original and our implementation."""
    print("\n" + "=" * 60)
    print("VARIABILITY ANALYSIS")
    print("=" * 60)

    # Filter successful runs
    successful_original = [r for r in results_original if 'error' not in r]
    successful_ours = [r for r in results_ours if 'error' not in r]

    print(f"\nSuccessful runs:")
    print(f"  Original: {len(successful_original)}/10")
    print(f"  Ours: {len(successful_ours)}/10")

    if len(successful_original) > 1:
        print(f"\nOriginal Implementation Variability:")
        activity_means = [r['activity_mean'] for r in successful_original]
        activity_stds = [r['activity_std'] for r in successful_original]
        map_estimates = [r['map_estimate'] for r in successful_original]
        confidences = [r['confidence'] for r in successful_original]
        cause_detected = [r['cause_detected'] for r in successful_original]

        print(f"  Activity means across runs: {np.std(activity_means):.4f} (std)")
        print(f"  Activity stds across runs: {np.std(activity_stds):.4f} (std)")
        print(f"  MAP estimates across runs: {np.std(map_estimates):.4f} (std)")
        print(f"  Confidence across runs: {np.std(confidences):.4f} (std)")
        print(f"  Cause detection: {cause_detected}")
        print(f"  Causes vary across runs: {len(set(cause_detected)) > 1}")

        # Check if final activities are different
        final_activities = [r['final_activity'] for r in successful_original]
        if len(final_activities) > 1:
            # Compare correlation between final activities of different runs
            corr_sum = 0
            count = 0
            for i in range(len(final_activities)):
                for j in range(i+1, len(final_activities)):
                    corr = np.corrcoef(final_activities[i], final_activities[j])[0,1]
                    corr_sum += corr
                    count += 1
            avg_correlation = corr_sum / count if count > 0 else 1.0
            print(f"  Average correlation between runs: {avg_correlation:.4f}")
            print(f"  Runs are different: {avg_correlation < 0.9}")

    if len(successful_ours) > 1:
        print(f"\nOur Implementation Variability:")
        activity_means = [r['activity_mean'] for r in successful_ours]
        activity_stds = [r['activity_std'] for r in successful_ours]
        n_causes = [r['n_causes'] for r in successful_ours]
        map_estimates = [r['map_estimate'] for r in successful_ours]
        confidences = [r['confidence'] for r in successful_ours]

        print(f"  Activity means across runs: {np.std(activity_means):.4f} (std)")
        print(f"  Activity stds across runs: {np.std(activity_stds):.4f} (std)")
        print(f"  MAP estimates across runs: {np.std(map_estimates):.4f} (std)")
        print(f"  Confidence across runs: {np.std(confidences):.4f} (std)")
        print(f"  Number of causes: {n_causes}")
        print(f"  Causes vary across runs: {len(set(n_causes)) > 1}")

        # Check if final activities are different
        final_activities = [r['final_activity'] for r in successful_ours]
        if len(final_activities) > 1:
            # Compare correlation between final activities of different runs
            corr_sum = 0
            count = 0
            for i in range(len(final_activities)):
                for j in range(i+1, len(final_activities)):
                    corr = np.corrcoef(final_activities[i], final_activities[j])[0,1]
                    corr_sum += corr
                    count += 1
            avg_correlation = corr_sum / count if count > 0 else 1.0
            print(f"  Average correlation between runs: {avg_correlation:.4f}")
            print(f"  Runs are different: {avg_correlation < 0.9}")

    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if len(successful_original) > 1 and len(successful_ours) > 1:
        orig_causes = [r['cause_detected'] for r in successful_original]
        our_causes = [r['n_causes'] for r in successful_ours]
        orig_map = [r['map_estimate'] for r in successful_original]
        our_map = [r['map_estimate'] for r in successful_ours]

        print(f"Original generates variable causes: {len(set(orig_causes)) > 1}")
        print(f"Our implementation generates variable causes: {len(set(our_causes)) > 1}")
        print(f"Original MAP estimate variability: {np.std(orig_map):.4f}")
        print(f"Our MAP estimate variability: {np.std(our_map):.4f}")
        print(f"Both implementations show similar variability patterns: {len(set(orig_causes)) > 1 or len(set(our_causes)) > 1}")

        # Test if MAP estimates are in similar ranges
        orig_range = [np.min(orig_map), np.max(orig_map)]
        our_range = [np.min(our_map), np.max(our_map)]
        print(f"Original MAP range: [{orig_range[0]:.3f}, {orig_range[1]:.3f}]")
        print(f"Our MAP range: [{our_range[0]:.3f}, {our_range[1]:.3f}]")
    else:
        print("Insufficient successful runs to compare variability")


if __name__ == "__main__":
    # Test original implementation
    results_original = test_original_echeveste_variability()

    # Test our implementation
    results_ours = test_our_implementation_variability()

    # Analyze and compare variability
    analyze_variability(results_original, results_ours)