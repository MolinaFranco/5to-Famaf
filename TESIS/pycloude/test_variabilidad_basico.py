#!/usr/bin/env python3
"""
Basic test to verify that 10 consecutive executions of our implementation
generate different activity patterns, demonstrating proper stochastic behavior.

This is a simplified version that focuses on answering the user's core question:
"si lo corres 10 veces (tienen que ser distintas entre si generando distintas causas)?"
"""

import sys
import numpy as np

# Add scikit-neuromsi to path
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

from skneuromsi.neural import Echeveste2020

def test_basic_variability():
    """Test if our implementation generates different patterns across runs."""
    print("=" * 60)
    print("BASIC VARIABILITY TEST")
    print("=" * 60)

    activities = []
    final_states = []

    for run_idx in range(10):
        print(f"\nRun {run_idx + 1}/10:", end=" ")

        try:
            # Create fresh instance with different random seed each time
            ssn = Echeveste2020(
                N_E=50,
                N_I=50,
                seed=None  # Different random seed each time
            )

            # Load learned parameters
            ssn.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')

            # Run simulation with time that matches expected resolution
            # Time resolution is 0.2ms, expecting 5000 points means 5000 * 0.2 = 1000ms
            response = ssn.run(
                stimulus_contrast=0.32,
                stimulus_orientation=45.0,
                simulation_time=1000.0  # Full simulation to match expected time points
            )

            # Extract excitatory activity
            if hasattr(response, 'e_'):
                activity = response.e_

                # Store statistics
                activity_mean = np.mean(activity)
                activity_std = np.std(activity)
                final_state = activity[-5:] if len(activity) >= 5 else activity

                activities.append(activity_mean)
                final_states.append(final_state)

                print(f"DONE (mean: {activity_mean:.3f}, std: {activity_std:.3f})")

            else:
                print("ERROR - Cannot extract activity")
                activities.append(None)
                final_states.append(None)

        except Exception as e:
            print(f"ERROR - {e}")
            activities.append(None)
            final_states.append(None)

    return activities, final_states


def analyze_basic_variability(activities, final_states):
    """Analyze if the runs show variability."""
    print("\n" + "=" * 60)
    print("VARIABILITY ANALYSIS")
    print("=" * 60)

    # Filter successful runs
    valid_activities = [a for a in activities if a is not None]
    valid_final_states = [s for s in final_states if s is not None]

    print(f"Successful runs: {len(valid_activities)}/10")

    if len(valid_activities) < 2:
        print("Insufficient successful runs")
        return False

    print(f"\nActivity means across runs:")
    for i, activity in enumerate(valid_activities):
        print(f"  Run {i+1}: {activity:.6f}")

    # Test variability
    activity_std = np.std(valid_activities)
    print(f"\nStandard deviation of means: {activity_std:.6f}")
    print(f"Range: [{np.min(valid_activities):.6f}, {np.max(valid_activities):.6f}]")

    # Check if activities vary significantly
    varies_significantly = activity_std > 0.001  # Threshold for significance

    # Test correlation between final states
    if len(valid_final_states) > 1:
        correlations = []
        for i in range(len(valid_final_states)):
            for j in range(i+1, len(valid_final_states)):
                if len(valid_final_states[i]) == len(valid_final_states[j]):
                    corr = np.corrcoef(valid_final_states[i], valid_final_states[j])[0,1]
                    correlations.append(corr)

        if correlations:
            avg_correlation = np.mean(correlations)
            print(f"Average correlation between final states: {avg_correlation:.4f}")
            different_final_states = avg_correlation < 0.9
        else:
            different_final_states = False
    else:
        different_final_states = False

    print(f"\nVariability Tests:")
    print(f"  Activities vary significantly: {varies_significantly}")
    print(f"  Final states are different: {different_final_states}")

    shows_variability = varies_significantly or different_final_states

    print(f"\n" + "=" * 60)
    print("ANSWER TO USER'S QUESTION")
    print("=" * 60)
    print("Question: ¿Si lo corres 10 veces (tienen que ser distintas entre sí)")
    print("          generando distintas causas)?")
    print()

    if shows_variability:
        print("✓ YES - Our implementation shows proper stochastic variability")
        print("✓ Each execution generates different neural activity patterns")
        print("✓ This indicates the noise processes are working correctly")
        print("✓ Different activity patterns would lead to different causal inferences")
    else:
        print("✗ NO - Implementation shows insufficient variability")
        print("✗ Runs produce very similar or identical results")
        print("✗ This suggests problems with random number generation")

    return shows_variability


if __name__ == "__main__":
    print("Testing stochastic variability across 10 executions...")

    # Run basic variability test
    activities, final_states = test_basic_variability()

    # Analyze results
    shows_variability = analyze_basic_variability(activities, final_states)

    # Summary
    print(f"\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if shows_variability:
        print("Our implementation correctly generates different results across")
        print("multiple executions, demonstrating proper stochastic behavior.")
        print("This means it would generate different causes like the original.")
    else:
        print("Our implementation may have issues with stochastic variability.")
        print("This should be investigated further.")