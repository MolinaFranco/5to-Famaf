"""
Comprehensive test of Echeveste2020 training with real GSM model.

This script tests the complete training pipeline including:
- GSM posterior computation
- Stage 1 optimization (connectivity parameters)
- Stage 2 optimization (noise covariance)
- Full two-stage training workflow
"""

import sys
import numpy as np

sys.path.insert(0, '../scikit-neuromsi')

from skneuromsi.neural import Echeveste2020  # noqa: E402
from skneuromsi.generative import GSM  # noqa: E402


def test_gsm_posterior_computation():
    """Test básico de cálculo del posterior del GSM."""
    print("=" * 70)
    print("TEST 1: GSM Posterior Computation")
    print("=" * 70)

    # Crear GSM pequeño para testing
    gsm = GSM(
        patch_size=16,
        n_orientations=25,  # Reducido para testing rápido
        use_pretrained=False,
        random_seed=42
    )

    print(f"GSM created: {gsm.n_orientations} orientations")
    print(f"A matrix shape: {gsm.A.shape}")
    print(f"C matrix shape: {gsm.C.shape}")

    # Generar un estímulo de prueba
    stim_data = gsm.generate_stimulus_patch(contrast=0.5)
    x = stim_data['x']

    print(f"\nGenerated stimulus patch: shape {x.shape}")
    print(f"True contrast: {stim_data['z']}")

    # Calcular posterior
    try:
        mu_post, Sigma_post = gsm.compute_posterior_moments(x, z_map=0.5)

        print("\n✓ Posterior computation successful")
        print(f"Posterior mean shape: {mu_post.shape}")
        print(f"Posterior covariance shape: {Sigma_post.shape}")
        print(f"Posterior mean norm: {np.linalg.norm(mu_post):.6f}")
        print(f"Posterior cov trace: {np.trace(Sigma_post):.6f}")

        # Verificar que Sigma_post es simétrica y positiva definida
        is_symmetric = np.allclose(Sigma_post, Sigma_post.T)
        eigenvalues = np.linalg.eigvals(Sigma_post)
        is_pos_def = np.all(eigenvalues > 0)

        print(f"Covariance is symmetric: {is_symmetric}")
        print(f"Covariance is positive definite: {is_pos_def}")
        print(f"Min eigenvalue: {eigenvalues.min():.6f}")

        return True

    except Exception as e:
        print(f"\n✗ Posterior computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gsm_training_targets():
    """Test generación de targets para entrenamiento del SSN."""
    print("\n" + "=" * 70)
    print("TEST 2: GSM Training Targets Generation")
    print("=" * 70)

    gsm = GSM(
        patch_size=16,
        n_orientations=25,
        use_pretrained=False,
        random_seed=42
    )

    contrast = 0.5
    n_samples = 20  # Pocas muestras para testing rápido

    print(f"Generating targets: contrast={contrast}, n_samples={n_samples}")

    try:
        targets = gsm.compute_posterior_for_ssn_training(
            contrast=contrast,
            n_samples=n_samples
        )

        print("\n✓ Target generation successful")
        print(f"Target mu shape: {targets['target_mu'].shape}")
        print(f"Target sigma shape: {targets['target_sigma'].shape}")
        print(f"H inputs shape: {targets['h_inputs'].shape}")
        print(f"Stimuli shape: {targets['stimuli'].shape}")

        print("\nTarget statistics:")
        print(f"  Mean norm: {np.linalg.norm(targets['target_mu']):.6f}")
        print(f"  Cov trace: {np.trace(targets['target_sigma']):.6f}")
        print(f"  H inputs mean: {targets['h_inputs'].mean():.6f}")

        return True

    except Exception as e:
        print(f"\n✗ Target generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stage1_with_real_gsm():
    """Test Stage 1 con targets reales del GSM."""
    print("\n" + "=" * 70)
    print("TEST 3: Stage 1 Training with Real GSM")
    print("=" * 70)

    # Crear GSM
    gsm = GSM(
        patch_size=16,
        n_orientations=10,  # Muy pequeño para testing rápido
        use_pretrained=False,
        random_seed=42
    )

    # Crear SSN matching GSM dimensions
    ssn = Echeveste2020(
        N_E=10,
        N_I=10,
        seed=42
    )

    print(f"GSM: {gsm.n_orientations} orientations")
    print(f"SSN: {ssn.N_E}E + {ssn.N_I}I neurons")

    # Parámetros de Stage 1 (muy limitados)
    stage1_params = {
        'max_iter': 2,  # Solo 2 iteraciones
        'dt': 1e-3,  # 1ms
        't_max': 0.01,  # 10ms
        't_subsamp': 5e-3,
        'contrast': 0.5,
        'n_samples_gsm': 10,  # Pocas muestras
    }

    print("\nStarting Stage 1 optimization...")
    print(f"Parameters: {stage1_params}")

    try:
        results = ssn._optimize_stage1(gsm, stage1_params)

        print("\n✓ Stage 1 optimization completed")
        print(f"Success: {results['success']}")
        print(f"Final cost: {results.get('final_cost', 'N/A')}")
        print(f"Iterations: {results.get('n_iterations', 'N/A')}")

        print("\nOptimized connectivity parameters:")
        for key in ['a_EE', 'a_EI', 'a_IE', 'a_II']:
            init = results['initial_params'][key]
            opt = results['optimized_params'][key]
            print(f"  {key}: {init:.6f} → {opt:.6f}")

        return True

    except Exception as e:
        print(f"\n✗ Stage 1 optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stage2_with_real_gsm():
    """Test Stage 2 con targets reales del GSM."""
    print("\n" + "=" * 70)
    print("TEST 4: Stage 2 Training with Real GSM")
    print("=" * 70)

    # Crear GSM
    gsm = GSM(
        patch_size=16,
        n_orientations=10,
        use_pretrained=False,
        random_seed=42
    )

    # Crear SSN y simular Stage 1 completado
    ssn = Echeveste2020(
        N_E=10,
        N_I=10,
        seed=42
    )

    # Marcar Stage 1 como completado (con parámetros default)
    ssn._stage1_completed = True
    # Inicializar parámetros de conectividad
    ssn._a_EE = 0.02
    ssn._a_EI = 0.02
    ssn._a_IE = 0.02
    ssn._a_II = 0.02
    ssn._d_EE = 0.8
    ssn._d_EI = 0.8
    ssn._d_IE = 0.8
    ssn._d_II = 0.8

    # Parámetros de Stage 2 (muy limitados)
    stage2_params = {
        'max_iter': 2,
        'dt': 1e-3,
        't_max': 0.01,
        't_subsamp': 5e-3,
        'contrast': 0.5,
        'n_samples_gsm': 10,
    }

    print("Starting Stage 2 optimization...")
    print(f"Parameters: {stage2_params}")

    try:
        results = ssn._optimize_stage2(gsm, stage2_params)

        print("\n✓ Stage 2 optimization completed")
        print(f"Success: {results['success']}")
        print(f"Final cost: {results.get('final_cost', 'N/A')}")
        print(f"Iterations: {results.get('n_iterations', 'N/A')}")

        print("\nOptimized noise parameters:")
        init_params = results['initial_params']
        opt_params = results['optimized_params']
        for key in ['width', 'std_e', 'std_i', 'rho']:
            print(f"  {key}: {init_params[key]:.4f} → "
                  f"{opt_params[key]:.4f}")

        return True

    except Exception as e:
        print(f"\n✗ Stage 2 optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_two_stage_training():
    """Test entrenamiento completo de dos etapas."""
    print("\n" + "=" * 70)
    print("TEST 5: Full Two-Stage Training")
    print("=" * 70)

    # Crear GSM
    gsm = GSM(
        patch_size=16,
        n_orientations=10,
        use_pretrained=False,
        random_seed=42
    )

    # Crear SSN
    ssn = Echeveste2020(
        N_E=10,
        N_I=10,
        seed=42
    )

    # Parámetros compartidos
    shared_params = {
        'contrast': 0.5,
        'n_samples_gsm': 10,
        'dt': 1e-3,
        't_max': 0.01,
        't_subsamp': 5e-3,
    }

    stage1_params = {
        **shared_params,
        'max_iter': 2,
    }

    stage2_params = {
        **shared_params,
        'max_iter': 2,
    }

    print("Starting full two-stage training...")
    print(f"Shared params: {shared_params}")

    try:
        results = ssn.train(
            gsm,
            stage1_params=stage1_params,
            stage2_params=stage2_params
        )

        print("\n✓ Full training completed successfully")
        print(f"Stage 1 success: {results['stage1']['success']}")
        print(f"Stage 2 success: {results['stage2']['success']}")
        print(f"Model trained: {results['convergence_info']['is_trained']}")

        print("\nFinal connectivity parameters:")
        conn_params = results['connectivity_params']
        for key, val in conn_params.items():
            print(f"  {key}: {val:.6f}")

        print("\nFinal noise parameters:")
        noise_params = results['stage2']['optimized_params']
        for key, val in noise_params.items():
            print(f"  {key}: {val:.4f}")

        return True

    except Exception as e:
        print(f"\n✗ Full training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("ECHEVESTE2020 + GSM COMPLETE TRAINING TESTS")
    print("=" * 70)

    results = []

    # Test 1: GSM posterior
    results.append(("GSM Posterior", test_gsm_posterior_computation()))

    # Test 2: Training targets
    results.append(("GSM Targets", test_gsm_training_targets()))

    # Test 3: Stage 1 with GSM
    results.append(("Stage 1 + GSM", test_stage1_with_real_gsm()))

    # Test 4: Stage 2 with GSM
    results.append(("Stage 2 + GSM", test_stage2_with_real_gsm()))

    # Test 5: Full training
    results.append(("Full Training", test_full_two_stage_training()))

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
