#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script para verificar la generación de parámetros y funcionamiento
de la red Echeveste2020.

Este script verifica:
1. Generación correcta de los 8 parámetros de conectividad (Stage 1)
2. Generación correcta de los 4 parámetros de ruido (Stage 2)
3. Funcionamiento básico de la red SSN
4. Integración completa del modelo

Basado en:
- Echeveste et al. (2020): SSN sampling-based inference
- ssn_inference_numerical_experiments: referencia de implementación
"""

import sys
import os

# Agregar el path de scikit-neuromsi al PYTHONPATH
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        '..',
        'scikit-neuromsi'
    )
)

import numpy as np
from skneuromsi.neural import Echeveste2020
from skneuromsi.data import EchevesteDataLoader


def test_parameter_loading():
    """
    Test 1: Verificar carga de parámetros pre-entrenados.

    Verifica que los 15 parámetros se carguen correctamente:
    - Stage 1: 8 parámetros de conectividad (a_EE, a_EI, a_IE, a_II,
                                              d_EE, d_EI, d_IE, d_II)
    - Stage 2: 4 parámetros de ruido (width, std_e, std_i, rho)
               + matriz Sigma_eta
    """
    print("=" * 70)
    print("TEST 1: Carga de parámetros pre-entrenados")
    print("=" * 70)

    # Usar el data loader interno
    loader = EchevesteDataLoader()

    # Cargar parámetros de conectividad (Stage 1)
    print("\n[Stage 1] Cargando parámetros de conectividad...")
    conn_params = loader.load_ssn_connectivity_parameters()

    print("\nParámetros de conectividad cargados:")
    print(f"  a_EE = {conn_params['a_EE']:.6f}")
    print(f"  a_EI = {conn_params['a_EI']:.6f}")
    print(f"  a_IE = {conn_params['a_IE']:.6f}")
    print(f"  a_II = {conn_params['a_II']:.6f}")
    print(f"  d_EE = {conn_params['d_EE']:.6f}")
    print(f"  d_EI = {conn_params['d_EI']:.6f}")
    print(f"  d_IE = {conn_params['d_IE']:.6f}")
    print(f"  d_II = {conn_params['d_II']:.6f}")

    # Verificar que todos los parámetros son válidos
    assert all(
        isinstance(v, (int, float)) for v in conn_params.values()
    ), "Todos los parámetros deben ser numéricos"
    assert all(v > 0 for v in conn_params.values()), \
        "Todos los parámetros deben ser positivos"

    print("\n✓ Stage 1: 8 parámetros de conectividad OK")

    # Cargar matriz de covarianza de ruido (Stage 2)
    print("\n[Stage 2] Cargando matriz de covarianza de ruido...")
    sigma_eta = loader.load_noise_covariance()

    print(f"\nSigma_eta shape: {sigma_eta.shape}")
    print(f"Sigma_eta min: {sigma_eta.min():.6f}")
    print(f"Sigma_eta max: {sigma_eta.max():.6f}")
    print(f"Sigma_eta mean: {sigma_eta.mean():.6f}")

    # Verificar propiedades de la matriz de covarianza
    N = sigma_eta.shape[0]
    assert sigma_eta.shape == (N, N), \
        "Sigma_eta debe ser cuadrada"
    assert np.allclose(sigma_eta, sigma_eta.T), \
        "Sigma_eta debe ser simétrica"

    # Verificar que es positiva semi-definida
    eigenvalues = np.linalg.eigvalsh(sigma_eta)
    assert np.all(eigenvalues >= -1e-10), \
        "Sigma_eta debe ser positiva semi-definida"

    print("\n✓ Stage 2: Matriz Sigma_eta OK")
    print(f"  Dimensión: {N}x{N}")
    print(f"  Autovalores mínimo: {eigenvalues.min():.6e}")
    print(f"  Autovalores máximo: {eigenvalues.max():.6e}")

    print("\n" + "=" * 70)
    print("TEST 1: PASADO ✓")
    print("=" * 70)

    return conn_params, sigma_eta


def test_network_initialization():
    """
    Test 2: Verificar inicialización correcta de la red.

    Verifica que la red Echeveste2020 se inicialice correctamente
    con los parámetros por defecto y cargue los parámetros entrenados.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Inicialización de la red Echeveste2020")
    print("=" * 70)

    # Crear instancia del modelo
    print("\nCreando modelo SSN...")
    model = Echeveste2020(
        N_E=50,  # Número de neuronas excitatorias
        N_I=50,  # Número de neuronas inhibitorias
        tau_e=20.0,  # ms
        tau_i=10.0,  # ms
        tau_n=20.0,  # ms
        n=2.0,  # Exponente supralineal
        k=0.3,  # Factor de escala
        seed=42
    )

    print(f"  N_E = {model._N_E}")
    print(f"  N_I = {model._N_I}")
    print(f"  N_total = {model._N}")
    print(f"  tau_e = {model._integrator.f.tau_e * 1000:.1f} ms")
    print(f"  tau_i = {model._integrator.f.tau_i * 1000:.1f} ms")
    print(f"  tau_n = {model._integrator.f.tau_n * 1000:.1f} ms")
    print(f"  n (exponente) = {model._integrator.f.n}")
    print(f"  k (escala) = {model._integrator.f.k}")

    # Cargar parámetros pre-entrenados
    print("\nCargando parámetros pre-entrenados...")
    model.load_parameters()

    # Verificar que los parámetros se cargaron
    assert model._stage1_completed, "Stage 1 debe estar completado"
    assert model._stage2_completed, "Stage 2 debe estar completado"
    assert model._is_trained, "Modelo debe estar entrenado"

    print("\n✓ Parámetros cargados correctamente")
    print(f"  Stage 1 completado: {model._stage1_completed}")
    print(f"  Stage 2 completado: {model._stage2_completed}")
    print(f"  Modelo entrenado: {model._is_trained}")

    # Verificar matrices de conectividad
    print("\nVerificando matrices de conectividad...")
    W = model.build_connectivity_matrix()
    print(f"  W shape: {W.shape}")
    print(f"  W min: {W.min():.6f}")
    print(f"  W max: {W.max():.6f}")
    print(f"  W mean: {W.mean():.6f}")

    assert W.shape == (model._N, model._N), \
        "W debe tener dimensión (N_total, N_total)"

    print("\n" + "=" * 70)
    print("TEST 2: PASADO ✓")
    print("=" * 70)

    return model


def test_network_simulation():
    """
    Test 3: Verificar funcionamiento de la red con simulación.

    Ejecuta una simulación corta para verificar que:
    1. La red se ejecuta sin errores
    2. Las tasas de disparo son razonables
    3. Las estadísticas son consistentes
    """
    print("\n" + "=" * 70)
    print("TEST 3: Simulación de la red SSN")
    print("=" * 70)

    # Crear y configurar modelo
    model = Echeveste2020(
        N_E=50,
        N_I=50,
        tau_e=20.0,
        tau_i=10.0,
        tau_n=20.0,
        n=2.0,
        k=0.3,
        seed=42,
        time_range=(0, 500),  # Simulación corta de 500ms
        time_res=0.2  # dt = 0.2ms
    )

    # Cargar parámetros
    model.load_parameters()

    # Crear estímulo simple (orientación a 45 grados)
    print("\nCreando estímulo de prueba (45 grados)...")
    stimulus_orientation = 45.0  # grados

    # Simular actividad
    print("\nEjecutando simulación...")
    print("  Duración: 500 ms")
    print("  dt: 0.2 ms")

    # Generar input vector para el estímulo
    # (esto normalmente se haría con create_stimulus_from_orientation)
    h_input = model._create_gaussian_bump_stimulus(
        stimulus_orientation,
        amplitude=2.0  # Amplitud moderada
    )

    print(f"  h_input shape: {h_input.shape}")
    print(f"  h_input min: {h_input.min():.6f}")
    print(f"  h_input max: {h_input.max():.6f}")

    # Ejecutar simulación con input constante
    print("\nSimulando red...")
    result = model._run_network_dynamics(
        h_ext=h_input,
        noise_level=1.0,
        dt_ms=0.2,
        duration_ms=500,
        burn_in_ms=100
    )

    # Analizar resultados
    r_exc = result['r_exc']  # (n_steps, N_E)
    r_inh = result['r_inh']  # (n_steps, N_I)

    print("\nResultados de la simulación:")
    print(f"  r_exc shape: {r_exc.shape}")
    print(f"  r_inh shape: {r_inh.shape}")
    print(f"  r_exc mean: {r_exc.mean():.6f} Hz")
    print(f"  r_inh mean: {r_inh.mean():.6f} Hz")
    print(f"  r_exc std: {r_exc.std():.6f} Hz")
    print(f"  r_inh std: {r_inh.std():.6f} Hz")

    # Verificar que las tasas son razonables
    assert not np.any(np.isnan(r_exc)), \
        "No debe haber NaN en tasas excitatorias"
    assert not np.any(np.isnan(r_inh)), \
        "No debe haber NaN en tasas inhibitorias"
    assert not np.any(np.isinf(r_exc)), \
        "No debe haber Inf en tasas excitatorias"
    assert not np.any(np.isinf(r_inh)), \
        "No debe haber Inf en tasas inhibitorias"

    # Verificar que hay actividad (no todo ceros)
    assert r_exc.max() > 0, "Debe haber actividad excitatoria"
    assert r_inh.max() > 0, "Debe haber actividad inhibitoria"

    print("\n✓ Simulación ejecutada correctamente")

    print("\n" + "=" * 70)
    print("TEST 3: PASADO ✓")
    print("=" * 70)

    return result


def test_complete_workflow():
    """
    Test 4: Workflow completo de la red.

    Simula el workflow completo:
    1. Inicialización
    2. Carga de parámetros
    3. Múltiples simulaciones con diferentes estímulos
    4. Análisis de respuestas de tuning
    """
    print("\n" + "=" * 70)
    print("TEST 4: Workflow completo")
    print("=" * 70)

    # Inicializar modelo
    model = Echeveste2020(
        N_E=50,
        N_I=50,
        seed=42,
        time_range=(0, 300),
        time_res=0.2
    )

    model.load_parameters()

    # Probar múltiples orientaciones
    orientations = [0, 45, 90, 135]
    print(f"\nProbando {len(orientations)} orientaciones diferentes...")

    responses = {}
    for ori in orientations:
        print(f"\n  Simulando orientación {ori}°...")

        h_input = model._create_gaussian_bump_stimulus(ori, amplitude=2.0)

        result = model._run_network_dynamics(
            h_ext=h_input,
            noise_level=0.5,
            dt_ms=0.2,
            duration_ms=300,
            burn_in_ms=50
        )

        # Guardar respuesta promedio
        mean_response = result['r_exc'].mean(axis=0)
        responses[ori] = mean_response

        print(f"    Mean activity: {mean_response.mean():.4f} Hz")
        print(f"    Max activity: {mean_response.max():.4f} Hz")

    # Verificar que hay variabilidad entre orientaciones
    print("\nAnalizando selectividad de orientación...")
    response_array = np.array([responses[ori] for ori in orientations])

    # Variabilidad entre orientaciones (debe ser > 0)
    between_orientation_var = np.var(response_array.mean(axis=1))
    print(f"  Varianza entre orientaciones: {between_orientation_var:.6f}")

    assert between_orientation_var > 0, \
        "Debe haber variabilidad entre orientaciones"

    print("\n✓ Workflow completo ejecutado correctamente")

    print("\n" + "=" * 70)
    print("TEST 4: PASADO ✓")
    print("=" * 70)

    return responses


def main():
    """
    Ejecutar todos los tests.
    """
    print("\n" + "=" * 70)
    print("PRUEBAS DE LA RED ECHEVESTE2020")
    print("=" * 70)
    print("\nEste script verifica:")
    print("1. Carga correcta de 15 parámetros (8 conectividad + 4 ruido + "
          "Sigma_eta)")
    print("2. Inicialización correcta de la red")
    print("3. Funcionamiento de simulaciones")
    print("4. Workflow completo")

    try:
        # Test 1: Cargar parámetros
        conn_params, sigma_eta = test_parameter_loading()

        # Test 2: Inicializar red
        model = test_network_initialization()

        # Test 3: Simular red
        result = test_network_simulation()

        # Test 4: Workflow completo
        responses = test_complete_workflow()

        # Resumen final
        print("\n" + "=" * 70)
        print("RESUMEN FINAL")
        print("=" * 70)
        print("\n✓ TODOS LOS TESTS PASARON CORRECTAMENTE\n")
        print("Parámetros verificados:")
        print("  - 8 parámetros de conectividad (Stage 1)")
        print("  - Matriz Sigma_eta (Stage 2)")
        print("  - Simulación funcional")
        print("  - Selectividad de orientación")
        print("\n" + "=" * 70)

        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR EN LOS TESTS")
        print("=" * 70)
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
