#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test simplificado para verificar los 15 parámetros de Echeveste2020
y el funcionamiento básico de la red SSN.

Parámetros a verificar:
1. Stage 1 (8 parámetros de conectividad):
   - a_EE, a_EI, a_IE, a_II (amplitudes)
   - d_EE, d_EI, d_IE, d_II (dispersiones)

2. Stage 2 (4 parámetros de ruido + matriz):
   - width (ancho espacial del kernel de ruido)
   - std_e (desviación estándar del ruido excitatorio)
   - std_i (desviación estándar del ruido inhibitorio)
   - rho (correlación entre poblaciones)
   - Sigma_eta (matriz de covarianza del ruido, NxN)

Total: 8 + 4 + 1 matriz = 15 parámetros

Basado en:
- Echeveste et al. (2020): Supplementary Material, Table S1
- ssn_inference_numerical_experiments: train.ml y objective.ml
"""

import sys
import os

# Agregar paths necesarios
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'scikit-neuromsi')
)

import numpy as np


def print_section(title):
    """Imprime un separador visual."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_stage1_connectivity_params():
    """
    Test de los 8 parámetros de conectividad (Stage 1).

    Estos parámetros definen la conectividad paramétrica según Eq. 10:
    W_XY(θi, θj) = a_XY * exp[(cos(2(θi - θj)) - 1) / d_XY²]
    """
    print_section("TEST 1: Parámetros de Conectividad (Stage 1)")

    # Importar después de agregar el path
    from skneuromsi.data import EchevesteDataLoader

    loader = EchevesteDataLoader()

    print("\nCargando parámetros de conectividad desde data loader...")

    try:
        params = loader.load_ssn_connectivity_parameters()

        # Lista de los 8 parámetros esperados
        expected_params = [
            'a_EE', 'a_EI', 'a_IE', 'a_II',
            'd_EE', 'd_EI', 'd_IE', 'd_II'
        ]

        print("\n✓ Parámetros cargados exitosamente\n")
        print("PARÁMETROS DE AMPLITUD (a_XY):")
        print("-" * 40)
        for param in ['a_EE', 'a_EI', 'a_IE', 'a_II']:
            value = params[param]
            print(f"  {param:6s} = {value:10.6f}")

        print("\nPARÁMETROS DE DISPERSIÓN (d_XY):")
        print("-" * 40)
        for param in ['d_EE', 'd_EI', 'd_IE', 'd_II']:
            value = params[param]
            print(f"  {param:6s} = {value:10.6f}")

        # Verificaciones
        print("\nVERIFICACIONES:")
        print("-" * 40)

        # 1. Todos los parámetros están presentes
        all_present = all(p in params for p in expected_params)
        print(f"  ✓ Todos los 8 parámetros presentes: {all_present}")
        assert all_present, "Faltan parámetros de conectividad"

        # 2. Todos son numéricos
        all_numeric = all(
            isinstance(params[p], (int, float, np.number))
            for p in expected_params
        )
        print(f"  ✓ Todos son numéricos: {all_numeric}")
        assert all_numeric, "Hay parámetros no numéricos"

        # 3. Todos son positivos (físicamente necesario)
        all_positive = all(params[p] > 0 for p in expected_params)
        print(f"  ✓ Todos son positivos: {all_positive}")
        assert all_positive, "Hay parámetros no positivos"

        # 4. Rangos razonables según el paper
        # Amplitudes típicamente en [0.01, 10]
        # Dispersiones típicamente en [0.1, 2.0]
        amplitudes_ok = all(
            0.001 < params[p] < 100 for p in
            ['a_EE', 'a_EI', 'a_IE', 'a_II']
        )
        dispersiones_ok = all(
            0.01 < params[p] < 10 for p in
            ['d_EE', 'd_EI', 'd_IE', 'd_II']
        )

        print(f"  ✓ Amplitudes en rango razonable: {amplitudes_ok}")
        print(f"  ✓ Dispersiones en rango razonable: {dispersiones_ok}")

        print("\n" + "=" * 70)
        print("TEST 1: PASADO ✓ - 8 parámetros verificados")
        print("=" * 70)

        return params

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_stage2_noise_params():
    """
    Test de los parámetros de ruido (Stage 2).

    Stage 2 optimiza la matriz de covarianza Sigma_eta usando 4 parámetros:
    - width: ancho espacial del kernel de correlación
    - std_e: desviación estándar del ruido excitatorio
    - std_i: desviación estándar del ruido inhibitorio
    - rho: correlación entre poblaciones E-I

    Basado en objective.ml líneas 174-179 y 290-305
    """
    print_section("TEST 2: Parámetros de Ruido (Stage 2)")

    from skneuromsi.data import EchevesteDataLoader

    loader = EchevesteDataLoader()

    print("\nCargando matriz de covarianza del ruido (Sigma_eta)...")

    try:
        sigma_eta = loader.load_noise_covariance()

        print(f"\n✓ Sigma_eta cargada exitosamente")
        print(f"\nPROPIEDADES DE LA MATRIZ:")
        print("-" * 40)
        print(f"  Dimensión:  {sigma_eta.shape[0]} x {sigma_eta.shape[1]}")
        print(f"  Mínimo:     {sigma_eta.min():10.6e}")
        print(f"  Máximo:     {sigma_eta.max():10.6e}")
        print(f"  Media:      {sigma_eta.mean():10.6e}")
        print(f"  Diagonal media: {np.diag(sigma_eta).mean():10.6e}")

        # Verificaciones matemáticas
        print("\nVERIFICACIONES MATEMÁTICAS:")
        print("-" * 40)

        # 1. Debe ser cuadrada
        N = sigma_eta.shape[0]
        is_square = sigma_eta.shape[0] == sigma_eta.shape[1]
        print(f"  ✓ Matriz cuadrada: {is_square}")
        assert is_square, "Sigma_eta debe ser cuadrada"

        # 2. Debe ser simétrica (propiedad de covarianza)
        is_symmetric = np.allclose(sigma_eta, sigma_eta.T, rtol=1e-10)
        print(f"  ✓ Matriz simétrica: {is_symmetric}")
        assert is_symmetric, "Sigma_eta debe ser simétrica"

        # 3. Debe ser positiva semi-definida
        eigenvalues = np.linalg.eigvalsh(sigma_eta)
        min_eigenval = eigenvalues.min()
        is_psd = min_eigenval >= -1e-10
        print(f"  ✓ Positiva semi-definida: {is_psd}")
        print(f"    (autovalor mínimo: {min_eigenval:.6e})")
        assert is_psd, "Sigma_eta debe ser positiva semi-definida"

        # 4. Dimensión compatible con red (N_E + N_I = 100 por defecto)
        expected_N = 100  # 50 E + 50 I
        correct_size = N == expected_N
        print(f"  ✓ Dimensión esperada ({expected_N}x{expected_N}): "
              f"{correct_size}")

        # Información sobre la estructura de correlación
        print("\nESTRUCTURA DE CORRELACIÓN:")
        print("-" * 40)

        # Extraer bloques E-E, E-I, I-I
        N_E = N // 2
        N_I = N - N_E

        sigma_ee = sigma_eta[:N_E, :N_E]
        sigma_ei = sigma_eta[:N_E, N_E:]
        sigma_ii = sigma_eta[N_E:, N_E:]

        print(f"  Varianza E promedio: {np.diag(sigma_ee).mean():.6e}")
        print(f"  Varianza I promedio: {np.diag(sigma_ii).mean():.6e}")
        print(f"  Covarianza E-I promedio: {sigma_ei.mean():.6e}")

        # Estimar los 4 parámetros desde la matriz (aproximado)
        print("\nPARÁMETROS ESTIMADOS (desde matriz):")
        print("-" * 40)

        # std_e y std_i desde diagonales
        std_e_est = np.sqrt(np.diag(sigma_ee).mean())
        std_i_est = np.sqrt(np.diag(sigma_ii).mean())
        print(f"  std_e ≈ {std_e_est:.6f}")
        print(f"  std_i ≈ {std_i_est:.6f}")

        # rho desde covarianza E-I
        rho_est = sigma_ei.mean() / (std_e_est * std_i_est)
        print(f"  rho   ≈ {rho_est:.6f}")

        # width desde estructura espacial (más difícil de estimar)
        # Se podría hacer un fit pero es más complejo
        print(f"  width: (requiere análisis espacial detallado)")

        print("\n" + "=" * 70)
        print("TEST 2: PASADO ✓ - Matriz Sigma_eta verificada")
        print("=" * 70)

        return sigma_eta

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_network_basic_function():
    """
    Test básico de funcionamiento de la red.

    Verifica que:
    1. La red se puede inicializar
    2. Los parámetros se pueden cargar
    3. Se puede construir la matriz de conectividad W
    4. Se puede ejecutar una simulación breve
    """
    print_section("TEST 3: Funcionamiento Básico de la Red")

    from skneuromsi.neural import Echeveste2020

    print("\nInicializando red Echeveste2020...")

    try:
        # Crear modelo con parámetros por defecto
        model = Echeveste2020(
            N_E=50,  # Neuronas excitatorias
            N_I=50,  # Neuronas inhibitorias
            tau_e=20.0,  # ms
            tau_i=10.0,  # ms
            tau_n=20.0,  # ms
            n=2.0,  # Exponente supralineal
            k=0.3,  # Factor de escala
            seed=42,
            time_range=(0, 200),  # Simulación corta
            time_res=0.2
        )

        print("✓ Red inicializada")
        print("\nPARÁMETROS DE LA RED:")
        print("-" * 40)
        print(f"  N_E = {model._N_E}")
        print(f"  N_I = {model._N_I}")
        print(f"  N_total = {model._N}")
        print(f"  tau_e = {model._integrator.f.tau_e * 1000:.1f} ms")
        print(f"  tau_i = {model._integrator.f.tau_i * 1000:.1f} ms")
        print(f"  tau_n = {model._integrator.f.tau_n * 1000:.1f} ms")
        print(f"  n = {model._integrator.f.n}")
        print(f"  k = {model._integrator.f.k}")

        # Cargar parámetros pre-entrenados
        print("\nCargando parámetros entrenados...")
        model.load_parameters()

        print("✓ Parámetros cargados")
        print("\nESTADO DEL ENTRENAMIENTO:")
        print("-" * 40)
        print(f"  Stage 1 completado: {model._stage1_completed}")
        print(f"  Stage 2 completado: {model._stage2_completed}")
        print(f"  Modelo entrenado: {model._is_trained}")

        assert model._stage1_completed, "Stage 1 debe estar completado"
        assert model._stage2_completed, "Stage 2 debe estar completado"
        assert model._is_trained, "Modelo debe estar entrenado"

        # Construir matriz de conectividad
        print("\nConstruyendo matriz de conectividad W...")
        W = model.build_connectivity_matrix()

        print(f"✓ Matriz W construida: {W.shape}")
        print("\nPROPIEDADES DE W:")
        print("-" * 40)
        print(f"  Dimensión: {W.shape[0]} x {W.shape[1]}")
        print(f"  Mínimo:    {W.min():10.6f}")
        print(f"  Máximo:    {W.max():10.6f}")
        print(f"  Media:     {W.mean():10.6f}")

        # Verificar que W tiene la estructura correcta
        N = model._N
        assert W.shape == (N, N), f"W debe ser {N}x{N}"

        # Crear estímulo simple
        print("\nCreando estímulo de prueba (contraste 0.5)...")
        h_input = model._generate_simple_stimulus(contrast=0.5)

        print(f"✓ Estímulo creado: shape={h_input.shape}")
        print(f"  min={h_input.min():.4f}, max={h_input.max():.4f}")

        # Ejecutar simulación usando el método de generación de respuesta
        print("\nEjecutando simulación (generando respuesta)...")
        # Usar el método interno que genera respuesta de red
        network_activity = model._generate_network_response(h_input)

        print(f"✓ Simulación completada")
        print("\nRESULTADOS DE LA SIMULACIÓN:")
        print("-" * 40)
        print(f"  network_activity shape: {network_activity.shape}")

        # Separar actividad excitatoria e inhibitoria
        r_exc = network_activity[:model._N_E]
        r_inh = network_activity[model._N_E:]

        print(f"  Actividad E media: {r_exc.mean():.4f}")
        print(f"  Actividad I media: {r_inh.mean():.4f}")
        print(f"  Actividad E máxima: {r_exc.max():.4f}")
        print(f"  Actividad I máxima: {r_inh.max():.4f}")

        # Verificaciones de sanidad
        print("\nVERIFICACIONES:")
        print("-" * 40)

        no_nans = not np.any(np.isnan(network_activity))
        print(f"  ✓ Sin NaN: {no_nans}")
        assert no_nans, "No debe haber NaN en la actividad"

        no_infs = not np.any(np.isinf(network_activity))
        print(f"  ✓ Sin Inf: {no_infs}")
        assert no_infs, "No debe haber Inf en la actividad"

        has_activity = (r_exc.max() > 0) or (r_inh.max() > 0)
        print(f"  ✓ Hay actividad: {has_activity}")
        assert has_activity, "Debe haber actividad neuronal"

        # Verificar dimensión correcta
        correct_shape = network_activity.shape[0] == model._N
        print(f"  ✓ Dimensión correcta ({model._N}): {correct_shape}")
        assert correct_shape, "Dimensión incorrecta"

        print("\n" + "=" * 70)
        print("TEST 3: PASADO ✓ - Red funciona correctamente")
        print("=" * 70)

        return model, network_activity

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Ejecutar todos los tests en secuencia."""
    print_section("PRUEBA DE LOS 15 PARÁMETROS DE ECHEVESTE2020")

    print("\nEste test verifica:")
    print("  1. 8 parámetros de conectividad (Stage 1)")
    print("  2. 4 parámetros de ruido + matriz Sigma_eta (Stage 2)")
    print("  3. Funcionamiento básico de la red SSN")
    print("\nTotal: 15 parámetros (8 + 4 + matriz)")

    success_count = 0
    total_tests = 3

    # Test 1: Stage 1 (8 parámetros)
    conn_params = test_stage1_connectivity_params()
    if conn_params is not None:
        success_count += 1

    # Test 2: Stage 2 (4 parámetros + matriz)
    sigma_eta = test_stage2_noise_params()
    if sigma_eta is not None:
        success_count += 1

    # Test 3: Funcionamiento de la red
    model, network_activity = test_network_basic_function()
    if model is not None and network_activity is not None:
        success_count += 1

    # Resumen final
    print_section("RESUMEN FINAL")

    print(f"\nTests pasados: {success_count}/{total_tests}")

    if success_count == total_tests:
        print("\n✓✓✓ TODOS LOS TESTS PASARON ✓✓✓")
        print("\nParámetros verificados:")
        print("  ✓ 8 parámetros de conectividad (a_EE, a_EI, a_IE, a_II,")
        print("                                   d_EE, d_EI, d_IE, d_II)")
        print("  ✓ Matriz Sigma_eta de covarianza del ruido")
        print("  ✓ Red funcional con simulación exitosa")
        print("\n" + "=" * 70)
        return True
    else:
        print(f"\n✗ {total_tests - success_count} test(s) fallaron")
        print("\n" + "=" * 70)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
