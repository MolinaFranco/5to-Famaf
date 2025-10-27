#!/usr/bin/env python3
"""
Investigar por qué nuestras actividades son ~12x más altas que el original.
Posibles causas:
1. Diferencias en escalas de variables (u vs r vs activity)
2. Diferencias en unidades de tiempo
3. Diferencias en funciones de activación
4. Diferencias en ruido
"""

import os
import sys
import numpy as np

# Configurar rutas
sys.path.append('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/SSN')
sys.path.append('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

# Cambiar al directorio original
os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/SSN')

# Importar código original
import methods as mt_original
from parameters import *

# Importar nuestra implementación
from skneuromsi.neural import Echeveste2020


def compare_activation_functions():
    """Comparar las funciones de activación."""
    print("=== COMPARACIÓN DE FUNCIONES DE ACTIVACIÓN ===")

    # Rango de valores de entrada
    u_test = np.linspace(-2, 10, 100)

    # Función original
    r_original = mt_original.get_r(u_test)

    # Nuestra función (extraer de integrador)
    ssn = Echeveste2020(seed=42)
    ssn.load_parameters()

    # Nuestra función de activación: r = k * [u]_+^n
    k_ours = ssn._integrator.f.k
    n_ours = ssn._integrator.f.n

    print(f"Parámetros de activación:")
    print(f"  Original k: {k}, n: {n}")
    print(f"  Nuestro k: {k_ours}, n: {n_ours}")

    # Calcular nuestra función de activación
    r_ours = k_ours * np.maximum(0, u_test) ** n_ours

    # Comparar valores para algunos puntos clave
    test_points = [0, 1, 2, 5, 10]
    print(f"\nComparación punto a punto:")
    print(f"  u\tOriginal\tNuestro\tRatio")
    for u_val in test_points:
        r_orig_val = mt_original.get_r(np.array([u_val]))[0]
        r_our_val = k_ours * max(0, u_val) ** n_ours
        ratio = r_our_val / r_orig_val if r_orig_val != 0 else np.inf

        print(f"  {u_val:.1f}\t{r_orig_val:.3f}\t\t{r_our_val:.3f}\t\t{ratio:.2f}")

    # Verificar si las funciones son idénticas
    r_ours_at_test = []
    for u_val in u_test:
        r_ours_at_test.append(k_ours * max(0, u_val) ** n_ours)
    r_ours_at_test = np.array(r_ours_at_test)

    max_diff = np.abs(r_original - r_ours_at_test).max()
    print(f"\nDiferencia máxima entre funciones: {max_diff:.8f}")

    if max_diff < 1e-10:
        print(f"✅ Funciones de activación son idénticas")
        return True
    else:
        print(f"❌ Funciones de activación son diferentes")
        return False


def compare_variable_scales():
    """Comparar las escalas de las variables."""
    print(f"\n=== COMPARACIÓN DE ESCALAS DE VARIABLES ===")

    # Configurar mismo estado inicial
    np.random.seed(42)

    # Cargar parámetros originales
    W = np.loadtxt('parameter_files/w_learn')
    Sigma_eta = np.loadtxt('parameter_files/sigma_eta_learn')
    h = np.loadtxt('parameter_files/h_true_0_learn')
    mu_0 = np.loadtxt('parameter_files/mu_evolved_net_0')
    Sigma_0 = np.loadtxt('parameter_files/sigma_evolved_net_0')

    # Estado inicial original
    u_orig = np.random.multivariate_normal(mean=mu_0, cov=Sigma_0)
    eta_orig = np.random.multivariate_normal(mean=np.zeros(N), cov=Sigma_eta)

    # Calcular r original
    r_orig = mt_original.get_r(u_orig)

    print(f"Variables originales (paso inicial):")
    print(f"  u: range=[{u_orig.min():.3f}, {u_orig.max():.3f}], mean={u_orig.mean():.3f}")
    print(f"  r: range=[{r_orig.min():.3f}, {r_orig.max():.3f}], mean={r_orig.mean():.3f}")
    print(f"  eta: range=[{eta_orig.min():.3f}, {eta_orig.max():.3f}], mean={eta_orig.mean():.3f}")
    print(f"  h: valor={h[0]:.6f}")

    # Ahora simular algunos pasos del original para ver evolución
    u = u_orig.copy()
    eta = eta_orig.copy()

    # Preparar ruido O-U
    L = np.linalg.cholesky(Sigma_eta)
    eps_1 = 1.0 - dt / tau_n
    eps_2 = np.sqrt(2.0 * dt / tau_n)

    for step in range(10):
        r = mt_original.get_r(u)

        # Evolucionar
        np.random.seed(42 + step)
        white_noise = L @ np.random.normal(0.0, 1.0, N)
        eta = eps_1 * eta + eps_2 * white_noise
        u = mt_original.new_u(u, r, h, W, eta)

    print(f"\nVariables originales (después de 10 pasos):")
    print(f"  u: range=[{u.min():.3f}, {u.max():.3f}], mean={u.mean():.3f}")
    print(f"  r: range=[{r.min():.3f}, {r.max():.3f}], mean={r.mean():.3f}")

    # Ahora comparar con nuestro resultado
    print(f"\n--- Nuestras variables ---")

    # Ejecutar nuestra simulación
    ssn = Echeveste2020(seed=42, time_range=(0, 2.0), time_res=0.2)  # Solo 10 pasos
    ssn.load_parameters()

    try:
        result = ssn.run(
            stimulus_contrast=h[0],
            simulation_time=2.0  # 10 pasos
        )

        data_exc = result.get_modes()['excitatory']
        data_inh = result.get_modes()['inhibitory']

        print(f"Nuestras variables (después de 10 pasos):")
        print(f"  activity_e: range=[{data_exc.min():.3f}, {data_exc.max():.3f}], mean={data_exc.mean():.3f}")
        print(f"  activity_i: range=[{data_inh.min():.3f}, {data_inh.max():.3f}], mean={data_inh.mean():.3f}")

        # AQUÍ ESTÁ LA CLAVE: ¿Qué representa exactamente "activity"?
        print(f"\n🔍 INVESTIGACIÓN CLAVE:")
        print(f"   ¿Nuestro 'activity' corresponde a 'u' o 'r' del original?")

        # Ratios
        our_max = max(data_exc.max(), data_inh.max())
        orig_u_max = np.abs(u).max()
        orig_r_max = np.abs(r).max()

        ratio_vs_u = our_max / orig_u_max
        ratio_vs_r = our_max / orig_r_max

        print(f"   Nuestro max: {our_max:.3f}")
        print(f"   Original u max: {orig_u_max:.3f} -> ratio: {ratio_vs_u:.2f}")
        print(f"   Original r max: {orig_r_max:.3f} -> ratio: {ratio_vs_r:.2f}")

        if abs(ratio_vs_r - 1.0) < abs(ratio_vs_u - 1.0):
            print(f"   ✅ Nuestro 'activity' parece corresponder más a 'r' (firing rate)")
        else:
            print(f"   ✅ Nuestro 'activity' parece corresponder más a 'u' (membrane potential)")

    except Exception as e:
        print(f"❌ Error en nuestra simulación: {e}")


def investigate_time_scaling():
    """Investigar diferencias en escalas de tiempo."""
    print(f"\n=== INVESTIGACIÓN DE ESCALAS DE TIEMPO ===")

    print(f"Configuración temporal original:")
    print(f"  dt: {dt}s = {dt*1000}ms")
    print(f"  tau_e: {tau_e}s = {tau_e*1000}ms")
    print(f"  tau_i: {tau_i}s = {tau_i*1000}ms")
    print(f"  tau_n: {tau_n}s = {tau_n*1000}ms")

    ssn = Echeveste2020()
    print(f"\nConfiguración temporal nuestra:")
    print(f"  time_res: {ssn.time_res}ms")
    print(f"  tau_e: {ssn._integrator.f.tau_e}ms")
    print(f"  tau_i: {ssn._integrator.f.tau_i}ms")

    # Verificar si hay diferencias en unidades
    tau_e_diff = abs(ssn._integrator.f.tau_e - tau_e*1000)
    tau_i_diff = abs(ssn._integrator.f.tau_i - tau_i*1000)

    print(f"\nDiferencias:")
    print(f"  tau_e: {tau_e_diff:.6f}ms")
    print(f"  tau_i: {tau_i_diff:.6f}ms")

    if tau_e_diff < 1e-6 and tau_i_diff < 1e-6:
        print(f"✅ Parámetros temporales son idénticos")
        return True
    else:
        print(f"❌ Parámetros temporales son diferentes")
        return False


def main():
    """Función principal."""
    print("INVESTIGACIÓN DE DIFERENCIAS DE ESCALA")
    print("=" * 60)

    # 1. Comparar funciones de activación
    activation_identical = compare_activation_functions()

    # 2. Comparar escalas de variables
    compare_variable_scales()

    # 3. Investigar escalas de tiempo
    time_identical = investigate_time_scaling()

    print(f"\n" + "=" * 60)
    print(f"DIAGNÓSTICO:")
    print(f"✅ Funciones de activación idénticas: {activation_identical}")
    print(f"✅ Parámetros temporales idénticos: {time_identical}")

    if activation_identical and time_identical:
        print(f"\n🔍 HIPÓTESIS PRINCIPAL:")
        print(f"   Las diferencias pueden estar en:")
        print(f"   1. Interpretación de variables (u vs r)")
        print(f"   2. Acumulación de diferencias numéricas pequeñas")
        print(f"   3. Diferencias en inicialización o ruido")
        print(f"   4. Diferencias en integración numérica")
        print(f"\n💡 Si las actividades están en rango consistente (~100-170),")
        print(f"   pueden ser simplemente una escala diferente pero ESTABLE.")
    else:
        print(f"\n❌ Hay diferencias fundamentales en la implementación")


if __name__ == "__main__":
    main()