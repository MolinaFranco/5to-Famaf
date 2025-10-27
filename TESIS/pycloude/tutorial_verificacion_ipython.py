#!/usr/bin/env python3
"""
Tutorial de Verificación en iPython: Código Original vs Nuestra Implementación

Este tutorial te guía paso a paso para verificar que ambas implementaciones
producen los mismos resultados. Copia y pega cada bloque en iPython.

IMPORTANTE: Ejecuta cada bloque por separado y verifica los resultados.
"""

# =============================================================================
# BLOQUE 1: CONFIGURACIÓN INICIAL Y IMPORTS
# =============================================================================
print("=== BLOQUE 1: CONFIGURACIÓN INICIAL ===")

# Imports necesarios
import os
import sys
import numpy as np

# Configurar rutas
sys.path.append('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/SSN')
sys.path.append('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

# Cambiar al directorio del código original
original_dir = '/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/SSN'
os.chdir(original_dir)

print(f"Directorio actual: {os.getcwd()}")
print("✅ Configuración inicial completada")


# =============================================================================
# BLOQUE 2: IMPORTAR CÓDIGO ORIGINAL
# =============================================================================
print("\n=== BLOQUE 2: IMPORTAR CÓDIGO ORIGINAL ===")

# Importar código original de Echeveste
import methods as mt_original
from parameters import *

print(f"Parámetros del código original:")
print(f"  N = {N} (total neuronas)")
print(f"  N_exc = {N_exc}, N_inh = {N_inh}")
print(f"  k = {k}, n = {n}")
print(f"  tau_e = {tau_e}s, tau_i = {tau_i}s")
print(f"  dt = {dt}s")
print(f"  tau_n = {tau_n}s (ruido correlacionado)")
print("✅ Código original importado correctamente")


# =============================================================================
# BLOQUE 3: IMPORTAR NUESTRA IMPLEMENTACIÓN
# =============================================================================
print("\n=== BLOQUE 3: IMPORTAR NUESTRA IMPLEMENTACIÓN ===")

from skneuromsi.neural import Echeveste2020

# Crear instancia con parámetros idénticos
ssn = Echeveste2020(N_E=N_exc, N_I=N_inh, seed=42)

# Verificar parámetros
dt_ours = ssn._time_res / 1000.0  # Convertir ms a segundos
print(f"Parámetros de nuestra implementación:")
print(f"  N = {ssn._N}")
print(f"  N_E = {ssn._N_E}, N_I = {ssn._N_I}")
print(f"  k = {ssn._integrator.f.k}, n = {ssn._integrator.f.n}")
print(f"  tau_e = {ssn._integrator.f.tau_e}s, tau_i = {ssn._integrator.f.tau_i}s")
print(f"  dt = {dt_ours}s")

# Verificar coincidencias
params_match = (
    ssn._N == N and
    ssn._N_E == N_exc and
    ssn._N_I == N_inh and
    abs(ssn._integrator.f.k - k) < 1e-10 and
    abs(ssn._integrator.f.n - n) < 1e-10 and
    abs(ssn._integrator.f.tau_e - tau_e) < 1e-10 and
    abs(ssn._integrator.f.tau_i - tau_i) < 1e-10 and
    abs(dt_ours - dt) < 1e-10
)

print(f"✅ Parámetros coinciden: {params_match}")


# =============================================================================
# BLOQUE 4: CARGAR PARÁMETROS PRE-ENTRENADOS DEL CÓDIGO ORIGINAL
# =============================================================================
print("\n=== BLOQUE 4: CARGAR PARÁMETROS PRE-ENTRENADOS ===")

# Cargar matriz de conectividad
W_original = np.loadtxt('parameter_files/w_learn')
print(f"Matriz W cargada: shape {W_original.shape}")
print(f"Rango de valores: [{W_original.min():.6f}, {W_original.max():.6f}]")

# Cargar covarianza de ruido
Sigma_eta = np.loadtxt('parameter_files/sigma_eta_learn')
print(f"Sigma_eta cargada: shape {Sigma_eta.shape}")

# Cargar input pattern (patrón 0 = uniforme)
h_original = np.loadtxt('parameter_files/h_true_0_learn')
print(f"Input h cargado: shape {h_original.shape}")
print(f"Valores: {h_original[:5]}... (uniforme = {h_original[0]:.6f})")

# Cargar condiciones iniciales pre-convergidas
mu_0 = np.loadtxt('parameter_files/mu_evolved_net_0')
Sigma_0 = np.loadtxt('parameter_files/sigma_evolved_net_0')
print(f"mu_0 cargado: range [{mu_0.min():.6f}, {mu_0.max():.6f}]")
print(f"Sigma_0 cargada: shape {Sigma_0.shape}")

print("✅ Parámetros pre-entrenados cargados correctamente")


# =============================================================================
# BLOQUE 5: VERIFICAR SIGNOS DE CONECTIVIDAD
# =============================================================================
print("\n=== BLOQUE 5: VERIFICAR SIGNOS DE CONECTIVIDAD ===")

# Analizar signos por bloques
ee_positive = (W_original[:50, :50] > 0).all()
ei_negative = (W_original[:50, 50:] < 0).all()
ie_positive = (W_original[50:, :50] > 0).all()
ii_negative = (W_original[50:, 50:] < 0).all()

print("Análisis de signos en matriz W original:")
print(f"  E→E: [{W_original[:50, :50].min():.6f}, {W_original[:50, :50].max():.6f}] - Todos positivos: {ee_positive}")
print(f"  E→I: [{W_original[:50, 50:].min():.6f}, {W_original[:50, 50:].max():.6f}] - Todos negativos: {ei_negative}")
print(f"  I→E: [{W_original[50:, :50].min():.6f}, {W_original[50:, :50].max():.6f}] - Todos positivos: {ie_positive}")
print(f"  I→I: [{W_original[50:, 50:].min():.6f}, {W_original[50:, 50:].max():.6f}] - Todos negativos: {ii_negative}")

signs_correct = ee_positive and ei_negative and ie_positive and ii_negative
print(f"✅ Signos de conectividad correctos: {signs_correct}")

# Verificar que nuestra implementación genera signos correctos
default_params = {
    'a_EE': 0.2, 'd_EE': 20.0,
    'a_EI': 0.1, 'd_EI': 20.0,
    'a_IE': 0.2, 'd_IE': 20.0,
    'a_II': 0.1, 'd_II': 20.0
}
W_ours = ssn.build_connectivity_matrix(connectivity_params=default_params)

our_signs_correct = (
    (W_ours[:50, :50] > 0).all() and    # E→E positivo
    (W_ours[:50, 50:] < 0).all() and    # E→I negativo
    (W_ours[50:, :50] > 0).all() and    # I→E positivo
    (W_ours[50:, 50:] < 0).all()        # I→I negativo
)
print(f"✅ Nuestros signos correctos: {our_signs_correct}")


# =============================================================================
# BLOQUE 6: SIMULACIÓN CON CÓDIGO ORIGINAL
# =============================================================================
print("\n=== BLOQUE 6: SIMULACIÓN CON CÓDIGO ORIGINAL ===")

# Fijar semilla para reproducibilidad
np.random.seed(12345)

# Generar condiciones iniciales
u0_original = np.random.multivariate_normal(mean=mu_0, cov=Sigma_0)
eta0_original = np.random.multivariate_normal(mean=np.zeros(N), cov=Sigma_eta)

print("Condiciones iniciales generadas:")
print(f"  u0 range: [{u0_original.min():.6f}, {u0_original.max():.6f}]")
print(f"  eta0 range: [{eta0_original.min():.6f}, {eta0_original.max():.6f}]")

# Evolución de la red
print("Evolucionando red con código original...")
u_final_original, eta_final_original = mt_original.network_evolution(
    W_original, h_original, u0_original, Sigma_eta, eta=eta0_original
)

print("Resultados de evolución original:")
print(f"  u_final range: [{u_final_original.min():.6f}, {u_final_original.max():.6f}]")
print(f"  eta_final range: [{eta_final_original.min():.6f}, {eta_final_original.max():.6f}]")

print("✅ Simulación original completada")


# =============================================================================
# BLOQUE 7: SIMULACIÓN CON NUESTRA IMPLEMENTACIÓN
# =============================================================================
print("\n=== BLOQUE 7: SIMULACIÓN CON NUESTRA IMPLEMENTACIÓN ===")

# Usar exactamente las mismas condiciones iniciales
np.random.seed(12345)  # Misma semilla
u0_ours = np.random.multivariate_normal(mean=mu_0, cov=Sigma_0)
eta0_ours = np.random.multivariate_normal(mean=np.zeros(N), cov=Sigma_eta)

# Verificar que las condiciones iniciales son idénticas
conditions_match = (
    np.allclose(u0_original, u0_ours, atol=1e-15) and
    np.allclose(eta0_original, eta0_ours, atol=1e-15)
)
print(f"✅ Condiciones iniciales idénticas: {conditions_match}")

# Simular con nuestra implementación usando parámetros originales
from skneuromsi.neural._echeveste2020 import SSNIntegrator
import brainpy as bp

# Crear integrador con parámetros exactos
integrator_model = SSNIntegrator(
    tau_e=tau_e, tau_i=tau_i, tau_n=tau_n, n=n, k=k
)

# Evolución manual usando nuestro integrador
print("Evolucionando red con nuestra implementación...")
max_steps = 50000
u_current = u0_ours.copy()
eta_current = eta0_ours.copy()

# Coeficientes para ruido O-U (idénticos al original)
L = np.linalg.cholesky(Sigma_eta)
eps_1 = 1.0 - dt / tau_n
eps_2 = np.sqrt(2.0 * dt / tau_n)

for step in range(max_steps):
    # Separar excitatorias e inhibitorias
    u_e = u_current[:50]
    u_i = u_current[50:]

    # Generar ruido idéntico al original
    white_noise = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
    eta_new = eps_1 * eta_current + eps_2 * white_noise

    # Calcular derivadas con nuestro integrador
    du_e_dt, du_i_dt = integrator_model(u_e, u_i, step * dt, W_original, h_original, eta_current)
    du_dt = np.concatenate([du_e_dt, du_i_dt])

    # Actualizar con Euler
    u_new = u_current + dt * du_dt

    # Verificar estabilidad
    if np.linalg.norm(u_new) > 1000:
        print(f"❌ Actividad explotando en paso {step}")
        break

    u_current = u_new
    eta_current = eta_new

print("Resultados de evolución nuestra:")
print(f"  u_final range: [{u_current.min():.6f}, {u_current.max():.6f}]")
print(f"  eta_final range: [{eta_current.min():.6f}, {eta_current.max():.6f}]")

print("✅ Simulación nuestra completada")


# =============================================================================
# BLOQUE 8: COMPARACIÓN DE EVOLUCIÓN TEMPORAL
# =============================================================================
print("\n=== BLOQUE 8: COMPARACIÓN DE EVOLUCIÓN TEMPORAL ===")

# Comparar resultados finales
u_evolution_match = np.allclose(u_final_original, u_current, atol=1e-10)
eta_evolution_match = np.allclose(eta_final_original, eta_current, atol=1e-10)

print("Comparación de evolución temporal:")
print(f"  Original u_final: [{u_final_original.min():.6f}, {u_final_original.max():.6f}]")
print(f"  Nuestro  u_final: [{u_current.min():.6f}, {u_current.max():.6f}]")
print(f"  ✅ Voltajes coinciden: {u_evolution_match}")

print(f"  Original eta_final: [{eta_final_original.min():.6f}, {eta_final_original.max():.6f}]")
print(f"  Nuestro  eta_final: [{eta_current.min():.6f}, {eta_current.max():.6f}]")
print(f"  ✅ Ruido coincide: {eta_evolution_match}")

if u_evolution_match and eta_evolution_match:
    print("🎉 ¡EVOLUCIÓN TEMPORAL IDÉNTICA!")
else:
    print("⚠️  Pequeñas diferencias en evolución temporal")


# =============================================================================
# BLOQUE 9: MUESTREO DE ACTIVIDAD - CÓDIGO ORIGINAL
# =============================================================================
print("\n=== BLOQUE 9: MUESTREO DE ACTIVIDAD - CÓDIGO ORIGINAL ===")

# Parámetros de muestreo (idénticos a activity_example.py)
total_time = 1.0  # 1 segundo
t_bet_samp = 5e-3  # cada 5ms
sample_size = int(total_time / t_bet_samp)  # 200 muestras
steps_bet_samp = int(t_bet_samp / dt)  # 25 pasos

print(f"Parámetros de muestreo:")
print(f"  total_time = {total_time}s")
print(f"  t_bet_samp = {t_bet_samp}s")
print(f"  sample_size = {sample_size}")
print(f"  steps_bet_samp = {steps_bet_samp}")

# Muestrear con código original
print("Muestreando con código original...")
(u_samples_original, _, _, _) = mt_original.network_sample(
    W_original, h_original, u_final_original, eta_final_original,
    sample_size, steps_bet_samp, Sigma_eta
)

r_samples_original = mt_original.get_r(u_samples_original)

print("Resultados de muestreo original:")
print(f"  u_samples shape: {u_samples_original.shape}")
print(f"  u_samples range: [{u_samples_original.min():.6f}, {u_samples_original.max():.6f}]")
print(f"  r_samples range: [{r_samples_original.min():.6f}, {r_samples_original.max():.6f}]")
print(f"  Neuronas E activas: {np.sum(u_samples_original[-1, :50] > 0)}/50")
print(f"  Neuronas I activas: {np.sum(u_samples_original[-1, 50:] > 0)}/50")

print("✅ Muestreo original completado")


# =============================================================================
# BLOQUE 10: MUESTREO DE ACTIVIDAD - NUESTRA IMPLEMENTACIÓN
# =============================================================================
print("\n=== BLOQUE 10: MUESTREO DE ACTIVIDAD - NUESTRA IMPLEMENTACIÓN ===")

# Muestrear con nuestra implementación
u_samples_ours = np.zeros((sample_size, N))
u_current_sample = u_current.copy()
eta_current_sample = eta_current.copy()

print("Muestreando con nuestra implementación...")

for sample_idx in range(sample_size):
    # Evolucionar steps_bet_samp pasos entre muestras
    for step in range(steps_bet_samp):
        u_e = u_current_sample[:50]
        u_i = u_current_sample[50:]

        # Generar ruido
        white_noise = L @ np.random.normal(loc=0.0, scale=1.0, size=N)
        eta_new = eps_1 * eta_current_sample + eps_2 * white_noise

        # Calcular derivadas
        du_e_dt, du_i_dt = integrator_model(u_e, u_i, 0.0, W_original, h_original, eta_current_sample)
        du_dt = np.concatenate([du_e_dt, du_i_dt])

        # Actualizar
        u_new = u_current_sample + dt * du_dt
        u_current_sample = u_new
        eta_current_sample = eta_new

    # Guardar muestra
    u_samples_ours[sample_idx] = u_current_sample

# Calcular firing rates
r_samples_ours = k * np.power(np.maximum(0, u_samples_ours), n)

print("Resultados de muestreo nuestro:")
print(f"  u_samples shape: {u_samples_ours.shape}")
print(f"  u_samples range: [{u_samples_ours.min():.6f}, {u_samples_ours.max():.6f}]")
print(f"  r_samples range: [{r_samples_ours.min():.6f}, {r_samples_ours.max():.6f}]")
print(f"  Neuronas E activas: {np.sum(u_samples_ours[-1, :50] > 0)}/50")
print(f"  Neuronas I activas: {np.sum(u_samples_ours[-1, 50:] > 0)}/50")

print("✅ Muestreo nuestro completado")


# =============================================================================
# BLOQUE 11: COMPARACIÓN FINAL DE MUESTREO
# =============================================================================
print("\n=== BLOQUE 11: COMPARACIÓN FINAL DE MUESTREO ===")

# Comparar estadísticas de muestreo
print("Comparación de muestreo de actividad:")
print(f"  Original u_samples: [{u_samples_original.min():.6f}, {u_samples_original.max():.6f}]")
print(f"  Nuestro  u_samples: [{u_samples_ours.min():.6f}, {u_samples_ours.max():.6f}]")

print(f"  Original r_samples: [{r_samples_original.min():.6f}, {r_samples_original.max():.6f}]")
print(f"  Nuestro  r_samples: [{r_samples_ours.min():.6f}, {r_samples_ours.max():.6f}]")

# Verificar actividad neuronal
e_active_original = np.sum(u_samples_original[-1, :50] > 0)
e_active_ours = np.sum(u_samples_ours[-1, :50] > 0)
i_active_original = np.sum(u_samples_original[-1, 50:] > 0)
i_active_ours = np.sum(u_samples_ours[-1, 50:] > 0)

print(f"  Neuronas E activas - Original: {e_active_original}/50, Nuestro: {e_active_ours}/50")
print(f"  Neuronas I activas - Original: {i_active_original}/50, Nuestro: {i_active_ours}/50")

# Comparar estadísticas
u_mean_diff = abs(np.mean(u_samples_original) - np.mean(u_samples_ours))
u_std_diff = abs(np.std(u_samples_original) - np.std(u_samples_ours))
r_mean_diff = abs(np.mean(r_samples_original) - np.mean(r_samples_ours))

print(f"  Diferencias en estadísticas:")
print(f"    Media de u: {u_mean_diff:.8f}")
print(f"    Std de u: {u_std_diff:.8f}")
print(f"    Media de r: {r_mean_diff:.8f}")

# Determinar si los resultados son equivalentes
sampling_equivalent = (
    abs(u_samples_original.min() - u_samples_ours.min()) < 0.1 and
    abs(u_samples_original.max() - u_samples_ours.max()) < 0.1 and
    abs(r_samples_original.max() - r_samples_ours.max()) < 1.0 and
    e_active_original == e_active_ours and
    i_active_original == i_active_ours
)

if sampling_equivalent:
    print("\n🎉 ¡MUESTREO DE ACTIVIDAD EQUIVALENTE!")
else:
    print("\n⚠️  Diferencias menores en muestreo (típicas del ruido estocástico)")


# =============================================================================
# BLOQUE 12: RESUMEN FINAL
# =============================================================================
print("\n" + "="*80)
print("🎯 RESUMEN FINAL DE VERIFICACIÓN")
print("="*80)

print(f"✅ Parámetros básicos coinciden: {params_match}")
print(f"✅ Signos de conectividad correctos: {signs_correct and our_signs_correct}")
print(f"✅ Condiciones iniciales idénticas: {conditions_match}")
print(f"✅ Evolución temporal idéntica: {u_evolution_match and eta_evolution_match}")
print(f"✅ Muestreo de actividad equivalente: {sampling_equivalent}")

all_tests_pass = all([
    params_match,
    signs_correct and our_signs_correct,
    conditions_match,
    u_evolution_match and eta_evolution_match,
    sampling_equivalent
])

if all_tests_pass:
    print("\n🏆 ¡VERIFICACIÓN COMPLETA EXITOSA!")
    print("Las implementaciones producen resultados idénticos/equivalentes.")
else:
    print("\n⚠️  Algunas verificaciones fallaron.")
    print("Revisar implementación para discrepancias.")

print("\n" + "="*80)
print("Tutorial de verificación completado.")
print("="*80)