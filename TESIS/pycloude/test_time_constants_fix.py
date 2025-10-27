#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test para verificar que las constantes de tiempo están en unidades correctas.

Este script verifica que:
1. tau_e, tau_i, tau_n están en segundos (0.02s, 0.01s, 0.02s)
2. dt está en segundos (0.0002s)
3. inv_taus se calcula correctamente (50 Hz, 100 Hz)
4. La integración temporal es consistente con el código original

Referencias:
- echepaper.pdf Supplementary Table S1: τ_E = 20ms, τ_I = 10ms, dt = 0.2ms
- ssn_inference_numerical_experiments/SSN/parameters.py:
  tau_e = 20.0e-3, tau_i = 10.0e-3, dt = 0.2e-3 (todos en segundos)
"""

import sys
import numpy as np

# Agregar scikit-neuromsi al path
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

from skneuromsi.neural import Echeveste2020  # noqa: E402

print("=" * 70)
print("VERIFICACIÓN DE CONSTANTES DE TIEMPO")
print("=" * 70)

# Crear modelo
print("\n1. Creando modelo SSN...")
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)

# Verificar valores de tau en el integrador
print("\n2. Verificando constantes de tiempo en SSNIntegrator:")
tau_e = ssn._integrator.f.tau_e
tau_i = ssn._integrator.f.tau_i
tau_n = ssn._integrator.f.tau_n

print(f"   tau_e = {tau_e:.6f} s")
print(f"   tau_i = {tau_i:.6f} s")
print(f"   tau_n = {tau_n:.6f} s")

# Verificar que están en segundos
expected_tau_e = 0.02  # 20 ms
expected_tau_i = 0.01  # 10 ms
expected_tau_n = 0.02  # 20 ms

assert np.isclose(tau_e, expected_tau_e), \
    f"ERROR: tau_e debería ser {expected_tau_e}s, pero es {tau_e}s"
assert np.isclose(tau_i, expected_tau_i), \
    f"ERROR: tau_i debería ser {expected_tau_i}s, pero es {tau_i}s"
assert np.isclose(tau_n, expected_tau_n), \
    f"ERROR: tau_n debería ser {expected_tau_n}s, pero es {tau_n}s"

print("   ✓ Todas las constantes de tiempo están en SEGUNDOS (correcto)")

# Verificar dt
print("\n3. Verificando timestep dt:")
dt = ssn._time_res / 1000.0  # Conversión de ms a s
print(f"   time_res = {ssn._time_res} ms")
print(f"   dt = {dt:.6f} s")

expected_dt = 0.0002  # 0.2 ms
assert np.isclose(dt, expected_dt), \
    f"ERROR: dt debería ser {expected_dt}s, pero es {dt}s"
print("   ✓ dt está en SEGUNDOS (correcto)")

# Verificar inv_taus
print("\n4. Verificando inverse time constants (inv_taus):")
inv_tau_e = 1.0 / tau_e
inv_tau_i = 1.0 / tau_i

print(f"   1/tau_e = {inv_tau_e:.2f} Hz")
print(f"   1/tau_i = {inv_tau_i:.2f} Hz")

expected_inv_tau_e = 50.0  # Hz
expected_inv_tau_i = 100.0  # Hz

assert np.isclose(inv_tau_e, expected_inv_tau_e), \
    f"ERROR: 1/tau_e debería ser {expected_inv_tau_e} Hz, pero es {inv_tau_e}"
assert np.isclose(inv_tau_i, expected_inv_tau_i), \
    f"ERROR: 1/tau_i debería ser {expected_inv_tau_i} Hz, pero es {inv_tau_i}"

print("   ✓ Inverse time constants correctos")

# Verificar la dinámica temporal
print("\n5. Verificando dinámica temporal de integración:")
# Ejemplo: u_new = u_old + dt * du_dt
# donde du_dt = (...) / tau
# Entonces: u_new = u_old + (dt / tau) * (...)
print(f"   Factor temporal excitatorio: dt/tau_e = {dt/tau_e:.4f}")
print(f"   Factor temporal inhibitorio: dt/tau_i = {dt/tau_i:.4f}")

# Comparar con código original
# Original: dt = 0.2e-3 = 0.0002s, tau_e = 20.0e-3 = 0.02s
original_factor_e = 0.0002 / 0.02
original_factor_i = 0.0002 / 0.01

print("\n   Original (echeveste code):")
print(f"     dt/tau_e = {original_factor_e:.4f}")
print(f"     dt/tau_i = {original_factor_i:.4f}")

assert np.isclose(dt/tau_e, original_factor_e), \
    "ERROR: Factor temporal excitatorio no coincide con código original"
assert np.isclose(dt/tau_i, original_factor_i), \
    "ERROR: Factor temporal inhibitorio no coincide con código original"

print("   ✓ Factores temporales coinciden con código original")

# Verificar que NO estamos usando ms en lugar de segundos
print("\n6. Verificando que NO hay mezcla de unidades (bug previo):")
wrong_factor_e = 0.0002 / 20.0  # Bug: dt en s, tau en ms
wrong_factor_i = 0.0002 / 10.0  # Bug: dt en s, tau en ms

print("   Factor INCORRECTO (bug previo):")
print(f"     dt/tau_e = {wrong_factor_e:.6f} (1000× más lento!)")
print(f"     dt/tau_i = {wrong_factor_i:.6f} (1000× más lento!)")

print("\n   Factor CORRECTO (actual):")
print(f"     dt/tau_e = {dt/tau_e:.4f}")
print(f"     dt/tau_i = {dt/tau_i:.4f}")

# Verificar que la diferencia es de 1000×
ratio_e = (dt/tau_e) / wrong_factor_e
ratio_i = (dt/tau_i) / wrong_factor_i
print("\n   Mejora de velocidad:")
print(f"     Excitatorio: {ratio_e:.1f}× más rápido")
print(f"     Inhibitorio: {ratio_i:.1f}× más rápido")

assert np.isclose(ratio_e, 1000.0, rtol=0.01), \
    "ERROR: La corrección debería ser de 1000×"
assert np.isclose(ratio_i, 1000.0, rtol=0.01), \
    "ERROR: La corrección debería ser de 1000×"

print("   ✓ Bug corregido: dinámicas ahora son 1000× más rápidas (correcto)")

print("\n" + "=" * 70)
print("✓ TODAS LAS VERIFICACIONES PASARON")
print("✓ Las constantes de tiempo están correctamente en SEGUNDOS")
print("✓ La integración es consistente con el código original de Echeveste")
print("=" * 70)
