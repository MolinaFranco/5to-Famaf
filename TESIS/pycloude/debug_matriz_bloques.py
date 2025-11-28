#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Debug: Comparar cómo se construyen los bloques de W."""

import sys
import os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'scikit-neuromsi')
)

import numpy as np
from skneuromsi.neural import Echeveste2020
from skneuromsi.data import EchevesteDataLoader

# Cargar parámetros
loader = EchevesteDataLoader()
params = loader.load_ssn_connectivity_parameters()

print("=" * 70)
print("DEBUG: Construcción de bloques de W")
print("=" * 70)

print("\nParámetros cargados:")
for key, val in params.items():
    print(f"  {key} = {val:.8f}")

# Simular cómo _build_connectivity_matrices construye la matriz
print("\n" + "=" * 70)
print("MÉTODO 1: _build_connectivity_matrices()")
print("=" * 70)

N_E = 50
N_I = 50
N = 100

orientations = np.linspace(0, np.pi, N, endpoint=False)
theta_e_v1 = orientations[:N_E]
theta_i_v1 = orientations[N_E:]

print(f"\nOrientaciones E (primeras 3):")
for i in range(3):
    print(f"  theta_e[{i}] = {theta_e_v1[i]:.8f} rad = "
          f"{np.degrees(theta_e_v1[i]):.4f}°")

print(f"\nOrientaciones I (primeras 3):")
for i in range(3):
    print(f"  theta_i[{i}] = {theta_i_v1[i]:.8f} rad = "
          f"{np.degrees(theta_i_v1[i]):.4f}°")

# Función para construir bloque
def build_block(theta_pre, theta_post, a, d):
    """Construcción según _build_parametric_matrix."""
    n_pre = len(theta_pre)
    n_post = len(theta_post)
    matrix = np.zeros((n_pre, n_post))
    for i in range(n_pre):
        for j in range(n_post):
            delta_theta = theta_pre[i] - theta_post[j]
            matrix[i, j] = a * np.exp(
                (np.cos(2 * delta_theta) - 1) / (d**2)
            )
    return matrix

W_IE_v1 = build_block(theta_i_v1, theta_e_v1,
                      params['a_IE'], params['d_IE'])

print(f"\nW_IE (I→E) [0:3, 0:3]:")
print(W_IE_v1[0:3, 0:3])
print(f"  W_IE[0,0] = {W_IE_v1[0,0]:.8f}")

# Simular cómo build_connectivity_matrix (CON FIX) construye la matriz
print("\n" + "=" * 70)
print("MÉTODO 2: build_connectivity_matrix() CON FIX")
print("=" * 70)

orientations_v2 = np.linspace(0, np.pi, N, endpoint=False)
theta_e_v2 = orientations_v2[:N_E]
theta_i_v2 = orientations_v2[N_E:]

print(f"\nOrientaciones E (primeras 3):")
for i in range(3):
    print(f"  theta_e[{i}] = {theta_e_v2[i]:.8f} rad = "
          f"{np.degrees(theta_e_v2[i]):.4f}°")

print(f"\nOrientaciones I (primeras 3):")
for i in range(3):
    print(f"  theta_i[{i}] = {theta_i_v2[i]:.8f} rad = "
          f"{np.degrees(theta_i_v2[i]):.4f}°")

# Función connectivity_block según build_connectivity_matrix
def connectivity_block(theta_pre, theta_post, a, d, sign=1):
    delta_theta = theta_pre[:, None] - theta_post[None, :]
    return sign * a * np.exp((np.cos(2 * delta_theta) - 1) / d**2)

W_IE_v2 = connectivity_block(theta_i_v2, theta_e_v2,
                             params['a_IE'], params['d_IE'], sign=1)

print(f"\nW_IE (I→E) [0:3, 0:3]:")
print(W_IE_v2[0:3, 0:3])
print(f"  W_IE[0,0] = {W_IE_v2[0,0]:.8f}")

# Cargar W_exact para comparar
print("\n" + "=" * 70)
print("W_exact (del archivo)")
print("=" * 70)

model = Echeveste2020(N_E=50, N_I=50, seed=42)
model.load_parameters()
W_exact = model._W_exact

W_IE_exact = W_exact[N_E:, :N_E]

print(f"\nW_IE_exact (I→E) [0:3, 0:3]:")
print(W_IE_exact[0:3, 0:3])
print(f"  W_IE_exact[0,0] = {W_IE_exact[0,0]:.8f}")

# Comparación
print("\n" + "=" * 70)
print("COMPARACIÓN:")
print("=" * 70)

print(f"\n¿Método 1 == Método 2?")
are_equal_12 = np.allclose(W_IE_v1, W_IE_v2)
print(f"  np.allclose(W_IE_v1, W_IE_v2) = {are_equal_12}")
if are_equal_12:
    print("  ✓ Ambos métodos generan la misma matriz")
else:
    print(f"  ✗ Diferencia máxima: {np.abs(W_IE_v1 - W_IE_v2).max():.2e}")

print(f"\n¿Método 1 == W_exact?")
are_equal_1e = np.allclose(W_IE_v1, W_IE_exact, rtol=1e-8)
print(f"  np.allclose(W_IE_v1, W_IE_exact) = {are_equal_1e}")
if not are_equal_1e:
    print(f"  Diferencia máxima: {np.abs(W_IE_v1 - W_IE_exact).max():.2e}")
    print(f"  W_IE_v1[0,0] = {W_IE_v1[0,0]:.8f}")
    print(f"  W_IE_exact[0,0] = {W_IE_exact[0,0]:.8f}")
    print(f"  Diff = {W_IE_v1[0,0] - W_IE_exact[0,0]:.8f}")

print(f"\n¿Método 2 == W_exact?")
are_equal_2e = np.allclose(W_IE_v2, W_IE_exact, rtol=1e-8)
print(f"  np.allclose(W_IE_v2, W_IE_exact) = {are_equal_2e}")
if not are_equal_2e:
    print(f"  Diferencia máxima: {np.abs(W_IE_v2 - W_IE_exact).max():.2e}")
