#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Debug: ver cómo se construyen las orientaciones."""

import sys
import os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'scikit-neuromsi')
)

import numpy as np
from skneuromsi.neural import Echeveste2020

model = Echeveste2020(N_E=50, N_I=50, seed=42)
model.load_parameters()

print("=" * 70)
print("DEBUG: Orientaciones y construcción de matrices")
print("=" * 70)

# Ver qué pasa en _build_connectivity_matrices
print("\n1. ¿Cómo construye las matrices _W_XX?")
print("-" * 70)

# Mirar el código relevante
if model._W_EE is not None:
    print("✓ Matrices _W_XX ya construidas en load_parameters()")
    print(f"  _W_EE[0,0] = {model._W_EE[0,0]:.8f}")
    print(f"  _W_EE[0,1] = {model._W_EE[0,1]:.8f}")
    print(f"  _W_EE[0,2] = {model._W_EE[0,2]:.8f}")

# Ver W_exact
print(f"\n  W_exact[0,0] = {model._W_exact[0,0]:.8f}")
print(f"  W_exact[0,1] = {model._W_exact[0,1]:.8f}")
print(f"  W_exact[0,2] = {model._W_exact[0,2]:.8f}")

print("\n2. Patron observable:")
print("-" * 70)
print("  _W_EE[0,1] == W_exact[0,0] ?",
      np.isclose(model._W_EE[0,1], model._W_exact[0,0]))
print("  _W_EE[0,2] == W_exact[0,1] ?",
      np.isclose(model._W_EE[0,2], model._W_exact[0,1]))

print("\n¿Hay un shift de 1 posición?")

# Verificar si hay un patrón de desplazamiento
print("\nComparando diagonales:")
for i in range(5):
    print(f"  _W_EE[{i},{i}] = {model._W_EE[i,i]:.8f}, "
          f"W_exact[{i},{i}] = {model._W_exact[i,i]:.8f}")

print("\n3. ¿Cómo se generan las orientaciones?")
print("-" * 70)

# Ver position_range
print(f"position_range: {model._position_range}")

# Reconstruir las orientaciones como lo hace build_connectivity_matrix
theta_e = np.linspace(
    np.radians(model._position_range[0]),
    np.radians(model._position_range[1]),
    model._N_E,
)

print(f"\nOrientaciones E (primeras 5):")
for i in range(5):
    print(f"  theta_e[{i}] = {theta_e[i]:.6f} rad = "
          f"{np.degrees(theta_e[i]):.2f}°")

# Ver qué orientaciones deberían usarse según el código original
print("\n4. ¿Y en _build_connectivity_matrices()?")
print("-" * 70)
print("En la línea 994-996 del código:")
print("  orientations = np.pi * np.arange(self._N) / self._N")
print("\nEso daría:")
orientations_v1 = np.pi * np.arange(100) / 100
print(f"  orientations[0] = {orientations_v1[0]:.6f} rad = "
      f"{np.degrees(orientations_v1[0]):.2f}°")
print(f"  orientations[1] = {orientations_v1[1]:.6f} rad = "
      f"{np.degrees(orientations_v1[1]):.2f}°")

print("\nPero build_connectivity_matrix usa:")
print("  theta_e = np.linspace(0, pi, N_E)")
print("\nEso da:")
theta_v2 = np.linspace(0, np.pi, 50)
print(f"  theta[0] = {theta_v2[0]:.6f} rad = "
      f"{np.degrees(theta_v2[0]):.2f}°")
print(f"  theta[1] = {theta_v2[1]:.6f} rad = "
      f"{np.degrees(theta_v2[1]):.2f}°")

print("\n5. Diferencia:")
print("-" * 70)
print(f"Método 1 (arange):  {np.degrees(orientations_v1[1]):.4f}°")
print(f"Método 2 (linspace): {np.degrees(theta_v2[1]):.4f}°")
print(f"Diferencia: {np.degrees(theta_v2[1] - orientations_v1[1]):.4f}°")

print("\n" + "=" * 70)
print("CONCLUSIÓN:")
print("=" * 70)
print("""
Hay DOS formas diferentes de generar orientaciones en el código:

1. _build_connectivity_matrices() usa:
   orientations = np.pi * np.arange(N) / N

2. build_connectivity_matrix() usa:
   theta = np.linspace(0, pi, N_E)

Estas NO dan los mismos valores!
Esto explica por qué W_reconstruida ≠ W_exact
""")
