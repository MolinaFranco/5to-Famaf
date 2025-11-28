#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análisis de W_exact para inferir cómo deberían ser las orientaciones.
"""

import sys
import os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'scikit-neuromsi')
)

import numpy as np
from skneuromsi.neural import Echeveste2020
from skneuromsi.data import EchevesteDataLoader

# Cargar W_exact y parámetros
model = Echeveste2020(N_E=50, N_I=50, seed=42)
model.load_parameters()
W_exact = model._W_exact

loader = EchevesteDataLoader()
params = loader.load_ssn_connectivity_parameters()

print("=" * 70)
print("ANÁLISIS: ¿Cómo están organizadas las orientaciones en W_exact?")
print("=" * 70)

N_E = 50
N_I = 50

# Extraer bloques
W_EE = W_exact[:N_E, :N_E]
W_EI = W_exact[:N_E, N_E:]
W_IE = W_exact[N_E:, :N_E]
W_II = W_exact[N_E:, N_E:]

print("\n1. Valores máximos de cada bloque:")
print("-" * 70)
print(f"  W_EE max: {W_EE.max():.8f}  (debería ser ≈ a_EE = "
      f"{params['a_EE']:.8f})")
print(f"  W_EI max: {np.abs(W_EI).max():.8f}  (debería ser ≈ a_EI = "
      f"{params['a_EI']:.8f})")
print(f"  W_IE max: {W_IE.max():.8f}  (debería ser ≈ a_IE = "
      f"{params['a_IE']:.8f})")
print(f"  W_II max: {np.abs(W_II).max():.8f}  (debería ser ≈ a_II = "
      f"{params['a_II']:.8f})")

print("\n2. ¿Dónde están los máximos?")
print("-" * 70)

# W_EE
idx_ee = np.unravel_index(W_EE.argmax(), W_EE.shape)
print(f"  W_EE máximo en {idx_ee}: W_EE[{idx_ee}] = {W_EE[idx_ee]:.8f}")

# W_IE
idx_ie = np.unravel_index(W_IE.argmax(), W_IE.shape)
print(f"  W_IE máximo en {idx_ie}: W_IE[{idx_ie}] = {W_IE[idx_ie]:.8f}")

print("\n3. Diagonal de W_IE (conexiones i→i, I→E):")
print("-" * 70)
print("  (¿Las neuronas I_i e E_i tienen la misma orientación?)")
print()
for i in range(5):
    print(f"  W_IE[{i},{i}] = {W_IE[i,i]:.8f}")

print("\n4. Primera fila de W_IE:")
print("-" * 70)
print("  (Conexiones de I_0 a todas las E_j)")
print()
for j in range(5):
    print(f"  W_IE[0,{j}] = {W_IE[0,j]:.8f}")

# Buscar el patrón
diag = np.diag(W_IE)
print(f"\n5. Patrón en la diagonal de W_IE:")
print(f"  Mínimo: {diag.min():.8f}")
print(f"  Máximo: {diag.max():.8f}")
print(f"  Media:  {diag.mean():.8f}")

# Comparar con valores esperados
print("\n6. Análisis:")
print("-" * 70)

if np.allclose(diag.max(), params['a_IE'], rtol=0.01):
    print("  ✓ La diagonal de W_IE contiene los valores máximos (≈ a_IE)")
    print("    → Las neuronas I_i y E_i tienen la MISMA orientación")
    print("    → theta_i = theta_e (no desplazadas)")
else:
    print("  ✗ La diagonal NO contiene los máximos")

    # Encontrar qué elemento es máximo
    max_val = W_IE.max()
    max_idx = np.unravel_index(W_IE.argmax(), W_IE.shape)
    print(f"  Máximo está en W_IE[{max_idx}] = {max_val:.8f}")

    if max_idx[0] == max_idx[1]:
        print("  → El máximo está en la diagonal")
    else:
        offset = max_idx[1] - max_idx[0]
        print(f"  → El máximo está desplazado {offset} posiciones")

print("\n7. Verificar si son estructuras circulantes:")
print("-" * 70)

# W_EE debe ser circulante (toeplitz circular)
print("  W_EE - primera fila:")
print("    ", W_EE[0, :5])
print("  W_EE - segunda fila:")
print("    ", W_EE[1, :5])

# Verificar si W_EE[1,:] es W_EE[0,:] rotado
is_circulant_ee = np.allclose(
    W_EE[1, :], np.roll(W_EE[0, :], 1), rtol=1e-6
)
print(f"  ¿W_EE es circulante? {is_circulant_ee}")

print("\n  W_IE - primera fila:")
print("    ", W_IE[0, :5])
print("  W_IE - segunda fila:")
print("    ", W_IE[1, :5])

is_circulant_ie = np.allclose(
    W_IE[1, :], np.roll(W_IE[0, :], 1), rtol=1e-6
)
print(f"  ¿W_IE es circulante? {is_circulant_ie}")

print("\n" + "=" * 70)
print("CONCLUSIÓN:")
print("=" * 70)

if np.allclose(diag.max(), params['a_IE'], rtol=0.01) and is_circulant_ie:
    print("""
✓ HALLAZGO CLAVE:

Las neuronas inhibitorias y excitatorias con el mismo índice
tienen la MISMA orientación preferida (no están desplazadas).

Es decir:
  theta_e[i] = theta_i[i]  para todo i

La red tiene 50 orientaciones, y cada orientación tiene:
  - 1 neurona excitatoria
  - 1 neurona inhibitoria

La forma correcta de generar orientaciones es:
  orientations = np.linspace(0, π, N_E, endpoint=False)
  theta_e = orientations
  theta_i = orientations  # ¡LAS MISMAS!
    """)
