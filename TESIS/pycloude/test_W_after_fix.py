#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test para verificar que W_reconstruida coincide con W_exact
después del fix.
"""

import sys
import os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'scikit-neuromsi')
)

import numpy as np
from skneuromsi.neural import Echeveste2020

print("=" * 70)
print("TEST: W matrices después del fix")
print("=" * 70)

# Crear modelo y cargar parámetros
model = Echeveste2020(N_E=50, N_I=50, seed=42)
model.load_parameters()

# FORZAR que se invaliden las matrices _W_XX
print("\n1. Invalidando matrices _W_XX guardadas...")
model._W_EE = None
model._W_EI = None
model._W_IE = None
model._W_II = None

# Ahora build_connectivity_matrix() debe reconstruir desde parámetros
print("2. Reconstruyendo W desde parámetros con el fix...")
W_recon = model.build_connectivity_matrix()

# Obtener W_exact
W_exact = model._W_exact

print("\n3. Comparando matrices...")
print(f"\nW_exact[0:3, 0:3]:")
print(W_exact[0:3, 0:3])

print(f"\nW_recon[0:3, 0:3]:")
print(W_recon[0:3, 0:3])

diff = W_recon - W_exact

print(f"\nDiferencia[0:3, 0:3]:")
print(diff[0:3, 0:3])

print("\n" + "=" * 70)
print("ESTADÍSTICAS:")
print("=" * 70)

print(f"\nDiferencia absoluta:")
print(f"  Mínima:  {np.abs(diff).min():.8e}")
print(f"  Máxima:  {np.abs(diff).max():.8e}")
print(f"  Media:   {np.abs(diff).mean():.8e}")

# Error relativo
rel_error = np.abs(diff) / (np.abs(W_exact) + 1e-10)
print(f"\nError relativo (%):")
print(f"  Media:   {rel_error.mean() * 100:.4f}%")
print(f"  Mediana: {np.median(rel_error) * 100:.4f}%")
print(f"  Máximo:  {rel_error.max() * 100:.4f}%")

# ¿Son aproximadamente iguales?
tolerances = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
print(f"\n¿Son aproximadamente iguales?")
for tol in tolerances:
    are_close = np.allclose(W_exact, W_recon, rtol=tol, atol=tol)
    print(f"  Con tolerancia {tol:.0e}: {are_close}")

print("\n" + "=" * 70)
print("CONCLUSIÓN:")
print("=" * 70)

if np.allclose(W_exact, W_recon, rtol=1e-8, atol=1e-8):
    print("\n✓✓✓ FIX EXITOSO ✓✓✓")
    print("W_reconstruida coincide con W_exact (tolerancia 1e-8)")
else:
    max_error = np.abs(diff).max()
    print(f"\n✗ Todavía hay diferencias (máximo: {max_error:.2e})")

    # Encontrar dónde está el máximo error
    idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    print(f"\nMáximo error en posición {idx}:")
    print(f"  W_exact[{idx}] = {W_exact[idx]:.8f}")
    print(f"  W_recon[{idx}] = {W_recon[idx]:.8f}")
    print(f"  Diferencia = {diff[idx]:.8f}")
