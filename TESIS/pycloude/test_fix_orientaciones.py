#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test para verificar que el fix de orientaciones funciona."""

import numpy as np

N_E = 50
N_I = 50
N_total = 100

print("=" * 70)
print("VERIFICACIÓN: Orientaciones después del fix")
print("=" * 70)

# Método 1: _build_connectivity_matrices (línea 994)
orientations_v1 = np.linspace(0, np.pi, N_total, endpoint=False)
theta_e_v1 = orientations_v1[:N_E]
theta_i_v1 = orientations_v1[N_E:]

print("\nMétodo 1: _build_connectivity_matrices()")
print(f"  orientations = np.linspace(0, np.pi, {N_total}, endpoint=False)")
print(f"  theta_e = orientations[:50]")
print(f"  theta_i = orientations[50:]")
print(f"\n  theta_e[0] = {theta_e_v1[0]:.8f} rad = "
      f"{np.degrees(theta_e_v1[0]):.4f}°")
print(f"  theta_e[1] = {theta_e_v1[1]:.8f} rad = "
      f"{np.degrees(theta_e_v1[1]):.4f}°")
print(f"  theta_e[49] = {theta_e_v1[49]:.8f} rad = "
      f"{np.degrees(theta_e_v1[49]):.4f}°")
print(f"\n  theta_i[0] = {theta_i_v1[0]:.8f} rad = "
      f"{np.degrees(theta_i_v1[0]):.4f}°")
print(f"  theta_i[1] = {theta_i_v1[1]:.8f} rad = "
      f"{np.degrees(theta_i_v1[1]):.4f}°")

# Método 2: build_connectivity_matrix CON FIX (línea 1873-1884)
position_range = (0, 180)
theta_e_v2 = np.linspace(
    np.radians(position_range[0]),
    np.radians(position_range[1]),
    N_E,
    endpoint=False,
)
theta_i_v2 = np.linspace(
    np.radians(position_range[0]),
    np.radians(position_range[1]),
    N_I,
    endpoint=False,
)

print("\n" + "-" * 70)
print("\nMétodo 2: build_connectivity_matrix() CON FIX")
print(f"  theta_e = np.linspace(0, π, {N_E}, endpoint=False)")
print(f"  theta_i = np.linspace(0, π, {N_I}, endpoint=False)")
print(f"\n  theta_e[0] = {theta_e_v2[0]:.8f} rad = "
      f"{np.degrees(theta_e_v2[0]):.4f}°")
print(f"  theta_e[1] = {theta_e_v2[1]:.8f} rad = "
      f"{np.degrees(theta_e_v2[1]):.4f}°")
print(f"  theta_e[49] = {theta_e_v2[49]:.8f} rad = "
      f"{np.degrees(theta_e_v2[49]):.4f}°")
print(f"\n  theta_i[0] = {theta_i_v2[0]:.8f} rad = "
      f"{np.degrees(theta_i_v2[0]):.4f}°")
print(f"  theta_i[1] = {theta_i_v2[1]:.8f} rad = "
      f"{np.degrees(theta_i_v2[1]):.4f}°")

print("\n" + "=" * 70)
print("COMPARACIÓN:")
print("=" * 70)

print(f"\n¿Son iguales theta_e?")
are_equal_e = np.allclose(theta_e_v1, theta_e_v2)
print(f"  np.allclose(theta_e_v1, theta_e_v2) = {are_equal_e}")
if not are_equal_e:
    print(f"  Diferencia máxima: {np.abs(theta_e_v1 - theta_e_v2).max():.2e}")
    print(f"  theta_e_v1[1] = {theta_e_v1[1]:.10f}")
    print(f"  theta_e_v2[1] = {theta_e_v2[1]:.10f}")
    print(f"  diff[1] = {theta_e_v2[1] - theta_e_v1[1]:.10f}")

print(f"\n¿Son iguales theta_i?")
are_equal_i = np.allclose(theta_i_v1, theta_i_v2)
print(f"  np.allclose(theta_i_v1, theta_i_v2) = {are_equal_i}")
if not are_equal_i:
    print(f"  Diferencia máxima: {np.abs(theta_i_v1 - theta_i_v2).max():.2e}")
    print(f"  theta_i_v1[0] = {theta_i_v1[0]:.10f}")
    print(f"  theta_i_v2[0] = {theta_i_v2[0]:.10f}")
    print(f"  diff[0] = {theta_i_v2[0] - theta_i_v1[0]:.10f}")

print("\n" + "=" * 70)
print("DIAGNÓSTICO:")
print("=" * 70)

if are_equal_e and are_equal_i:
    print("\n✓ PERFECTO: Ambos métodos generan las mismas orientaciones")
else:
    print("\n✗ PROBLEMA: Los métodos generan orientaciones diferentes")
    print("\nRazón:")
    print("  Método 1 genera 100 orientaciones y luego las divide 50/50")
    print("  Método 2 genera 50 orientaciones para E y 50 para I por separado")
    print("\n  Con Método 1:")
    print(f"    theta_e = [0, π/100, 2π/100, ..., 49π/100]")
    print(f"    theta_i = [50π/100, 51π/100, ..., 99π/100]")
    print("\n  Con Método 2:")
    print(f"    theta_e = [0, π/50, 2π/50, ..., 49π/50]")
    print(f"    theta_i = [0, π/50, 2π/50, ..., 49π/50]")
    print("\n  ¡Las orientaciones inhibitorias empiezan en valores diferentes!")
