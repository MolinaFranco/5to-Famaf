#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análisis exhaustivo del problema del endpoint en las orientaciones.
"""

import numpy as np

print("=" * 70)
print("ANÁLISIS: endpoint=False vs endpoint=True")
print("=" * 70)

N_E = 50
position_range = (0, 180)

# Método 1: linspace SIN endpoint (CORRECTO según el código original)
theta_correcto = np.linspace(0, np.pi, N_E, endpoint=False)

# Método 2: linspace CON endpoint (BUG en línea 1871)
theta_bug = np.linspace(
    np.radians(position_range[0]),
    np.radians(position_range[1]),
    N_E
)
# Nota: cuando no se especifica, endpoint=True por defecto

print("\n1. MÉTODO CORRECTO (endpoint=False):")
print("-" * 70)
print("Código: np.linspace(0, np.pi, N_E, endpoint=False)")
print(f"\nPrimeros 5 valores:")
for i in range(5):
    print(f"  theta[{i}] = {theta_correcto[i]:.8f} rad = "
          f"{np.degrees(theta_correcto[i]):.4f}°")

print(f"\nÚltimos 2 valores:")
for i in range(N_E-2, N_E):
    print(f"  theta[{i}] = {theta_correcto[i]:.8f} rad = "
          f"{np.degrees(theta_correcto[i]):.4f}°")

print(f"\nDiferencia entre valores consecutivos:")
diff_correcto = theta_correcto[1] - theta_correcto[0]
print(f"  Δθ = {diff_correcto:.8f} rad = "
      f"{np.degrees(diff_correcto):.4f}°")

print(f"\nFórmula equivalente: θ[i] = π * i / N")
print(f"  θ[0] = π * 0 / {N_E} = {np.pi * 0 / N_E:.8f}")
print(f"  θ[1] = π * 1 / {N_E} = {np.pi * 1 / N_E:.8f}")
print(f"  θ[49] = π * 49 / {N_E} = {np.pi * 49 / N_E:.8f}")

print("\n" + "=" * 70)
print("2. MÉTODO CON BUG (endpoint=True por defecto):")
print("-" * 70)
print("Código: np.linspace(np.radians(0), np.radians(180), N_E)")
print("        (SIN especificar endpoint, usa True por defecto)")

print(f"\nPrimeros 5 valores:")
for i in range(5):
    print(f"  theta[{i}] = {theta_bug[i]:.8f} rad = "
          f"{np.degrees(theta_bug[i]):.4f}°")

print(f"\nÚltimos 2 valores:")
for i in range(N_E-2, N_E):
    print(f"  theta[{i}] = {theta_bug[i]:.8f} rad = "
          f"{np.degrees(theta_bug[i]):.4f}°")

print(f"\nDiferencia entre valores consecutivos:")
diff_bug = theta_bug[1] - theta_bug[0]
print(f"  Δθ = {diff_bug:.8f} rad = {np.degrees(diff_bug):.4f}°")

print(f"\nFórmula equivalente: θ[i] = π * i / (N - 1)")
print(f"  θ[0] = π * 0 / {N_E-1} = {np.pi * 0 / (N_E-1):.8f}")
print(f"  θ[1] = π * 1 / {N_E-1} = {np.pi * 1 / (N_E-1):.8f}")
print(f"  θ[49] = π * 49 / {N_E-1} = {np.pi * 49 / (N_E-1):.8f}")

print("\n" + "=" * 70)
print("3. COMPARACIÓN DIRECTA:")
print("=" * 70)

print(f"\n{'i':<4} {'Correcto (rad)':<18} {'Bug (rad)':<18} "
      f"{'Diferencia (rad)':<18} {'Diff (°)':<10}")
print("-" * 70)

for i in [0, 1, 2, 10, 25, 48, 49]:
    diff_val = theta_bug[i] - theta_correcto[i]
    print(f"{i:<4} {theta_correcto[i]:<18.8f} {theta_bug[i]:<18.8f} "
          f"{diff_val:<18.8f} {np.degrees(diff_val):<10.4f}")

print("\n" + "=" * 70)
print("4. IMPACTO EN LA MATRIZ W:")
print("=" * 70)

# Simular cómo afecta esto a W usando la fórmula de conectividad
a_EE = 0.331089  # Valor real del paper
d_EE = 0.802756

def calc_connectivity(theta_i, theta_j, a, d):
    """Ecuación 10 del paper."""
    delta = theta_i - theta_j
    return a * np.exp((np.cos(2 * delta) - 1) / d**2)

# Calcular W[0,1] con ambos métodos
W_01_correcto = calc_connectivity(
    theta_correcto[0],
    theta_correcto[1],
    a_EE,
    d_EE
)

W_01_bug = calc_connectivity(
    theta_bug[0],
    theta_bug[1],
    a_EE,
    d_EE
)

print(f"\nConectividad W[0,1] (neurona 0 → neurona 1):")
print(f"  Con método correcto: {W_01_correcto:.8f}")
print(f"  Con método bug:      {W_01_bug:.8f}")
print(f"  Diferencia:          {W_01_bug - W_01_correcto:.8f}")
print(f"  Error relativo:      {abs(W_01_bug - W_01_correcto) / W_01_correcto * 100:.2f}%")

# Calcular error promedio en toda la matriz
W_correcto_full = np.zeros((N_E, N_E))
W_bug_full = np.zeros((N_E, N_E))

for i in range(N_E):
    for j in range(N_E):
        W_correcto_full[i, j] = calc_connectivity(
            theta_correcto[i], theta_correcto[j], a_EE, d_EE
        )
        W_bug_full[i, j] = calc_connectivity(
            theta_bug[i], theta_bug[j], a_EE, d_EE
        )

diff_matrix = W_bug_full - W_correcto_full
rel_error_matrix = np.abs(diff_matrix) / (np.abs(W_correcto_full) + 1e-10)

print(f"\nEstadísticas de error en toda la matriz W_EE:")
print(f"  Error absoluto medio:    {np.abs(diff_matrix).mean():.8f}")
print(f"  Error absoluto máximo:   {np.abs(diff_matrix).max():.8f}")
print(f"  Error relativo medio:    {rel_error_matrix.mean() * 100:.2f}%")
print(f"  Error relativo máximo:   {rel_error_matrix.max() * 100:.2f}%")

print("\n" + "=" * 70)
print("5. CONCLUSIÓN:")
print("=" * 70)

print("""
El BUG está en build_connectivity_matrix() líneas 1871-1880:

INCORRECTO (actual):
    theta_e = np.linspace(
        np.radians(self._position_range[0]),
        np.radians(self._position_range[1]),
        self._N_E,
    )

CORRECTO (debe ser):
    theta_e = np.linspace(
        np.radians(self._position_range[0]),
        np.radians(self._position_range[1]),
        self._N_E,
        endpoint=False    # ← AGREGAR ESTO
    )

Razón:
- Sin endpoint=False, linspace incluye el último valor (180°)
- Esto da θ[i] = π * i / (N-1) en lugar de θ[i] = π * i / N
- La diferencia angular es 3.67° en lugar de 3.6°
- Esto causa errores de hasta ~10-20% en los valores de W
""")

# Verificar qué tan cerca están
are_close = np.allclose(theta_correcto, theta_bug, rtol=0.01)
print(f"\n¿Son aproximadamente iguales (1% tolerancia)? {are_close}")
print(f"→ No, porque el último valor difiere significativamente:")
print(f"  Correcto: {np.degrees(theta_correcto[-1]):.4f}°")
print(f"  Bug:      {np.degrees(theta_bug[-1]):.4f}°")
print(f"  Diferencia: {np.degrees(theta_bug[-1] - theta_correcto[-1]):.4f}°")
