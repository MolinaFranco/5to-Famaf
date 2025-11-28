#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis profundo del mapeo de orientaciones en el modelo de Echeveste et al. (2020)

Este script documenta exhaustivamente cómo se distribuyen las orientaciones
en las neuronas de la red y cómo se mapea la matriz de conectividad w_learn.

Referencias:
- Echeveste et al. (2020): "Cortical-like dynamics in recurrent circuits
  optimized for sampling-based probabilistic inference"
- Repositorio original: ssn_inference_numerical_experiments
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("ANÁLISIS DEL MAPEO DE ORIENTACIONES EN ECHEVESTE ET AL. (2020)")
print("="*80)
print()

# ============================================================================
# 1. PARÁMETROS BÁSICOS DE LA RED
# ============================================================================
print("1. PARÁMETROS BÁSICOS DE LA RED")
print("-"*80)

N_E = 50  # Neuronas excitatorias
N_I = 50  # Neuronas inhibitorias
N = N_E + N_I  # Total: 100 neuronas

print(f"Número de neuronas excitatorias (E): {N_E}")
print(f"Número de neuronas inhibitorias (I): {N_I}")
print(f"Número total de neuronas: {N}")
print()

# ============================================================================
# 2. MAPEO DE ORIENTACIONES: θ ∈ [0, π]
# ============================================================================
print("2. MAPEO DE ORIENTACIONES")
print("-"*80)

# CRUCIAL: El código original usa θ ∈ [0, π] radianes
# Esto equivale a orientaciones en el rango [0°, 180°)
# que representa todas las orientaciones posibles (la orientación es π-periódica)
theta_rad = np.linspace(0, np.pi, N_E, endpoint=False)
theta_deg = np.degrees(theta_rad)

# Sistema centrado en 0: restamos 90° para tener [-90°, 90°)
theta_deg_centered = theta_deg - 90.0

print(f"Rango de θ: [0, π] radianes = [0°, 180°)")
print(f"Espaciado entre neuronas: Δθ = {np.degrees(theta_rad[1] - theta_rad[0]):.2f}°")
print(f"Total de orientaciones únicas: {N_E}")
print()

print("Mapeo de índices de neuronas a orientaciones preferidas:")
print()
print("  Índice | θ (rad) | θ (deg) | θ centered (deg)")
print("  " + "-"*50)

# Mostrar algunos ejemplos representativos
indices_to_show = [0, 1, 12, 13, 24, 25, 26, 37, 38, 49]
for i in indices_to_show:
    print(f"  E[{i:2d}] | {theta_rad[i]:7.4f} | {theta_deg[i]:7.2f}° | {theta_deg_centered[i]:7.2f}°")

print()
print("INTERPRETACIÓN:")
print("- Neurona E[0]:  representa θ = 0° = -90° (vertical, orientación hacia abajo)")
print("- Neurona E[25]: representa θ = 90° = 0° (horizontal)")
print("- Neurona E[49]: representa θ ≈ 176.4° = 86.4° (casi vertical hacia arriba)")
print()
print("IMPORTANTE: Las orientaciones visuales son π-periódicas, por lo que")
print("una línea a 0° es idéntica a una a 180°. Por eso solo necesitamos")
print("el rango [0°, 180°) para representar todas las orientaciones posibles.")
print()

# ============================================================================
# 3. ESTRUCTURA DE ANILLOS (RING TOPOLOGY)
# ============================================================================
print("3. ESTRUCTURA DE ANILLOS (RING TOPOLOGY)")
print("-"*80)

print("La red tiene una topología de anillo donde:")
print("- Las neuronas están organizadas espacialmente según su orientación preferida")
print("- Cada neurona E tiene una neurona I 'emparejada' con la misma orientación")
print("- La conectividad es CIRCULANTE: W[i,j] depende solo de |θ_i - θ_j|")
print()

print("Organización de las 100 neuronas:")
print("  Índices [0-49]:   Neuronas excitatorias (E), θ ∈ [0°, 180°)")
print("  Índices [50-99]:  Neuronas inhibitorias (I), θ ∈ [0°, 180°)")
print()
print("Ejemplo de emparejamiento E-I:")
print("  E[0]  (θ=0°)    <-> I[50] (θ=0°)")
print("  E[25] (θ=90°)   <-> I[75] (θ=90°)")
print("  E[49] (θ≈176°)  <-> I[99] (θ≈176°)")
print()

# ============================================================================
# 4. MATRIZ DE CONECTIVIDAD W_LEARN
# ============================================================================
print("4. MATRIZ DE CONECTIVIDAD W_LEARN")
print("-"*80)

# Cargar la matriz w_learn
w_learn_path = '/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/SSN/parameter_files/w_learn'
w_learn = np.loadtxt(w_learn_path)

print(f"Shape de w_learn: {w_learn.shape}")
print(f"Tipo: matriz de conectividad completa (100x100)")
print()

# Extraer los 4 bloques
W_EE = w_learn[:N_E, :N_E]       # E -> E
W_EI = w_learn[:N_E, N_E:]       # I -> E (inhibitorio)
W_IE = w_learn[N_E:, :N_E]       # E -> I
W_II = w_learn[N_E:, N_E:]       # I -> I (inhibitorio)

print("Estructura de bloques de W:")
print()
print("        | E (cols 0-49)  | I (cols 50-99)")
print("  ------|----------------|----------------")
print("  E     | W_EE (50x50)   | W_EI (50x50)")
print("  (rows)| (excitatorio)  | (inhibitorio)")
print("  0-49  |                |")
print("  ------|----------------|----------------")
print("  I     | W_IE (50x50)   | W_II (50x50)")
print("  (rows)| (excitatorio)  | (inhibitorio)")
print("  50-99 |                |")
print()

print("Estadísticas de cada bloque:")
print(f"  W_EE: min={W_EE.min():.6f}, max={W_EE.max():.6f}, mean={W_EE.mean():.6f}")
print(f"  W_EI: min={W_EI.min():.6f}, max={W_EI.max():.6f}, mean={W_EI.mean():.6f}")
print(f"  W_IE: min={W_IE.min():.6f}, max={W_IE.max():.6f}, mean={W_IE.mean():.6f}")
print(f"  W_II: min={W_II.min():.6f}, max={W_II.max():.6f}, mean={W_II.mean():.6f}")
print()

print("SIGNOS DE LOS BLOQUES:")
print("  W_EE > 0: Excitatorio (E -> E)")
print("  W_EI < 0: Inhibitorio (I -> E)")
print("  W_IE > 0: Excitatorio (E -> I)")
print("  W_II < 0: Inhibitorio (I -> I)")
print()

# ============================================================================
# 5. PROPIEDAD CIRCULANTE DE LA CONECTIVIDAD
# ============================================================================
print("5. PROPIEDAD CIRCULANTE (RING TOPOLOGY)")
print("-"*80)

print("En una red con topología de anillo, la conectividad es circulante:")
print("  W[i,j] = W[i',j'] si |θ_i - θ_j| = |θ_i' - θ_j'| (módulo π)")
print()

# Verificar propiedad circulante en W_EE
print("Verificación en W_EE:")
print(f"  W_EE[0,1] = {W_EE[0,1]:.6f}")
print(f"  W_EE[1,2] = {W_EE[1,2]:.6f}")
print(f"  ¿Son iguales? {np.isclose(W_EE[0,1], W_EE[1,2])}")
print()
print(f"  W_EE[0,5] = {W_EE[0,5]:.6f}")
print(f"  W_EE[10,15] = {W_EE[10,15]:.6f}")
print(f"  ¿Son iguales? {np.isclose(W_EE[0,5], W_EE[10,15])}")
print()

print("La diagonal de W_EE (neuronas con orientación idéntica) tiene el valor máximo:")
print(f"  Diagonal W_EE[i,i] = {W_EE[0,0]:.6f} (constante para todo i)")
print()

# ============================================================================
# 6. FÓRMULA PARAMÉTRICA DE CONECTIVIDAD (Ecuación 10 del paper)
# ============================================================================
print("6. FÓRMULA PARAMÉTRICA DE CONECTIVIDAD")
print("-"*80)

# Cargar parámetros
params_path = '/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/SSN/parameter_files/'

a_ee = float(np.loadtxt(params_path + 'w_ee_height_learn'))
d_ee = float(np.loadtxt(params_path + 'w_ee_width_learn'))
a_ei = float(np.loadtxt(params_path + 'w_ei_height_learn'))
d_ei = float(np.loadtxt(params_path + 'w_ei_width_learn'))
a_ie = float(np.loadtxt(params_path + 'w_ie_height_learn'))
d_ie = float(np.loadtxt(params_path + 'w_ie_width_learn'))
a_ii = float(np.loadtxt(params_path + 'w_ii_height_learn'))
d_ii = float(np.loadtxt(params_path + 'w_ii_width_learn'))

print("Ecuación 10 (Echeveste et al. 2020):")
print()
print("  W_XY(θ_i, θ_j) = a_XY * exp[(cos(2*(θ_i - θ_j)) - 1) / d_XY²]")
print()
print("donde:")
print("  - a_XY: amplitud (altura del kernel)")
print("  - d_XY: ancho del kernel (en radianes)")
print("  - θ_i, θ_j: orientaciones preferidas de neuronas i, j")
print("  - El factor 2 en cos(2*Δθ) refleja la periodicidad π de las orientaciones")
print()

print("Parámetros aprendidos:")
print(f"  a_EE = {a_ee:.6f},  d_EE = {d_ee:.6f} rad = {np.degrees(d_ee):.2f}°")
print(f"  a_EI = {a_ei:.6f},  d_EI = {d_ei:.6f} rad = {np.degrees(d_ei):.2f}°")
print(f"  a_IE = {a_ie:.6f},  d_IE = {d_ie:.6f} rad = {np.degrees(d_ie):.2f}°")
print(f"  a_II = {a_ii:.6f},  d_II = {d_ii:.6f} rad = {np.degrees(d_ii):.2f}°")
print()

# Verificar que w_learn se puede reconstruir con la fórmula paramétrica
def spatial_kernel(delta_theta, width):
    """Kernel espacial de la Ecuación 10."""
    return np.exp((np.cos(2 * delta_theta) - 1) / (width**2))

# Reconstruir W_EE usando la fórmula
theta_E = np.linspace(0, np.pi, N_E, endpoint=False)
delta_theta = theta_E[:, None] - theta_E[None, :]
W_EE_reconstructed = a_ee * spatial_kernel(delta_theta, d_ee)

print("Verificación: reconstrucción de W_EE con la fórmula paramétrica")
max_error = np.abs(W_EE - W_EE_reconstructed).max()
print(f"  Error máximo: {max_error:.10f}")
print(f"  ¿Reconstrucción exitosa? {max_error < 1e-6}")
print()

# ============================================================================
# 7. CÓMO SE LEE w_learn EN NUESTRO CÓDIGO
# ============================================================================
print("7. CÓMO SE USA w_learn EN SCIKIT-NEUROMSI")
print("-"*80)

print("En _echeveste2020.py, el archivo w_learn se carga como:")
print()
print("  w_learn_path = Path(DATA_DIR) / 'w_learn'")
print("  self._W = np.loadtxt(w_learn_path)")
print()
print("Estructura de self._W:")
print("  - Shape: (100, 100)")
print("  - Filas [0-49]: conexiones hacia neuronas E")
print("  - Filas [50-99]: conexiones hacia neuronas I")
print("  - Columnas [0-49]: conexiones desde neuronas E")
print("  - Columnas [50-99]: conexiones desde neuronas I")
print()
print("Acceso a bloques:")
print("  W_EE = self._W[:N_E, :N_E]    # E -> E")
print("  W_EI = self._W[:N_E, N_E:]    # I -> E (negativo)")
print("  W_IE = self._W[N_E:, :N_E]    # E -> I")
print("  W_II = self._W[N_E:, N_E:]    # I -> I (negativo)")
print()

# ============================================================================
# 8. VISUALIZACIÓN DE LA CONECTIVIDAD
# ============================================================================
print("8. GENERANDO VISUALIZACIONES...")
print("-"*80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Estructura de la Matriz de Conectividad w_learn', fontsize=16, fontweight='bold')

# Matriz completa
ax = axes[0, 0]
im = ax.imshow(w_learn, cmap='RdBu_r', aspect='auto')
ax.set_title('Matriz completa W (100x100)')
ax.set_xlabel('Neurona post-sináptica')
ax.set_ylabel('Neurona pre-sináptica')
ax.axhline(49.5, color='white', linewidth=2, linestyle='--')
ax.axvline(49.5, color='white', linewidth=2, linestyle='--')
ax.text(25, 25, 'W_EE', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
ax.text(75, 25, 'W_EI', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
ax.text(25, 75, 'W_IE', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
ax.text(75, 75, 'W_II', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# W_EE
ax = axes[0, 1]
im = ax.imshow(W_EE, cmap='Reds', aspect='auto')
ax.set_title('W_EE (E → E, excitatorio)')
ax.set_xlabel('Neurona E post-sináptica')
ax.set_ylabel('Neurona E pre-sináptica')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# W_EI
ax = axes[0, 2]
im = ax.imshow(W_EI, cmap='Blues_r', aspect='auto')
ax.set_title('W_EI (I → E, inhibitorio)')
ax.set_xlabel('Neurona I')
ax.set_ylabel('Neurona E')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# W_IE
ax = axes[1, 0]
im = ax.imshow(W_IE, cmap='Reds', aspect='auto')
ax.set_title('W_IE (E → I, excitatorio)')
ax.set_xlabel('Neurona E')
ax.set_ylabel('Neurona I')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# W_II
ax = axes[1, 1]
im = ax.imshow(W_II, cmap='Blues_r', aspect='auto')
ax.set_title('W_II (I → I, inhibitorio)')
ax.set_xlabel('Neurona I post-sináptica')
ax.set_ylabel('Neurona I pre-sináptica')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Perfil de conectividad: W_EE[0, :]
ax = axes[1, 2]
ax.plot(theta_deg_centered, W_EE[0, :], 'o-', linewidth=2, markersize=4, label='W_EE[0, j]')
ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Neurona E[0]')
ax.set_xlabel('Orientación preferida θ_j (grados, centrado)')
ax.set_ylabel('Peso de conexión')
ax.set_title('Perfil de conectividad desde E[0]')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()

output_path = '/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs/10_perfil_conectividad_w_learn.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Figura guardada en: {output_path}")
print()

# ============================================================================
# 9. RESUMEN Y CONCLUSIONES
# ============================================================================
print("="*80)
print("RESUMEN Y CONCLUSIONES")
print("="*80)
print()

print("1. MAPEO DE ORIENTACIONES:")
print("   - 50 neuronas E y 50 neuronas I, cada una representa una orientación")
print("   - θ ∈ [0, π] radianes = [0°, 180°) = [-90°, 90°) centrado en 0°")
print("   - Espaciado uniforme: Δθ = 3.6°")
print("   - Neurona E[i] tiene la misma orientación que neurona I[i+50]")
print()

print("2. TOPOLOGÍA DE ANILLO:")
print("   - Conectividad CIRCULANTE: W[i,j] depende solo de |θ_i - θ_j|")
print("   - Máxima conexión entre neuronas con orientación idéntica")
print("   - Conexión decae exponencialmente con diferencia angular")
print()

print("3. MATRIZ w_learn:")
print("   - Shape: (100, 100)")
print("   - Bloques: [W_EE, W_EI; W_IE, W_II]")
print("   - W_EE, W_IE > 0 (excitatorio)")
print("   - W_EI, W_II < 0 (inhibitorio)")
print()

print("4. FÓRMULA PARAMÉTRICA:")
print("   - W_XY(θ_i, θ_j) = a_XY * exp[(cos(2*(θ_i - θ_j)) - 1) / d_XY²]")
print("   - El factor 2 en cos(2*Δθ) es crucial: refleja periodicidad π")
print("   - Parámetros (a_XY, d_XY) fueron aprendidos mediante optimización")
print()

print("5. CONVENCIÓN FUNDAMENTAL:")
print("   *** SIEMPRE USAMOS θ ∈ [0, π] EN EL CÓDIGO ***")
print("   - En Python: theta = np.linspace(0, np.pi, N_E, endpoint=False)")
print("   - Equivale a [-90°, 90°) si centramos en 0°")
print("   - Esta convención es consistente con el código original de Echeveste")
print()

print("="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
