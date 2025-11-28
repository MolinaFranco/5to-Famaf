#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para generar Figura 3 de Echeveste et al. (2020).

Este script carga simulaciones pre-generadas y genera gráficos de generalización:
- Fig3a: Generalización en contraste (media y std vs contraste)
- Fig3b: Tuning curves para diferentes contrastes
- Fig3c: Correlaciones entre neuronas

Referencias:
- Echeveste et al. (2020), Nature Neuroscience
- Figure 3: "Generalization in the optimized network"
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración de matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
})

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Directorios
BASE_DIR = Path(__file__).parent.parent
SIMULATION_DIR = BASE_DIR / 'data' / 'simulations'
OUTPUT_DIR = Path(__file__).parent

# Niveles de contraste disponibles
CONTRAST_LEVELS = [0.0, 0.125, 0.25, 0.5, 1.0]

print("=" * 80)
print("GENERACIÓN DE FIGURA 3 - ECHEVESTE ET AL. (2020)")
print("=" * 80)
print(f"\nDirectorio de datos: {SIMULATION_DIR}")
print(f"Directorio de salida: {OUTPUT_DIR}")
print(f"Niveles de contraste: {CONTRAST_LEVELS}")
print()

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================


def load_simulation_data(contrast, suffix='_h_true_correcto'):
    """
    Cargar datos de simulación.

    Parameters
    ----------
    contrast : float
        Nivel de contraste
    suffix : str
        Sufijo del directorio

    Returns
    -------
    dict
        Datos de simulación
    """
    dir_name = f"contrast_{contrast:.3f}{suffix}"
    data_path = SIMULATION_DIR / dir_name / "simulation_data.npz"

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró: {data_path}")

    data = np.load(data_path)
    return {
        'u_E': data['u_E'],
        'u_I': data['u_I'],
        'r_E': data['r_E'],
        'r_I': data['r_I'],
        'time': data['time'],
        'orientations': data['orientations'],
        'dt': data['dt']
    }


# =============================================================================
# CARGAR DATOS
# =============================================================================

print("Cargando datos de simulación...")
all_data = {}
for contrast in CONTRAST_LEVELS:
    print(f"  Cargando contraste {contrast}...")
    all_data[contrast] = load_simulation_data(contrast)
    print(f"    ✓ Shape: {all_data[contrast]['u_E'].shape}")

orientations = all_data[0.0]['orientations']
print(f"\n✓ Datos cargados: {len(CONTRAST_LEVELS)} niveles de contraste")
print(f"✓ Orientaciones: {len(orientations)} neuronas")
print()

# =============================================================================
# FIGURA 3A: GENERALIZACIÓN POR CONTRASTE
# =============================================================================

print("Generando Fig3a: Generalización por contraste...")

# Calcular estadísticas por contraste (estado estacionario: últimos 200ms)
mean_rates = []
std_rates = []

for contrast in CONTRAST_LEVELS:
    data = all_data[contrast]
    time = data['time']
    r_E = data['r_E']

    # Tomar estado estacionario (últimos 200ms)
    steady_idx = time > (time[-1] - 200.0)
    r_E_steady = r_E[steady_idx]

    # Calcular media y std temporal para cada neurona
    mean_rates.append(np.mean(r_E_steady))
    std_rates.append(np.std(r_E_steady))

# Crear figura
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel izquierdo: Media
axes[0].plot(CONTRAST_LEVELS, mean_rates, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Contrast')
axes[0].set_ylabel('Mean firing rate (Hz)')
axes[0].set_title('Network response - Mean')
axes[0].grid(True, alpha=0.3)

# Panel derecho: Std
axes[1].plot(CONTRAST_LEVELS, std_rates, 'o-', linewidth=2, markersize=8, color='C1')
axes[1].set_xlabel('Contrast')
axes[1].set_ylabel('Std firing rate (Hz)')
axes[1].set_title('Network response - Std')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig3a_path = OUTPUT_DIR / 'Fig3a_contrast_generalization.png'
plt.savefig(fig3a_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Guardada: {fig3a_path}")

# =============================================================================
# FIGURA 3B: TUNING CURVES POR CONTRASTE
# =============================================================================

print("Generando Fig3b: Tuning curves...")

fig, axes = plt.subplots(1, len(CONTRAST_LEVELS), figsize=(20, 4), sharey=True)

for idx, contrast in enumerate(CONTRAST_LEVELS):
    data = all_data[contrast]
    time = data['time']
    r_E = data['r_E']
    orients = data['orientations']

    # Estado estacionario
    steady_idx = time > (time[-1] - 200.0)
    r_E_steady = r_E[steady_idx]

    # Media temporal
    mean_r_E = np.mean(r_E_steady, axis=0)

    # Graficar tuning curve
    axes[idx].plot(orients, mean_r_E, 'o-', linewidth=1.5)
    axes[idx].set_xlabel('Orientation (deg)')
    axes[idx].set_title(f'Contrast = {contrast}')
    axes[idx].grid(True, alpha=0.3)

    if idx == 0:
        axes[idx].set_ylabel('Mean firing rate (Hz)')

plt.suptitle('Tuning curves across contrasts', y=1.02)
plt.tight_layout()
fig3b_path = OUTPUT_DIR / 'Fig3b_tuning_curves.png'
plt.savefig(fig3b_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Guardada: {fig3b_path}")

# =============================================================================
# FIGURA 3C: MATRICES DE CORRELACIÓN
# =============================================================================

print("Generando Fig3c: Matrices de correlación...")

fig, axes = plt.subplots(1, len(CONTRAST_LEVELS), figsize=(20, 4))

for idx, contrast in enumerate(CONTRAST_LEVELS):
    data = all_data[contrast]
    time = data['time']
    r_E = data['r_E']

    # Estado estacionario
    steady_idx = time > (time[-1] - 200.0)
    r_E_steady = r_E[steady_idx]

    # Calcular correlaciones
    corr = np.corrcoef(r_E_steady.T)

    # Graficar
    im = axes[idx].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[idx].set_xlabel('Neuron index')
    axes[idx].set_title(f'Contrast = {contrast}')

    if idx == 0:
        axes[idx].set_ylabel('Neuron index')

# Colorbar compartido
fig.colorbar(im, ax=axes, label='Correlation', fraction=0.046, pad=0.04)

plt.suptitle('Correlation matrices across contrasts', y=1.02)
plt.tight_layout()
fig3c_path = OUTPUT_DIR / 'Fig3c_correlation_matrices.png'
plt.savefig(fig3c_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Guardada: {fig3c_path}")

# =============================================================================
# RESUMEN
# =============================================================================

print()
print("=" * 80)
print("RESUMEN DE FIGURAS GENERADAS")
print("=" * 80)
print()
print("✓ Fig3a: Generalización por contraste")
print(f"  - Archivo: {fig3a_path}")
print(f"  - Media y std vs contraste")
print()
print("✓ Fig3b: Tuning curves")
print(f"  - Archivo: {fig3b_path}")
print(f"  - Curvas de sintonización para {len(CONTRAST_LEVELS)} contrastes")
print()
print("✓ Fig3c: Matrices de correlación")
print(f"  - Archivo: {fig3c_path}")
print(f"  - Correlaciones entre neuronas para {len(CONTRAST_LEVELS)} contrastes")
print()
print("=" * 80)
print("✓ FIGURA 3 COMPLETA GENERADA")
print("=" * 80)
