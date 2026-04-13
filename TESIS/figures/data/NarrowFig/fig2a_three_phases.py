#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generar figura de actividad poblacional con tres fases temporales.

Esta figura muestra la actividad de la red neuronal durante:
1. Pre-estímulo: actividad espontánea (sin estímulo)
2. Estímulo: respuesta a un estímulo de alto contraste
3. Post-estímulo: retorno a actividad espontánea

Usa datos de simulación real con transiciones naturales.

Referencias:
- Echeveste et al. (2020), Nature Neuroscience
"""

import sys
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# Configuracion de matplotlib
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
    'axes.grid': False,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
})


# =============================================================================
# CONFIGURACION
# =============================================================================

# Directorios
SIMULATION_DIR = Path(__file__).parent.parent / 'simulations'
OUTPUT_DIR = Path(__file__).parent

# Nombre de la simulacion con tres fases
THREE_PHASE_DIR = 'three_phases_1000ms_c1.0'

# Parametros
GAUSSIAN_SIGMA = 10  # Sigma para suavizado gaussiano temporal

print("=" * 70)
print("GENERACION DE FIGURA CON TRES FASES TEMPORALES")
print("=" * 70)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def smooth_temporal(data, sigma):
    """
    Aplicar suavizado gaussiano a lo largo del eje temporal.

    Parameters
    ----------
    data : np.ndarray
        Datos de forma (time, neurons)
    sigma : float
        Desviacion estandar del filtro gaussiano

    Returns
    -------
    np.ndarray
        Datos suavizados
    """
    smoothed = np.zeros_like(data)
    for i in range(data.shape[1]):
        smoothed[:, i] = gaussian_filter1d(data[:, i], sigma=sigma)
    return smoothed


# =============================================================================
# CARGAR DATOS
# =============================================================================

print("\nCargando datos de simulacion con tres fases...")

data_path = SIMULATION_DIR / THREE_PHASE_DIR / "simulation_data.npz"
if not data_path.exists():
    raise FileNotFoundError(f"No se encontro: {data_path}")

data = np.load(data_path)

u_E = data['u_E']
time = data['time']
orientations = data['orientations']
dt = float(data['dt'])
h_input = data['h_input']
h_temporal = data['h_temporal']

# Indices de fases
n_pre = int(data['n_pre'])
n_stim = int(data['n_stim'])
n_post = int(data['n_post'])

# Duraciones de fases
t_init = float(data['t_init'])
t_stimulus = float(data['t_stimulus'])
t_final = float(data['t_final'])

print(f"  Datos cargados: {u_E.shape}")
print(f"  Tiempo total: {time[-1]:.1f} ms")
print(f"  Fases: Pre={t_init}ms, Stim={t_stimulus}ms, Post={t_final}ms")

# Calcular limites de las fases
pre_end = t_init
stim_end = t_init + t_stimulus
T_max = time[-1]


# =============================================================================
# GENERAR FIGURA PRINCIPAL
# =============================================================================

print("\nGenerando figura principal...")

# Figura con dos paneles de IGUAL altura
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 1, figure=fig, hspace=0.25, height_ratios=[1, 1])

# Aplicar suavizado gaussiano a la actividad
u_E_smoothed = smooth_temporal(u_E, GAUSSIAN_SIGMA)

# Calcular escala de colores basada en los datos
VMIN = u_E_smoothed.min()
VMAX = u_E_smoothed.max()

# Escala de h
h_max = h_temporal.max()
h_vmin = 0
h_vmax = h_max if h_max > 0 else 1.0

# -----------------------------------------------------------------------------
# Panel 1: Actividad poblacional (u_E)
# -----------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])

im1 = ax1.imshow(
    u_E_smoothed.T,  # Transponer para (neurons, time)
    aspect='auto',
    cmap='coolwarm',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=VMIN,
    vmax=VMAX
)

# Lineas verticales para delimitar fases
ax1.axvline(x=pre_end, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.axvline(x=stim_end, color='white', linestyle='--', linewidth=1.5, alpha=0.8)

# Etiquetas de fases en la parte superior
ax1.text(t_init / 2, 85, 'Pre', ha='center', va='center',
         fontsize=11, fontweight='bold', color='white',
         bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7))
ax1.text(pre_end + t_stimulus / 2, 85, 'Stimulus', ha='center', va='center',
         fontsize=11, fontweight='bold', color='white',
         bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.7))
ax1.text(stim_end + t_final / 2, 85, 'Post', ha='center', va='center',
         fontsize=11, fontweight='bold', color='white',
         bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7))

ax1.set_ylabel('Preferred Orientation (deg)')
ax1.set_title('Population Activity ($u_E$): Three Temporal Phases', fontsize=13)
ax1.set_ylim(orientations[0], orientations[-1])
ax1.set_yticks([-90, -45, 0, 45, 90])
ax1.set_xlim(0, T_max)
ax1.set_xticks(np.arange(0, T_max + 1, 100))
ax1.set_xticklabels([])

# Colorbar
cbar1 = plt.colorbar(im1, ax=ax1, label='$u_E$ (a.u.)', pad=0.02)

# -----------------------------------------------------------------------------
# Panel 2: Entrada del estimulo (h)
# -----------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[1, 0])

im2 = ax2.imshow(
    h_temporal,
    aspect='auto',
    cmap='gist_heat_r',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=h_vmin,
    vmax=h_vmax
)

# Lineas verticales para delimitar fases
ax2.axvline(x=pre_end, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
ax2.axvline(x=stim_end, color='white', linestyle='--', linewidth=1.5, alpha=0.8)

ax2.set_ylabel('Preferred Orientation (deg)')
ax2.set_xlabel('Time (ms)')
ax2.set_title('Stimulus Input (h)', fontsize=12)
ax2.set_ylim(orientations[0], orientations[-1])
ax2.set_yticks([-90, -45, 0, 45, 90])
ax2.set_xlim(0, T_max)
ax2.set_xticks(np.arange(0, T_max + 1, 100))

# Colorbar
cbar2 = plt.colorbar(im2, ax=ax2, label='h (a.u.)', pad=0.02)

plt.tight_layout()

# Guardar figura
output_path = OUTPUT_DIR / 'fig2a_three_phases.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nGuardada: {output_path}")
plt.close()


# =============================================================================
# GENERAR VERSION COMPARATIVA (Espontaneo vs Tres Fases vs Estimulo continuo)
# =============================================================================

print("\nGenerando version comparativa...")

# Cargar datos de contraste 0 y 1.0 para comparacion
data_spont = np.load(
    SIMULATION_DIR / 'contrast_0.000_narrow_wf0.01' / 'simulation_data.npz'
)
data_stim = np.load(
    SIMULATION_DIR / 'contrast_1.000_narrow_wf0.01' / 'simulation_data.npz'
)

u_E_spont = data_spont['u_E']
u_E_stim_const = data_stim['u_E']
h_input_stim = data_stim['h_input']

# Suavizar
u_E_spont_smooth = smooth_temporal(u_E_spont, GAUSSIAN_SIGMA)
u_E_stim_smooth = smooth_temporal(u_E_stim_const, GAUSSIAN_SIGMA)

# Usar solo los primeros 1000ms para que coincida
n_samples = len(time)
u_E_spont_smooth = u_E_spont_smooth[:n_samples, :]
u_E_stim_smooth = u_E_stim_smooth[:n_samples, :]

# Escala comun para los tres paneles de actividad
all_data = [u_E_spont_smooth, u_E_smoothed, u_E_stim_smooth]
VMIN_ALL = min(d.min() for d in all_data)
VMAX_ALL = max(d.max() for d in all_data)

# Crear figura comparativa con cuatro paneles de IGUAL altura
fig2 = plt.figure(figsize=(14, 12))
gs2 = GridSpec(4, 1, figure=fig2, hspace=0.25, height_ratios=[1, 1, 1, 1])

# Panel 1: Actividad espontanea (contraste 0)
ax_spont = fig2.add_subplot(gs2[0, 0])
im_spont = ax_spont.imshow(
    u_E_spont_smooth.T,
    aspect='auto',
    cmap='coolwarm',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=VMIN_ALL,
    vmax=VMAX_ALL
)
ax_spont.set_ylabel('Orientation (deg)')
ax_spont.set_title('Spontaneous Activity (Contrast = 0)', fontsize=12)
ax_spont.set_ylim(orientations[0], orientations[-1])
ax_spont.set_yticks([-90, -45, 0, 45, 90])
ax_spont.set_xlim(0, T_max)
ax_spont.set_xticks(np.arange(0, T_max + 1, 100))
ax_spont.set_xticklabels([])
cbar_spont = plt.colorbar(im_spont, ax=ax_spont, label='', pad=0.02)
cbar_spont.ax.set_visible(False)

# Panel 2: Tres fases (simulacion real)
ax_three = fig2.add_subplot(gs2[1, 0])
im_three = ax_three.imshow(
    u_E_smoothed.T,
    aspect='auto',
    cmap='coolwarm',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=VMIN_ALL,
    vmax=VMAX_ALL
)

# Lineas de fases
ax_three.axvline(x=pre_end, color='white', linestyle='--',
                 linewidth=1.5, alpha=0.8)
ax_three.axvline(x=stim_end, color='white', linestyle='--',
                 linewidth=1.5, alpha=0.8)

# Etiquetas de fases
ax_three.text(t_init / 2, 85, 'Pre', ha='center', va='center',
              fontsize=10, fontweight='bold', color='white',
              bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7))
ax_three.text(pre_end + t_stimulus / 2, 85, 'Stim', ha='center', va='center',
              fontsize=10, fontweight='bold', color='white',
              bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.7))
ax_three.text(stim_end + t_final / 2, 85, 'Post', ha='center', va='center',
              fontsize=10, fontweight='bold', color='white',
              bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7))

ax_three.set_ylabel('Orientation (deg)')
ax_three.set_title('Three-Phase Protocol: Pre → Stimulus → Post', fontsize=12)
ax_three.set_ylim(orientations[0], orientations[-1])
ax_three.set_yticks([-90, -45, 0, 45, 90])
ax_three.set_xlim(0, T_max)
ax_three.set_xticks(np.arange(0, T_max + 1, 100))
ax_three.set_xticklabels([])
cbar_three = plt.colorbar(im_three, ax=ax_three, label='$u_E$ (a.u.)', pad=0.02)

# Panel 3: Estimulo continuo (contraste 1.0)
ax_stim = fig2.add_subplot(gs2[2, 0])
im_stim = ax_stim.imshow(
    u_E_stim_smooth.T,
    aspect='auto',
    cmap='coolwarm',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=VMIN_ALL,
    vmax=VMAX_ALL
)
ax_stim.set_ylabel('Orientation (deg)')
ax_stim.set_title('Constant Stimulus (Contrast = 1.0)', fontsize=12)
ax_stim.set_ylim(orientations[0], orientations[-1])
ax_stim.set_yticks([-90, -45, 0, 45, 90])
ax_stim.set_xlim(0, T_max)
ax_stim.set_xticks(np.arange(0, T_max + 1, 100))
ax_stim.set_xticklabels([])
cbar_stim = plt.colorbar(im_stim, ax=ax_stim, label='', pad=0.02)
cbar_stim.ax.set_visible(False)

# Panel 4: Perfil del estimulo (h) para las tres fases
ax_h = fig2.add_subplot(gs2[3, 0])
im_h = ax_h.imshow(
    h_temporal,
    aspect='auto',
    cmap='gist_heat_r',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=0,
    vmax=h_vmax
)

ax_h.axvline(x=pre_end, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
ax_h.axvline(x=stim_end, color='white', linestyle='--', linewidth=1.5, alpha=0.8)

ax_h.set_ylabel('Orientation (deg)')
ax_h.set_xlabel('Time (ms)')
ax_h.set_title('Stimulus Profile (h) for Three-Phase Protocol', fontsize=12)
ax_h.set_ylim(orientations[0], orientations[-1])
ax_h.set_yticks([-90, -45, 0, 45, 90])
ax_h.set_xlim(0, T_max)
ax_h.set_xticks(np.arange(0, T_max + 1, 100))
cbar_h = plt.colorbar(im_h, ax=ax_h, label='h (a.u.)', pad=0.02)

plt.tight_layout()

output_path2 = OUTPUT_DIR / 'fig2a_three_phases_comparison.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Guardada: {output_path2}")
plt.close()


# =============================================================================
# RESUMEN
# =============================================================================

print("\n" + "=" * 70)
print("FIGURAS GENERADAS")
print("=" * 70)

print(f"\n1. fig2a_three_phases.png")
print(f"   - Panel superior: Actividad poblacional con tres fases")
print(f"   - Panel inferior: Perfil temporal del estimulo")
print(f"   - Fases: Pre ({t_init:.0f}ms) → Stim ({t_stimulus:.0f}ms) → "
      f"Post ({t_final:.0f}ms)")
print(f"   - Transiciones naturales (simulacion real)")
print(f"   - Paneles con igual altura")

print(f"\n2. fig2a_three_phases_comparison.png")
print(f"   - Panel 1: Actividad espontanea continua (contraste=0)")
print(f"   - Panel 2: Protocolo de tres fases (simulacion real)")
print(f"   - Panel 3: Estimulo continuo (contraste=1.0)")
print(f"   - Panel 4: Perfil temporal del estimulo")
print(f"   - Escala de colores compartida para comparacion directa")
print(f"   - Paneles con igual altura")

print("\n" + "=" * 70)
print("COMPLETADO")
print("=" * 70)
