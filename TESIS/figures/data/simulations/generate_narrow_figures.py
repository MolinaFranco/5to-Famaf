#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generar figuras para simulaciones con estímulos angostos.

Este script genera los gráficos de las simulaciones con:
- noise_level: 0.05
- width_factor: 0.01 (estímulos muy angostos)
- Alturas calibradas para coincidir con h_true de Echeveste

Figuras generadas:
- Fig_narrow_a: Population activity (heatmaps de u_E) para cada contraste
- Fig_narrow_b: Mean y std comparando con ideal observer
- Fig_narrow_c: Correlation matrices

Referencias:
- Echeveste et al. (2020), Nature Neuroscience
"""

import sys
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from skneuromsi.generative import GSM
from skneuromsi.data import EchevesteDataLoader, GSMDataLoader

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
    'axes.grid': False,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
})


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Directorios
BASE_DIR = Path(__file__).parent.parent.parent
SIMULATION_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent.parent / 'NarrowFig'

# Crear directorio de salida
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Niveles de contraste (5 niveles)
CONTRAST_LEVELS = [0.0, 0.125, 0.25, 0.5, 1.0]

# Colores
CONTRAST_COLORS = {
    0.0: '#2E4057',
    0.125: '#048A81',
    0.25: '#54C6EB',
    0.5: '#F18F01',
    1.0: '#C73E1D',
}

# Parámetros
N_E = 50
SEED = 42

print("=" * 70)
print("GENERACIÓN DE FIGURAS - ESTÍMULOS ANGOSTOS")
print("=" * 70)
print(f"\nDirectorio de datos: {SIMULATION_DIR}")
print(f"Directorio de salida: {OUTPUT_DIR}")
print(f"Niveles de contraste: {CONTRAST_LEVELS}")


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def load_simulation_data(contrast, suffix='_narrow_wf0.01'):
    """
    Cargar datos de simulación con estímulos angostos.

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
        'dt': float(data['dt']),
        'u_E_mean': data['u_E_mean'],
        'u_E_std': data['u_E_std'],
        'u_E_cov': data['u_E_cov'],
        'u_E_corr': data['u_E_corr'],
        'h_input': data['h_input'],
    }


def compute_posterior_moments_gsm_narrow(gsm, contrast, width_factor=0.01,
                                         dominant_idx=25):
    """
    Calcular momentos del posterior GSM para estímulo angosto.

    Parameters
    ----------
    gsm : GSM
        Modelo GSM
    contrast : float
        Nivel de contraste
    width_factor : float
        Factor de ancho del estímulo
    dominant_idx : int
        Índice de orientación dominante

    Returns
    -------
    mean, std, cov : np.ndarray
        Momentos del posterior
    """
    # Cargar datos para prior
    loader = EchevesteDataLoader()
    C = loader.load_prior_covariance()

    if contrast == 0.0:
        # Actividad espontánea
        mean = np.zeros(N_E)
        cov = C
        std = np.sqrt(np.diag(cov))
    else:
        # Amplitudes calibradas
        amplitude_map = {
            0.125: 7.7341,
            0.25: 15.4641,
            0.5: 30.9287,
            1.0: 61.8572,
        }
        amplitude = amplitude_map.get(contrast, 10.0)

        # Generar estímulo angosto
        y = gsm.generate_bump_stimulus(
            dominant_orientation_idx=dominant_idx,
            amplitude=amplitude,
            width_factor=width_factor,
            normalize=False
        )

        # Generar patch
        x_patch = gsm.generate_stimulus_from_y(y, contrast=1.0, add_noise=False)

        # Calcular posterior
        mean, cov = gsm.compute_posterior_moments(
            x_patch, z_map=contrast, add_baseline=True
        )
        std = np.sqrt(np.diag(cov))

    return mean, std, cov


# =============================================================================
# CARGAR DATOS
# =============================================================================

print("\nCargando datos de simulación...")
all_data = {}
for contrast in CONTRAST_LEVELS:
    print(f"  Cargando contraste {contrast}...")
    all_data[contrast] = load_simulation_data(contrast)
    print(f"    ✓ Shape: {all_data[contrast]['u_E'].shape}")
    print(f"    h_input max: {all_data[contrast]['h_input'].max():.4f}")

orientations = all_data[0.0]['orientations']

# Cargar GSM
print("\nCargando GSM...")
gsm_loader = GSMDataLoader()
params = gsm_loader.load_transformation_parameters()
gsm = GSM(
    use_pretrained=True,
    alpha_h=params['alpha_h'],
    beta_h=params['beta_h'],
    gamma_h=params['gamma_h']
)


# =============================================================================
# FIG NARROW A: POPULATION ACTIVITY
# =============================================================================

print("\nGenerando Fig_narrow_a: Population activity...")

fig_a = plt.figure(figsize=(14, 20))
fig_a.suptitle('SSN Responses - Narrow Stimuli (width_factor=0.01, noise=0.05)',
               fontsize=14, fontweight='bold')

# GridSpec: 5 contrastes x 2 columnas (heatmap + h_input)
gs = GridSpec(5, 3, figure=fig_a, hspace=0.35, wspace=0.15,
              width_ratios=[1, 0.3, 0.05])

# Escalas fijas
VMIN_U = 0.0
VMAX_U = 10.0
T_max = all_data[0.0]['time'][-1]

# Calcular escala de h
h_max_global = max(d['h_input'].max() for d in all_data.values())
h_min_global = min(d['h_input'].min() for d in all_data.values())

for idx, contrast in enumerate(CONTRAST_LEVELS):
    data = all_data[contrast]
    label = ['spontaneous', 'low', 'medium', 'high', 'very_high'][idx]

    # Columna 1: Heatmap u_E
    ax1 = fig_a.add_subplot(gs[idx, 0])
    im1 = ax1.imshow(
        data['u_E'].T,
        aspect='auto',
        cmap='coolwarm',
        extent=[0, T_max, orientations[0], orientations[-1]],
        origin='lower',
        vmin=VMIN_U,
        vmax=VMAX_U
    )
    ax1.set_ylabel('Orientation (°)')
    ax1.set_title(f'Contrast {contrast} ({label})')
    ax1.set_ylim(orientations[0], orientations[-1])
    ax1.set_yticks([-90, -45, 0, 45, 90])
    ax1.set_xlim(0, T_max)
    ax1.set_xticks(np.arange(0, T_max + 1, 200))

    if idx == len(CONTRAST_LEVELS) - 1:
        ax1.set_xlabel('Time (ms)')
    else:
        ax1.set_xticklabels([])

    # Columna 2: h_input
    ax2 = fig_a.add_subplot(gs[idx, 1])
    ax2.plot(data['h_input'], orientations, 'k-', linewidth=2)
    ax2.fill_betweenx(orientations, 0, data['h_input'], alpha=0.3, color='C0')
    ax2.set_ylim(orientations[0], orientations[-1])
    ax2.set_xlim(h_min_global, h_max_global * 1.1)
    ax2.set_yticks([])
    ax2.set_title(f'h (max={data["h_input"].max():.2f})')

    if idx == len(CONTRAST_LEVELS) - 1:
        ax2.set_xlabel('h (a.u.)')

# Colorbar compartida
cax = fig_a.add_subplot(gs[:, 2])
cbar = fig_a.colorbar(im1, cax=cax, label='$u_E$ (a.u.)')

plt.tight_layout()
fig_a_path = OUTPUT_DIR / 'narrow_fig_a_population_activity.png'
plt.savefig(fig_a_path, dpi=300, bbox_inches='tight')
print(f"✓ Guardada: {fig_a_path}")
plt.close()


# =============================================================================
# FIG NARROW B: MEAN AND STD
# =============================================================================

print("\nGenerando Fig_narrow_b: Mean and std...")

# Recolectar momentos
all_post_means = []
all_post_stds = []
all_net_means = []
all_net_stds = []

for contrast in CONTRAST_LEVELS:
    # Red
    data = all_data[contrast]
    all_net_means.append(data['u_E_mean'])
    all_net_stds.append(data['u_E_std'])

    # Ideal observer con estímulo angosto
    post_mean, post_std, post_cov = compute_posterior_moments_gsm_narrow(
        gsm, contrast, width_factor=0.01, dominant_idx=25
    )
    all_post_means.append(post_mean)
    all_post_stds.append(post_std)

# Crear figura
fig_b = plt.figure(figsize=(12, 8))
fig_b.suptitle('Mean and Std - Narrow Stimuli vs Ideal Observer',
               fontsize=14, fontweight='bold')
gs = GridSpec(2, 2, figure=fig_b, hspace=0.3, wspace=0.3)

# Determinar escala Y consistente
max_mean = max(max(m.max() for m in all_post_means),
               max(m.max() for m in all_net_means))
max_std = max(max(s.max() for s in all_post_stds),
              max(s.max() for s in all_net_stds))
y_max_mean = max_mean * 1.1
y_max_std = max_std * 1.1

# Subplot 1: Ideal Observer mean
ax1 = fig_b.add_subplot(gs[0, 0])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax1.plot(orientations, all_post_means[i], color=CONTRAST_COLORS[contrast],
             linewidth=2, label=f'c={contrast:.3f}')
ax1.set_xlabel('Preferred Orientation (°)')
ax1.set_ylabel('Mean $u_E$ (mV)')
ax1.set_title('Ideal Observer: Mean')
ax1.set_xlim(-90, 90)
ax1.set_ylim(0, y_max_mean)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=8)

# Subplot 2: Network mean
ax2 = fig_b.add_subplot(gs[0, 1])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax2.plot(orientations, all_net_means[i], color=CONTRAST_COLORS[contrast],
             linewidth=2, label=f'c={contrast:.3f}')
ax2.set_xlabel('Preferred Orientation (°)')
ax2.set_ylabel('Mean $u_E$ (mV)')
ax2.set_title('Network: Mean')
ax2.set_xlim(-90, 90)
ax2.set_ylim(0, y_max_mean)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=8)

# Subplot 3: Ideal Observer std
ax3 = fig_b.add_subplot(gs[1, 0])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax3.plot(orientations, all_post_stds[i], color=CONTRAST_COLORS[contrast],
             linewidth=2, label=f'c={contrast:.3f}')
ax3.set_xlabel('Preferred Orientation (°)')
ax3.set_ylabel('Std $u_E$ (mV)')
ax3.set_title('Ideal Observer: Standard Deviation')
ax3.set_xlim(-90, 90)
ax3.set_ylim(0, y_max_std)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best', fontsize=8)

# Subplot 4: Network std
ax4 = fig_b.add_subplot(gs[1, 1])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax4.plot(orientations, all_net_stds[i], color=CONTRAST_COLORS[contrast],
             linewidth=2, label=f'c={contrast:.3f}')
ax4.set_xlabel('Preferred Orientation (°)')
ax4.set_ylabel('Std $u_E$ (mV)')
ax4.set_title('Network: Standard Deviation')
ax4.set_xlim(-90, 90)
ax4.set_ylim(0, y_max_std)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='best', fontsize=8)

plt.tight_layout()
fig_b_path = OUTPUT_DIR / 'narrow_fig_b_mean_std.png'
plt.savefig(fig_b_path, dpi=300, bbox_inches='tight')
print(f"✓ Guardada: {fig_b_path}")
plt.close()


# =============================================================================
# FIG NARROW C: CORRELATION MATRICES
# =============================================================================

print("\nGenerando Fig_narrow_c: Correlation matrices...")

fig_c = plt.figure(figsize=(12, 20))
fig_c.suptitle('Correlation Matrices - Narrow Stimuli',
               fontsize=14, fontweight='bold')
gs = GridSpec(5, 2, figure=fig_c, hspace=0.4, wspace=0.3)

all_axes = []

for idx, contrast in enumerate(CONTRAST_LEVELS):
    print(f"  Procesando contraste {contrast}...")

    # Red neuronal
    data = all_data[contrast]
    net_corr = data['u_E_corr']

    # Ideal observer
    post_mean, post_std, post_cov = compute_posterior_moments_gsm_narrow(
        gsm, contrast, width_factor=0.01, dominant_idx=25
    )

    # Convertir covarianza a correlación
    post_corr = np.zeros_like(post_cov)
    for i in range(N_E):
        for j in range(N_E):
            denom = np.sqrt(post_cov[i, i]) * np.sqrt(post_cov[j, j])
            if denom > 0:
                post_corr[i, j] = post_cov[i, j] / denom
            else:
                post_corr[i, j] = 0.0

    # Subplot: Ideal observer
    ax1 = fig_c.add_subplot(gs[idx, 0])
    im1 = ax1.imshow(post_corr, cmap='RdBu_r', aspect='auto',
                     extent=[-90, 90, -90, 90], origin='lower',
                     vmin=-1, vmax=1)
    ax1.set_xlabel('Orientation (°)')
    ax1.set_ylabel('Orientation (°)')
    ax1.set_title(f'Ideal Observer (c={contrast})')
    all_axes.append(ax1)

    # Subplot: Network
    ax2 = fig_c.add_subplot(gs[idx, 1])
    im2 = ax2.imshow(net_corr, cmap='RdBu_r', aspect='auto',
                     extent=[-90, 90, -90, 90], origin='lower',
                     vmin=-1, vmax=1)
    ax2.set_xlabel('Orientation (°)')
    ax2.set_ylabel('Orientation (°)')
    ax2.set_title(f'Network (c={contrast})')
    all_axes.append(ax2)

# Colorbar compartida
cbar = fig_c.colorbar(im2, ax=all_axes, label='Correlation',
                      orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

plt.tight_layout()
fig_c_path = OUTPUT_DIR / 'narrow_fig_c_correlations.png'
plt.savefig(fig_c_path, dpi=300, bbox_inches='tight')
print(f"✓ Guardada: {fig_c_path}")
plt.close()


# =============================================================================
# FIG 2a STYLE: SAMPLE POPULATION ACTIVITY (como Fig2a original)
# =============================================================================

print("\nGenerando narrow_fig2a: Sample population activity (estilo Fig2a)...")

fig_2a = plt.figure(figsize=(10, 12))
gs_2a = GridSpec(3, 1, figure=fig_2a, hspace=0.35, height_ratios=[1, 1, 1])

# Escala de colores fija
VMIN = 0.0
VMAX = 10.0
T_max = all_data[0.0]['time'][-1]

# Panel 1: Contraste 0.0 (espontáneo)
ax1 = fig_2a.add_subplot(gs_2a[0, 0])
data_zero = all_data[0.0]
u_E_plot = data_zero['u_E'].T

im1 = ax1.imshow(
    u_E_plot,
    aspect='auto',
    cmap='coolwarm',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=VMIN,
    vmax=VMAX
)

ax1.set_ylabel('Preferred Orientation (°)')
ax1.set_title('Zero Contrast (Spontaneous) - Narrow Stimulus')
ax1.set_ylim(orientations[0], orientations[-1])
ax1.set_yticks([-90, -45, 0, 45, 90])
ax1.set_xlim(0, T_max)
ax1.set_xticks(np.arange(0, T_max + 1, 200))
ax1.set_xticklabels([])

# Colorbar invisible pero mantiene espacio
cbar1 = plt.colorbar(im1, ax=ax1, label='')
cbar1.ax.set_visible(False)

# Panel 2: Contraste 1.0 (alto)
ax2 = fig_2a.add_subplot(gs_2a[1, 0])
data_high = all_data[1.0]
u_E_plot_high = data_high['u_E'].T

im2 = ax2.imshow(
    u_E_plot_high,
    aspect='auto',
    cmap='coolwarm',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=VMIN,
    vmax=VMAX
)

ax2.set_ylabel('Preferred Orientation (°)')
ax2.set_title('High Contrast (1.0) - Narrow Stimulus')
ax2.set_ylim(orientations[0], orientations[-1])
ax2.set_yticks([-90, -45, 0, 45, 90])
ax2.set_xlim(0, T_max)
ax2.set_xticks(np.arange(0, T_max + 1, 200))
ax2.set_xticklabels([])

# Colorbar en panel medio
cbar2 = plt.colorbar(im2, ax=ax2, label='$u_E$ (a.u.)')

# Panel 3: Estímulo (h_input) para contraste 1.0
ax3 = fig_2a.add_subplot(gs_2a[2, 0])

# Usar h_input de la simulación narrow
h_E_stimulus = data_high['h_input']

# Modo single-phase: estímulo constante
h_temporal = np.tile(h_E_stimulus[:, None], (1, int(T_max)))

im3 = ax3.imshow(
    h_temporal,
    aspect='auto',
    cmap='gist_heat_r',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=0,
    vmax=h_E_stimulus.max()
)

ax3.set_xlim(0, T_max)
ax3.set_xticks(np.arange(0, T_max + 1, 200))
ax3.set_ylabel('Preferred Orientation (°)')
ax3.set_xlabel('Time (ms)')
ax3.set_title('Narrow Stimulus Input (h) - width_factor=0.01')
ax3.set_ylim(orientations[0], orientations[-1])
ax3.set_yticks([-90, 0, 90])

# Colorbar
cbar3 = plt.colorbar(im3, ax=ax3, label='h (a.u.)')

plt.tight_layout()
fig_2a_path = OUTPUT_DIR / 'narrow_fig2a_population_activity.png'
plt.savefig(fig_2a_path, dpi=300, bbox_inches='tight')
print(f"✓ Guardada: {fig_2a_path}")
plt.close()


# =============================================================================
# FIG 2a STYLE OPEN: SAMPLE POPULATION ACTIVITY (escala libre compartida)
# =============================================================================

print("\nGenerando narrow_fig2a_open: Sample population activity (escala libre)...")

fig_2a_open = plt.figure(figsize=(10, 12))
gs_2a_open = GridSpec(3, 1, figure=fig_2a_open, hspace=0.35, height_ratios=[1, 1, 1])

T_max = all_data[0.0]['time'][-1]

# Calcular escala compartida (libre) para ambos paneles
data_zero = all_data[0.0]
data_high = all_data[1.0]
u_E_plot_zero = data_zero['u_E'].T
u_E_plot_high = data_high['u_E'].T

# Escala compartida basada en los datos reales
VMIN_OPEN = min(u_E_plot_zero.min(), u_E_plot_high.min())
VMAX_OPEN = max(u_E_plot_zero.max(), u_E_plot_high.max())

# Panel 1: Contraste 0.0 (espontáneo) - escala libre compartida
ax1_open = fig_2a_open.add_subplot(gs_2a_open[0, 0])

im1_open = ax1_open.imshow(
    u_E_plot_zero,
    aspect='auto',
    cmap='coolwarm',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=VMIN_OPEN,
    vmax=VMAX_OPEN
)

ax1_open.set_ylabel('Preferred Orientation (°)')
ax1_open.set_title('Zero Contrast (Spontaneous) - Narrow Stimulus')
ax1_open.set_ylim(orientations[0], orientations[-1])
ax1_open.set_yticks([-90, -45, 0, 45, 90])
ax1_open.set_xlim(0, T_max)
ax1_open.set_xticks(np.arange(0, T_max + 1, 200))
ax1_open.set_xticklabels([])

# Colorbar invisible pero mantiene espacio (igual que versión fija)
cbar1_open = plt.colorbar(im1_open, ax=ax1_open, label='')
cbar1_open.ax.set_visible(False)

# Panel 2: Contraste 1.0 (alto) - escala libre compartida
ax2_open = fig_2a_open.add_subplot(gs_2a_open[1, 0])

im2_open = ax2_open.imshow(
    u_E_plot_high,
    aspect='auto',
    cmap='coolwarm',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=VMIN_OPEN,
    vmax=VMAX_OPEN
)

ax2_open.set_ylabel('Preferred Orientation (°)')
ax2_open.set_title('High Contrast (1.0) - Narrow Stimulus')
ax2_open.set_ylim(orientations[0], orientations[-1])
ax2_open.set_yticks([-90, -45, 0, 45, 90])
ax2_open.set_xlim(0, T_max)
ax2_open.set_xticks(np.arange(0, T_max + 1, 200))
ax2_open.set_xticklabels([])

# Colorbar visible solo en panel 2
cbar2_open = plt.colorbar(im2_open, ax=ax2_open, label='$u_E$ (a.u.)')

# Panel 3: Estímulo (h_input) para contraste 1.0
ax3_open = fig_2a_open.add_subplot(gs_2a_open[2, 0])

h_E_stimulus = data_high['h_input']
h_temporal = np.tile(h_E_stimulus[:, None], (1, int(T_max)))

im3_open = ax3_open.imshow(
    h_temporal,
    aspect='auto',
    cmap='gist_heat_r',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=0,
    vmax=h_E_stimulus.max()
)

ax3_open.set_xlim(0, T_max)
ax3_open.set_xticks(np.arange(0, T_max + 1, 200))
ax3_open.set_ylabel('Preferred Orientation (°)')
ax3_open.set_xlabel('Time (ms)')
ax3_open.set_title('Narrow Stimulus Input (h) - width_factor=0.01')
ax3_open.set_ylim(orientations[0], orientations[-1])
ax3_open.set_yticks([-90, 0, 90])

cbar3_open = plt.colorbar(im3_open, ax=ax3_open, label='h (a.u.)')

plt.tight_layout()
fig_2a_open_path = OUTPUT_DIR / 'narrow_fig2a_population_activity_open.png'
plt.savefig(fig_2a_open_path, dpi=300, bbox_inches='tight')
print(f"✓ Guardada: {fig_2a_open_path}")
plt.close()


# =============================================================================
# RESUMEN
# =============================================================================

print("\n" + "=" * 70)
print("RESUMEN DE FIGURAS GENERADAS")
print("=" * 70)
print(f"\n✓ Fig_narrow_a: Population activity (5 contrastes)")
print(f"  - Archivo: {fig_a_path}")
print(f"  - 5 niveles de contraste con heatmaps y h_input")

print(f"\n✓ Fig_narrow_b: Mean and std")
print(f"  - Archivo: {fig_b_path}")
print(f"  - Comparación ideal observer vs network")

print(f"\n✓ Fig_narrow_c: Correlation matrices")
print(f"  - Archivo: {fig_c_path}")
print(f"  - Correlaciones para cada contraste")

print(f"\n✓ Narrow_fig2a: Sample population activity (estilo Fig2a)")
print(f"  - Archivo: {fig_2a_path}")
print(f"  - Contraste 0 y 1.0 con estímulo narrow (escala fija 0-10)")

print(f"\n✓ Narrow_fig2a_open: Sample population activity (escala libre)")
print(f"  - Archivo: {fig_2a_open_path}")
print(f"  - Contraste 0 y 1.0 con estímulo narrow (escala libre)")

print("\n" + "=" * 70)
print("✓ FIGURAS DE ESTÍMULOS ANGOSTOS COMPLETADAS")
print("=" * 70)
