#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generar Figura 2 completa del paper Echeveste et al. (2020).

Este script genera todos los paneles de la Figura 2 a partir de los datos
de simulación previamente guardados en figuras/data/simulations/:

- Fig2a: Sample population activity (heatmaps de u_E)
- Fig2b: Covariance ellipses (ideal observer vs red)
- Fig2c: Mean y std de variables latentes vs potenciales membrana
- Fig2d: Correlation matrices (posterior vs red)

Referencias:
- Echeveste et al. (2020), Nature Neuroscience
- Fig. 2: "Inference and responses in the optimized network"
"""

import sys
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from pathlib import Path

from skneuromsi.generative import GSM  # noqa: E402
from skneuromsi.data import EchevesteDataLoader, GSMDataLoader  # noqa: E402

# Configuración de matplotlib para figuras de paper
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
BASE_DIR = Path(__file__).parent.parent
SIMULATION_DIR = BASE_DIR / 'data' / 'simulations'
OUTPUT_DIR = Path(__file__).parent

# Niveles de contraste disponibles
CONTRAST_LEVELS = [0.0, 0.125, 0.25, 0.5, 1.0]

# Colores para cada nivel de contraste
CONTRAST_COLORS = {
    0.0: '#2E4057',    # Azul oscuro (contraste cero)
    0.125: '#048A81',  # Verde azulado
    0.25: '#54C6EB',   # Azul cielo
    0.5: '#F18F01',    # Naranja
    1.0: '#C73E1D'     # Rojo (contraste alto)
}

# Parámetros
N_E = 50
SEED = 42

print("="*70)
print("GENERACIÓN DE FIGURA 2 COMPLETA")
print("="*70)
print(f"\nDirectorio de datos: {SIMULATION_DIR}")
print(f"Directorio de salida: {OUTPUT_DIR}")
print(f"Niveles de contraste: {CONTRAST_LEVELS}")


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
        'dt': float(data['dt']),
        'u_E_mean': data['u_E_mean'],
        'u_E_std': data['u_E_std'],
        'u_E_cov': data['u_E_cov'],
        'u_E_corr': data['u_E_corr'],
    }


def compute_stationary_moments(u_E, time, burn_in=200.0):
    """
    Compute mean, std, and covariance from stationary distribution.

    Parameters
    ----------
    u_E : np.ndarray, shape (n_times, N_E)
        Excitatory membrane potentials
    time : np.ndarray
        Time vector
    burn_in : float
        Burn-in time in ms

    Returns
    -------
    mean : np.ndarray, shape (N_E,)
    std : np.ndarray, shape (N_E,)
    cov : np.ndarray, shape (N_E, N_E)
    """
    stationary_idx = time >= burn_in
    u_stationary = u_E[stationary_idx, :]

    mean = np.mean(u_stationary, axis=0)
    std = np.std(u_stationary, axis=0)
    cov = np.cov(u_stationary.T)

    return mean, std, cov


def covariance_ellipse(cov, mean, n_std=2.0, **kwargs):
    """
    Create a matplotlib Ellipse representing a 2D covariance matrix.

    Parameters
    ----------
    cov : np.ndarray, shape (2, 2)
        2D covariance matrix
    mean : np.ndarray, shape (2,)
        2D mean vector
    n_std : float
        Number of standard deviations

    Returns
    -------
    ellipse : matplotlib.patches.Ellipse
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        **kwargs
    )

    return ellipse


def compute_posterior_moments_gsm(gsm_data, contrast, dominant_orientation=0.0):
    """
    Compute posterior moments from GSM for a given contrast.

    Para replicar las figuras del paper, generamos un patch sintético
    con orientación dominante específica (0° por defecto).

    Parameters
    ----------
    gsm_data : GSMDataLoader
        GSM data loader
    contrast : float
        Contrast level
    dominant_orientation : float
        Dominant orientation in degrees (default: 0°)

    Returns
    -------
    mean : np.ndarray, shape (N_E,)
    std : np.ndarray, shape (N_E,)
    cov : np.ndarray, shape (N_E, N_E)
    """
    # Cargar GSM pre-entrenado
    loader = EchevesteDataLoader()
    C = loader.load_prior_covariance()
    A = loader.load_gabor_filters()

    # Crear GSM
    gsm = GSM(
        patch_size=16,
        n_orientations=N_E,
        h_scale=1.0/15.0,
        random_seed=SEED,
        use_pretrained=True
    )

    if contrast == 0.0:
        # Actividad espontánea: prior del GSM
        mean = np.zeros(N_E)
        cov = C
        std = np.sqrt(np.diag(cov))
    else:
        # Encontrar índice de la orientación dominante
        orientations = np.linspace(-90, 90, N_E, endpoint=False)
        idx_dominant = np.argmin(np.abs(orientations - dominant_orientation))

        # Generar bump gaussiano usando el método del GSM
        # Ref: GSM.generate_bump_stimulus() implementa GSM.py líneas 339-348
        y = gsm.generate_bump_stimulus(
            dominant_orientation_idx=idx_dominant,
            amplitude=6.0,
            width_factor=0.15
        )

        # Generar patch sintético desde y
        # Ref: GSM.generate_stimulus_from_y() implementa GSM.py línea 364
        x_patch = gsm.generate_stimulus_from_y(y, contrast=contrast,
                                               add_noise=False)

        # Calcular posterior con baseline para comparación con red SSN
        # Ref: add_baseline=True añade 3.0 mV (GSM.py líneas 236, 468)
        mean, cov = gsm.compute_posterior_moments(
            x_patch, z_map=contrast, add_baseline=True
        )
        std = np.sqrt(np.diag(cov))

    return mean, std, cov


# =============================================================================
# CARGAR TODOS LOS DATOS DE SIMULACIÓN
# =============================================================================

print("\nCargando datos de simulación...")
all_data = {}
for contrast in CONTRAST_LEVELS:
    print(f"  Cargando contraste {contrast}...")
    all_data[contrast] = load_simulation_data(contrast)
    print(f"    ✓ Shape: {all_data[contrast]['u_E'].shape}")

orientations = all_data[0.0]['orientations']


# =============================================================================
# FIG 2a: SAMPLE POPULATION ACTIVITY
# =============================================================================

print("\nGenerando Fig2a: Sample population activity...")

fig2a = plt.figure(figsize=(10, 12))
gs = GridSpec(3, 1, figure=fig2a, hspace=0.35, height_ratios=[1, 1, 1])

# Escala de colores fija (misma para ambos paneles)
VMIN = 0.0
VMAX = 10.0

# Panel 1: Contraste 0.0 (espontáneo)
ax1 = fig2a.add_subplot(gs[0, 0])
data_zero = all_data[0.0]
u_E_plot = data_zero['u_E'].T
T_max = data_zero['time'][-1]

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
ax1.set_title('Zero Contrast (Spontaneous)')
ax1.set_ylim(orientations[0], orientations[-1])
ax1.set_yticks([-90, -45, 0, 45, 90])
ax1.set_xlim(0, T_max)
ax1.set_xticks(np.arange(0, T_max + 1, 200))
ax1.set_xticklabels([])  # Sin labels en el panel superior

# Colorbar invisible pero mantiene el espacio para igualar ancho con otros plots
cbar1 = plt.colorbar(im1, ax=ax1, label='')
cbar1.ax.set_visible(False)  # Hace invisible todo el colorbar pero mantiene espacio

# Panel 2: Contraste 1.0 (alto)
ax2 = fig2a.add_subplot(gs[1, 0])
data_high = all_data[1.0]
u_E_plot = data_high['u_E'].T

im2 = ax2.imshow(
    u_E_plot,
    aspect='auto',
    cmap='coolwarm',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=VMIN,
    vmax=VMAX
)

ax2.set_ylabel('Preferred Orientation (°)')
ax2.set_title('High Contrast (1.0)')
ax2.set_ylim(orientations[0], orientations[-1])
ax2.set_yticks([-90, -45, 0, 45, 90])
ax2.set_xlim(0, T_max)
ax2.set_xticks(np.arange(0, T_max + 1, 200))
ax2.set_xticklabels([])  # Sin labels en el panel medio

# Colorbar en panel medio
cbar2 = plt.colorbar(im2, ax=ax2, label='$u_E$ (a.u.)')

# Panel 3: Estímulo (h_input) para contraste 1.0
ax3 = fig2a.add_subplot(gs[2, 0])

# Cargar parámetros de la simulación para saber si usa tres fases
import json
params_path_high = SIMULATION_DIR / f"contrast_{1.0:.3f}_h_true_correcto" / "parameters.json"
with open(params_path_high, 'r') as f:
    sim_params = json.load(f)

use_three_phases = sim_params.get('use_three_phases', False)

# Cargar el h_input usado
from skneuromsi.data import GSMDataLoader
gsm_loader = GSMDataLoader()
h_input_high = gsm_loader.load_h_vec_transformed(4)  # h_true_4 (con estímulo)
h_E_stimulus = h_input_high[:50]  # Solo neuronas excitatorias

if use_three_phases:
    # MODO TRES FASES: mostrar estructura temporal
    # Leer info de fases desde parámetros (o usar defaults de transients.py)
    t_init = sim_params.get('t_init', 225.0)  # ms
    t_stimulus = sim_params.get('t_stimulus', 100.0)  # ms
    t_final = sim_params.get('t_final', 125.0)  # ms
    total_phases = t_init + t_stimulus + t_final

    # Cargar h_baseline para fases espontáneas
    h_input_baseline = gsm_loader.load_h_vec_transformed(0)  # h_true_0
    h_E_baseline = h_input_baseline[:50]

    # Crear array temporal con tres fases
    n_time_points = int(total_phases)
    h_temporal = np.zeros((N_E, n_time_points))

    idx_phase1_end = int(t_init)
    idx_phase2_end = int(t_init + t_stimulus)

    # Fase 1: Espontánea
    h_temporal[:, 0:idx_phase1_end] = h_E_baseline[:, None]
    # Fase 2: Estímulo
    h_temporal[:, idx_phase1_end:idx_phase2_end] = h_E_stimulus[:, None]
    # Fase 3: Espontánea
    h_temporal[:, idx_phase2_end:] = h_E_baseline[:, None]

    # Graficar con estructura de fases
    im3 = ax3.imshow(
        h_temporal,
        aspect='auto',
        cmap='gist_heat_r',
        extent=[0, total_phases, orientations[0], orientations[-1]],
        origin='lower',
        vmin=0,
        vmax=h_E_stimulus.max()
    )

    ax3.set_xlim(0, total_phases)
    ax3.set_xticks([0, t_init, t_init+t_stimulus, total_phases])
    ax3.set_xticklabels(['0', f'{t_init:.0f}\n(stim ON)',
                         f'{t_init+t_stimulus:.0f}\n(stim OFF)',
                         f'{total_phases:.0f}'])

    # Líneas verticales para transiciones
    ax3.axvline(t_init, color='white', linestyle='--', linewidth=1.5,
                alpha=0.8)
    ax3.axvline(t_init+t_stimulus, color='white', linestyle='--',
                linewidth=1.5, alpha=0.8)

    # Anotaciones de fases
    ax3.text(0.02, 0.98, 'Phase 1:\nSpontaneous', transform=ax3.transAxes,
             verticalalignment='top', fontsize=8, color='white',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax3.text(0.52, 0.98, 'Phase 2:\nStimulus', transform=ax3.transAxes,
             verticalalignment='top', fontsize=8, color='white',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax3.text(0.85, 0.98, 'Phase 3:\nSpontaneous', transform=ax3.transAxes,
             verticalalignment='top', fontsize=8, color='white',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    title_str = 'Stimulus Input (h) - Three Phases Structure'

else:
    # MODO SINGLE-PHASE: estímulo constante
    # Repetir h_E_stimulus en todo el tiempo para visualización
    h_temporal = np.tile(h_E_stimulus[:, None], (1, int(T_max)))

    # Graficar con estímulo constante
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

    title_str = 'Stimulus Input (h) - Constant Stimulus'

# Configuración común
ax3.set_ylabel('Preferred Orientation (°)')
ax3.set_xlabel('Time (ms)')
ax3.set_title(title_str)
ax3.set_ylim(orientations[0], orientations[-1])
ax3.set_yticks([-90, 0, 90])

# Colorbar
cbar3 = plt.colorbar(im3, ax=ax3, label='h (a.u.)')

plt.tight_layout()
fig2a_path = OUTPUT_DIR / 'Fig2a_population_activity.png'
plt.savefig(fig2a_path, dpi=300, bbox_inches='tight')
print(f"✓ Guardada: {fig2a_path}")
plt.close()


# =============================================================================
# FIG 2b: COVARIANCE ELLIPSES
# =============================================================================

print("\nGenerando Fig2b: Covariance ellipses...")

# Seleccionar dos neuronas representativas (42° y 16° aprox)
target_orientations = [42.0, 16.0]
neuron_indices = []
for target_ori in target_orientations:
    idx = np.argmin(np.abs(orientations - target_ori))
    neuron_indices.append(idx)
    print(f"  Neurona {idx}: orientación {orientations[idx]:.1f}°")

# Crear figura con 5 subplots verticales
fig2b = plt.figure(figsize=(6, 20))
gs = GridSpec(5, 1, figure=fig2b, hspace=0.4)

# Cargar GSM data
gsm_loader = GSMDataLoader()

for idx, contrast in enumerate(CONTRAST_LEVELS):
    print(f"  Procesando contraste {contrast}...")

    # Obtener datos de la red
    data = all_data[contrast]
    net_mean = data['u_E_mean']
    net_cov = data['u_E_cov']

    # Obtener datos del posterior (ideal observer) con orientación 0°
    post_mean, post_std, post_cov = compute_posterior_moments_gsm(
        gsm_loader, contrast, dominant_orientation=0.0
    )

    # Extraer submatrices 2D
    net_mean_2d = net_mean[neuron_indices]
    net_cov_2d = net_cov[np.ix_(neuron_indices, neuron_indices)]

    post_mean_2d = post_mean[neuron_indices]
    post_cov_2d = post_cov[np.ix_(neuron_indices, neuron_indices)]

    # Crear subplot
    ax = fig2b.add_subplot(gs[idx, 0])

    # Elipse del ideal observer (verde, línea punteada)
    ellipse_post = covariance_ellipse(
        post_cov_2d, post_mean_2d, n_std=2.0,
        facecolor='none',
        edgecolor='green',
        linewidth=2,
        linestyle='--',
        alpha=0.7
    )
    ax.add_patch(ellipse_post)

    # Elipse de la red (rojo, línea sólida)
    ellipse_net = covariance_ellipse(
        net_cov_2d, net_mean_2d, n_std=2.0,
        facecolor='none',
        edgecolor='red',
        linewidth=2,
        alpha=0.7
    )
    ax.add_patch(ellipse_net)

    # Graficar trayectoria de la red (últimos 500ms)
    traj_idx = data['time'] >= 500.0
    u_E_traj = data['u_E'][traj_idx, :][:, neuron_indices]

    color = CONTRAST_COLORS[contrast]
    ax.plot(
        u_E_traj[:, 0], u_E_traj[:, 1],
        color=color, alpha=0.3, linewidth=0.5
    )

    # Configurar ejes
    ax.set_xlabel(f'E cell at {orientations[neuron_indices[0]]:.0f}°')
    ax.set_ylabel(f'E cell at {orientations[neuron_indices[1]]:.0f}°')
    ax.set_title(f'Contrast = {contrast:.3f}')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Leyenda solo en el primer subplot
    if idx == 0:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, linestyle='--',
                   label='Ideal observer'),
            Line2D([0], [0], color='red', lw=2, label='Network')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
fig2b_path = OUTPUT_DIR / 'Fig2b_covariance_ellipses.png'
plt.savefig(fig2b_path, dpi=300, bbox_inches='tight')
print(f"✓ Guardada: {fig2b_path}")
plt.close()


# =============================================================================
# FIG 2c: MEAN AND STD
# =============================================================================

print("\nGenerando Fig2c: Mean and std...")

# Recolectar momentos para todos los contrastes
all_post_means = []
all_post_stds = []
all_net_means = []
all_net_stds = []

for contrast in CONTRAST_LEVELS:
    # Red
    data = all_data[contrast]
    all_net_means.append(data['u_E_mean'])
    all_net_stds.append(data['u_E_std'])

    # Ideal observer con orientación 0°
    post_mean, post_std, post_cov = compute_posterior_moments_gsm(
        gsm_loader, contrast, dominant_orientation=0.0
    )
    all_post_means.append(post_mean)
    all_post_stds.append(post_std)

# Crear figura
fig2c = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig2c, hspace=0.3, wspace=0.3)

# Subplot 1: Posterior mean (ideal observer)
ax1 = fig2c.add_subplot(gs[0, 0])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax1.plot(
        orientations, all_post_means[i],
        color=CONTRAST_COLORS[contrast],
        linewidth=2,
        label=f'c={contrast:.3f}'
    )
ax1.set_xlabel('Preferred Orientation (°)')
ax1.set_ylabel('Mean $u_E$ (mV)')
ax1.set_title('Ideal Observer: Mean')
ax1.set_xlim(-90, 90)
ax1.set_ylim(0, 8)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=8)

# Subplot 2: Network mean
ax2 = fig2c.add_subplot(gs[0, 1])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax2.plot(
        orientations, all_net_means[i],
        color=CONTRAST_COLORS[contrast],
        linewidth=2,
        label=f'c={contrast:.3f}'
    )
ax2.set_xlabel('Preferred Orientation (°)')
ax2.set_ylabel('Mean $u_E$ (mV)')
ax2.set_title('Network: Mean')
ax2.set_xlim(-90, 90)
ax2.set_ylim(0, 8)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=8)

# Subplot 3: Posterior std (ideal observer)
ax3 = fig2c.add_subplot(gs[1, 0])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax3.plot(
        orientations, all_post_stds[i],
        color=CONTRAST_COLORS[contrast],
        linewidth=2,
        label=f'c={contrast:.3f}'
    )
ax3.set_xlabel('Preferred Orientation (°)')
ax3.set_ylabel('Std $u_E$ (mV)')
ax3.set_title('Ideal Observer: Standard Deviation')
ax3.set_xlim(-90, 90)
ax3.set_ylim(0, 8)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best', fontsize=8)

# Subplot 4: Network std
ax4 = fig2c.add_subplot(gs[1, 1])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax4.plot(
        orientations, all_net_stds[i],
        color=CONTRAST_COLORS[contrast],
        linewidth=2,
        label=f'c={contrast:.3f}'
    )
ax4.set_xlabel('Preferred Orientation (°)')
ax4.set_ylabel('Std $u_E$ (mV)')
ax4.set_title('Network: Standard Deviation')
ax4.set_xlim(-90, 90)
ax4.set_ylim(0, 8)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='best', fontsize=8)

plt.tight_layout()
fig2c_path = OUTPUT_DIR / 'Fig2c_mean_std.png'
plt.savefig(fig2c_path, dpi=300, bbox_inches='tight')
print(f"✓ Guardada: {fig2c_path}")
plt.close()


# =============================================================================
# FIG 2d: CORRELATION MATRICES
# =============================================================================

print("\nGenerando Fig2d: Correlation matrices...")

# Crear figura con 5 filas x 2 columnas
fig2d = plt.figure(figsize=(12, 25))
gs = GridSpec(5, 2, figure=fig2d, hspace=0.4, wspace=0.3)

# Guardar los axes para crear colorbar compartida después
all_axes = []

for idx, contrast in enumerate(CONTRAST_LEVELS):
    print(f"  Procesando contraste {contrast}...")

    # Red neuronal
    data = all_data[contrast]
    net_corr = data['u_E_corr']

    # Ideal observer con orientación 0°
    post_mean, post_std, post_cov = compute_posterior_moments_gsm(
        gsm_loader, contrast, dominant_orientation=0.0
    )

    # Convertir covarianza a correlación
    post_corr = np.zeros_like(post_cov)
    for i in range(N_E):
        for j in range(N_E):
            post_corr[i, j] = post_cov[i, j] / (
                np.sqrt(post_cov[i, i]) * np.sqrt(post_cov[j, j])
            )

    # Subplot 1: Ideal observer
    ax1 = fig2d.add_subplot(gs[idx, 0])
    im1 = ax1.imshow(
        post_corr,
        cmap='RdBu_r',
        aspect='auto',
        extent=[-90, 90, -90, 90],
        origin='lower',
        vmin=-1, vmax=1
    )
    ax1.set_xlabel('Preferred Orientation (°)')
    ax1.set_ylabel('Preferred Orientation (°)')
    ax1.set_title(f'Ideal Observer (c={contrast:.3f})')
    ax1.set_xlim(-90, 90)
    ax1.set_ylim(-90, 90)
    all_axes.append(ax1)

    # Subplot 2: Network
    ax2 = fig2d.add_subplot(gs[idx, 1])
    im2 = ax2.imshow(
        net_corr,
        cmap='RdBu_r',
        aspect='auto',
        extent=[-90, 90, -90, 90],
        origin='lower',
        vmin=-1, vmax=1
    )
    ax2.set_xlabel('Preferred Orientation (°)')
    ax2.set_ylabel('Preferred Orientation (°)')
    ax2.set_title(f'Network (c={contrast:.3f})')
    ax2.set_xlim(-90, 90)
    ax2.set_ylim(-90, 90)
    all_axes.append(ax2)

# Añadir una única colorbar compartida al final (abajo derecha)
# Usar el último subplot (ax2 del último contraste)
cbar = fig2d.colorbar(im2, ax=all_axes, label='Correlation',
                      orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

plt.tight_layout()
fig2d_path = OUTPUT_DIR / 'Fig2d_correlation_matrices.png'
plt.savefig(fig2d_path, dpi=300, bbox_inches='tight')
print(f"✓ Guardada: {fig2d_path}")
plt.close()


# =============================================================================
# RESUMEN
# =============================================================================

print("\n" + "="*70)
print("RESUMEN DE FIGURAS GENERADAS")
print("="*70)
print("\n✓ Fig2a: Sample population activity")
print(f"  - Archivo: {fig2a_path}")
print("  - Contraste 0.0 (arriba) y 1.0 (abajo)")
print("  - Escala fija: 0-10 (misma en ambos paneles)")
print("  - Eje X: cada 200ms")

print("\n✓ Fig2b: Covariance ellipses")
print(f"  - Archivo: {fig2b_path}")
print(f"  - Neuronas: {orientations[neuron_indices[0]]:.0f}° y "
      f"{orientations[neuron_indices[1]]:.0f}°")
print("  - 5 niveles de contraste")
print("  - Verde: Ideal observer, Rojo: Red")

print("\n✓ Fig2c: Mean and standard deviation")
print(f"  - Archivo: {fig2c_path}")
print("  - Izquierda: Ideal observer")
print("  - Derecha: Red neuronal")
print("  - Orientación dominante: 0°")

print("\n✓ Fig2d: Correlation matrices")
print(f"  - Archivo: {fig2d_path}")
print("  - 5 niveles de contraste")
print("  - Ideal observer vs Red")

print("\n" + "="*70)
print("✓ FIGURA 2 COMPLETA GENERADA")
print("="*70)
