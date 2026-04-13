#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generar fig2c final con estilo Echeveste.

Combina:
- Izquierda: Ideal observer (posterior GSM transformado por no linealidad)
- Derecha: Network (SSN)

Basado en Echeveste et al. (2020), Nature Neuroscience, Figure 2c.
"""

import sys
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gamma as gamma_dist, multivariate_normal, norm
from pathlib import Path

from skneuromsi.generative import GSM
from skneuromsi.data import EchevesteDataLoader, GSMDataLoader

# =============================================================================
# CONFIGURACION
# =============================================================================

N_E = 50
SIMULATION_DIR = Path(__file__).parent.parent / 'simulations'
OUTPUT_DIR = Path(__file__).parent

# Niveles de contraste
CONTRAST_LEVELS = [0.0, 0.125, 0.25, 0.5, 1.0]

# Colores estilo Echeveste
# Verdes para posterior (de claro a oscuro con contraste)
GREEN_COLORS = {
    0.0: '#a8ddb5',    # muy claro
    0.125: '#7bccc4',  # claro
    0.25: '#4eb3d3',   # medio (más hacia azul-verde)
    0.5: '#2b8cbe',    # oscuro
    1.0: '#08589e',    # muy oscuro
}

# Naranjas/rojos para network
ORANGE_COLORS = {
    0.0: '#fec44f',    # amarillo
    0.125: '#fe9929',  # naranja claro
    0.25: '#ec7014',   # naranja
    0.5: '#cc4c02',    # naranja oscuro
    1.0: '#8c2d04',    # rojo oscuro
}

# Parametros de la no linealidad (Echeveste)
NL_SCALE = 2.4
NL_POWER = 0.6
NL_BASELINE = (3.5 / NL_SCALE) ** (1.0 / NL_POWER)
BASELINE = 3.0  # baseline en mV

# Parametros GSM
GAMMA_K = 2.0
GAMMA_THETA = 2.0
s_x_2 = 100.0


# =============================================================================
# FUNCIONES
# =============================================================================

def np_threshold(u):
    return np.maximum(u, 0)


def nl_function(u):
    return NL_SCALE * (np_threshold(u + NL_BASELINE) ** NL_POWER)


def transform_moments_nonlinear(mu_gsm, std_gsm, n_points=201):
    """Transformar momentos GSM a traves de la no linealidad del SSN."""
    n_neurons = len(mu_gsm)
    points_1d = np.linspace(-4.0, 4.0, n_points)

    mu_nl = np.empty(n_neurons)
    var_nl = np.empty(n_neurons)

    for i in range(n_neurons):
        mu_i = mu_gsm[i]
        std_i = std_gsm[i]
        points_resc = std_i * points_1d + mu_i
        values_dist = norm.pdf(points_resc, loc=mu_i, scale=std_i)
        norm_f = np.sum(values_dist)
        values_fun = nl_function(points_resc)
        mu_nl[i] = np.sum(values_dist * values_fun) / norm_f
        var_nl[i] = np.sum(values_dist * (values_fun - mu_nl[i])**2) / norm_f

    return mu_nl, np.sqrt(var_nl)


def load_simulation_data(contrast, suffix='_narrow_wf0.01'):
    """Cargar datos de simulacion."""
    dir_name = f"contrast_{contrast:.3f}{suffix}"
    data_path = SIMULATION_DIR / dir_name / "simulation_data.npz"
    data = np.load(data_path)
    return {
        'u_E_mean': data['u_E_mean'],
        'u_E_std': data['u_E_std'],
        'orientations': data['orientations'],
    }


def compute_posterior_with_nl(gsm, A, C, C_inv, ATA, ACAT, contrast,
                               width_factor=0.01, dominant_idx=25):
    """Calcular posterior GSM con transformacion no lineal."""
    if contrast == 0.0:
        mu_gsm = np.zeros(N_E) + BASELINE
        std_gsm = np.sqrt(np.diag(C))
        mu_nl, std_nl = transform_moments_nonlinear(mu_gsm, std_gsm)
        return mu_nl, std_nl

    # Amplitudes calibradas
    amplitude_map = {0.125: 7.7341, 0.25: 15.4641, 0.5: 30.9287, 1.0: 61.8572}
    amplitude = amplitude_map.get(contrast, 10.0)

    # Generar estimulo
    y = gsm.generate_bump_stimulus(
        dominant_orientation_idx=dominant_idx,
        amplitude=amplitude,
        width_factor=width_factor,
        normalize=False
    )
    x = contrast * (A @ y)

    # P(z|x)
    z_range = np.linspace(0.001, 5.0, 201)
    n_points = len(z_range)
    D_x = len(x)
    log_p = np.empty(n_points)
    mean_vec = np.zeros(D_x)

    for i in range(n_points):
        z = z_range[i]
        cov_x = z * z * ACAT + s_x_2 * np.identity(D_x)
        log_p[i] = (gamma_dist.logpdf(z, GAMMA_K, loc=0, scale=GAMMA_THETA) +
                    multivariate_normal.logpdf(x, mean_vec, cov_x))

    max_lp = np.max(log_p)
    p_z = np.exp(log_p - max_lp)
    dz = z_range[1] - z_range[0]
    norm_val = np.sum(p_z) * dz
    p_z = p_z / norm_val

    # Full inference
    mu_post = np.zeros(N_E)
    Sigma_post = np.zeros((N_E, N_E))

    threshold_val = 10000.0
    i_p_max = np.argmax(p_z)
    p_max = p_z[i_p_max]

    i_min, i_max = 0, n_points
    for i in range(i_p_max, -1, -1):
        if p_z[i] <= (p_max / threshold_val):
            i_min = i
            break
    for i in range(i_p_max, n_points):
        if p_z[i] <= (p_max / threshold_val):
            i_max = i
            break

    for i in range(i_min, i_max):
        z = z_range[i]
        if z < 1e-10:
            Sigma_z = C
            mu_z = np.zeros(N_E)
        else:
            M = C_inv + (z**2 / s_x_2) * ATA
            Sigma_z = np.linalg.inv(M)
            mu_z = (z / s_x_2) * Sigma_z @ (A.T @ x)

        mu_post += mu_z * p_z[i]
        Sigma_post += (Sigma_z + np.outer(mu_z, mu_z)) * p_z[i]

    mu_post *= dz
    Sigma_post *= dz
    Sigma_post -= np.outer(mu_post, mu_post)

    # Agregar baseline y transformar
    mu_gsm = mu_post + BASELINE
    std_gsm = np.sqrt(np.diag(Sigma_post))

    mu_nl, std_nl = transform_moments_nonlinear(mu_gsm, std_gsm)
    return mu_nl, std_nl


# =============================================================================
# CARGAR DATOS
# =============================================================================

print("Cargando datos...")

# GSM y matrices
gsm_loader = GSMDataLoader()
params = gsm_loader.load_transformation_parameters()
gsm = GSM(
    use_pretrained=True,
    alpha_h=params['alpha_h'],
    beta_h=params['beta_h'],
    gamma_h=params['gamma_h']
)

loader = EchevesteDataLoader()
A = loader.load_gabor_filters()
C = loader.load_prior_covariance()
C_inv = np.linalg.inv(C)
ATA = A.T @ A
ACAT = A @ C @ A.T

# Datos de simulacion
all_data = {}
for contrast in CONTRAST_LEVELS:
    all_data[contrast] = load_simulation_data(contrast)

orientations = all_data[0.0]['orientations']

# Calcular momentos
print("Calculando momentos del posterior (con NL)...")
post_means = []
post_stds = []
net_means = []
net_stds = []

for contrast in CONTRAST_LEVELS:
    print(f"  Contraste {contrast}...")
    # Posterior con NL
    mu_nl, std_nl = compute_posterior_with_nl(
        gsm, A, C, C_inv, ATA, ACAT, contrast
    )
    post_means.append(mu_nl)
    post_stds.append(std_nl)

    # Network
    net_means.append(all_data[contrast]['u_E_mean'])
    net_stds.append(all_data[contrast]['u_E_std'])


# =============================================================================
# CREAR FIGURA ESTILO ECHEVESTE
# =============================================================================

print("\nGenerando figura estilo Echeveste...")

# Configuracion de matplotlib para estilo paper
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

# Ejes x compartido
x_ticks = [-90, 0, 90]
x_labels = ['-90°', '0°', '90°']

# --- Row 0: Mean ---

# Subplot [0,0]: Posterior mean
ax_post_mean = fig.add_subplot(gs[0, 0])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax_post_mean.plot(orientations, post_means[i],
                      color=GREEN_COLORS[contrast], linewidth=2)
ax_post_mean.set_ylabel(r'mean $u_E$ [mV]')
ax_post_mean.set_title('posterior', fontweight='bold')
ax_post_mean.set_xlim(-90, 90)
ax_post_mean.set_ylim(0, 10)
ax_post_mean.set_xticks(x_ticks)
ax_post_mean.set_xticklabels([])  # Sin labels en fila superior
ax_post_mean.spines['top'].set_visible(False)
ax_post_mean.spines['right'].set_visible(False)

# Subplot [0,1]: Network mean
ax_net_mean = fig.add_subplot(gs[0, 1])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax_net_mean.plot(orientations, net_means[i],
                     color=ORANGE_COLORS[contrast], linewidth=2)
ax_net_mean.set_title('network', fontweight='bold')
ax_net_mean.set_xlim(-90, 90)
ax_net_mean.set_ylim(0, 10)
ax_net_mean.set_xticks(x_ticks)
ax_net_mean.set_xticklabels([])
ax_net_mean.set_yticklabels([])  # Compartir escala con izquierda
ax_net_mean.spines['top'].set_visible(False)
ax_net_mean.spines['right'].set_visible(False)

# --- Row 1: Std ---

# Subplot [1,0]: Posterior std
ax_post_std = fig.add_subplot(gs[1, 0])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax_post_std.plot(orientations, post_stds[i],
                     color=GREEN_COLORS[contrast], linewidth=2)
ax_post_std.set_ylabel(r'$u_E$ std. [mV]')
ax_post_std.set_xlabel('pref. ori.')
ax_post_std.set_xlim(-90, 90)
ax_post_std.set_ylim(0, 3)
ax_post_std.set_xticks(x_ticks)
ax_post_std.set_xticklabels(x_labels)
ax_post_std.spines['top'].set_visible(False)
ax_post_std.spines['right'].set_visible(False)

# Subplot [1,1]: Network std
ax_net_std = fig.add_subplot(gs[1, 1])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax_net_std.plot(orientations, net_stds[i],
                    color=ORANGE_COLORS[contrast], linewidth=2)
ax_net_std.set_xlabel('pref. ori.')
ax_net_std.set_xlim(-90, 90)
ax_net_std.set_ylim(0, 3)
ax_net_std.set_xticks(x_ticks)
ax_net_std.set_xticklabels(x_labels)
ax_net_std.set_yticklabels([])
ax_net_std.spines['top'].set_visible(False)
ax_net_std.spines['right'].set_visible(False)

# Agregar letra "c" en esquina superior izquierda (estilo paper)
fig.text(0.02, 0.95, 'c', fontsize=16, fontweight='bold',
         transform=fig.transFigure)

plt.tight_layout()

# Guardar
output_path = OUTPUT_DIR / 'fig2c_echeveste_style.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Guardada: {output_path}")

# Version PDF para paper
output_pdf = OUTPUT_DIR / 'fig2c_echeveste_style.pdf'
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"Guardada: {output_pdf}")

plt.close()

# =============================================================================
# IMPRIMIR VALORES PARA VERIFICACION
# =============================================================================

print("\n" + "="*60)
print("VALORES DE VERIFICACION")
print("="*60)

print("\n--- Posterior (con NL) ---")
for i, c in enumerate(CONTRAST_LEVELS):
    print(f"c={c}: mean=[{post_means[i].min():.2f}, {post_means[i].max():.2f}], "
          f"std=[{post_stds[i].min():.2f}, {post_stds[i].max():.2f}]")

print("\n--- Network ---")
for i, c in enumerate(CONTRAST_LEVELS):
    print(f"c={c}: mean=[{net_means[i].min():.2f}, {net_means[i].max():.2f}], "
          f"std=[{net_stds[i].min():.2f}, {net_stds[i].max():.2f}]")

print("\n" + "="*60)
print("FIGURA COMPLETADA")
print("="*60)
