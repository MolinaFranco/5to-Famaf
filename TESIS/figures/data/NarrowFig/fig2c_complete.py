#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generar fig2c completa: posterior (izquierda) + network (derecha).

Basado en Echeveste et al. (2020), Nature Neuroscience, Figure 2c.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

# =============================================================================
# CONFIGURACION
# =============================================================================

N_exc = 50
SIMULATION_DIR = Path('/home/molina/FAMAF/5to-Famaf/TESIS/figures/data/simulations')

# Contrastes disponibles en nuestras simulaciones
CONTRAST_LEVELS = [0.0, 0.125, 0.25, 0.5, 1.0]

# Colores estilo Echeveste
GREEN_COLORS = plt.cm.Greens(np.linspace(0.3, 0.9, len(CONTRAST_LEVELS)))
ORANGE_COLORS = plt.cm.Oranges(np.linspace(0.3, 0.9, len(CONTRAST_LEVELS)))

# Parametros de la no linealidad (Echeveste)
nl_scale = 2.4
nl_power = 0.6
nl_baseline = (3.5 / nl_scale) ** (1.0 / nl_power)
baseline = 3.0  # mV

# Directorio de resultados GSM de Echeveste
GSM_RESULTS = Path('/home/molina/FAMAF/5to-Famaf/TESIS/'
                   'ssn_inference_numerical_experiments/GSM/bumps/no_noise/results')


# =============================================================================
# FUNCIONES
# =============================================================================

def np_th(u):
    """Rectificación (threshold en 0)."""
    return np.maximum(u, 0)


def nl_fun(u):
    """Función no lineal del SSN."""
    return nl_scale * (np_th(u + nl_baseline) ** nl_power)


def get_mean_var_nl(mu, std, n_points=201):
    """
    Calcular media y varianza después de la no linealidad.
    """
    points_1d = np.linspace(-4.0, 4.0, n_points)
    new_mean = np.empty(N_exc)
    new_var = np.empty(N_exc)

    for i in range(N_exc):
        mu_i = mu[i]
        std_i = std[i]
        points_resc = std_i * points_1d + mu_i
        values_dist = norm.pdf(points_resc, loc=mu_i, scale=std_i)
        norm_f = np.sum(values_dist)
        values_fun = nl_fun(points_resc)
        new_mean[i] = np.sum(values_dist * values_fun) / norm_f
        new_var[i] = np.sum(values_dist * (values_fun - new_mean[i])**2) / norm_f

    return new_mean, np.sqrt(new_var)


def load_network_data(contrast):
    """Cargar datos de simulación de la red SSN."""
    dir_name = f"contrast_{contrast:.3f}_narrow_wf0.01"
    data_path = SIMULATION_DIR / dir_name / "simulation_data.npz"
    data = np.load(data_path)
    return {
        'u_E_mean': data['u_E_mean'],
        'u_E_std': data['u_E_std'],
        'orientations': data['orientations'],
    }


# =============================================================================
# CARGAR Y PROCESAR DATOS
# =============================================================================

print("Cargando datos...")

# Mapeo de nuestros contrastes a los índices de Echeveste
# Echeveste usa: [0.0, 0.125, 0.25, 0.5, 1.0, 2.0] (índices 0-5)
# Nosotros usamos: [0.0, 0.125, 0.25, 0.5, 1.0] (índices 0-4)
echeveste_indices = {0.0: 0, 0.125: 1, 0.25: 2, 0.5: 3, 1.0: 4}

# Almacenar datos
post_means = []
post_stds = []
net_means = []
net_stds = []
orientations = None

for contrast in CONTRAST_LEVELS:
    print(f"  Procesando contraste {contrast}...")

    # --- POSTERIOR (GSM + NL transformation) ---
    alpha = echeveste_indices[contrast]

    if contrast == 0.0:
        # Para contraste 0, usar valores del prior
        mu_gsm = np.zeros(N_exc)
        std_gsm = np.ones(N_exc) * 2.0  # sqrt(C[0,0]) = sqrt(4) = 2
    else:
        mu_gsm = np.loadtxt(GSM_RESULTS / f"mu_true_z_{alpha}")
        std_gsm = np.loadtxt(GSM_RESULTS / f"std_true_z_{alpha}")

    # Agregar baseline y aplicar NL
    mu_gsm_shifted = baseline + mu_gsm
    mu_nl, std_nl = get_mean_var_nl(mu_gsm_shifted, std_gsm)

    post_means.append(mu_nl)
    post_stds.append(std_nl)

    # --- NETWORK (datos de simulación SSN) ---
    net_data = load_network_data(contrast)
    net_means.append(net_data['u_E_mean'])
    net_stds.append(net_data['u_E_std'])

    if orientations is None:
        orientations = net_data['orientations']

print(f"\nOrientaciones: {orientations[0]:.1f}° a {orientations[-1]:.1f}°")


# =============================================================================
# CREAR FIGURA
# =============================================================================

print("\nGenerando figura...")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Configurar estilo
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
})

# --- Fila superior: MEAN ---

# Posterior mean (izquierda)
ax = axes[0, 0]
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax.plot(orientations, post_means[i], color=GREEN_COLORS[i], linewidth=2,
            label=f'c={contrast}')
ax.set_ylabel('mean $u_E$ [mV]')
ax.set_title('posterior', fontweight='bold')
ax.set_xlim(-90, 90)
ax.set_ylim(4, 9)
ax.set_xticks([-90, -45, 0, 45, 90])
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', fontsize=8)

# Network mean (derecha)
ax = axes[0, 1]
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax.plot(orientations, net_means[i], color=ORANGE_COLORS[i], linewidth=2,
            label=f'c={contrast}')
ax.set_title('network', fontweight='bold')
ax.set_xlim(-90, 90)
ax.set_ylim(0, 10)
ax.set_xticks([-90, -45, 0, 45, 90])
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', fontsize=8)

# --- Fila inferior: STD ---

# Posterior std (izquierda)
ax = axes[1, 0]
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax.plot(orientations, post_stds[i], color=GREEN_COLORS[i], linewidth=2,
            label=f'c={contrast}')
ax.set_ylabel('$u_E$ std. [mV]')
ax.set_xlabel('pref. ori.')
ax.set_xlim(-90, 90)
ax.set_ylim(0.5, 2.0)
ax.set_xticks([-90, -45, 0, 45, 90])
ax.set_xticklabels(['-90°', '-45°', '0°', '45°', '90°'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Network std (derecha)
ax = axes[1, 1]
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax.plot(orientations, net_stds[i], color=ORANGE_COLORS[i], linewidth=2,
            label=f'c={contrast}')
ax.set_xlabel('pref. ori.')
ax.set_xlim(-90, 90)
ax.set_ylim(0, 3)
ax.set_xticks([-90, -45, 0, 45, 90])
ax.set_xticklabels(['-90°', '-45°', '0°', '45°', '90°'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Guardar
output_path = Path('/home/molina/FAMAF/5to-Famaf/TESIS/figures/data/NarrowFig/'
                   'fig2c_complete.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Guardada: {output_path}")

# PDF
output_pdf = output_path.with_suffix('.pdf')
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"Guardada: {output_pdf}")

plt.close()

# =============================================================================
# IMPRIMIR VALORES
# =============================================================================

print("\n" + "="*60)
print("VALORES DE VERIFICACIÓN")
print("="*60)

print("\n--- Posterior (con NL) ---")
for i, c in enumerate(CONTRAST_LEVELS):
    print(f"c={c}: mean=[{post_means[i].min():.2f}, {post_means[i].max():.2f}], "
          f"std=[{post_stds[i].min():.2f}, {post_stds[i].max():.2f}]")

print("\n--- Network ---")
for i, c in enumerate(CONTRAST_LEVELS):
    print(f"c={c}: mean=[{net_means[i].min():.2f}, {net_means[i].max():.2f}], "
          f"std=[{net_stds[i].min():.2f}, {net_stds[i].max():.2f}]")

print("\n¡Figura completada!")
