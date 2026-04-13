#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generar figuras limpias para simulaciones con estimulos angostos.

Este script genera los graficos sin mencionar "narrow" ni "width_factor".
Los mapas de calor usan suavizado gaussiano para reducir ruido.

Referencias:
- Echeveste et al. (2020), Nature Neuroscience
"""

import sys
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gamma as gamma_dist, multivariate_normal, norm
from pathlib import Path

from skneuromsi.generative import GSM
from skneuromsi.data import EchevesteDataLoader, GSMDataLoader

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
BASE_DIR = Path(__file__).parent.parent.parent
SIMULATION_DIR = Path(__file__).parent.parent / 'simulations'
OUTPUT_DIR = Path(__file__).parent

# Crear directorio de salida
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Niveles de contraste (5 niveles)
CONTRAST_LEVELS = [0.0, 0.125, 0.25, 0.5, 1.0]

# Colores para cada contraste
CONTRAST_COLORS = {
    0.0: '#2E4057',
    0.125: '#048A81',
    0.25: '#54C6EB',
    0.5: '#F18F01',
    1.0: '#C73E1D',
}

# Parametros
N_E = 50
SEED = 42
# Sigma para suavizado gaussiano temporal (en numero de muestras)
GAUSSIAN_SIGMA = 10

# Parametros del prior de contraste (gamma distribution)
GAMMA_K = 2.0
GAMMA_THETA = 2.0

print("=" * 70)
print("GENERACION DE FIGURAS (VERSION LIMPIA)")
print("=" * 70)
print(f"\nDirectorio de datos: {SIMULATION_DIR}")
print(f"Directorio de salida: {OUTPUT_DIR}")
print(f"Niveles de contraste: {CONTRAST_LEVELS}")
print(f"Suavizado gaussiano sigma: {GAUSSIAN_SIGMA} muestras")


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

# =============================================================================
# TRANSFORMACION NO LINEAL (SSN Power-law rectification)
# =============================================================================

# Parametros de la no linealidad (del codigo de Echeveste)
NL_SCALE = 2.4
NL_POWER = 0.6
NL_BASELINE = (3.5 / NL_SCALE) ** (1.0 / NL_POWER)


def np_threshold(u):
    """Rectificacion (threshold en 0)."""
    return np.maximum(u, 0)


def nl_function(u):
    """
    Funcion no lineal del SSN: rectified power law.

    f(u) = scale * (max(u + baseline, 0))^power

    Based on Echeveste transformed_moments_nonlinearity.py
    """
    return NL_SCALE * (np_threshold(u + NL_BASELINE) ** NL_POWER)


def transform_moments_nonlinear(mu_gsm, std_gsm, n_points=201):
    """
    Transformar momentos del GSM a traves de la no linealidad del SSN.

    Para cada neurona i con distribucion N(mu_i, sigma_i^2), calcula
    E[f(y_i)] y sqrt(Var[f(y_i)]) donde f es la rectified power law.

    Based on Echeveste GSM/transformed_moments_nonlinearity.py

    Parameters
    ----------
    mu_gsm : np.ndarray
        Media del posterior GSM (con baseline ya incluido)
    std_gsm : np.ndarray
        Desviacion estandar del posterior GSM
    n_points : int
        Numero de puntos para integracion numerica

    Returns
    -------
    mu_nl, std_nl : np.ndarray
        Media y std despues de la transformacion no lineal
    """
    n_neurons = len(mu_gsm)
    points_1d = np.linspace(-4.0, 4.0, n_points)

    mu_nl = np.empty(n_neurons)
    var_nl = np.empty(n_neurons)

    for i in range(n_neurons):
        mu_i = mu_gsm[i]
        std_i = std_gsm[i]

        # Puntos reescalados para integracion
        points_resc = std_i * points_1d + mu_i

        # Densidad de probabilidad N(mu_i, std_i^2)
        values_dist = norm.pdf(points_resc, loc=mu_i, scale=std_i)
        norm_f = np.sum(values_dist)

        # Valores de la funcion no lineal
        values_fun = nl_function(points_resc)

        # E[f(y_i)] y Var[f(y_i)]
        mu_nl[i] = np.sum(values_dist * values_fun) / norm_f
        var_nl[i] = np.sum(values_dist * (values_fun - mu_nl[i])**2) / norm_f

    return mu_nl, np.sqrt(var_nl)


def transform_covariance_nonlinear(mu_gsm, std_gsm, cov_gsm, n_points=101):
    """
    Transformar covarianza del GSM a traves de la no linealidad.

    Based on Echeveste GSM/transformed_moments_nonlinearity.py get_cov()

    Parameters
    ----------
    mu_gsm : np.ndarray
        Media del posterior GSM (con baseline)
    std_gsm : np.ndarray
        Desviacion estandar del posterior GSM
    cov_gsm : np.ndarray
        Matriz de covarianza del posterior GSM
    n_points : int
        Numero de puntos para integracion

    Returns
    -------
    cov_nl : np.ndarray
        Matriz de covarianza despues de la transformacion
    """
    n_neurons = len(mu_gsm)
    points_1d = np.linspace(-4.0, 4.0, n_points)

    # Primero calcular la media y varianza transformadas
    mu_nl, std_nl = transform_moments_nonlinear(mu_gsm, std_gsm, n_points)
    var_nl = std_nl ** 2

    cov_nl = np.empty((n_neurons, n_neurons))

    for i in range(n_neurons):
        cov_nl[i, i] = var_nl[i]

        for j in range(i + 1, n_neurons):
            mu_i = mu_gsm[i]
            std_i = std_gsm[i]
            mu_j = mu_gsm[j]
            std_j = std_gsm[j]

            # Submatriz 2x2 de covarianza
            mean_red = np.array([mu_i, mu_j])
            cov_red = np.array([
                [cov_gsm[i, i], cov_gsm[i, j]],
                [cov_gsm[j, i], cov_gsm[j, j]]
            ])

            # Puntos reescalados
            points_resc_i = std_i * points_1d + mu_i
            points_resc_j = std_j * points_1d + mu_j

            # Malla 2D
            X, Y = np.meshgrid(points_resc_i, points_resc_j)
            points_2d = np.vstack((X.flatten(), Y.flatten())).T

            # Densidad bivariada
            values_dist = multivariate_normal.pdf(
                points_2d, mean=mean_red, cov=cov_red
            ).reshape(n_points, n_points)

            norm_f = np.sum(values_dist)

            # Cov[f(y_i), f(y_j)] = E[(f(y_i) - E[f(y_i)])(f(y_j) - E[f(y_j)])]
            values_fun_2 = np.outer(
                nl_function(points_resc_i) - mu_nl[i],
                nl_function(points_resc_j) - mu_nl[j]
            )

            cov_nl[i, j] = np.sum(values_dist * values_fun_2) / norm_f

    # Simetrizar
    for i in range(n_neurons):
        for j in range(i):
            cov_nl[i, j] = cov_nl[j, i]

    return cov_nl


def load_simulation_data(contrast, suffix='_narrow_wf0.01'):
    """
    Cargar datos de simulacion con estimulos angostos.

    Parameters
    ----------
    contrast : float
        Nivel de contraste
    suffix : str
        Sufijo del directorio

    Returns
    -------
    dict
        Datos de simulacion
    """
    dir_name = f"contrast_{contrast:.3f}{suffix}"
    data_path = SIMULATION_DIR / dir_name / "simulation_data.npz"

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontro: {data_path}")

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
    # Aplicar filtro gaussiano a cada neurona por separado
    smoothed = np.zeros_like(data)
    for i in range(data.shape[1]):
        smoothed[:, i] = gaussian_filter1d(data[:, i], sigma=sigma)
    return smoothed


def compute_posterior_full_inference(gsm, A, C, C_inv, ATA, ACAT,
                                     contrast, width_factor=0.01,
                                     dominant_idx=25, s_x_2=100.0,
                                     k=2.0, theta=2.0,
                                     apply_nonlinearity=True):
    """
    Calcular momentos del posterior GSM con full inference sobre z.

    Integra sobre la distribucion P(z|x) para obtener E[y|x] y Cov[y|x].
    Opcionalmente aplica la transformacion no lineal del SSN para obtener
    momentos comparables con la red.

    Based on Echeveste et al. GSM.py get_post_moments_full_inference()
    and transformed_moments_nonlinearity.py

    Parameters
    ----------
    gsm : GSM
        Modelo GSM
    A : np.ndarray
        Matriz de filtros Gabor
    C : np.ndarray
        Matriz de covarianza del prior
    C_inv : np.ndarray
        Inversa de C
    ATA : np.ndarray
        A.T @ A
    ACAT : np.ndarray
        A @ C @ A.T
    contrast : float
        Nivel de contraste verdadero
    width_factor : float
        Factor de ancho del estimulo
    dominant_idx : int
        Indice de orientacion dominante
    s_x_2 : float
        Varianza del ruido de observacion
    k : float
        Shape parameter de gamma prior
    theta : float
        Scale parameter de gamma prior
    apply_nonlinearity : bool
        Si True, aplica la transformacion no lineal del SSN a los momentos

    Returns
    -------
    mean, std, cov : np.ndarray
        Momentos del posterior (transformados si apply_nonlinearity=True)
    """
    # Baseline (en mV) - se agrega antes de la no linealidad
    BASELINE = 3.0

    if contrast == 0.0:
        # Actividad espontanea - usar prior con baseline
        mu_gsm = np.zeros(N_E) + BASELINE
        cov_gsm = C.copy()
        std_gsm = np.sqrt(np.diag(cov_gsm))

        if apply_nonlinearity:
            mu_nl, std_nl = transform_moments_nonlinear(mu_gsm, std_gsm)
            cov_nl = transform_covariance_nonlinear(mu_gsm, std_gsm, cov_gsm)
            return mu_nl, std_nl, cov_nl
        else:
            return mu_gsm, std_gsm, cov_gsm

    # Amplitudes calibradas para estimulos angostos
    amplitude_map = {
        0.125: 7.7341,
        0.25: 15.4641,
        0.5: 30.9287,
        1.0: 61.8572,
    }
    amplitude = amplitude_map.get(contrast, 10.0)

    # Generar estimulo angosto (y) y patch (x)
    y = gsm.generate_bump_stimulus(
        dominant_orientation_idx=dominant_idx,
        amplitude=amplitude,
        width_factor=width_factor,
        normalize=False
    )
    x = contrast * (A @ y)

    # Calcular P(z|x)
    z_range = np.linspace(0.001, 5.0, 201)
    n_points = len(z_range)
    D_x = len(x)
    log_p = np.empty(n_points)
    mean_vec = np.zeros(D_x)

    for i in range(n_points):
        z = z_range[i]
        cov_x = z * z * ACAT + s_x_2 * np.identity(D_x)
        log_p[i] = (gamma_dist.logpdf(z, k, loc=0, scale=theta) +
                    multivariate_normal.logpdf(x, mean_vec, cov_x))

    max_lp = np.max(log_p)
    p_z = np.exp(log_p - max_lp)
    dz = z_range[1] - z_range[0]
    norm_val = np.sum(p_z) * dz
    p_z = p_z / norm_val

    # Full inference - integrar sobre z
    mu_post = np.zeros(N_E)
    Sigma_post = np.zeros((N_E, N_E))

    threshold_val = 10000.0
    i_p_max = np.argmax(p_z)
    p_max = p_z[i_p_max]

    # Encontrar rango efectivo de integracion
    i_min = 0
    for i in range(i_p_max, -1, -1):
        if p_z[i] <= (p_max / threshold_val):
            i_min = i
            break

    i_max = n_points
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
        # E[Var_z] + Var_z(mu)
        Sigma_post += (Sigma_z + np.outer(mu_z, mu_z)) * p_z[i]

    mu_post *= dz
    Sigma_post *= dz
    # Restar E(mu)E(mu)T
    Sigma_post -= np.outer(mu_post, mu_post)

    # Agregar baseline ANTES de la no linealidad
    mu_gsm = mu_post + BASELINE
    std_gsm = np.sqrt(np.diag(Sigma_post))
    cov_gsm = Sigma_post

    if apply_nonlinearity:
        # Transformar a traves de la no linealidad del SSN
        mu_nl, std_nl = transform_moments_nonlinear(mu_gsm, std_gsm)
        cov_nl = transform_covariance_nonlinear(mu_gsm, std_gsm, cov_gsm)
        return mu_nl, std_nl, cov_nl
    else:
        return mu_gsm, std_gsm, cov_gsm


def covariance_ellipse(cov, mean, n_std=2.0, **kwargs):
    """
    Crear una elipse de matplotlib representando una matriz de covarianza 2D.

    Parameters
    ----------
    cov : np.ndarray, shape (2, 2)
        Matriz de covarianza 2D
    mean : np.ndarray, shape (2,)
        Vector de media 2D
    n_std : float
        Numero de desviaciones estandar

    Returns
    -------
    ellipse : matplotlib.patches.Ellipse
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues, 0))

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        **kwargs
    )

    return ellipse


# =============================================================================
# CARGAR DATOS
# =============================================================================

print("\nCargando datos de simulacion...")
all_data = {}
for contrast in CONTRAST_LEVELS:
    print(f"  Cargando contraste {contrast}...")
    all_data[contrast] = load_simulation_data(contrast)
    print(f"    Shape: {all_data[contrast]['u_E'].shape}")
    print(f"    h_input max: {all_data[contrast]['h_input'].max():.4f}")

orientations = all_data[0.0]['orientations']

# Cargar GSM y matrices necesarias
print("\nCargando GSM y matrices...")
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
s_x_2 = 100.0  # noise variance (s_x = 10.0)


# =============================================================================
# FIG 2a: SAMPLE POPULATION ACTIVITY (CON SUAVIZADO)
# =============================================================================

print("\nGenerando fig2a: Sample population activity...")

fig_2a = plt.figure(figsize=(10, 12))
gs_2a = GridSpec(3, 1, figure=fig_2a, hspace=0.35, height_ratios=[1, 1, 1])

# Escala de colores fija
VMIN = 0.0
VMAX = 10.0
T_max = all_data[0.0]['time'][-1]

# Panel 1: Contraste 0.0 (espontaneo)
ax1 = fig_2a.add_subplot(gs_2a[0, 0])
data_zero = all_data[0.0]
# Aplicar suavizado gaussiano
u_E_plot = smooth_temporal(data_zero['u_E'], GAUSSIAN_SIGMA).T

im1 = ax1.imshow(
    u_E_plot,
    aspect='auto',
    cmap='coolwarm',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=VMIN,
    vmax=VMAX
)

ax1.set_ylabel('Preferred Orientation (deg)')
ax1.set_title('Zero Contrast (Spontaneous)')
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
# Aplicar suavizado gaussiano
u_E_plot_high = smooth_temporal(data_high['u_E'], GAUSSIAN_SIGMA).T

im2 = ax2.imshow(
    u_E_plot_high,
    aspect='auto',
    cmap='coolwarm',
    extent=[0, T_max, orientations[0], orientations[-1]],
    origin='lower',
    vmin=VMIN,
    vmax=VMAX
)

ax2.set_ylabel('Preferred Orientation (deg)')
ax2.set_title('High Contrast (1.0)')
ax2.set_ylim(orientations[0], orientations[-1])
ax2.set_yticks([-90, -45, 0, 45, 90])
ax2.set_xlim(0, T_max)
ax2.set_xticks(np.arange(0, T_max + 1, 200))
ax2.set_xticklabels([])

# Colorbar en panel medio
cbar2 = plt.colorbar(im2, ax=ax2, label='$u_E$ (a.u.)')

# Panel 3: Estimulo (h_input) para contraste 1.0
ax3 = fig_2a.add_subplot(gs_2a[2, 0])

# Usar h_input de la simulacion
h_E_stimulus = data_high['h_input']

# Modo single-phase: estimulo constante
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
ax3.set_ylabel('Preferred Orientation (deg)')
ax3.set_xlabel('Time (ms)')
ax3.set_title('Stimulus Input (h)')
ax3.set_ylim(orientations[0], orientations[-1])
ax3.set_yticks([-90, 0, 90])

# Colorbar
cbar3 = plt.colorbar(im3, ax=ax3, label='h (a.u.)')

plt.tight_layout()
fig_2a_path = OUTPUT_DIR / 'fig2a_population_activity.png'
plt.savefig(fig_2a_path, dpi=300, bbox_inches='tight')
print(f"Guardada: {fig_2a_path}")
plt.close()


# =============================================================================
# FIG 2a OPEN: SAMPLE POPULATION ACTIVITY (escala libre compartida)
# =============================================================================

print("\nGenerando fig2a_open: Sample population activity (escala libre)...")

fig_2a_open = plt.figure(figsize=(10, 12))
gs_2a_open = GridSpec(3, 1, figure=fig_2a_open, hspace=0.35,
                      height_ratios=[1, 1, 1])

T_max = all_data[0.0]['time'][-1]

# Calcular escala compartida (libre) para ambos paneles
data_zero = all_data[0.0]
data_high = all_data[1.0]
u_E_plot_zero = smooth_temporal(data_zero['u_E'], GAUSSIAN_SIGMA).T
u_E_plot_high = smooth_temporal(data_high['u_E'], GAUSSIAN_SIGMA).T

# Escala compartida basada en los datos reales
VMIN_OPEN = min(u_E_plot_zero.min(), u_E_plot_high.min())
VMAX_OPEN = max(u_E_plot_zero.max(), u_E_plot_high.max())

# Panel 1: Contraste 0.0 (espontaneo) - escala libre compartida
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

ax1_open.set_ylabel('Preferred Orientation (deg)')
ax1_open.set_title('Zero Contrast (Spontaneous)')
ax1_open.set_ylim(orientations[0], orientations[-1])
ax1_open.set_yticks([-90, -45, 0, 45, 90])
ax1_open.set_xlim(0, T_max)
ax1_open.set_xticks(np.arange(0, T_max + 1, 200))
ax1_open.set_xticklabels([])

# Colorbar invisible pero mantiene espacio
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

ax2_open.set_ylabel('Preferred Orientation (deg)')
ax2_open.set_title('High Contrast (1.0)')
ax2_open.set_ylim(orientations[0], orientations[-1])
ax2_open.set_yticks([-90, -45, 0, 45, 90])
ax2_open.set_xlim(0, T_max)
ax2_open.set_xticks(np.arange(0, T_max + 1, 200))
ax2_open.set_xticklabels([])

# Colorbar visible solo en panel 2
cbar2_open = plt.colorbar(im2_open, ax=ax2_open, label='$u_E$ (a.u.)')

# Panel 3: Estimulo (h_input) para contraste 1.0
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
ax3_open.set_ylabel('Preferred Orientation (deg)')
ax3_open.set_xlabel('Time (ms)')
ax3_open.set_title('Stimulus Input (h)')
ax3_open.set_ylim(orientations[0], orientations[-1])
ax3_open.set_yticks([-90, 0, 90])

cbar3_open = plt.colorbar(im3_open, ax=ax3_open, label='h (a.u.)')

plt.tight_layout()
fig_2a_open_path = OUTPUT_DIR / 'fig2a_population_activity_open.png'
plt.savefig(fig_2a_open_path, dpi=300, bbox_inches='tight')
print(f"Guardada: {fig_2a_open_path}")
plt.close()


# =============================================================================
# FIG 2b: COVARIANCE ELLIPSES
# =============================================================================

print("\nGenerando Fig2b: Covariance ellipses...")

# Seleccionar dos neuronas representativas (42 y 16 grados aprox)
target_orientations = [42.0, 16.0]
neuron_indices = []
for target_ori in target_orientations:
    idx = np.argmin(np.abs(orientations - target_ori))
    neuron_indices.append(idx)
    print(f"  Neurona {idx}: orientacion {orientations[idx]:.1f} deg")

# Crear figura con 2 columnas: izquierda (estimulos), derecha (elipses)
fig2b = plt.figure(figsize=(14, 16))
gs = GridSpec(5, 2, figure=fig2b, hspace=0.35, wspace=0.3,
              width_ratios=[0.4, 1])

for idx, contrast in enumerate(CONTRAST_LEVELS):
    print(f"  Procesando contraste {contrast}...")
    label = ['spontaneous', 'low', 'medium', 'high', 'very high'][idx]

    # Obtener datos de la red
    data = all_data[contrast]
    net_mean = data['u_E_mean']
    net_cov = data['u_E_cov']

    # Obtener datos del posterior (ideal observer) con full inference
    post_mean, post_std, post_cov = compute_posterior_full_inference(
        gsm, A, C, C_inv, ATA, ACAT,
        contrast, width_factor=0.01, dominant_idx=25,
        s_x_2=s_x_2, k=GAMMA_K, theta=GAMMA_THETA
    )

    # Extraer submatrices 2D
    net_mean_2d = net_mean[neuron_indices]
    net_cov_2d = net_cov[np.ix_(neuron_indices, neuron_indices)]

    post_mean_2d = post_mean[neuron_indices]
    post_cov_2d = post_cov[np.ix_(neuron_indices, neuron_indices)]

    # Columna izquierda: Estimulo (h_input)
    ax_stim = fig2b.add_subplot(gs[idx, 0])
    h_input = data['h_input']
    ax_stim.plot(h_input, orientations, 'k-', linewidth=2)
    ax_stim.fill_betweenx(orientations, 0, h_input,
                          alpha=0.3, color=CONTRAST_COLORS[contrast])
    ax_stim.set_ylim(-90, 90)
    ax_stim.set_yticks([-90, -45, 0, 45, 90])
    ax_stim.set_ylabel('Orientation (deg)')
    ax_stim.set_title(f'c={contrast} ({label})')
    if idx == len(CONTRAST_LEVELS) - 1:
        ax_stim.set_xlabel('h (a.u.)')

    # Columna derecha: Elipses de covarianza
    ax = fig2b.add_subplot(gs[idx, 1])

    # Elipse del ideal observer (verde, linea punteada)
    ellipse_post = covariance_ellipse(
        post_cov_2d, post_mean_2d, n_std=2.0,
        facecolor='none',
        edgecolor='green',
        linewidth=2,
        linestyle='--',
        alpha=0.7
    )
    ax.add_patch(ellipse_post)

    # Elipse de la red (rojo, linea solida)
    ellipse_net = covariance_ellipse(
        net_cov_2d, net_mean_2d, n_std=2.0,
        facecolor='none',
        edgecolor='red',
        linewidth=2,
        alpha=0.7
    )
    ax.add_patch(ellipse_net)

    # Graficar trayectoria de la red (ultimos 500ms)
    traj_idx = data['time'] >= 500.0
    u_E_traj = data['u_E'][traj_idx, :][:, neuron_indices]

    color = CONTRAST_COLORS[contrast]
    ax.plot(
        u_E_traj[:, 0], u_E_traj[:, 1],
        color=color, alpha=0.3, linewidth=0.5
    )

    # Configurar ejes
    ax.set_xlabel(f'E cell at {orientations[neuron_indices[0]]:.0f} deg')
    ax.set_ylabel(f'E cell at {orientations[neuron_indices[1]]:.0f} deg')
    ax.set_title(f'Covariance Ellipses (c={contrast})')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Leyenda solo en el primer subplot
    if idx == 0:
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, linestyle='--',
                   label='Ideal observer'),
            Line2D([0], [0], color='red', lw=2, label='Network'),
            Line2D([0], [0], color='gray', lw=0.5, alpha=0.5,
                   label='Network trajectory (500ms)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
fig2b_path = OUTPUT_DIR / 'fig2b_covariance_ellipses.png'
plt.savefig(fig2b_path, dpi=300, bbox_inches='tight')
print(f"Guardada: {fig2b_path}")
plt.close()


# =============================================================================
# FIG 2c: MEAN AND STD (con full inference para ideal observer)
# =============================================================================

print("\nGenerando Fig2c: Mean and std (con full inference)...")

# Recolectar momentos
all_post_means = []
all_post_stds = []
all_net_means = []
all_net_stds = []

for contrast in CONTRAST_LEVELS:
    print(f"  Procesando contraste {contrast}...")

    # Red
    data = all_data[contrast]
    all_net_means.append(data['u_E_mean'])
    all_net_stds.append(data['u_E_std'])

    # Ideal observer con full inference
    post_mean, post_std, post_cov = compute_posterior_full_inference(
        gsm, A, C, C_inv, ATA, ACAT,
        contrast, width_factor=0.01, dominant_idx=25,
        s_x_2=s_x_2, k=GAMMA_K, theta=GAMMA_THETA
    )
    all_post_means.append(post_mean)
    all_post_stds.append(post_std)

# Crear figura
fig2c = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig2c, hspace=0.3, wspace=0.3)

# Determinar escala Y consistente
max_mean = max(max(m.max() for m in all_post_means),
               max(m.max() for m in all_net_means))
max_std = max(max(s.max() for s in all_post_stds),
              max(s.max() for s in all_net_stds))
y_max_mean = max_mean * 1.1
y_max_std = max_std * 1.1

# Subplot 1: Ideal Observer mean
ax1 = fig2c.add_subplot(gs[0, 0])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax1.plot(orientations, all_post_means[i], color=CONTRAST_COLORS[contrast],
             linewidth=2, label=f'c={contrast:.3f}')
ax1.set_xlabel('Preferred Orientation (deg)')
ax1.set_ylabel('Mean $u_E$ (mV)')
ax1.set_title('Ideal Observer: Mean')
ax1.set_xlim(-90, 90)
ax1.set_ylim(0, y_max_mean)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=8)

# Subplot 2: Network mean
ax2 = fig2c.add_subplot(gs[0, 1])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax2.plot(orientations, all_net_means[i], color=CONTRAST_COLORS[contrast],
             linewidth=2, label=f'c={contrast:.3f}')
ax2.set_xlabel('Preferred Orientation (deg)')
ax2.set_ylabel('Mean $u_E$ (mV)')
ax2.set_title('Network: Mean')
ax2.set_xlim(-90, 90)
ax2.set_ylim(0, y_max_mean)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=8)

# Subplot 3: Ideal Observer std
ax3 = fig2c.add_subplot(gs[1, 0])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax3.plot(orientations, all_post_stds[i], color=CONTRAST_COLORS[contrast],
             linewidth=2, label=f'c={contrast:.3f}')
ax3.set_xlabel('Preferred Orientation (deg)')
ax3.set_ylabel('Std $u_E$ (mV)')
ax3.set_title('Ideal Observer: Standard Deviation')
ax3.set_xlim(-90, 90)
ax3.set_ylim(0, y_max_std)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best', fontsize=8)

# Subplot 4: Network std
ax4 = fig2c.add_subplot(gs[1, 1])
for i, contrast in enumerate(CONTRAST_LEVELS):
    ax4.plot(orientations, all_net_stds[i], color=CONTRAST_COLORS[contrast],
             linewidth=2, label=f'c={contrast:.3f}')
ax4.set_xlabel('Preferred Orientation (deg)')
ax4.set_ylabel('Std $u_E$ (mV)')
ax4.set_title('Network: Standard Deviation')
ax4.set_xlim(-90, 90)
ax4.set_ylim(0, y_max_std)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='best', fontsize=8)

plt.tight_layout()
fig2c_path = OUTPUT_DIR / 'fig2c_mean_std.png'
plt.savefig(fig2c_path, dpi=300, bbox_inches='tight')
print(f"Guardada: {fig2c_path}")
plt.close()


# =============================================================================
# FIG 2d: CORRELATION MATRICES
# =============================================================================

print("\nGenerando Fig2d: Correlation matrices...")

fig2d = plt.figure(figsize=(10, 22))
gs = GridSpec(5, 2, figure=fig2d, hspace=0.25, wspace=0.25)

all_axes = []
all_ims = []

for idx, contrast in enumerate(CONTRAST_LEVELS):
    print(f"  Procesando contraste {contrast}...")

    # Red neuronal
    data = all_data[contrast]
    net_corr = data['u_E_corr']

    # Ideal observer con full inference
    post_mean, post_std, post_cov = compute_posterior_full_inference(
        gsm, A, C, C_inv, ATA, ACAT,
        contrast, width_factor=0.01, dominant_idx=25,
        s_x_2=s_x_2, k=GAMMA_K, theta=GAMMA_THETA
    )

    # Convertir covarianza a correlacion
    post_corr = np.zeros_like(post_cov)
    for i in range(N_E):
        for j in range(N_E):
            denom = np.sqrt(post_cov[i, i]) * np.sqrt(post_cov[j, j])
            if denom > 0:
                post_corr[i, j] = post_cov[i, j] / denom
            else:
                post_corr[i, j] = 0.0

    # Subplot: Ideal observer
    ax1 = fig2d.add_subplot(gs[idx, 0])
    im1 = ax1.imshow(post_corr, cmap='RdBu_r', aspect='equal',
                     extent=[-90, 90, -90, 90], origin='lower',
                     vmin=-1, vmax=1)
    ax1.set_title(f'Ideal Observer (c={contrast})', pad=3)
    ax1.set_xticks([-90, -45, 0, 45, 90])
    ax1.set_yticks([-90, -45, 0, 45, 90])
    if idx == len(CONTRAST_LEVELS) - 1:
        ax1.set_xlabel('Orientation (deg)')
    ax1.set_ylabel('Orientation (deg)')
    all_axes.append(ax1)
    all_ims.append(im1)

    # Subplot: Network
    ax2 = fig2d.add_subplot(gs[idx, 1])
    im2 = ax2.imshow(net_corr, cmap='RdBu_r', aspect='equal',
                     extent=[-90, 90, -90, 90], origin='lower',
                     vmin=-1, vmax=1)
    ax2.set_title(f'Network (c={contrast})', pad=3)
    ax2.set_xticks([-90, -45, 0, 45, 90])
    ax2.set_yticks([-90, -45, 0, 45, 90])
    if idx == len(CONTRAST_LEVELS) - 1:
        ax2.set_xlabel('Orientation (deg)')
    ax2.set_ylabel('Orientation (deg)')
    all_axes.append(ax2)
    all_ims.append(im2)

# Colorbar unica a la derecha
fig2d.subplots_adjust(right=0.88)
cbar_ax = fig2d.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig2d.colorbar(all_ims[-1], cax=cbar_ax)
cbar.set_label('Correlation')
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

fig2d_path = OUTPUT_DIR / 'fig2d_correlations.png'
plt.savefig(fig2d_path, dpi=300, bbox_inches='tight')
print(f"Guardada: {fig2d_path}")
plt.close()


# =============================================================================
# RESUMEN
# =============================================================================

print("\n" + "=" * 70)
print("RESUMEN DE FIGURAS GENERADAS")
print("=" * 70)

print(f"\nFig2a: Sample population activity")
print(f"  - Archivo: {fig_2a_path}")
print(f"  - Contraste 0 y 1.0 (escala fija 0-10)")
print(f"  - Suavizado gaussiano sigma={GAUSSIAN_SIGMA}")

print(f"\nFig2a_open: Sample population activity (escala libre)")
print(f"  - Archivo: {fig_2a_open_path}")
print(f"  - Contraste 0 y 1.0 (escala libre)")
print(f"  - Suavizado gaussiano sigma={GAUSSIAN_SIGMA}")

print(f"\nFig2b: Covariance ellipses")
print(f"  - Archivo: {fig2b_path}")
print(f"  - Neuronas: {orientations[neuron_indices[0]]:.0f} deg y "
      f"{orientations[neuron_indices[1]]:.0f} deg")
print(f"  - 5 niveles de contraste")
print(f"  - Verde: Ideal observer, Rojo: Red")

print(f"\nFig2c: Mean and std (con full inference)")
print(f"  - Archivo: {fig2c_path}")
print(f"  - Comparacion ideal observer vs network")
print(f"  - Ideal observer usa full inference sobre P(z|x)")

print(f"\nFig2d: Correlation matrices")
print(f"  - Archivo: {fig2d_path}")
print(f"  - Correlaciones para cada contraste")

print("\n" + "=" * 70)
print("FIGURAS COMPLETADAS")
print("=" * 70)
