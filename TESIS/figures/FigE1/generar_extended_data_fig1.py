#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para generar Extended Data Figure 1 de Echeveste et al. (2020).

Este script genera los gráficos solicitados:
- Fig1b: Covarianza previa en el GSM (prior covariance C)
- Fig1c: Pesos recurrentes después del entrenamiento (3 subplots)
- Fig1e: No linealidad de transformación de activaciones

Referencias:
- Echeveste et al. (2020), Nature Neuroscience
- Extended Data Fig. 1: "GSM and network parameters"
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Agregar scikit-neuromsi al path
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

from skneuromsi.data import EchevesteDataLoader  # noqa: E402

# Configuración de matplotlib para figuras de paper
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.grid": False,
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.0,
    }
)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros del modelo según Echeveste et al. (2020) Supp. Table S1
N_E = 50  # Número de neuronas excitatorias
N_I = 50  # Número de neuronas inhibitorias
N = N_E + N_I  # Total de neuronas

# Parámetros de la transformación de entrada h (Eq. 14)
# Valores optimizados según parametros.md
ALPHA_H = 1.96
BETA_H = 0.10
GAMMA_H = 2.03

# Colores consistentes para tipos de conexiones
COLORS = {
    "EE": "#2E86AB",  # Azul para E→E
    "EI": "#A23B72",  # Magenta para E→I
    "IE": "#F18F01",  # Naranja para I→E
    "II": "#C73E1D",  # Rojo para I→I
}

print("=" * 80)
print("GENERACIÓN DE EXTENDED DATA FIG. 1 - ECHEVESTE ET AL. (2020)")
print("=" * 80)
print(f"\nDirectorio de salida: {OUTPUT_DIR}")
print(f"Parámetros: N_E={N_E}, N_I={N_I}")
print()

# ============================================================================
# CARGAR DATOS PRE-ENTRENADOS
# ============================================================================

print("Cargando datos pre-entrenados...")
loader = EchevesteDataLoader()

# Cargar datos del GSM
C = loader.load_prior_covariance()  # Prior covariance (50x50)
A = loader.load_gabor_filters()  # Gabor filters (256x50)

# Cargar parámetros de conectividad optimizados
conn_params = loader.load_ssn_connectivity_parameters()
a_EE = conn_params["a_EE"]
a_EI = conn_params["a_EI"]
a_IE = conn_params["a_IE"]
a_II = conn_params["a_II"]
d_EE = conn_params["d_EE"]
d_EI = conn_params["d_EI"]
d_IE = conn_params["d_IE"]
d_II = conn_params["d_II"]

# Cargar matriz de conectividad completa
W_full = loader.load_exact_connectivity_matrix()  # (100x100)

# Cargar covarianza del ruido de proceso
Sigma_eta = loader.load_noise_covariance()  # (100x100)

print(f"✓ C shape: {C.shape}")
print(f"✓ A shape: {A.shape}")
print(f"✓ W_full shape: {W_full.shape}")
print(f"✓ Sigma_eta shape: {Sigma_eta.shape}")
print(
    f"✓ Connectivity params: a_EE={a_EE:.3f}, a_EI={a_EI:.3f}, "
    f"a_IE={a_IE:.3f}, a_II={a_II:.3f}"
)
print()

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================


def compute_preferred_orientations(n_neurons):
    """
    Compute preferred orientations for ring network.

    Parameters
    ----------
    n_neurons : int
        Number of neurons

    Returns
    -------
    orientations : np.ndarray
        Preferred orientations in degrees from -90 to 90
    """
    # Ring topology: orientaciones uniformemente distribuidas
    # Echeveste et al. (2020) usa orientaciones de -90 a 90 grados
    # Centramos en 0 para que el índice 0 corresponda a aprox. -90°
    # y tengamos orientaciones simétricas alrededor de 0
    step = 180 / n_neurons
    return np.arange(n_neurons) * step - 90 + step / 2


def extract_circulant_row(matrix, n_e):
    """
    Extract one row from each block of circulant connectivity matrix.

    Como la matriz W es circulante por bloques, una sola fila de cada
    cuadrante contiene toda la información de conectividad.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Full connectivity matrix
    n_e : int
        Number of excitatory neurons

    Returns
    -------
    dict
        Dictionary with keys 'EE', 'EI', 'IE', 'II' containing
        connectivity profiles as 1D arrays
    """
    # Extraer primera fila de cada cuadrante
    # EE: conexiones E→E (primera fila del cuadrante superior izquierdo)
    w_ee = matrix[0, :n_e]

    # EI: conexiones E→I (primera fila del cuadrante superior derecho)
    w_ei = matrix[0, n_e:]

    # IE: conexiones I→E (primera fila E del cuadrante inferior izquierdo)
    w_ie = matrix[n_e, :n_e]

    # II: conexiones I→I (primera fila I del cuadrante inferior derecho)
    w_ii = matrix[n_e, n_e:]

    return {"EE": w_ee, "EI": w_ei, "IE": w_ie, "II": w_ii}


def normalize_weights(weights_dict):
    """
    Normalize absolute values of weights for comparison.

    Parameters
    ----------
    weights_dict : dict
        Dictionary with connectivity profiles

    Returns
    -------
    dict
        Normalized weights (max absolute value = 1 for each type)
    """
    normalized = {}
    for key, w in weights_dict.items():
        abs_w = np.abs(w)
        max_val = np.max(abs_w)
        if max_val > 0:
            normalized[key] = abs_w / max_val
        else:
            normalized[key] = abs_w
    return normalized


def extract_noise_covariance_row(sigma_eta, n_e):
    """
    Extract noise covariance profiles from circulant matrix.

    Parameters
    ----------
    sigma_eta : np.ndarray, shape (N, N)
        Process noise covariance matrix
    n_e : int
        Number of excitatory neurons

    Returns
    -------
    dict
        Dictionary with keys 'EE', 'EI=IE', 'II' containing
        covariance profiles (mV^2)
    """
    # Extraer primera fila de cada bloque
    cov_ee = sigma_eta[0, :n_e]  # Covar entre E cells
    cov_ei = sigma_eta[0, n_e:]  # Covar entre E-I cells
    cov_ii = sigma_eta[n_e, n_e:]  # Covar entre I cells

    # Según el paper, EI = IE (simetría)
    return {"EE": cov_ee, "EI=IE": cov_ei, "II": cov_ii}


# ============================================================================
# FIG 1b: PRIOR COVARIANCE IN THE GSM
# ============================================================================

print("Generando Fig1b: Prior covariance in the GSM...")

# Calcular orientaciones preferidas
orientations = compute_preferred_orientations(C.shape[0])

fig1b = plt.figure(figsize=(6, 5))
ax = fig1b.add_subplot(111)

# Graficar matriz de covarianza previa C
# Forzar límites de colormap entre -4 y 4
# Eje Y: -90 abajo, 90 arriba (extent: [-90, 90, -90, 90])
im = ax.imshow(
    C,
    cmap="RdBu_r",
    aspect="auto",
    extent=[-90, 90, -90, 90],
    interpolation="nearest",
    vmin=-4,
    vmax=4,
    origin="lower",
)  # origin='lower' para -90 abajo

# Configurar ejes para mostrar límites -90 y 90 claramente
ax.set_xlabel("Preferred Orientation (degrees)")
ax.set_ylabel("Preferred Orientation (degrees)")
ax.set_title("Prior Covariance in the GSM (C)")
ax.set_xlim(-90, 90)
ax.set_ylim(-90, 90)  # -90 abajo, 90 arriba

# Colorbar con límites explícitos
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Covariance")
cbar.set_ticks([-4, -2, 0, 2, 4])

plt.tight_layout()
fig1b_path = os.path.join(OUTPUT_DIR, "FigE1b_prior_covariance_GSM.png")
plt.savefig(fig1b_path, dpi=300, bbox_inches="tight")
print(f"✓ Guardada: {fig1b_path}")
plt.close()

# ============================================================================
# FIG E1d: RECURRENT WEIGHTS AFTER TRAINING
# ============================================================================

print("Generando FigE1d: Recurrent weights after training...")

# Extraer perfiles de conectividad (una fila de cada cuadrante)
weights = extract_circulant_row(W_full, N_E)
weights_norm = normalize_weights(weights)
noise_cov = extract_noise_covariance_row(Sigma_eta, N_E)

# Calcular diferencias de orientación relativas
# En una matriz circulante, queremos mostrar diferencias desde -90 a 90
# Para evitar discontinuidades, reordenamos los datos
orientations_e = compute_preferred_orientations(N_E)

# Crear array de diferencias de orientación de -90 a 90
# Esto representa diferencias circulares desde la primera neurona
orientation_diff = np.linspace(-90, 90, N_E, endpoint=False)

# Reordenar los pesos para que correspondan a estas diferencias
# La primera neurona (índice 0) tiene diferencia 0
# Necesitamos rotar los arrays para que el índice 0 esté en el medio
mid_idx = N_E // 2
# Crear índices rotados
indices = np.roll(np.arange(N_E), -mid_idx)

# Reordenar todos los arrays de pesos
for key in weights.keys():
    weights[key] = weights[key][indices]
    weights_norm[key] = weights_norm[key][indices]

for key in noise_cov.keys():
    noise_cov[key] = noise_cov[key][indices]

# Crear figura con 3 subplots verticales
fig1d, axes = plt.subplots(3, 1, figsize=(7, 9))

# --- Subplot 1: Raw recurrent weights (×10^-1) ---
ax1 = axes[0]
for conn_type in ["EE", "EI", "IE", "II"]:
    w = weights[conn_type] * 10  # Convertir a unidades de 10^-1
    ax1.plot(
        orientation_diff, w,
        label=conn_type, color=COLORS[conn_type], linewidth=2
    )

ax1.set_xlabel("Orientation difference (degrees)")
ax1.set_ylabel("rec. weights (×10⁻¹)")
ax1.set_title("Recurrent Weights After Training")
ax1.set_xlim(-90, 90)
ax1.legend(loc="best", frameon=True)
ax1.axhline(0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
ax1.axvline(0, color="k", linestyle=":", linewidth=0.5, alpha=0.5)
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Normalized absolute recurrent weights ---
ax2 = axes[1]
for conn_type in ["EE", "EI", "IE", "II"]:
    w_norm = weights_norm[conn_type]
    ax2.plot(
        orientation_diff, w_norm,
        label=conn_type, color=COLORS[conn_type], linewidth=2
    )

ax2.set_xlabel("Orientation difference (degrees)")
ax2.set_ylabel("norm. rec. weights")
ax2.set_title("Normalized Absolute Recurrent Weights")
ax2.set_xlim(-90, 90)
ax2.legend(loc="best", frameon=True)
ax2.axvline(0, color="k", linestyle=":", linewidth=0.5, alpha=0.5)
ax2.grid(True, alpha=0.3)

# --- Subplot 3: Process noise covariance (mV²) ---
ax3 = axes[2]
for noise_type in ["EE", "EI=IE", "II"]:
    cov = noise_cov[noise_type]
    # Usar los mismos colores, pero EI=IE usa color promedio
    if noise_type == "EI=IE":
        color = COLORS["EI"]  # Usar color EI para representar EI=IE
    else:
        color = COLORS[noise_type]

    ax3.plot(orientation_diff, cov, label=noise_type, color=color, linewidth=2)

ax3.set_xlabel("Orientation difference (degrees)")
ax3.set_ylabel("process noise cov (mV²)")
ax3.set_title("Process Noise Covariance")
ax3.set_xlim(-90, 90)
ax3.legend(loc="best", frameon=True)
ax3.axvline(0, color="k", linestyle=":", linewidth=0.5, alpha=0.5)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
fig1d_path = os.path.join(OUTPUT_DIR, "FigE1d_recurrent_weights.png")
plt.savefig(fig1d_path, dpi=300, bbox_inches="tight")
print(f"✓ Guardada: {fig1d_path}")
plt.close()

# ============================================================================
# FIG E1e: INPUT NONLINEARITY
# ============================================================================

print("Generando FigE1e: Input nonlinearity...")

# Cargar el GSM para generar estímulos de entrenamiento
from skneuromsi.generative import GSM  # noqa: E402

# Crear instancia del GSM con parámetros de Echeveste
# use_pretrained=True carga automáticamente A y C del paper
gsm = GSM(
    n_orientations=N_E,  # 50 orientaciones
    use_pretrained=True,  # Cargar filtros pre-entrenados
    alpha_h=ALPHA_H,
    beta_h=BETA_H,
    gamma_h=GAMMA_H,
)

print(f"  GSM cargado: A shape = {gsm.A.shape}, C shape = {gsm.C.shape}")

# Generar N_samples estímulos del training set
N_samples = 1000  # Número de estímulos para simular el training set
print(f"  Generando {N_samples} estímulos del training set...")

# Generar muestras del GSM: y ~ N(0, z*C), x = A @ y + noise
# Queremos la distribución de (W_ff @ x)_i ANTES de la transformación
# y también los inputs h DESPUÉS de la transformación
wff_x_samples = []  # Activaciones del campo receptivo (pueden ser negativas)
h_samples = []      # Inputs a la red (siempre positivos)

# Rango de contrastes típico del training set (de 0.1 a 1.0)
contrasts = np.random.uniform(0.1, 1.0, N_samples)

for i, contrast in enumerate(contrasts):
    # Generar muestra del GSM
    stimulus = gsm.generate_stimulus_patch(contrast)
    x = stimulus['x']  # Patch de imagen (256,)

    # Calcular activación del campo receptivo: (A.T @ x)
    # Normalizado por el factor 15.0 del paper (h_scale)
    wff_x = (gsm.A.T @ x) / 15.0

    # Aplicar transformación no-lineal para obtener h
    h = ALPHA_H * np.maximum(wff_x + BETA_H, 0) ** GAMMA_H

    # Guardar ambas distribuciones
    wff_x_samples.extend(wff_x)
    h_samples.extend(h)

wff_x_samples = np.array(wff_x_samples)
h_samples = np.array(h_samples)

print(f"  ✓ {len(wff_x_samples)} activaciones generadas")
print(
    f"  (W_ff·x)_i - Rango: [{wff_x_samples.min():.2f}, "
    f"{wff_x_samples.max():.2f}], "
    f"Media: {wff_x_samples.mean():.2f}, Std: {wff_x_samples.std():.2f}"
)
print(
    f"  h_i - Rango: [{h_samples.min():.2f}, {h_samples.max():.2f}], "
    f"Media: {h_samples.mean():.2f}, Std: {h_samples.std():.2f}"
)

# Definir la transformación no-lineal de entrada
# h_i = α_h * [W_ff @ x + β_h]_+^γ_h
# Nota: En el código de Echeveste, la transformación es:
# h = nl_scale * max(W_ff @ x + nl_baseline, 0)^nl_power
# donde nl_baseline es un threshold POSITIVO que se RESTA


def input_nonlinearity(wff_x, alpha_h, beta_h, gamma_h):
    """
    Compute input nonlinearity h = α_h * [W_ff @ x + β_h]_+^γ_h.

    Parameters
    ----------
    wff_x : np.ndarray
        Feedforward receptive field activations (W_ff @ x)_i
    alpha_h : float
        Scale parameter
    beta_h : float
        Baseline/threshold parameter
    gamma_h : float
        Power parameter

    Returns
    -------
    h : np.ndarray
        Network input after nonlinear transformation
    """
    # Aplicar threshold y no-linealidad
    return alpha_h * np.maximum(wff_x + beta_h, 0) ** gamma_h


# Crear figura FigE1e
fig1e = plt.figure(figsize=(6, 5))
ax = fig1e.add_subplot(111)

# Graficar la transformación no-lineal como curva negra
x_range = np.linspace(-2, 2, 500)
h_transformed = input_nonlinearity(x_range, ALPHA_H, BETA_H, GAMMA_H)

ax.plot(
    x_range,
    h_transformed,
    "k-",
    linewidth=2.5,
    label=f"$h_i = {ALPHA_H:.2f} \\cdot "
    f"\\max(W_{{ff}} \\cdot x + {BETA_H:.2f}, 0)^{{{GAMMA_H:.2f}}}$",
)

# Graficar línea roja punteada para el threshold
threshold_x = -BETA_H
ax.axvline(
    threshold_x,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Threshold: {threshold_x:.2f}",
)

# Configurar ejes y etiquetas
ax.set_xlabel("$(W_{ff} \\cdot x)_i$")
ax.set_ylabel("$h_i$")
ax.set_title("Input Nonlinearity (Eq. 14)")
ax.set_xlim(-2, 2)
ax.set_ylim(0, 8)
ax.legend(loc="upper left", frameon=True, fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig1e_path = os.path.join(OUTPUT_DIR, "FigE1e_input_nonlinearity.png")
plt.savefig(fig1e_path, dpi=300, bbox_inches="tight")
print(f"✓ Guardada: {fig1e_path}")
plt.close()

# ============================================================================
# RESUMEN
# ============================================================================

print()
print("=" * 80)
print("RESUMEN DE FIGURAS GENERADAS - EXTENDED DATA FIG. E1")
print("=" * 80)
print("✓ FigE1b: Prior covariance in the GSM")
print(f"  - Archivo: {fig1b_path}")
print("  - Eje Y: -90° (abajo) a 90° (arriba)")
print("  - Colorbar: -4 a 4")
print()
print("✓ FigE1d: Recurrent weights after training (3 subplots)")
print(f"  - Archivo: {fig1d_path}")
print("  - Eje X: Diferencias de orientación -90° a 90°")
print("  - Pico en 0° (diferencia = 0)")
print("  - Subplot 1: Raw recurrent weights (×10⁻¹)")
print("  - Subplot 2: Normalized absolute recurrent weights")
print("  - Subplot 3: Process noise covariance (mV²)")
print()
print("✓ FigE1e: Input nonlinearity (Eq. 14)")
print(f"  - Archivo: {fig1e_path}")
print(
    f"  - Parámetros: α_h={ALPHA_H:.2f}, β_h={BETA_H:.2f}, "
    f"γ_h={GAMMA_H:.2f}"
)
print("  - Eje X: -2 a 2 (rango completo)")
print("  - Distribución gris: (W_ff·x)_i del training set")
print("  - Curva negra: transformación h_i = f((W_ff·x)_i)")
print("  - Línea roja punteada: threshold β_h con etiqueta")
print()
print(f"Todas las figuras se guardaron en: {OUTPUT_DIR}")
print("=" * 80)
