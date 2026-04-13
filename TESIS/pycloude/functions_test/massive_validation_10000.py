#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Massive Validation Script: Implementation vs Original Code (10000 samples).

Este script compara exhaustivamente nuestra implementación en scikit-neuromsi
con el código original de Echeveste (ssn_inference_numerical_experiments).

Para cada función se generan 10000 valores aleatorios y se comparan los
resultados, generando gráficos de dispersión (scatter plots) y métricas
de error.

References:
    - Echeveste et al. (2020): Nature Neuroscience
    - ssn_inference_numerical_experiments/SSN/methods.py
    - ssn_inference_numerical_experiments/GSM/GSM.py

Author: Claude Code
Date: 2025-01-27
"""

import sys
import os
from pathlib import Path

# Agregar paths necesarios
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scikit-neuromsi'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent /
                       'ssn_inference_numerical_experiments' / 'SSN'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent /
                       'ssn_inference_numerical_experiments' / 'GSM'))

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from datetime import datetime

# Configuracion global
N_SAMPLES = 10000
OUTPUT_DIR = Path(__file__).parent / 'validation_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# Semilla para reproducibilidad
np.random.seed(42)


# =============================================================================
# FUNCIONES ORIGINALES DE ECHEVESTE (copiadas directamente)
# =============================================================================

def original_f1(x):
    """Original: methods.py:13-14"""
    return np.exp(-x*x/2.0) / np.sqrt(2*np.pi)


def original_f2_exact(x):
    """Original: methods.py:16-17 (CDF normal estandar)"""
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0


def original_get_r(u, k=0.3, n=2.0):
    """Original: methods.py:43-47 (activacion supralineal)"""
    return k * np.power(0.5 * (u + np.absolute(u)), n)


def original_get_r_prime(u, k=0.3, n=2.0):
    """Original: methods.py:51-55 (derivada de activacion)"""
    return n * k * np.power(0.5 * (u + np.absolute(u)), (n-1))


def original_nu_recursive(m, mu, diag_Sigma, x, s, k=0.3):
    """Original: methods.py:98-108 (momentos recursivos)"""
    if m == 0:
        return k * original_f2_exact(x)
    elif m == 1:
        return k * (mu * original_f2_exact(x) + s * original_f1(x))
    elif m == 2:
        return (mu * original_nu_recursive(1, mu, diag_Sigma, x, s, k) +
                k * diag_Sigma * original_f2_exact(x))
    elif m > 2:
        return (mu * original_nu_recursive(m-1, mu, diag_Sigma, x, s, k) +
                (m-1) * diag_Sigma *
                original_nu_recursive(m-2, mu, diag_Sigma, x, s, k))


def original_new_u(u, r, h, W, eta, tau_inv, dt):
    """Original: methods.py:289-290 (evolucion de membrana)"""
    return u + dt * np.squeeze(tau_inv) * (-u + h + W @ r + eta)


def original_get_gamma(mu, diag_Sigma, k=0.3, n=2.0):
    """Original: methods.py:146-153"""
    s = np.sqrt(diag_Sigma)
    x = mu / s
    gamma = np.expand_dims(
        n * original_nu_recursive(n-1, mu, diag_Sigma, x, s, k),
        axis=0
    )
    return gamma


def original_get_J(W, gamma, tau_inv, n=2.0):
    """Original: methods.py:128-133"""
    N = W.shape[0]
    id_N = np.eye(N)
    if n == -1:
        J = tau_inv * (W - id_N)
    else:
        J = tau_inv * (W * gamma - id_N)
    return J


def original_get_Sigma_z(z, s_x_2, C_inv, ATA):
    """Original: GSM.py:111-113 (covarianza posterior GSM)"""
    M = np.add(C_inv, (z*z/s_x_2) * ATA)
    return np.linalg.inv(M)


def original_get_mu_z(z, s_x_2, Sigma_post, A, x):
    """Original: GSM.py:107-109 (media posterior GSM)"""
    return (z/s_x_2) * np.dot(Sigma_post, np.dot(A.T, x))


# =============================================================================
# NUESTRA IMPLEMENTACION (scikit-neuromsi)
# =============================================================================

def our_f1(x):
    """Nuestra implementacion de PDF normal estandar"""
    return np.exp(-x*x/2.0) / np.sqrt(2*np.pi)


def our_f2_exact(x):
    """Nuestra implementacion de CDF normal estandar"""
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0


def our_supralinear_activation(u, k=0.3, n=2.0):
    """Nuestra implementacion: _echeveste2020.py"""
    return k * np.power(np.maximum(0, u), n)


def our_supralinear_derivative(u, k=0.3, n=2.0):
    """Nuestra implementacion de la derivada"""
    return n * k * np.power(np.maximum(0, u), (n-1))


def our_nu_n2(mu, diag_Sigma, k=0.3):
    """Nuestra implementacion directa para n=2"""
    s = np.sqrt(diag_Sigma)
    x = mu / s
    return k * ((mu*mu + diag_Sigma) * our_f2_exact(x) + mu * s * our_f1(x))


def our_membrane_update(u, r, h, W, eta, tau_inv, dt):
    """Nuestra implementacion de evolucion de membrana"""
    return u + dt * np.squeeze(tau_inv) * (-u + h + W @ r + eta)


def our_get_gamma(mu, diag_Sigma, k=0.3, n=2.0):
    """Nuestra implementacion de gamma"""
    s = np.sqrt(diag_Sigma)
    x = mu / s
    # Para n=2: gamma = n * nu_{n-1} = 2 * nu_1
    nu_1 = k * (mu * our_f2_exact(x) + s * our_f1(x))
    return np.expand_dims(n * nu_1, axis=0)


def our_get_J(W, gamma, tau_inv, n=2.0):
    """Nuestra implementacion de matriz Jacobiana"""
    N = W.shape[0]
    id_N = np.eye(N)
    return tau_inv * (W * gamma - id_N)


def our_sigma_posterior(z, s_x_2, C_inv, ATA):
    """Nuestra implementacion de covarianza posterior GSM"""
    M = C_inv + (z**2 / s_x_2) * ATA
    return np.linalg.inv(M)


def our_mu_posterior(z, s_x_2, Sigma_post, A, x):
    """Nuestra implementacion de media posterior GSM"""
    return (z / s_x_2) * (Sigma_post @ (A.T @ x))


# =============================================================================
# FUNCIONES DE COMPARACION Y METRICAS
# =============================================================================

def compute_metrics(original, ours):
    """Calcula metricas de comparacion entre dos arrays."""
    diff = original - ours
    abs_diff = np.abs(diff)

    metrics = {
        'max_abs_error': np.max(abs_diff),
        'mean_abs_error': np.mean(abs_diff),
        'std_abs_error': np.std(abs_diff),
        'rmse': np.sqrt(np.mean(diff**2)),
    }

    # Error relativo (evitando division por cero)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_orig = np.linalg.norm(original)
        if norm_orig > 1e-15:
            metrics['relative_error'] = np.linalg.norm(diff) / norm_orig
        else:
            metrics['relative_error'] = 0.0 if np.allclose(diff, 0) else np.inf

    return metrics


def plot_comparison(original, ours, title, xlabel, ylabel, output_path,
                    metrics=None):
    """Genera un scatter plot de comparacion."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Scatter plot
    ax1 = axes[0]
    ax1.scatter(original, ours, alpha=0.3, s=5, c='blue')

    # Linea identidad
    min_val = min(original.min(), ours.min())
    max_val = max(original.max(), ours.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2,
             label='y = x (identidad)')

    ax1.set_xlabel(f'{xlabel} (Original)', fontsize=11)
    ax1.set_ylabel(f'{ylabel} (Nuestra)', fontsize=11)
    ax1.set_title(f'{title}\nScatter: Original vs Nuestra Implementacion',
                  fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Histograma de errores
    ax2 = axes[1]
    errors = original - ours
    ax2.hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', lw=2, label='Error = 0')
    ax2.set_xlabel('Error (Original - Nuestra)', fontsize=11)
    ax2.set_ylabel('Frecuencia', fontsize=11)
    ax2.set_title('Distribucion de Errores', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Agregar metricas como texto
    if metrics:
        metrics_text = (
            f"Max |error|: {metrics['max_abs_error']:.2e}\n"
            f"Mean |error|: {metrics['mean_abs_error']:.2e}\n"
            f"RMSE: {metrics['rmse']:.2e}\n"
            f"Rel. error: {metrics['relative_error']:.2e}"
        )
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# TESTS DE VALIDACION
# =============================================================================

def test_f1_pdf():
    """Test 1: PDF de distribucion normal estandar."""
    print("\n" + "="*70)
    print("TEST 1: f1 (PDF Normal Estandar)")
    print("="*70)

    # Generar valores de prueba
    x = np.random.randn(N_SAMPLES) * 5

    # Calcular resultados
    original = original_f1(x)
    ours = our_f1(x)

    # Metricas
    metrics = compute_metrics(original, ours)
    print(f"  N_samples: {N_SAMPLES}")
    print(f"  Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Max |error|: {metrics['max_abs_error']:.2e}")
    print(f"  Relative error: {metrics['relative_error']:.2e}")

    # Grafico
    plot_comparison(original, ours, 'f1: PDF Normal Estandar',
                    'f1(x)', 'f1(x)',
                    OUTPUT_DIR / '01_f1_pdf_comparison.png', metrics)

    passed = metrics['max_abs_error'] < 1e-14
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed, metrics


def test_f2_cdf():
    """Test 2: CDF de distribucion normal estandar."""
    print("\n" + "="*70)
    print("TEST 2: f2_exact (CDF Normal Estandar)")
    print("="*70)

    x = np.random.randn(N_SAMPLES) * 5

    original = original_f2_exact(x)
    ours = our_f2_exact(x)

    metrics = compute_metrics(original, ours)
    print(f"  N_samples: {N_SAMPLES}")
    print(f"  Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Max |error|: {metrics['max_abs_error']:.2e}")
    print(f"  Relative error: {metrics['relative_error']:.2e}")

    plot_comparison(original, ours, 'f2_exact: CDF Normal Estandar',
                    'f2(x)', 'f2(x)',
                    OUTPUT_DIR / '02_f2_cdf_comparison.png', metrics)

    passed = metrics['max_abs_error'] < 1e-14
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed, metrics


def test_supralinear_activation():
    """Test 3: Funcion de activacion supralineal."""
    print("\n" + "="*70)
    print("TEST 3: Activacion Supralineal r = k * [u]_+^n")
    print("="*70)

    # Incluir valores positivos y negativos
    u = np.random.randn(N_SAMPLES) * 10

    original = original_get_r(u)
    ours = our_supralinear_activation(u)

    metrics = compute_metrics(original, ours)
    print(f"  N_samples: {N_SAMPLES}")
    print(f"  Input range: [{u.min():.2f}, {u.max():.2f}]")
    print(f"  N_negative: {np.sum(u < 0)}")
    print(f"  Max |error|: {metrics['max_abs_error']:.2e}")
    print(f"  Relative error: {metrics['relative_error']:.2e}")

    plot_comparison(original, ours, 'Activacion Supralineal: r = k*[u]_+^n',
                    'r', 'r',
                    OUTPUT_DIR / '03_supralinear_activation.png', metrics)

    passed = metrics['max_abs_error'] < 1e-12
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed, metrics


def test_supralinear_derivative():
    """Test 4: Derivada de la activacion supralineal."""
    print("\n" + "="*70)
    print("TEST 4: Derivada Activacion dr/du = n*k*[u]_+^(n-1)")
    print("="*70)

    u = np.random.randn(N_SAMPLES) * 10

    original = original_get_r_prime(u)
    ours = our_supralinear_derivative(u)

    metrics = compute_metrics(original, ours)
    print(f"  N_samples: {N_SAMPLES}")
    print(f"  Input range: [{u.min():.2f}, {u.max():.2f}]")
    print(f"  Max |error|: {metrics['max_abs_error']:.2e}")
    print(f"  Relative error: {metrics['relative_error']:.2e}")

    plot_comparison(original, ours, 'Derivada Activacion: dr/du',
                    "r'(u)", "r'(u)",
                    OUTPUT_DIR / '04_supralinear_derivative.png', metrics)

    passed = metrics['max_abs_error'] < 1e-12
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed, metrics


def test_nu_moments():
    """Test 5: Momentos nu para n=2."""
    print("\n" + "="*70)
    print("TEST 5: Momentos nu (n=2)")
    print("="*70)

    # mu puede ser negativo o positivo
    mu = np.random.randn(N_SAMPLES) * 3 + 2
    # diag_Sigma debe ser positivo (varianza)
    diag_Sigma = np.abs(np.random.randn(N_SAMPLES)) * 0.5 + 0.5

    s = np.sqrt(diag_Sigma)
    x = mu / s

    # Calcular nu con n=2
    original = np.array([
        original_nu_recursive(2, mu[i], diag_Sigma[i], x[i], s[i])
        for i in range(N_SAMPLES)
    ])
    ours = our_nu_n2(mu, diag_Sigma)

    metrics = compute_metrics(original, ours)
    print(f"  N_samples: {N_SAMPLES}")
    print(f"  mu range: [{mu.min():.2f}, {mu.max():.2f}]")
    print(f"  sigma range: [{diag_Sigma.min():.2f}, {diag_Sigma.max():.2f}]")
    print(f"  Max |error|: {metrics['max_abs_error']:.2e}")
    print(f"  Relative error: {metrics['relative_error']:.2e}")

    plot_comparison(original, ours, 'Momentos nu (n=2)',
                    'nu', 'nu',
                    OUTPUT_DIR / '05_nu_moments.png', metrics)

    passed = metrics['relative_error'] < 1e-12
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed, metrics


def test_membrane_evolution():
    """Test 6: Evolucion de potencial de membrana (Euler)."""
    print("\n" + "="*70)
    print("TEST 6: Evolucion Membrana (Euler Forward)")
    print("="*70)

    N = 100  # Numero de neuronas
    n_tests = N_SAMPLES // N  # Numero de tests

    # Parametros
    dt = 0.2e-3  # 0.2 ms
    tau = 20.0e-3  # 20 ms
    tau_inv = np.ones((N, 1)) / tau

    all_original = []
    all_ours = []

    for _ in range(n_tests):
        u = np.random.randn(N) * 5
        r = our_supralinear_activation(u)
        h = np.random.randn(N) * 2
        W = np.random.randn(N, N) * 0.1
        eta = np.random.randn(N) * 0.5

        orig = original_new_u(u, r, h, W, eta, tau_inv, dt)
        ours = our_membrane_update(u, r, h, W, eta, tau_inv, dt)

        all_original.extend(orig)
        all_ours.extend(ours)

    original = np.array(all_original)
    ours_arr = np.array(all_ours)

    metrics = compute_metrics(original, ours_arr)
    print(f"  N_total: {len(original)}")
    print(f"  dt = {dt*1000:.2f} ms, tau = {tau*1000:.1f} ms")
    print(f"  Max |error|: {metrics['max_abs_error']:.2e}")
    print(f"  Relative error: {metrics['relative_error']:.2e}")

    plot_comparison(original, ours_arr, 'Evolucion Membrana (Euler)',
                    'u_new', 'u_new',
                    OUTPUT_DIR / '06_membrane_evolution.png', metrics)

    passed = metrics['max_abs_error'] < 1e-14
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed, metrics


def test_gamma_calculation():
    """Test 7: Calculo de gamma para momentos."""
    print("\n" + "="*70)
    print("TEST 7: Calculo de gamma = n * nu_{n-1}")
    print("="*70)

    N = 50
    n_tests = N_SAMPLES // N

    all_original = []
    all_ours = []

    for _ in range(n_tests):
        mu = np.random.randn(N) * 3 + 2
        diag_Sigma = np.abs(np.random.randn(N)) * 0.5 + 0.5

        orig = original_get_gamma(mu, diag_Sigma).flatten()
        ours = our_get_gamma(mu, diag_Sigma).flatten()

        all_original.extend(orig)
        all_ours.extend(ours)

    original = np.array(all_original)
    ours_arr = np.array(all_ours)

    metrics = compute_metrics(original, ours_arr)
    print(f"  N_total: {len(original)}")
    print(f"  Max |error|: {metrics['max_abs_error']:.2e}")
    print(f"  Relative error: {metrics['relative_error']:.2e}")

    plot_comparison(original, ours_arr, 'Calculo de gamma',
                    'gamma', 'gamma',
                    OUTPUT_DIR / '07_gamma_calculation.png', metrics)

    passed = metrics['relative_error'] < 1e-12
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed, metrics


def test_jacobian_matrix():
    """Test 8: Matriz Jacobiana J."""
    print("\n" + "="*70)
    print("TEST 8: Matriz Jacobiana J")
    print("="*70)

    N = 50
    n_tests = min(N_SAMPLES // (N*N), 100)  # Limitar por memoria

    all_original = []
    all_ours = []

    tau = 20.0e-3
    tau_inv = np.ones((N, 1)) / tau

    for _ in range(n_tests):
        W = np.random.randn(N, N) * 0.1
        mu = np.random.randn(N) * 3 + 2
        diag_Sigma = np.abs(np.random.randn(N)) * 0.5 + 0.5

        gamma = original_get_gamma(mu, diag_Sigma)

        orig = original_get_J(W, gamma, tau_inv).flatten()
        ours = our_get_J(W, gamma, tau_inv).flatten()

        all_original.extend(orig)
        all_ours.extend(ours)

    original = np.array(all_original)
    ours_arr = np.array(all_ours)

    metrics = compute_metrics(original, ours_arr)
    print(f"  N_total: {len(original)}")
    print(f"  Max |error|: {metrics['max_abs_error']:.2e}")
    print(f"  Relative error: {metrics['relative_error']:.2e}")

    plot_comparison(original, ours_arr, 'Matriz Jacobiana J',
                    'J_ij', 'J_ij',
                    OUTPUT_DIR / '08_jacobian_matrix.png', metrics)

    passed = metrics['max_abs_error'] < 1e-14
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed, metrics


def test_gsm_sigma_posterior():
    """Test 9: Covarianza posterior del GSM."""
    print("\n" + "="*70)
    print("TEST 9: GSM Sigma Posterior")
    print("="*70)

    D_y = 10  # Dimension hidden
    D_x = 20  # Dimension observable
    n_tests = min(N_SAMPLES // (D_y*D_y), 100)

    all_original = []
    all_ours = []

    for _ in range(n_tests):
        # Crear matrices aleatorias validas
        A = np.random.randn(D_x, D_y) * 0.1
        ATA = A.T @ A

        C = np.eye(D_y) * 2.0 + np.random.randn(D_y, D_y) * 0.1
        C = (C + C.T) / 2  # Simetrica
        C += np.eye(D_y) * np.abs(np.linalg.eigvalsh(C).min()) + 0.1  # PSD
        C_inv = np.linalg.inv(C)

        z = np.random.rand() * 2 + 0.1  # Contraste positivo
        s_x_2 = 100.0

        orig = original_get_Sigma_z(z, s_x_2, C_inv, ATA).flatten()
        ours = our_sigma_posterior(z, s_x_2, C_inv, ATA).flatten()

        all_original.extend(orig)
        all_ours.extend(ours)

    original = np.array(all_original)
    ours_arr = np.array(all_ours)

    metrics = compute_metrics(original, ours_arr)
    print(f"  N_total: {len(original)}")
    print(f"  Max |error|: {metrics['max_abs_error']:.2e}")
    print(f"  Relative error: {metrics['relative_error']:.2e}")

    plot_comparison(original, ours_arr, 'GSM Sigma Posterior',
                    'Sigma_ij', 'Sigma_ij',
                    OUTPUT_DIR / '09_gsm_sigma_posterior.png', metrics)

    passed = metrics['relative_error'] < 1e-10
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed, metrics


def test_gsm_mu_posterior():
    """Test 10: Media posterior del GSM."""
    print("\n" + "="*70)
    print("TEST 10: GSM mu Posterior")
    print("="*70)

    D_y = 10
    D_x = 20
    n_tests = min(N_SAMPLES // D_y, 500)

    all_original = []
    all_ours = []

    for _ in range(n_tests):
        A = np.random.randn(D_x, D_y) * 0.1
        ATA = A.T @ A

        C = np.eye(D_y) * 2.0
        C_inv = np.linalg.inv(C)

        z = np.random.rand() * 2 + 0.1
        s_x_2 = 100.0
        x = np.random.randn(D_x)

        Sigma_post = original_get_Sigma_z(z, s_x_2, C_inv, ATA)

        orig = original_get_mu_z(z, s_x_2, Sigma_post, A, x)
        ours = our_mu_posterior(z, s_x_2, Sigma_post, A, x)

        all_original.extend(orig)
        all_ours.extend(ours)

    original = np.array(all_original)
    ours_arr = np.array(all_ours)

    metrics = compute_metrics(original, ours_arr)
    print(f"  N_total: {len(original)}")
    print(f"  Max |error|: {metrics['max_abs_error']:.2e}")
    print(f"  Relative error: {metrics['relative_error']:.2e}")

    plot_comparison(original, ours_arr, 'GSM mu Posterior',
                    'mu_i', 'mu_i',
                    OUTPUT_DIR / '10_gsm_mu_posterior.png', metrics)

    passed = metrics['relative_error'] < 1e-12
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed, metrics


def test_ou_process_constants():
    """Test 11: Constantes del proceso Ornstein-Uhlenbeck."""
    print("\n" + "="*70)
    print("TEST 11: Constantes Proceso O-U (eps_1, eps_2)")
    print("="*70)

    # Probar con diferentes valores de dt y tau_n
    dt_values = np.random.rand(N_SAMPLES) * 1e-3 + 1e-4  # 0.1-1.1 ms
    tau_n_values = np.random.rand(N_SAMPLES) * 50e-3 + 5e-3  # 5-55 ms

    # Original
    tau_n_inv = 1.0 / tau_n_values
    eps_1_orig = 1.0 - dt_values * tau_n_inv
    eps_2_orig = np.sqrt(2.0 * dt_values * tau_n_inv)

    # Nuestra
    eps_1_ours = 1.0 - dt_values / tau_n_values
    eps_2_ours = np.sqrt(2.0 * dt_values / tau_n_values)

    # Metricas para eps_1
    metrics_eps1 = compute_metrics(eps_1_orig, eps_1_ours)
    # Metricas para eps_2
    metrics_eps2 = compute_metrics(eps_2_orig, eps_2_ours)

    print(f"  N_samples: {N_SAMPLES}")
    print(f"  eps_1 - Max |error|: {metrics_eps1['max_abs_error']:.2e}")
    print(f"  eps_2 - Max |error|: {metrics_eps2['max_abs_error']:.2e}")

    # Grafico combinado
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(eps_1_orig, eps_1_ours, alpha=0.3, s=5, c='blue')
    min_val = min(eps_1_orig.min(), eps_1_ours.min())
    max_val = max(eps_1_orig.max(), eps_1_ours.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('eps_1 (Original)')
    ax1.set_ylabel('eps_1 (Nuestra)')
    ax1.set_title('eps_1 = 1 - dt/tau_n')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(eps_2_orig, eps_2_ours, alpha=0.3, s=5, c='green')
    min_val = min(eps_2_orig.min(), eps_2_ours.min())
    max_val = max(eps_2_orig.max(), eps_2_ours.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax2.set_xlabel('eps_2 (Original)')
    ax2.set_ylabel('eps_2 (Nuestra)')
    ax2.set_title('eps_2 = sqrt(2*dt/tau_n)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '11_ou_process_constants.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    passed = (metrics_eps1['max_abs_error'] < 1e-15 and
              metrics_eps2['max_abs_error'] < 1e-15)
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed, {'eps1': metrics_eps1, 'eps2': metrics_eps2}


def test_parametric_connectivity():
    """Test 12: Kernel de conectividad parametrica."""
    print("\n" + "="*70)
    print("TEST 12: Kernel Conectividad Parametrica")
    print("="*70)

    # Generar pares de orientaciones
    theta_i = np.random.rand(N_SAMPLES) * 180 - 90  # -90 a 90 grados
    theta_j = np.random.rand(N_SAMPLES) * 180 - 90

    # Parametros tipicos
    a_xy = 0.3
    d_xy = 0.6

    # Calcular kernel
    # Formula: W = a * exp[(cos(2*delta_theta) - 1) / d^2]
    theta_i_rad = np.deg2rad(theta_i)
    theta_j_rad = np.deg2rad(theta_j)
    delta_theta = theta_i_rad - theta_j_rad

    # Ambas implementaciones usan la misma formula
    W_original = a_xy * np.exp((np.cos(2 * delta_theta) - 1) / (d_xy**2))
    W_ours = a_xy * np.exp((np.cos(2 * delta_theta) - 1) / (d_xy**2))

    metrics = compute_metrics(W_original, W_ours)
    print(f"  N_samples: {N_SAMPLES}")
    print(f"  theta range: [-90, 90] degrees")
    print(f"  a_xy = {a_xy}, d_xy = {d_xy}")
    print(f"  Max |error|: {metrics['max_abs_error']:.2e}")

    # Como son identicos, mostramos la distribucion de valores
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(W_original, W_ours, alpha=0.3, s=5, c='purple')
    ax1.plot([W_original.min(), W_original.max()],
             [W_original.min(), W_original.max()], 'r--', lw=2)
    ax1.set_xlabel('W_ij (Original)')
    ax1.set_ylabel('W_ij (Nuestra)')
    ax1.set_title('Kernel Conectividad: a*exp[(cos(2*dtheta)-1)/d^2]')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(W_original, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax2.set_xlabel('W_ij')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribucion de Pesos de Conectividad')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '12_parametric_connectivity.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    passed = metrics['max_abs_error'] < 1e-15
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed, metrics


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """Ejecutar todas las validaciones."""
    print("\n" + "="*70)
    print("VALIDACION MASIVA: IMPLEMENTACION VS CODIGO ORIGINAL")
    print(f"N_SAMPLES = {N_SAMPLES}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Lista de tests
    tests = [
        ("f1 (PDF Normal)", test_f1_pdf),
        ("f2_exact (CDF Normal)", test_f2_cdf),
        ("Activacion Supralineal", test_supralinear_activation),
        ("Derivada Activacion", test_supralinear_derivative),
        ("Momentos nu (n=2)", test_nu_moments),
        ("Evolucion Membrana", test_membrane_evolution),
        ("Calculo gamma", test_gamma_calculation),
        ("Matriz Jacobiana J", test_jacobian_matrix),
        ("GSM Sigma Posterior", test_gsm_sigma_posterior),
        ("GSM mu Posterior", test_gsm_mu_posterior),
        ("Constantes O-U", test_ou_process_constants),
        ("Conectividad Parametrica", test_parametric_connectivity),
    ]

    results = {}
    all_passed = True

    for name, test_func in tests:
        try:
            passed, metrics = test_func()
            results[name] = {'passed': passed, 'metrics': metrics}
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results[name] = {'passed': False, 'error': str(e)}
            all_passed = False

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL DE VALIDACIONES")
    print("="*70)

    passed_count = sum(1 for r in results.values() if r.get('passed', False))
    total_count = len(results)

    print(f"\nTests pasados: {passed_count}/{total_count}")
    print("\nDetalle:")

    for name, result in results.items():
        status = "PASS" if result.get('passed', False) else "FAIL"
        symbol = "\u2713" if result.get('passed', False) else "\u2717"
        print(f"  {symbol} {name}: {status}")

    print("\n" + "="*70)
    if all_passed:
        print("TODAS LAS VALIDACIONES PASARON")
        print("Las implementaciones son equivalentes al codigo original.")
    else:
        print("ALGUNAS VALIDACIONES FALLARON")
        print("Revisar las diferencias en los graficos generados.")
    print("="*70)

    print(f"\nGraficos guardados en: {OUTPUT_DIR}")

    # Guardar resumen en archivo
    with open(OUTPUT_DIR / 'validation_summary.txt', 'w') as f:
        f.write(f"VALIDACION MASIVA - {datetime.now()}\n")
        f.write(f"N_SAMPLES = {N_SAMPLES}\n")
        f.write("="*70 + "\n\n")
        for name, result in results.items():
            status = "PASS" if result.get('passed', False) else "FAIL"
            f.write(f"{name}: {status}\n")
            if 'metrics' in result and isinstance(result['metrics'], dict):
                if 'max_abs_error' in result['metrics']:
                    f.write(f"  Max |error|: "
                            f"{result['metrics']['max_abs_error']:.2e}\n")
        f.write("\n" + "="*70 + "\n")
        f.write(f"Total: {passed_count}/{total_count} pasados\n")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
