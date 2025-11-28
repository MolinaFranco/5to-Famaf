#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark CPU vs GPU para operaciones JAX.

Este script benchmarkea las operaciones JAX que se usan en:
- Training (fit_stage1, fit_stage2)
- Análisis (compute_posterior_from_activity)

Genera gráficos comparativos y guarda los resultados.
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin display
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Verificar que JAX GPU esté disponible
import jax
import jax.numpy as jnp

print("=" * 70)
print("VERIFICACIÓN DE GPU")
print("=" * 70)
print(f"JAX version: {jax.__version__}")
print(f"Dispositivos JAX: {jax.devices()}")
print(f"Backend por defecto: {jax.default_backend()}")
gpu_available = any(d.platform == 'gpu' for d in jax.devices())
print(f"GPU disponible: {gpu_available}")
print("=" * 70)

if not gpu_available:
    print("\n⚠️  GPU no detectada.")
    sys.exit(1)


def benchmark_matrix_operations(device='cpu', size=100, n_trials=20):
    """
    Benchmark de operaciones matriciales básicas.

    Operaciones típicas:
    - Multiplicación matriz-vector (W @ r)
    - Multiplicación matriz-matriz (A @ B)
    - Operaciones element-wise
    """
    print(f"\n  Benchmark: Matrix Operations {size}x{size} ({device.upper()})")

    # Configurar dispositivo
    if device == 'cpu':
        target_device = jax.devices('cpu')[0]
    else:
        target_device = jax.devices('gpu')[0]

    times = []

    for trial in range(n_trials):
        with jax.default_device(target_device):
            # Crear matrices
            key = jax.random.PRNGKey(trial)
            A = jax.random.normal(key, (size, size))
            B = jax.random.normal(key, (size, size))
            v = jax.random.normal(key, (size,))

            start = time.time()

            # Operaciones
            C = jnp.dot(A, B)
            w = jnp.dot(A, v)
            D = jnp.dot(C, B.T)
            result = jnp.sum(D) + jnp.sum(w)

            # Forzar ejecución
            _ = result.block_until_ready()

            elapsed = time.time() - start
            times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"    {avg_time*1000:.3f} ± {std_time*1000:.3f}ms")

    return avg_time, std_time


def benchmark_nonlinear_moments(device='cpu', size=100, n_trials=20):
    """
    Benchmark de cálculo de momentos no lineales.

    Simula compute_nonlinear_moments() del training:
    - Cálculo de PDF/CDF gaussiana
    - Operaciones element-wise complejas
    """
    print(f"\n  Benchmark: Nonlinear Moments ({device.upper()})")

    if device == 'cpu':
        target_device = jax.devices('cpu')[0]
    else:
        target_device = jax.devices('gpu')[0]

    times = []

    for trial in range(n_trials):
        with jax.default_device(target_device):
            key = jax.random.PRNGKey(trial)
            mu = jax.random.normal(key, (size,))
            sigma = jax.random.normal(key, (size, size))
            sigma = jnp.dot(sigma, sigma.T)

            start = time.time()

            # Simular compute_nonlinear_moments
            sigma2 = jnp.diag(sigma)
            sigma_std = jnp.sqrt(sigma2)
            ratio = jnp.where(sigma_std > 1e-10, mu / sigma_std, 0.0)

            from jax.scipy import stats as jax_stats
            phi = jax_stats.norm.pdf(ratio)
            psi = jax_stats.norm.cdf(ratio)

            nu1 = mu * psi + sigma_std * phi
            nu = 0.3 * (mu * nu1 + sigma2 * psi)
            gamma = 2 * 0.3 * nu1

            result = jnp.sum(nu) + jnp.sum(gamma)
            _ = result.block_until_ready()

            elapsed = time.time() - start
            times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"    {avg_time*1000:.3f} ± {std_time*1000:.3f}ms")

    return avg_time, std_time


def benchmark_adf_evolution(device='cpu', size=100, n_trials=15):
    """
    Benchmark de evolución ADF.

    Simula adf_evolution_step():
    - Múltiples productos matriciales
    - Operaciones de simetría
    - Cálculo de derivadas
    """
    print(f"\n  Benchmark: ADF Evolution ({device.upper()})")

    if device == 'cpu':
        target_device = jax.devices('cpu')[0]
    else:
        target_device = jax.devices('gpu')[0]

    times = []

    for trial in range(n_trials):
        with jax.default_device(target_device):
            key = jax.random.PRNGKey(trial)
            W = jax.random.normal(key, (size, size)) * 0.1
            mu = jax.random.normal(key, (size,))
            Sigma = jax.random.normal(key, (size, size))
            Sigma = jnp.dot(Sigma, Sigma.T)
            h = jax.random.normal(key, (size,))

            start = time.time()

            # Simular ADF evolution
            sigma2 = jnp.diag(Sigma)
            sigma_std = jnp.sqrt(sigma2)
            ratio = jnp.where(sigma_std > 1e-10, mu / sigma_std, 0.0)

            from jax.scipy import stats as jax_stats
            phi = jax_stats.norm.pdf(ratio)
            psi = jax_stats.norm.cdf(ratio)

            nu1 = mu * psi + sigma_std * phi
            gamma = 2 * 0.3 * nu1

            # Evolución de mu
            nu = 0.3 * (mu * nu1 + sigma2 * psi)
            dmu = -mu + jnp.dot(W, nu) + h

            # Evolución de Sigma
            Gamma_diag = jnp.diag(gamma)
            W_Gamma = jnp.dot(W, Gamma_diag)
            dSigma = jnp.dot(W_Gamma, Sigma) + jnp.dot(Sigma, W_Gamma.T)

            result = jnp.sum(dmu) + jnp.sum(dSigma)
            _ = result.block_until_ready()

            elapsed = time.time() - start
            times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"    {avg_time*1000:.3f} ± {std_time*1000:.3f}ms")

    return avg_time, std_time


def benchmark_cost_computation(device='cpu', size=100, n_trials=15):
    """
    Benchmark de cómputo de función de costo.

    Simula compute_cost():
    - Diferencias y normas
    - Pesos y sumas ponderadas
    """
    print(f"\n  Benchmark: Cost Computation ({device.upper()})")

    if device == 'cpu':
        target_device = jax.devices('cpu')[0]
    else:
        target_device = jax.devices('gpu')[0]

    times = []

    for trial in range(n_trials):
        with jax.default_device(target_device):
            key = jax.random.PRNGKey(trial)
            mu_ssn = jax.random.normal(key, (size,))
            mu_gsm = jax.random.normal(key, (size,))
            Sigma_ssn = jax.random.normal(key, (size, size))
            Sigma_gsm = jax.random.normal(key, (size, size))

            start = time.time()

            # Simular compute_cost
            diff_mu = mu_ssn - mu_gsm
            cost_mu = jnp.dot(diff_mu, diff_mu)

            var_ssn = jnp.diag(Sigma_ssn)
            var_gsm = jnp.diag(Sigma_gsm)
            diff_var = var_ssn - var_gsm
            cost_var = jnp.dot(diff_var, diff_var)

            diff_Sigma = Sigma_ssn - Sigma_gsm
            cost_Sigma = jnp.sum(diff_Sigma ** 2)

            total_cost = cost_mu + cost_var + cost_Sigma
            _ = total_cost.block_until_ready()

            elapsed = time.time() - start
            times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"    {avg_time*1000:.3f} ± {std_time*1000:.3f}ms")

    return avg_time, std_time


def benchmark_posterior_analysis(device='cpu', n_trials=10):
    """
    Benchmark de análisis posterior.

    Simula compute_posterior_from_activity():
    - Reconstrucción de estímulo
    - Cálculo de covarianzas
    - Loop de evaluación de contraste
    """
    print(f"\n  Benchmark: Posterior Analysis ({device.upper()})")

    if device == 'cpu':
        target_device = jax.devices('cpu')[0]
    else:
        target_device = jax.devices('gpu')[0]

    N_E = 50
    D_x = 256
    n_contrasts = 100

    times = []

    for trial in range(n_trials):
        with jax.default_device(target_device):
            key = jax.random.PRNGKey(trial)
            A = jax.random.normal(key, (D_x, N_E))
            C = jax.random.normal(key, (N_E, N_E))
            C = jnp.dot(C, C.T)
            r_e = jax.random.normal(key, (N_E,))

            start = time.time()

            # Reconstrucción
            x_reconstructed = jnp.dot(A, r_e)
            ACA_T = jnp.dot(A, jnp.dot(C, A.T))

            # Loop de contraste
            log_p = jnp.zeros(n_contrasts)
            for i in range(n_contrasts):
                z = float(i + 1) / n_contrasts
                cov = z * z * ACA_T + 100.0 * jnp.eye(D_x)
                log_like = -0.5 * jnp.sum(x_reconstructed ** 2) / (z + 0.1)
                log_p = log_p.at[i].set(log_like)

            # Normalización
            max_log_p = jnp.max(log_p)
            p_unnorm = jnp.exp(log_p - max_log_p)
            probs = p_unnorm / jnp.sum(p_unnorm)

            _ = probs.block_until_ready()

            elapsed = time.time() - start
            times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"    {avg_time*1000:.3f} ± {std_time*1000:.3f}ms")

    return avg_time, std_time


def run_all_benchmarks():
    """Ejecutar todos los benchmarks."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "BENCHMARK JAX: CPU vs GPU" + " "*28 + "║")
    print("╚" + "="*68 + "╝")
    print()

    results = {}
    benchmark_fns = [
        ('Matrix Operations', benchmark_matrix_operations, 100),
        ('Nonlinear Moments', benchmark_nonlinear_moments, 100),
        ('ADF Evolution', benchmark_adf_evolution, 100),
        ('Cost Computation', benchmark_cost_computation, 100),
        ('Posterior Analysis', benchmark_posterior_analysis, None)
    ]

    for name, fn, size in benchmark_fns:
        print(f"\n{'#'*70}")
        print(f"# {name}")
        print(f"{'#'*70}")

        results[name] = {}

        # CPU
        print("\n--- CPU ---")
        if size:
            cpu_time, cpu_std = fn(device='cpu', size=size)
        else:
            cpu_time, cpu_std = fn(device='cpu')
        results[name]['cpu'] = (cpu_time, cpu_std)

        # GPU
        print("\n--- GPU ---")
        if size:
            gpu_time, gpu_std = fn(device='gpu', size=size)
        else:
            gpu_time, gpu_std = fn(device='gpu')
        results[name]['gpu'] = (gpu_time, gpu_std)

        speedup = cpu_time / gpu_time
        print(f"\n🚀 Speedup: {speedup:.2f}x")

    return results


def generate_plots(results, output_dir='pycloude/outputs'):
    """Generar gráficos."""
    print("\n" + "="*70)
    print("GENERANDO GRÁFICOS")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    benchmarks = list(results.keys())
    cpu_times = []
    gpu_times = []
    cpu_stds = []
    gpu_stds = []
    speedups = []

    for bench in benchmarks:
        cpu_time, cpu_std = results[bench]['cpu']
        gpu_time, gpu_std = results[bench]['gpu']

        cpu_times.append(cpu_time * 1000)  # a ms
        gpu_times.append(gpu_time * 1000)
        cpu_stds.append(cpu_std * 1000)
        gpu_stds.append(gpu_std * 1000)
        speedups.append(cpu_time / gpu_time)

    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(benchmarks))
    width = 0.35

    # Gráfico de tiempos
    bars1 = ax1.bar(x - width/2, cpu_times, width, yerr=cpu_stds,
                    label='CPU', color='#4472C4', alpha=0.8,
                    capsize=5, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, gpu_times, width, yerr=gpu_stds,
                    label='GPU', color='#FF6B6B', alpha=0.8,
                    capsize=5, edgecolor='black', linewidth=0.5)

    ax1.set_ylabel('Tiempo (ms)', fontsize=13, fontweight='bold')
    ax1.set_title('Comparación de Tiempos: CPU vs GPU\nOperaciones JAX',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks, rotation=20, ha='right', fontsize=10)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Valores en barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=8)

    # Gráfico de speedups
    colors = ['#2ECC71' if s > 1 else '#E74C3C' for s in speedups]
    bars3 = ax2.bar(x, speedups, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2,
                label='Sin mejora (1x)', zorder=0)

    ax2.set_ylabel('Speedup (x veces más rápido)',
                   fontsize=13, fontweight='bold')
    ax2.set_title('Aceleración GPU vs CPU\nOperaciones JAX',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(benchmarks, rotation=20, ha='right', fontsize=10)
    ax2.legend(fontsize=12, loc='upper left')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Valores en barras
    for bar, speedup in zip(bars3, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Guardar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/benchmark_jax_gpu_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {filename}")

    plt.close()
    return filename


def save_results(results, output_dir='pycloude/outputs'):
    """Guardar resultados."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/benchmark_jax_results_{timestamp}.md'

    with open(filename, 'w') as f:
        f.write("# Benchmark JAX: CPU vs GPU\n\n")
        f.write(f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**GPU**: {jax.devices()[0]}\n")
        f.write(f"**JAX Version**: {jax.__version__}\n\n")
        f.write("---\n\n")

        f.write("## Resultados\n\n")
        f.write("| Benchmark | CPU (ms) | GPU (ms) | Speedup |\n")
        f.write("|-----------|----------|----------|----------|\n")

        speedups = []
        for name in results.keys():
            cpu_time, cpu_std = results[name]['cpu']
            gpu_time, gpu_std = results[name]['gpu']
            speedup = cpu_time / gpu_time
            speedups.append(speedup)

            f.write(f"| {name} | {cpu_time*1000:.2f} ± {cpu_std*1000:.2f} | "
                   f"{gpu_time*1000:.2f} ± {gpu_std*1000:.2f} | "
                   f"**{speedup:.2f}x** |\n")

        avg_speedup = np.mean(speedups)
        f.write(f"\n**Speedup promedio**: {avg_speedup:.2f}x\n\n")

        f.write("## Interpretación\n\n")
        f.write("Estos benchmarks miden las operaciones JAX que se usan en:\n\n")
        f.write("1. **Matrix Operations**: Multiplicaciones matriciales "
               "básicas (W@r, A@B)\n")
        f.write("2. **Nonlinear Moments**: Cálculo de momentos para "
               "activación supralineal\n")
        f.write("3. **ADF Evolution**: Evolución de momentos en "
               "Assumed Density Filtering\n")
        f.write("4. **Cost Computation**: Función de costo del entrenamiento\n")
        f.write("5. **Posterior Analysis**: Análisis de posterior "
               "bayesiano\n\n")

        f.write(f"**Conclusión**: Las operaciones JAX son en promedio "
               f"**{avg_speedup:.2f}x más rápidas** en GPU.\n")

    print(f"✓ Resultados guardados: {filename}")
    return filename


def main():
    """Ejecutar benchmark completo."""
    results = run_all_benchmarks()
    plot_file = generate_plots(results)
    md_file = save_results(results)

    # Resumen
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)

    speedups = []
    for name in results.keys():
        cpu_time, _ = results[name]['cpu']
        gpu_time, _ = results[name]['gpu']
        speedup = cpu_time / gpu_time
        speedups.append(speedup)
        print(f"{name:25s}: {speedup:.2f}x más rápido en GPU")

    avg_speedup = np.mean(speedups)
    print("-" * 70)
    print(f"{'PROMEDIO':25s}: {avg_speedup:.2f}x más rápido en GPU")
    print("="*70)

    print(f"\n📊 Gráfico: {plot_file}")
    print(f"📄 Reporte: {md_file}")
    print()


if __name__ == "__main__":
    main()
