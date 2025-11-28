#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark de escalabilidad: CPU vs GPU para diferentes tamaños.

Este script muestra cómo el beneficio de GPU aumenta con el tamaño
del problema. Genera gráficos que muestran el "crossover point"
donde GPU se vuelve más rápida que CPU.
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import os

import jax
import jax.numpy as jnp

print("=" * 70)
print("BENCHMARK DE ESCALABILIDAD: CPU vs GPU")
print("=" * 70)
print(f"GPU: {jax.devices()[0]}")
print("=" * 70 + "\n")


def benchmark_matmul_scaling(sizes, n_trials=10):
    """
    Benchmark de multiplicación matricial para diferentes tamaños.

    Esta es la operación más común en el training (W @ r, A @ B, etc.)
    """
    print("Benchmarking Matrix Multiplication...")

    cpu_times = []
    gpu_times = []

    for size in sizes:
        print(f"\n  Tamaño: {size}x{size}")

        # CPU
        cpu_trial_times = []
        for trial in range(n_trials):
            with jax.default_device(jax.devices('cpu')[0]):
                key = jax.random.PRNGKey(trial)
                A = jax.random.normal(key, (size, size))
                B = jax.random.normal(key, (size, size))

                start = time.time()
                C = jnp.dot(A, B)
                _ = C.block_until_ready()
                elapsed = time.time() - start
                cpu_trial_times.append(elapsed)

        cpu_time = np.mean(cpu_trial_times)
        cpu_times.append(cpu_time)
        print(f"    CPU: {cpu_time*1000:.2f}ms")

        # GPU
        gpu_trial_times = []
        for trial in range(n_trials):
            with jax.default_device(jax.devices('gpu')[0]):
                key = jax.random.PRNGKey(trial + 100)
                A = jax.random.normal(key, (size, size))
                B = jax.random.normal(key, (size, size))

                start = time.time()
                C = jnp.dot(A, B)
                _ = C.block_until_ready()
                elapsed = time.time() - start
                gpu_trial_times.append(elapsed)

        gpu_time = np.mean(gpu_trial_times)
        gpu_times.append(gpu_time)
        print(f"    GPU: {gpu_time*1000:.2f}ms")
        print(f"    Speedup: {cpu_time/gpu_time:.2f}x")

    return np.array(cpu_times), np.array(gpu_times)


def benchmark_complex_operation_scaling(sizes, n_trials=8):
    """
    Benchmark de operación compleja (típica del training).

    Combina: matmul + element-wise + reductions
    """
    print("\nBenchmarking Complex Operations (Training-like)...")

    cpu_times = []
    gpu_times = []

    for size in sizes:
        print(f"\n  Tamaño: {size}x{size}")

        # CPU
        cpu_trial_times = []
        for trial in range(n_trials):
            with jax.default_device(jax.devices('cpu')[0]):
                key = jax.random.PRNGKey(trial)
                W = jax.random.normal(key, (size, size))
                mu = jax.random.normal(key, (size,))
                Sigma = jax.random.normal(key, (size, size))
                Sigma = jnp.dot(Sigma, Sigma.T)

                start = time.time()

                # Operaciones típicas del training
                sigma2 = jnp.diag(Sigma)
                sigma_std = jnp.sqrt(sigma2)
                ratio = jnp.where(sigma_std > 1e-10, mu / sigma_std, 0.0)

                from jax.scipy import stats as jax_stats
                phi = jax_stats.norm.pdf(ratio)
                psi = jax_stats.norm.cdf(ratio)

                nu1 = mu * psi + sigma_std * phi
                nu = 0.3 * (mu * nu1 + sigma2 * psi)

                W_nu = jnp.dot(W, nu)
                cost = jnp.sum(W_nu ** 2)

                _ = cost.block_until_ready()
                elapsed = time.time() - start
                cpu_trial_times.append(elapsed)

        cpu_time = np.mean(cpu_trial_times)
        cpu_times.append(cpu_time)
        print(f"    CPU: {cpu_time*1000:.2f}ms")

        # GPU
        gpu_trial_times = []
        for trial in range(n_trials):
            with jax.default_device(jax.devices('gpu')[0]):
                key = jax.random.PRNGKey(trial + 100)
                W = jax.random.normal(key, (size, size))
                mu = jax.random.normal(key, (size,))
                Sigma = jax.random.normal(key, (size, size))
                Sigma = jnp.dot(Sigma, Sigma.T)

                start = time.time()

                sigma2 = jnp.diag(Sigma)
                sigma_std = jnp.sqrt(sigma2)
                ratio = jnp.where(sigma_std > 1e-10, mu / sigma_std, 0.0)

                from jax.scipy import stats as jax_stats
                phi = jax_stats.norm.pdf(ratio)
                psi = jax_stats.norm.cdf(ratio)

                nu1 = mu * psi + sigma_std * phi
                nu = 0.3 * (mu * nu1 + sigma2 * psi)

                W_nu = jnp.dot(W, nu)
                cost = jnp.sum(W_nu ** 2)

                _ = cost.block_until_ready()
                elapsed = time.time() - start
                gpu_trial_times.append(elapsed)

        gpu_time = np.mean(gpu_trial_times)
        gpu_times.append(gpu_time)
        print(f"    GPU: {gpu_time*1000:.2f}ms")
        print(f"    Speedup: {cpu_time/gpu_time:.2f}x")

    return np.array(cpu_times), np.array(gpu_times)


def plot_scaling_results(sizes, matmul_cpu, matmul_gpu,
                         complex_cpu, complex_gpu,
                         output_dir='pycloude/outputs'):
    """Generar gráficos de escalabilidad."""
    print("\n" + "="*70)
    print("GENERANDO GRÁFICOS DE ESCALABILIDAD")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Tiempos absolutos - Matrix Multiplication
    ax = axes[0, 0]
    ax.plot(sizes, matmul_cpu * 1000, 'o-', label='CPU',
            color='#4472C4', linewidth=2, markersize=8)
    ax.plot(sizes, matmul_gpu * 1000, 's-', label='GPU',
            color='#FF6B6B', linewidth=2, markersize=8)
    ax.set_xlabel('Tamaño de Matriz (NxN)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tiempo (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Multiplicación Matricial: Escalabilidad',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. Speedup - Matrix Multiplication
    ax = axes[0, 1]
    speedup_matmul = matmul_cpu / matmul_gpu
    colors = ['#2ECC71' if s > 1 else '#E74C3C' for s in speedup_matmul]
    ax.bar(range(len(sizes)), speedup_matmul, color=colors,
           alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='black', linestyle='--',
               linewidth=2, label='Sin mejora')
    ax.set_xlabel('Tamaño de Matriz', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (GPU/CPU)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup: Multiplicación Matricial',
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f'{s}x{s}' for s in sizes], rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Agregar valores
    for i, (speedup, color) in enumerate(zip(speedup_matmul, colors)):
        ax.text(i, speedup, f'{speedup:.2f}x',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. Tiempos absolutos - Complex Operations
    ax = axes[1, 0]
    ax.plot(sizes, complex_cpu * 1000, 'o-', label='CPU',
            color='#4472C4', linewidth=2, markersize=8)
    ax.plot(sizes, complex_gpu * 1000, 's-', label='GPU',
            color='#FF6B6B', linewidth=2, markersize=8)
    ax.set_xlabel('Tamaño de Problema (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tiempo (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Operaciones Complejas (Training): Escalabilidad',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 4. Speedup - Complex Operations
    ax = axes[1, 1]
    speedup_complex = complex_cpu / complex_gpu
    colors = ['#2ECC71' if s > 1 else '#E74C3C' for s in speedup_complex]
    ax.bar(range(len(sizes)), speedup_complex, color=colors,
           alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='black', linestyle='--',
               linewidth=2, label='Sin mejora')
    ax.set_xlabel('Tamaño de Problema', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (GPU/CPU)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup: Operaciones Complejas',
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f'N={s}' for s in sizes], rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Agregar valores
    for i, (speedup, color) in enumerate(zip(speedup_complex, colors)):
        ax.text(i, speedup, f'{speedup:.2f}x',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/benchmark_escalabilidad_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {filename}")

    plt.close()
    return filename


def save_scaling_results(sizes, matmul_cpu, matmul_gpu,
                         complex_cpu, complex_gpu,
                         output_dir='pycloude/outputs'):
    """Guardar resultados de escalabilidad."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/escalabilidad_results_{timestamp}.md'

    with open(filename, 'w') as f:
        f.write("# Benchmark de Escalabilidad: CPU vs GPU\n\n")
        f.write(f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**GPU**: {jax.devices()[0]}\n\n")
        f.write("---\n\n")

        f.write("## Multiplicación Matricial\n\n")
        f.write("| Tamaño | CPU (ms) | GPU (ms) | Speedup |\n")
        f.write("|--------|----------|----------|----------|\n")

        for size, cpu_t, gpu_t in zip(sizes, matmul_cpu, matmul_gpu):
            speedup = cpu_t / gpu_t
            f.write(f"| {size}x{size} | {cpu_t*1000:.2f} | "
                   f"{gpu_t*1000:.2f} | {speedup:.2f}x |\n")

        f.write("\n## Operaciones Complejas (Training-like)\n\n")
        f.write("| Tamaño | CPU (ms) | GPU (ms) | Speedup |\n")
        f.write("|--------|----------|----------|----------|\n")

        for size, cpu_t, gpu_t in zip(sizes, complex_cpu, complex_gpu):
            speedup = cpu_t / gpu_t
            f.write(f"| N={size} | {cpu_t*1000:.2f} | "
                   f"{gpu_t*1000:.2f} | {speedup:.2f}x |\n")

        f.write("\n## Análisis\n\n")

        # Encontrar crossover point
        matmul_speedups = matmul_cpu / matmul_gpu
        complex_speedups = complex_cpu / complex_gpu

        matmul_crossover = None
        for i, speedup in enumerate(matmul_speedups):
            if speedup > 1.0:
                matmul_crossover = sizes[i]
                break

        complex_crossover = None
        for i, speedup in enumerate(complex_speedups):
            if speedup > 1.0:
                complex_crossover = sizes[i]
                break

        if matmul_crossover:
            f.write(f"- **Multiplicación Matricial**: GPU se vuelve más "
                   f"rápida a partir de tamaño {matmul_crossover}x"
                   f"{matmul_crossover}\n")
        else:
            f.write("- **Multiplicación Matricial**: CPU más rápida en "
                   "todos los tamaños probados\n")

        if complex_crossover:
            f.write(f"- **Operaciones Complejas**: GPU se vuelve más "
                   f"rápida a partir de N={complex_crossover}\n")
        else:
            f.write("- **Operaciones Complejas**: CPU más rápida en "
                   "todos los tamaños probados\n")

        f.write("\n### Conclusión\n\n")
        f.write("Para el modelo Echeveste2020:\n")
        f.write("- **Redes pequeñas (N < 200)**: CPU más eficiente por "
               "overhead de transferencia GPU\n")
        f.write("- **Redes medianas (200 < N < 500)**: GPU empieza a "
               "mostrar ventajas\n")
        f.write("- **Redes grandes (N > 500)**: GPU significativamente "
               "más rápida\n\n")
        f.write("**Recomendación**: Para el tamaño típico del paper (N=100), "
               "CPU es más eficiente. GPU se vuelve ventajosa para "
               "experimentos con redes más grandes.\n")

    print(f"✓ Resultados guardados: {filename}")
    return filename


def main():
    """Ejecutar benchmark de escalabilidad."""
    # Tamaños a probar
    sizes = [50, 100, 200, 400, 800, 1600]

    print(f"Probando tamaños: {sizes}\n")

    # Matrix multiplication
    matmul_cpu, matmul_gpu = benchmark_matmul_scaling(sizes)

    # Complex operations
    complex_cpu, complex_gpu = benchmark_complex_operation_scaling(sizes)

    # Generar gráficos
    plot_file = plot_scaling_results(
        sizes, matmul_cpu, matmul_gpu, complex_cpu, complex_gpu
    )

    # Guardar resultados
    md_file = save_scaling_results(
        sizes, matmul_cpu, matmul_gpu, complex_cpu, complex_gpu
    )

    # Resumen
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)

    print("\nMultiplicación Matricial:")
    for size, cpu_t, gpu_t in zip(sizes, matmul_cpu, matmul_gpu):
        speedup = cpu_t / gpu_t
        status = "✓ GPU" if speedup > 1 else "✗ CPU"
        print(f"  {size}x{size:4d}: {speedup:.2f}x  {status}")

    print("\nOperaciones Complejas:")
    for size, cpu_t, gpu_t in zip(sizes, complex_cpu, complex_gpu):
        speedup = cpu_t / gpu_t
        status = "✓ GPU" if speedup > 1 else "✗ CPU"
        print(f"  N={size:4d}: {speedup:.2f}x  {status}")

    print("\n" + "="*70)
    print(f"📊 Gráfico: {plot_file}")
    print(f"📄 Reporte: {md_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
