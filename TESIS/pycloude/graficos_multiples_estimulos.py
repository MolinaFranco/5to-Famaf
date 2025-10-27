#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gráficos de dinámicas neuronales para múltiples niveles de estímulo.

Este script reproduce los diferentes niveles de contraste del paper de
Echeveste et al. (2020), generando gráficos comparativos de:
- Potenciales de membrana u(t)
- Firing rates r(t)
- Actividad poblacional

Basado en los 5 patrones del código original:
- Pattern 0: contrast=0.019  (baseline, muy bajo)
- Pattern 1: contrast=0.1    (bajo)
- Pattern 2: contrast=0.3    (medio)
- Pattern 3: contrast=0.5    (alto)
- Pattern 4: contrast=0.8    (muy alto)

Referencias:
- echepaper.pdf: Figura 2 (respuestas a diferentes contrastes)
- activity_example.py: Código original con N_pat=5 patrones
"""

import sys
import os

# Agregar scikit-neuromsi al path
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

import matplotlib  # noqa: E402
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from skneuromsi.neural import Echeveste2020  # noqa: E402
from skneuromsi.neural._echeveste_visualization import (  # noqa: E402
    plot_membrane_potentials,
    plot_firing_rates,
)


def run_simulation_for_contrast(ssn, contrast_level, pattern_name,
                                simulation_time=500.0):
    """
    Ejecuta simulación para un nivel de contraste específico.

    Parameters
    ----------
    ssn : Echeveste2020
        Modelo SSN entrenado
    contrast_level : float
        Nivel de contraste (0-1)
    pattern_name : str
        Nombre descriptivo del patrón
    simulation_time : float
        Tiempo de simulación en ms

    Returns
    -------
    result : NDResult
        Resultado de la simulación
    """
    print(f"\n{'='*70}")
    print(f"Simulando patrón: {pattern_name} (contrast={contrast_level})")
    print(f"{'='*70}")

    result = ssn.run(
        stimulus_contrast=contrast_level,
        stimulus_orientation=0.0,
        noise_level=0.1,
        simulation_time=simulation_time,
    )

    # Extraer datos para análisis
    df = result.get_modes()
    n_times = len(result.times_)

    u_e = df['excitatory_potential'].values.reshape(n_times, -1)
    r_e = df['excitatory_firing_rate'].values.reshape(n_times, -1)
    u_i = df['inhibitory_potential'].values.reshape(n_times, -1)
    r_i = df['inhibitory_firing_rate'].values.reshape(n_times, -1)

    # Estadísticas
    print("\nEstadísticas excitatorias (primeras 5 neuronas):")
    print("  Potencial u(t):")
    print(f"    Min: {u_e[:, :5].min():.4f}, Max: {u_e[:, :5].max():.4f}, "
          f"Mean: {u_e[:, :5].mean():.4f}, Std: {u_e[:, :5].std():.4f}")
    print("  Firing rate r(t):")
    print(f"    Min: {r_e[:, :5].min():.4f}, Max: {r_e[:, :5].max():.4f}, "
          f"Mean: {r_e[:, :5].mean():.4f}, Std: {r_e[:, :5].std():.4f}")

    print("\nEstadísticas inhibitorias (primeras 5 neuronas):")
    print("  Potencial u(t):")
    print(f"    Min: {u_i[:, :5].min():.4f}, Max: {u_i[:, :5].max():.4f}, "
          f"Mean: {u_i[:, :5].mean():.4f}, Std: {u_i[:, :5].std():.4f}")
    print("  Firing rate r(t):")
    print(f"    Min: {r_i[:, :5].min():.4f}, Max: {r_i[:, :5].max():.4f}, "
          f"Mean: {r_i[:, :5].mean():.4f}, Std: {r_i[:, :5].std():.4f}")

    return result


def create_comparison_plots(results_dict, output_dir):
    """
    Crea gráficos comparativos para todos los niveles de contraste.

    Parameters
    ----------
    results_dict : dict
        Diccionario con resultados de simulaciones
        {pattern_name: result}
    output_dir : str
        Directorio de salida
    """
    print(f"\n{'='*70}")
    print("Generando gráficos comparativos")
    print(f"{'='*70}")

    n_patterns = len(results_dict)

    # Gráfico 1: Comparación de potenciales de membrana (excitatorias)
    print("\n1. Potenciales de membrana - comparación entre contrastes...")
    fig1, axes1 = plt.subplots(n_patterns, 1, figsize=(14, 3*n_patterns),
                               sharex=True)
    if n_patterns == 1:
        axes1 = [axes1]

    for idx, (pattern_name, result) in enumerate(results_dict.items()):
        df = result.get_modes()
        n_times = len(result.times_)
        u_e = df['excitatory_potential'].values.reshape(n_times, -1)

        # Graficar primeras 5 neuronas excitatorias
        for neuron_idx in range(5):
            axes1[idx].plot(result.times_, u_e[:, neuron_idx],
                            alpha=0.7, label=f'E{neuron_idx}')

        axes1[idx].set_ylabel('u (a.u.)')
        axes1[idx].set_title(f'{pattern_name} - Membrane potentials')
        axes1[idx].legend(loc='upper right', fontsize=8, ncol=5)
        axes1[idx].grid(True, alpha=0.3)

    axes1[-1].set_xlabel('Time (ms)')
    fig1.suptitle('Membrane Potentials u(t) - Excitatory Neurons\n'
                  'Comparison across contrast levels',
                  fontsize=14, y=0.995)
    plt.tight_layout()
    fig1.savefig(
        f'{output_dir}/comparison_membrane_potentials.png',
        dpi=150, bbox_inches='tight'
    )
    print("   ✓ Guardado: comparison_membrane_potentials.png")

    # Gráfico 2: Comparación de firing rates (excitatorias)
    print("2. Firing rates - comparación entre contrastes...")
    fig2, axes2 = plt.subplots(n_patterns, 1, figsize=(14, 3*n_patterns),
                               sharex=True)
    if n_patterns == 1:
        axes2 = [axes2]

    for idx, (pattern_name, result) in enumerate(results_dict.items()):
        df = result.get_modes()
        n_times = len(result.times_)
        r_e = df['excitatory_firing_rate'].values.reshape(n_times, -1)

        # Graficar primeras 5 neuronas excitatorias
        for neuron_idx in range(5):
            axes2[idx].plot(result.times_, r_e[:, neuron_idx],
                            alpha=0.7, label=f'E{neuron_idx}')

        axes2[idx].set_ylabel('r (Hz)')
        axes2[idx].set_title(f'{pattern_name} - Firing rates')
        axes2[idx].legend(loc='upper right', fontsize=8, ncol=5)
        axes2[idx].grid(True, alpha=0.3)

    axes2[-1].set_xlabel('Time (ms)')
    fig2.suptitle('Firing Rates r(t) - Excitatory Neurons\n'
                  'Comparison across contrast levels',
                  fontsize=14, y=0.995)
    plt.tight_layout()
    fig2.savefig(
        f'{output_dir}/comparison_firing_rates.png',
        dpi=150, bbox_inches='tight'
    )
    print("   ✓ Guardado: comparison_firing_rates.png")

    # Gráfico 3: Media poblacional vs contraste
    print("3. Media poblacional - comparación entre contrastes...")
    fig3, (ax_u, ax_r) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for pattern_name, result in results_dict.items():
        df = result.get_modes()
        n_times = len(result.times_)
        u_e = df['excitatory_potential'].values.reshape(n_times, -1)
        r_e = df['excitatory_firing_rate'].values.reshape(n_times, -1)

        # Media poblacional
        u_mean = np.mean(u_e, axis=1)
        r_mean = np.mean(r_e, axis=1)

        ax_u.plot(result.times_, u_mean, label=pattern_name, alpha=0.8, lw=2)
        ax_r.plot(result.times_, r_mean, label=pattern_name, alpha=0.8, lw=2)

    ax_u.set_ylabel('Mean membrane potential u (a.u.)')
    ax_u.set_title('Population mean - Membrane potentials')
    ax_u.legend(loc='best')
    ax_u.grid(True, alpha=0.3)

    ax_r.set_xlabel('Time (ms)')
    ax_r.set_ylabel('Mean firing rate r (Hz)')
    ax_r.set_title('Population mean - Firing rates')
    ax_r.legend(loc='best')
    ax_r.grid(True, alpha=0.3)

    fig3.suptitle('Population Mean Activity - Comparison across contrasts',
                  fontsize=14, y=0.995)
    plt.tight_layout()
    fig3.savefig(
        f'{output_dir}/comparison_population_mean.png',
        dpi=150, bbox_inches='tight'
    )
    print("   ✓ Guardado: comparison_population_mean.png")

    plt.close('all')


def main():
    """Función principal."""
    print("=" * 70)
    print("GRÁFICOS DE MÚLTIPLES NIVELES DE ESTÍMULO")
    print("Echeveste et al. (2020) - SSN Model")
    print("=" * 70)

    # Directorio de salida
    output_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Crear modelo
    print("\n1. Creando modelo SSN...")
    ssn = Echeveste2020(N_E=50, N_I=50, seed=42)

    # Cargar parámetros
    print("2. Cargando parámetros entrenados...")
    ssn.load_parameters()
    print("   ✓ Modelo entrenado cargado")

    # Definir niveles de contraste basados en el código original
    # Pattern 0: h_true_0 = 0.019 (uniforme, baseline)
    # Pattern 1: h_true_1 ≈ 0.1 (bajo, orientado)
    # Pattern 2: h_true_2 ≈ 0.3 (medio, orientado)
    # Pattern 3: h_true_3 ≈ 0.5 (alto, orientado)
    # Pattern 4: h_true_4 ≈ 0.8 (muy alto, orientado)

    contrast_levels = [
        (0.019, "Pattern 0: Baseline (c=0.019)"),
        (0.1, "Pattern 1: Low contrast (c=0.1)"),
        (0.3, "Pattern 2: Medium contrast (c=0.3)"),
        (0.5, "Pattern 3: High contrast (c=0.5)"),
        (0.8, "Pattern 4: Very high contrast (c=0.8)"),
    ]

    # Ejecutar simulaciones para cada nivel de contraste
    print("\n3. Ejecutando simulaciones para cada nivel de contraste...")
    results = {}

    for contrast, pattern_name in contrast_levels:
        try:
            result = run_simulation_for_contrast(
                ssn,
                contrast_level=contrast,
                pattern_name=pattern_name,
                simulation_time=500.0  # 500 ms para visualización rápida
            )
            results[pattern_name] = result
        except Exception as e:
            print(f"\n✗ Error en {pattern_name}: {e}")
            print("   Continuando con siguiente patrón...")
            continue

    # Generar gráficos individuales para cada patrón
    print("\n4. Generando gráficos individuales...")
    for pattern_name, result in results.items():
        safe_name = pattern_name.replace(" ", "_").replace(":", "").\
            replace("(", "").replace(")", "").replace("=", "")

        # Potenciales de membrana
        try:
            fig_u, _ = plot_membrane_potentials(
                result,
                neuron_indices=[0, 1, 2, 3, 4],
                population="both",
                title=f"{pattern_name}\nMembrane potentials u(t)"
            )
            fig_u.savefig(f'{output_dir}/{safe_name}_potentials.png',
                          dpi=150, bbox_inches='tight')
            print(f"   ✓ Guardado: {safe_name}_potentials.png")
            plt.close(fig_u)
        except Exception as e:
            print(f"   ✗ Error graficando potenciales: {e}")

        # Firing rates
        try:
            fig_r, _ = plot_firing_rates(
                result,
                neuron_indices=[0, 1, 2, 3, 4],
                population="both",
                title=f"{pattern_name}\nFiring rates r(t)"
            )
            fig_r.savefig(f'{output_dir}/{safe_name}_firing_rates.png',
                          dpi=150, bbox_inches='tight')
            print(f"   ✓ Guardado: {safe_name}_firing_rates.png")
            plt.close(fig_r)
        except Exception as e:
            print(f"   ✗ Error graficando firing rates: {e}")

    # Generar gráficos comparativos
    print("\n5. Generando gráficos comparativos...")
    if len(results) > 0:
        create_comparison_plots(results, output_dir)
    else:
        print("   ✗ No hay resultados para comparar")

    print("\n" + "=" * 70)
    print("✓ PROCESO COMPLETADO")
    print(f"✓ Gráficos guardados en: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
