#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualizaciones completas del modelo SSN Echeveste2020.

Este script genera visualizaciones profesionales incluyendo:
1. Heatmaps de actividad neuronal (100 neuronas × tiempo)
2. Comparación de neuronas espaciadas uniformemente
3. Análisis de estructura espacial (tuning curves)
4. Dinámica temporal de poblaciones
5. Comparación entre diferentes contrastes

Referencias:
- echepaper.pdf: Figuras de ejemplo del paper
- Diseñado para mostrar la estructura de ring topology y selectividad espacial
"""

import sys
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from skneuromsi.neural import Echeveste2020  # noqa: E402


def plot_heatmap_activity(result, population='excitatory', output_file=None):
    """
    Gráfico de calor: Actividad de TODAS las neuronas vs tiempo.

    Parameters
    ----------
    result : NDResult
        Resultado de simulación
    population : str
        'excitatory' o 'inhibitory'
    output_file : str
        Ruta para guardar el gráfico
    """
    df = result.get_modes()
    n_times = len(result.times_)

    if population == 'excitatory':
        u = df['excitatory_potential'].values.reshape(n_times, -1)
        r = df['excitatory_firing_rate'].values.reshape(n_times, -1)
        pop_name = 'Excitatory'
        color_u = 'RdBu_r'
        color_r = 'viridis'
    else:
        u = df['inhibitory_potential'].values.reshape(n_times, -1)
        r = df['inhibitory_firing_rate'].values.reshape(n_times, -1)
        pop_name = 'Inhibitory'
        color_u = 'RdBu_r'
        color_r = 'plasma'

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8],
                  hspace=0.3, wspace=0.3)

    # Panel 1: Heatmap de potenciales de membrana
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(u.T, aspect='auto', cmap=color_u,
                     extent=[result.times_[0], result.times_[-1],
                             0, u.shape[1]],
                     origin='lower', interpolation='nearest')
    ax1.set_ylabel('Neuron ID')
    ax1.set_title(f'{pop_name} Population - Membrane Potentials u(t)\n'
                  f'Heatmap: {u.shape[1]} neurons × time')
    plt.colorbar(im1, ax=ax1, label='u (a.u.)')
    ax1.set_xlabel('Time (ms)')

    # Panel 2: Heatmap de firing rates
    ax2 = fig.add_subplot(gs[1, :])
    im2 = ax2.imshow(r.T, aspect='auto', cmap=color_r,
                     extent=[result.times_[0], result.times_[-1],
                             0, r.shape[1]],
                     origin='lower', interpolation='nearest')
    ax2.set_ylabel('Neuron ID')
    ax2.set_title(f'{pop_name} Population - Firing Rates r(t)\n'
                  f'Heatmap: {r.shape[1]} neurons × time')
    plt.colorbar(im2, ax=ax2, label='r (Hz)')
    ax2.set_xlabel('Time (ms)')

    # Panel 3: Promedio temporal por neurona (tuning curve espacial)
    ax3 = fig.add_subplot(gs[2, 0])
    r_mean_time = np.mean(r, axis=0)
    neuron_ids = np.arange(len(r_mean_time))
    ax3.bar(neuron_ids, r_mean_time, alpha=0.7, edgecolor='black', lw=0.5)
    ax3.set_xlabel('Neuron ID')
    ax3.set_ylabel('Mean firing rate (Hz)')
    ax3.set_title('Spatial tuning: Average activity per neuron')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Promedio poblacional en el tiempo
    ax4 = fig.add_subplot(gs[2, 1])
    r_mean_pop = np.mean(r, axis=1)
    ax4.plot(result.times_, r_mean_pop, lw=2, color='darkblue')
    ax4.fill_between(result.times_,
                     r_mean_pop - np.std(r, axis=1),
                     r_mean_pop + np.std(r, axis=1),
                     alpha=0.3)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Population mean ± std (Hz)')
    ax4.set_title('Population dynamics over time')
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'{pop_name} Population - Complete Activity Profile',
                 fontsize=16, y=0.995)

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Guardado: {output_file}")

    return fig


def plot_spatial_sampling(result, n_samples=10, output_file=None):
    """
    Comparación de neuronas muestreadas espacialmente.

    Selecciona n_samples neuronas espaciadas uniformemente a lo largo
    del ring para mostrar diversidad de respuestas.
    """
    df = result.get_modes()
    n_times = len(result.times_)

    r_e = df['excitatory_firing_rate'].values.reshape(n_times, -1)
    r_i = df['inhibitory_firing_rate'].values.reshape(n_times, -1)
    u_e = df['excitatory_potential'].values.reshape(n_times, -1)
    u_i = df['inhibitory_potential'].values.reshape(n_times, -1)

    # Seleccionar neuronas espaciadas uniformemente
    n_e = r_e.shape[1]
    n_i = r_i.shape[1]
    sample_indices_e = np.linspace(0, n_e-1, n_samples, dtype=int)
    sample_indices_i = np.linspace(0, n_i-1, n_samples, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: Potenciales de membrana - Excitatorias
    ax1 = axes[0, 0]
    for idx in sample_indices_e:
        orientation = idx * 180.0 / n_e  # Asumiendo ring de 0-180°
        ax1.plot(result.times_, u_e[:, idx], alpha=0.7,
                 label=f'E{idx} (θ={orientation:.1f}°)')
    ax1.set_ylabel('Membrane potential u (a.u.)')
    ax1.set_title(f'Excitatory neurons - Sampled spatially '
                  f'({n_samples}/{n_e} neurons)')
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Firing rates - Excitatorias
    ax2 = axes[0, 1]
    for idx in sample_indices_e:
        orientation = idx * 180.0 / n_e
        ax2.plot(result.times_, r_e[:, idx], alpha=0.7,
                 label=f'E{idx} (θ={orientation:.1f}°)')
    ax2.set_ylabel('Firing rate (Hz)')
    ax2.set_title('Excitatory neurons - Firing rates')
    ax2.legend(loc='best', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Potenciales de membrana - Inhibitorias
    ax3 = axes[1, 0]
    for idx in sample_indices_i:
        orientation = idx * 180.0 / n_i
        ax3.plot(result.times_, u_i[:, idx], alpha=0.7,
                 label=f'I{idx} (θ={orientation:.1f}°)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Membrane potential u (a.u.)')
    ax3.set_title(f'Inhibitory neurons - Sampled spatially '
                  f'({n_samples}/{n_i} neurons)')
    ax3.legend(loc='best', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Firing rates - Inhibitorias
    ax4 = axes[1, 1]
    for idx in sample_indices_i:
        orientation = idx * 180.0 / n_i
        ax4.plot(result.times_, r_i[:, idx], alpha=0.7,
                 label=f'I{idx} (θ={orientation:.1f}°)')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Firing rate (Hz)')
    ax4.set_title('Inhibitory neurons - Firing rates')
    ax4.legend(loc='best', fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Spatial Sampling: Neurons distributed across ring topology',
                 fontsize=14, y=0.995)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Guardado: {output_file}")

    return fig


def plot_contrast_comparison(ssn, contrasts=None, sim_time=200.0,
                             output_file=None):
    """
    Comparación de actividad para diferentes niveles de contraste.

    Genera heatmaps lado a lado mostrando cómo cambia la actividad
    con el contraste del estímulo.
    """
    if contrasts is None:
        contrasts = [0.1, 0.3, 0.5]

    print(f"\n   Ejecutando {len(contrasts)} simulaciones...")

    results = []
    for c in contrasts:
        print(f"   - Contraste {c}...")
        result = ssn.run(
            stimulus_contrast=c,
            stimulus_orientation=0.0,
            noise_level=0.1,
            simulation_time=sim_time,
        )
        results.append(result)

    # Extraer datos
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, len(contrasts), figure=fig, hspace=0.35, wspace=0.35)

    for col, (result, c) in enumerate(zip(results, contrasts)):
        df = result.get_modes()
        n_times = len(result.times_)
        r_e = df['excitatory_firing_rate'].values.reshape(n_times, -1)
        r_i = df['inhibitory_firing_rate'].values.reshape(n_times, -1)

        # Fila 1: Heatmap excitatorias
        ax1 = fig.add_subplot(gs[0, col])
        im1 = ax1.imshow(r_e.T, aspect='auto', cmap='viridis',
                         extent=[result.times_[0], result.times_[-1],
                                 0, r_e.shape[1]],
                         origin='lower', interpolation='nearest')
        ax1.set_title(f'Contrast = {c}')
        if col == 0:
            ax1.set_ylabel('Excitatory\nNeuron ID')
        plt.colorbar(im1, ax=ax1, label='r (Hz)' if col == len(contrasts)-1
                     else '')

        # Fila 2: Heatmap inhibitorias
        ax2 = fig.add_subplot(gs[1, col])
        im2 = ax2.imshow(r_i.T, aspect='auto', cmap='plasma',
                         extent=[result.times_[0], result.times_[-1],
                                 0, r_i.shape[1]],
                         origin='lower', interpolation='nearest')
        if col == 0:
            ax2.set_ylabel('Inhibitory\nNeuron ID')
        plt.colorbar(im2, ax=ax2, label='r (Hz)' if col == len(contrasts)-1
                     else '')

        # Fila 3: Media poblacional
        ax3 = fig.add_subplot(gs[2, col])
        r_e_mean = np.mean(r_e, axis=1)
        r_i_mean = np.mean(r_i, axis=1)
        ax3.plot(result.times_, r_e_mean, label='Excitatory', lw=2)
        ax3.plot(result.times_, r_i_mean, label='Inhibitory', lw=2)
        ax3.set_xlabel('Time (ms)')
        if col == 0:
            ax3.set_ylabel('Mean rate (Hz)')
            ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

    fig.suptitle('Contrast Comparison: Network Activity Across Stimulus '
                 'Intensities',
                 fontsize=16, y=0.995)

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Guardado: {output_file}")

    return fig


def main():
    """Función principal."""
    print("=" * 70)
    print("VISUALIZACIONES COMPLETAS - SSN Echeveste2020")
    print("=" * 70)

    output_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs"

    # Crear modelo
    print("\n1. Creando modelo SSN...")
    ssn = Echeveste2020(N_E=50, N_I=50, seed=42)

    print("2. Cargando parámetros entrenados...")
    ssn.load_parameters()

    # Ejecutar simulación base
    print("\n3. Ejecutando simulación base (contraste 0.5)...")
    result = ssn.run(
        stimulus_contrast=0.5,
        stimulus_orientation=0.0,
        noise_level=0.1,
        simulation_time=200.0,
    )

    # Visualización 1: Heatmaps completos
    print("\n4. Generando heatmaps de actividad completa...")
    print("   4a. Población excitatoria...")
    plot_heatmap_activity(
        result,
        population='excitatory',
        output_file=f'{output_dir}/heatmap_excitatory.png'
    )
    plt.close()

    print("   4b. Población inhibitoria...")
    plot_heatmap_activity(
        result,
        population='inhibitory',
        output_file=f'{output_dir}/heatmap_inhibitory.png'
    )
    plt.close()

    # Visualización 2: Muestreo espacial
    print("\n5. Generando comparación de neuronas espaciadas...")
    plot_spatial_sampling(
        result,
        n_samples=10,
        output_file=f'{output_dir}/spatial_sampling.png'
    )
    plt.close()

    # Visualización 3: Comparación de contrastes
    print("\n6. Generando comparación de contrastes...")
    plot_contrast_comparison(
        ssn,
        contrasts=[0.1, 0.3, 0.5],
        sim_time=200.0,
        output_file=f'{output_dir}/contrast_comparison.png'
    )
    plt.close()

    print("\n" + "=" * 70)
    print("✓ TODAS LAS VISUALIZACIONES GENERADAS")
    print(f"✓ Archivos guardados en: {output_dir}/")
    print("\nArchivos generados:")
    print("  - heatmap_excitatory.png: Heatmap completo población E")
    print("  - heatmap_inhibitory.png: Heatmap completo población I")
    print("  - spatial_sampling.png: Neuronas espaciadas uniformemente")
    print("  - contrast_comparison.png: Comparación entre contrastes")
    print("=" * 70)


if __name__ == "__main__":
    main()
