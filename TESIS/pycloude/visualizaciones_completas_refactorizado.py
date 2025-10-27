#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualizaciones completas del modelo SSN Echeveste2020.

Este script demuestra cómo usar las funciones de visualización
del módulo skneuromsi.neural._echeveste_visualization para
crear gráficos profesionales.

IMPORTANTE: Este script ahora usa las funciones oficiales del módulo,
no define funciones propias.
"""

import sys
import os
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from skneuromsi.neural import Echeveste2020  # noqa: E402
from skneuromsi.neural._echeveste_visualization import (  # noqa: E402
    plot_heatmap_activity,
    plot_spatial_sampling,
)


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
    print("Usando funciones del módulo skneuromsi.neural")
    print("=" * 70)

    output_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs"
    os.makedirs(output_dir, exist_ok=True)

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

    # Visualización 1: Heatmaps completos (usando función del módulo)
    print("\n4. Generando heatmaps de actividad completa...")
    print("   4a. Población excitatoria...")
    fig_e, _ = plot_heatmap_activity(
        result,
        population='excitatory'
    )
    fig_e.savefig(f'{output_dir}/heatmap_excitatory.png',
                  dpi=150, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_dir}/heatmap_excitatory.png")
    plt.close(fig_e)

    print("   4b. Población inhibitoria...")
    fig_i, _ = plot_heatmap_activity(
        result,
        population='inhibitory'
    )
    fig_i.savefig(f'{output_dir}/heatmap_inhibitory.png',
                  dpi=150, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_dir}/heatmap_inhibitory.png")
    plt.close(fig_i)

    # Visualización 2: Muestreo espacial (usando función del módulo)
    print("\n5. Generando comparación de neuronas espaciadas...")
    fig_spatial, _ = plot_spatial_sampling(result, n_samples=10)
    fig_spatial.savefig(f'{output_dir}/spatial_sampling.png',
                        dpi=150, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_dir}/spatial_sampling.png")
    plt.close(fig_spatial)

    # Visualización 3: Comparación de contrastes
    print("\n6. Generando comparación de contrastes...")
    fig_contrast = plot_contrast_comparison(
        ssn,
        contrasts=[0.1, 0.3, 0.5],
        sim_time=200.0,
        output_file=f'{output_dir}/contrast_comparison.png'
    )
    plt.close(fig_contrast)

    print("\n" + "=" * 70)
    print("✓ TODAS LAS VISUALIZACIONES GENERADAS")
    print(f"✓ Archivos guardados en: {output_dir}/")
    print("\nArchivos generados:")
    print("  - heatmap_excitatory.png: Heatmap completo población E")
    print("  - heatmap_inhibitory.png: Heatmap completo población I")
    print("  - spatial_sampling.png: Neuronas espaciadas uniformemente")
    print("  - contrast_comparison.png: Comparación entre contrastes")
    print("\nNOTA: Este script usa las funciones oficiales del módulo:")
    print("  - plot_heatmap_activity()")
    print("  - plot_spatial_sampling()")
    print("  Disponibles en skneuromsi.neural._echeveste_visualization")
    print("=" * 70)


if __name__ == "__main__":
    main()
