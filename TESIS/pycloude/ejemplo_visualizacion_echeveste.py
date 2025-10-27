#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de ejemplo para visualización de dinámicas neuronales del modelo
Echeveste et al. (2020).

Este script demuestra cómo usar el módulo de visualización para analizar
los estados neuronales durante la simulación del modelo SSN.

Basado en las figuras del paper:
    - echepaper.pdf: Figuras 2-7
    - Análisis de dinámicas: potenciales de membrana, oscilaciones,
      transitorios
"""

import sys
import os
import matplotlib.pyplot as plt

# Agregar el path de scikit-neuromsi al PYTHONPATH
sys.path.insert(
    0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi"
)
from skneuromsi.neural import Echeveste2020  # noqa: E402
from skneuromsi.neural._echeveste_visualization import (  # noqa: E402
    plot_membrane_potentials,
    plot_firing_rates,
    plot_population_activity,
    plot_mean_firing_rates,
    plot_autocorrelation,
    plot_power_spectrum,
    plot_transient_analysis,
    plot_fano_factor,
    plot_neural_dynamics_summary,
)


def main():
    """
    Ejemplo de uso del módulo de visualización.

    Ejecuta una simulación del modelo SSN y genera múltiples gráficos
    para analizar las dinámicas neuronales.
    """
    print("=" * 70)
    print("Ejemplo de visualización - Modelo Echeveste et al. (2020)")
    print("=" * 70)

    # Paso 1: Crear instancia del modelo
    print("\n1. Creando modelo SSN...")
    ssn = Echeveste2020(
        N_E=50,  # 50 neuronas excitatorias
        N_I=50,  # 50 neuronas inhibitorias
        seed=42,  # semilla para reproducibilidad
    )

    # Paso 2: Cargar parámetros pre-entrenados
    print("2. Cargando parámetros entrenados...")
    try:
        ssn.load_parameters()
        print("   ✓ Parámetros cargados exitosamente")
    except Exception as e:
        print(f"   ✗ Error cargando parámetros: {e}")
        print("   Intentando entrenar modelo...")
        ssn.train()

    # Paso 3: Ejecutar simulación
    print("\n3. Ejecutando simulación...")
    print("   Parámetros:")
    print("   - Contraste del estímulo: 0.019")
    print("   - Orientación: 0.0 radianes")
    print("   - Nivel de ruido: 0.1")
    print("   - Tiempo de simulación: 1000 ms")

    result = ssn.run(
        stimulus_contrast=0.019,  # contraste del estímulo GSM
        stimulus_orientation=0.0,  # orientación en radianes
        noise_level=0.1,  # nivel de ruido η
        simulation_time=1000.0,  # 1 segundo de simulación
    )

    print("   ✓ Simulación completada")
    print(f"   - Forma excitatorias: {result.modes['excitatory'].shape}")
    print(f"   - Forma inhibitorias: {result.modes['inhibitory'].shape}")
    print(f"   - Eje temporal: {len(result.time_axis)} pasos")

    # Paso 4: Generar visualizaciones
    print("\n4. Generando visualizaciones...")

    # 4a. Potenciales de membrana individuales
    print("   a) Potenciales de membrana u(t) (primeras 5 neuronas)...")
    fig1, _ = plot_membrane_potentials(
        result,
        neuron_indices=[0, 1, 2, 3, 4],
        population="both",
        title="Membrane potentials u(t) - First 5 neurons per population",
    )
    output_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs"
    os.makedirs(output_dir, exist_ok=True)
    fig1.savefig(f"{output_dir}/echeveste_membrane_potentials.png", dpi=150)
    print(
        f"      ✓ Guardado en "
        f"{output_dir}/echeveste_membrane_potentials.png"
    )

    # 4b. Firing rates individuales
    print("   b) Firing rates r(t) (primeras 5 neuronas)...")
    fig2, _ = plot_firing_rates(
        result,
        neuron_indices=[0, 1, 2, 3, 4],
        population="both",
        title="Firing rates r(t) - First 5 neurons per population",
    )
    fig2.savefig(f"{output_dir}/echeveste_firing_rates.png", dpi=150)
    print(
        f"      ✓ Guardado en "
        f"{output_dir}/echeveste_firing_rates.png"
    )

    # 4c. Actividad poblacional (raster-like)
    print("   c) Actividad poblacional (raster)...")
    fig3, _ = plot_population_activity(
        result, n_neurons=50, title="Population Activity (all neurons)"
    )
    fig3.savefig(f"{output_dir}/echeveste_population_activity.png", dpi=150)
    print(
        f"      ✓ Guardado en "
        f"{output_dir}/echeveste_population_activity.png"
    )

    # 4d. Firing rates promedio
    print("   d) Firing rates promedio con variabilidad...")
    fig4, _ = plot_mean_firing_rates(
        result, window_size=20, title="Mean Population Firing Rates"
    )
    fig4.savefig(f"{output_dir}/echeveste_mean_firing_rates.png", dpi=150)
    print(f"      ✓ Guardado en {output_dir}/echeveste_mean_firing_rates.png")

    # 4e. Autocorrelación temporal
    print("   e) Autocorrelación temporal (excitatorias)...")
    fig5, _ = plot_autocorrelation(
        result,
        max_lag=200,
        population="excitatory",
        title="Temporal Autocorrelation - Excitatory Population",
    )
    fig5.savefig(f"{output_dir}/echeveste_autocorrelation.png", dpi=150)
    print(f"      ✓ Guardado en {output_dir}/echeveste_autocorrelation.png")

    # 4f. Espectro de potencia (oscilaciones gamma)
    print("   f) Espectro de potencia (0-100 Hz)...")
    fig6, _ = plot_power_spectrum(
        result,
        population="excitatory",
        freq_range=(0, 100),
        title="Power Spectrum - Gamma Oscillations (20-80 Hz)",
    )
    fig6.savefig(f"{output_dir}/echeveste_power_spectrum.png", dpi=150)
    print(f"      ✓ Guardado en {output_dir}/echeveste_power_spectrum.png")

    # 4g. Análisis de transitorios
    print("   g) Análisis de transitorios (overshoots)...")
    fig7, _ = plot_transient_analysis(
        result,
        baseline_window=(50, 150),
        transient_window=(150, 350),
        title="Transient Overshoot Analysis",
    )
    fig7.savefig(f"{output_dir}/echeveste_transients.png", dpi=150)
    print(f"      ✓ Guardado en {output_dir}/echeveste_transients.png")

    # 4h. Fano factor (variabilidad)
    print("   h) Fano factor (varianza/media)...")
    fig8, _ = plot_fano_factor(
        result, window_size=50, title="Fano Factor - Neural Variability"
    )
    fig8.savefig(f"{output_dir}/echeveste_fano_factor.png", dpi=150)
    print(f"      ✓ Guardado en {output_dir}/echeveste_fano_factor.png")

    # 4i. Resumen completo (multi-panel)
    print("   i) Resumen de dinámicas (multi-panel)...")
    fig9, _ = plot_neural_dynamics_summary(result, figsize=(16, 12))
    fig9.savefig(f"{output_dir}/echeveste_dynamics_summary.png", dpi=150)
    print(f"      ✓ Guardado en {output_dir}/echeveste_dynamics_summary.png")

    print("\n" + "=" * 70)
    print("✓ Todas las visualizaciones generadas exitosamente")
    print(f"✓ Archivos guardados en: {output_dir}/")
    print("=" * 70)

    # Mostrar todas las figuras
    plt.show()


if __name__ == "__main__":
    main()
