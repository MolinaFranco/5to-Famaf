#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script para verificar visualizaciones del modelo Echeveste (2020).

Este script usa los parámetros exactos del paper (Table S1) para generar
visualizaciones y compararlas con las figuras del paper original.

Parámetros base (de Table S1 del Supplementary Material):
    - N_E = 50 neuronas excitatorias
    - N_I = 50 neuronas inhibitorias
    - Simulation time = 500 ms
    - Contraste típico: 0.019 (valor medio de los 5 niveles usados)
"""

import sys
import os

# Agregar el path de scikit-neuromsi al PYTHONPATH
sys.path.insert(
    0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi"
)

from skneuromsi.neural import Echeveste2020  # noqa: E402
from skneuromsi.neural._echeveste_visualization import (  # noqa: E402
    plot_membrane_potentials,
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
    Test con parámetros exactos del paper Echeveste et al. (2020).

    Basado en Table S1 del Supplementary Material.
    """
    print("=" * 80)
    print("TEST: Visualización Echeveste et al. (2020) - Parámetros del paper")
    print("=" * 80)

    # Parámetros exactos del paper (Table S1)
    N_E = 50  # Number of E cells
    N_I = 50  # Number of I cells
    simulation_time = 500.0  # Tmax: 500 ms
    stimulus_contrast = 0.019  # Contraste medio de los 5 niveles
    stimulus_orientation = 0.0  # Orientación en radianes
    noise_level = 0.1  # Nivel de ruido estándar

    print("\nParámetros de la simulación (basados en Table S1):")
    print(f"  - N_E (excitatory neurons): {N_E}")
    print(f"  - N_I (inhibitory neurons): {N_I}")
    print("  - τ_E (E time constant): 20 ms (del modelo)")
    print("  - τ_I (I time constant): 10 ms (del modelo)")
    print("  - τ_η (noise time constant): 20 ms (del modelo)")
    print(f"  - Simulation time (Tmax): {simulation_time} ms")
    print(f"  - Stimulus contrast: {stimulus_contrast}")
    print(f"  - Stimulus orientation: {stimulus_orientation} rad")
    print(f"  - Noise level: {noise_level}")
    print("  - dt (time-step): 0.2 ms (del modelo)")

    # Paso 1: Crear instancia del modelo
    print("\n" + "-" * 80)
    print("1. Creando modelo SSN con parámetros del paper...")
    print("-" * 80)

    ssn = Echeveste2020(
        N_E=N_E,
        N_I=N_I,
        seed=42,  # Para reproducibilidad
    )
    print("   ✓ Modelo creado")

    # Paso 2: Cargar parámetros entrenados
    print("\n" + "-" * 80)
    print("2. Cargando parámetros entrenados (W, Σ_η)...")
    print("-" * 80)

    try:
        ssn.load_parameters()
        print("   ✓ Parámetros cargados desde archivos")
    except Exception as e:
        print(f"   ⚠ No se pudieron cargar parámetros: {e}")
        print("   → Esto es esperado si aún no se han entrenado")
        print("   → Las visualizaciones mostrarán dinámicas sin entrenar")

    # Paso 3: Ejecutar simulación
    print("\n" + "-" * 80)
    print("3. Ejecutando simulación...")
    print("-" * 80)

    result = ssn.run(
        stimulus_contrast=stimulus_contrast,
        stimulus_orientation=stimulus_orientation,
        noise_level=noise_level,
        simulation_time=simulation_time,
    )

    print("   ✓ Simulación completada")
    exc_data = result.get_modes()['excitatory'].values.reshape(
        len(result.times_), -1
    )
    inh_data = result.get_modes()['inhibitory'].values.reshape(
        len(result.times_), -1
    )
    print(f"   - Shape excitatory: {exc_data.shape}")
    print(f"   - Shape inhibitory: {inh_data.shape}")
    print(f"   - Time axis length: {len(result.times_)} steps")
    print(
        f"   - Time resolution: "
        f"{result.times_[1] - result.times_[0]:.2f} ms"
    )

    # Paso 4: Crear directorio para gráficos
    print("\n" + "-" * 80)
    print("4. Preparando directorio para gráficos...")
    print("-" * 80)

    graphics_dir = (
        "/home/molina/FAMAF/5to-Famaf/TESIS/"
        "scikit-neuromsi/graphics/echeveste"
    )
    os.makedirs(graphics_dir, exist_ok=True)
    print(f"   ✓ Directorio creado: {graphics_dir}")

    # Paso 5: Generar visualizaciones
    print("\n" + "-" * 80)
    print("5. Generando visualizaciones (comparar con paper)...")
    print("-" * 80)

    # 5a. Membrane potentials (Fig. 2, 4, 5)
    print("\n   a) Potenciales de membrana (primeras 5 neuronas)")
    print("      → Comparar con: Main paper Fig. 2, 4, 5")
    fig1, _ = plot_membrane_potentials(
        result,
        neuron_indices=[0, 1, 2, 3, 4],
        population="both",
        title=(
            "Neural firing rates - Echeveste et al. (2020)\n"
            f"Contrast={stimulus_contrast}, Time={simulation_time}ms"
        ),
    )
    fig1.savefig(f"{graphics_dir}/fig_membrane_potentials.png", dpi=150)
    print(f"      ✓ Guardado: {graphics_dir}/fig_membrane_potentials.png")

    # 5b. Population activity (Fig. 2a)
    print("\n   b) Actividad poblacional (heatmap)")
    print("      → Comparar con: Main paper Fig. 2a")
    fig2, _ = plot_population_activity(
        result,
        n_neurons=50,
        title=(
            "Population Activity - Echeveste et al. (2020)\n"
            f"Contrast={stimulus_contrast}"
        ),
    )
    fig2.savefig(f"{graphics_dir}/fig_population_activity.png", dpi=150)
    print(f"      ✓ Guardado: {graphics_dir}/fig_population_activity.png")

    # 5c. Mean firing rates (Fig. 2b)
    print("\n   c) Firing rates promedio ± std")
    print("      → Comparar con: Main paper Fig. 2b")
    fig3, _ = plot_mean_firing_rates(
        result,
        window_size=20,
        title=(
            "Mean Population Firing Rates - Echeveste et al. (2020)\n"
            f"Contrast={stimulus_contrast}"
        ),
    )
    fig3.savefig(f"{graphics_dir}/fig_mean_firing_rates.png", dpi=150)
    print(f"      ✓ Guardado: {graphics_dir}/fig_mean_firing_rates.png")

    # 5d. Autocorrelation (Fig. 4a)
    print("\n   d) Autocorrelación temporal")
    print("      → Comparar con: Main paper Fig. 4a")
    print("      → Debe mostrar oscilaciones gamma (~20-80 Hz)")
    fig4, _ = plot_autocorrelation(
        result,
        max_lag=200,
        population="excitatory",
        title=(
            "Temporal Autocorrelation - Echeveste et al. (2020)\n"
            "Excitatory population"
        ),
    )
    fig4.savefig(f"{graphics_dir}/fig_autocorrelation.png", dpi=150)
    print(f"      ✓ Guardado: {graphics_dir}/fig_autocorrelation.png")

    # 5e. Power spectrum (Fig. 5c, 6)
    print("\n   e) Espectro de potencia (gamma oscillations)")
    print("      → Comparar con: Main paper Fig. 5c, Fig. 6")
    print("      → Debe mostrar pico en banda gamma (20-80 Hz)")
    fig5, _ = plot_power_spectrum(
        result,
        population="excitatory",
        freq_range=(0, 100),
        title=(
            "Power Spectrum - Echeveste et al. (2020)\n"
            "Gamma oscillations (20-80 Hz)"
        ),
    )
    fig5.savefig(f"{graphics_dir}/fig_power_spectrum.png", dpi=150)
    print(f"      ✓ Guardado: {graphics_dir}/fig_power_spectrum.png")

    # 5f. Transient analysis (Fig. 7)
    print("\n   f) Análisis de transitorios (overshoots)")
    print("      → Comparar con: Main paper Fig. 7")
    print("      → Debe mostrar overshoot al inicio del estímulo")
    fig6, _ = plot_transient_analysis(
        result,
        baseline_window=(50, 100),
        transient_window=(100, 250),
        title="Transient Overshoot - Echeveste et al. (2020)",
    )
    fig6.savefig(f"{graphics_dir}/fig_transients.png", dpi=150)
    print(f"      ✓ Guardado: {graphics_dir}/fig_transients.png")

    # 5g. Fano factor (Fig. 5d)
    print("\n   g) Fano factor (variabilidad)")
    print("      → Comparar con: Main paper Fig. 5d")
    fig7, _ = plot_fano_factor(
        result,
        window_size=50,
        title=(
            "Fano Factor - Echeveste et al. (2020)\n"
            "Neural Variability (Var/Mean)"
        ),
    )
    fig7.savefig(f"{graphics_dir}/fig_fano_factor.png", dpi=150)
    print(f"      ✓ Guardado: {graphics_dir}/fig_fano_factor.png")

    # 5h. Resumen completo (multi-panel)
    print("\n   h) Resumen de dinámicas (multi-panel)")
    print("      → Resumen de todas las dinámicas en una figura")
    fig8, _ = plot_neural_dynamics_summary(result, figsize=(16, 12))
    fig8.savefig(f"{graphics_dir}/fig_dynamics_summary.png", dpi=150)
    print(f"      ✓ Guardado: {graphics_dir}/fig_dynamics_summary.png")

    # Resumen final
    print("\n" + "=" * 80)
    print("✓ TEST COMPLETADO")
    print("=" * 80)
    print(f"\nTodas las figuras guardadas en: {graphics_dir}/")
    print("\nFiguras generadas:")
    print("  1. fig_membrane_potentials.png  → Fig. 2, 4, 5 del paper")
    print("  2. fig_population_activity.png  → Fig. 2a del paper")
    print("  3. fig_mean_firing_rates.png    → Fig. 2b del paper")
    print("  4. fig_autocorrelation.png      → Fig. 4a del paper")
    print("  5. fig_power_spectrum.png       → Fig. 5c, 6 del paper")
    print("  6. fig_transients.png           → Fig. 7 del paper")
    print("  7. fig_fano_factor.png          → Fig. 5d del paper")
    print("  8. fig_dynamics_summary.png     → Resumen multi-panel")
    print("\n" + "=" * 80)
    print("PRÓXIMOS PASOS:")
    print("=" * 80)
    print("1. Revisar cada figura y compararla con el paper original")
    print("2. Verificar que las oscilaciones gamma estén presentes")
    print("3. Verificar que los transitorios muestren overshoots")
    print("4. Verificar que el Fano factor sea razonable")
    print("=" * 80)


if __name__ == "__main__":
    main()
