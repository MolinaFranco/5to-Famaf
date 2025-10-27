#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de prueba para verificar los potenciales de membrana.

Este script verifica que:
1. Los potenciales de membrana u(t) varían con el tiempo
2. Las firing rates r(t) se calculan correctamente
3. Las visualizaciones muestran las variables correctas con unidades correctas
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# Agregar el path de scikit-neuromsi al PYTHONPATH
sys.path.insert(
    0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi"
)
from skneuromsi.neural import Echeveste2020  # noqa: E402
from skneuromsi.neural._echeveste_visualization import (  # noqa: E402
    plot_membrane_potentials,
    plot_firing_rates,
)


def main():
    """Ejecutar prueba de potenciales de membrana y firing rates."""
    print("=" * 70)
    print("PRUEBA: Potenciales de membrana vs Firing rates")
    print("=" * 70)

    # Crear modelo
    print("\n1. Creando modelo SSN...")
    ssn = Echeveste2020(N_E=50, N_I=50, seed=42)

    # Cargar parámetros
    print("2. Cargando parámetros entrenados...")
    ssn.load_parameters()

    # Ejecutar simulación
    print("\n3. Ejecutando simulación (100 ms)...")
    result = ssn.run(
        stimulus_contrast=0.019,
        stimulus_orientation=0.0,
        noise_level=0.1,
        simulation_time=100.0,
    )

    # Verificar datos disponibles
    print("\n4. Verificando datos en el resultado...")
    df = result.get_modes()
    print(f"   Modos disponibles: {list(df.columns)}")

    # Extraer datos
    exc_potential = df['excitatory_potential'].values
    exc_firing = df['excitatory'].values
    # inh_potential = df['inhibitory_potential'].values  # noqa: F841
    # inh_firing = df['inhibitory'].values  # noqa: F841

    # Reshape para análisis
    n_times = len(result.times_)
    exc_pot_reshaped = exc_potential.reshape(n_times, -1)
    exc_fir_reshaped = exc_firing.reshape(n_times, -1)
    # inh_pot_reshaped = inh_potential.reshape(n_times, -1)  # noqa: F841
    # inh_fir_reshaped = inh_firing.reshape(n_times, -1)  # noqa: F841

    print(f"\n   Shape potenciales E: {exc_pot_reshaped.shape}")
    print(f"   Shape firing rates E: {exc_fir_reshaped.shape}")

    # Estadísticas para neurona 0 excitatorias
    print("\n5. Estadísticas para neurona E0:")
    print("   Potencial u(t):")
    print(f"     - Min: {exc_pot_reshaped[:, 0].min():.4f} (a.u.)")
    print(f"     - Max: {exc_pot_reshaped[:, 0].max():.4f} (a.u.)")
    print(f"     - Mean: {exc_pot_reshaped[:, 0].mean():.4f} (a.u.)")
    print(f"     - Std: {exc_pot_reshaped[:, 0].std():.4f} (a.u.)")
    print("   Firing rate r(t):")
    print(f"     - Min: {exc_fir_reshaped[:, 0].min():.4f} Hz")
    print(f"     - Max: {exc_fir_reshaped[:, 0].max():.4f} Hz")
    print(f"     - Mean: {exc_fir_reshaped[:, 0].mean():.4f} Hz")
    print(f"     - Std: {exc_fir_reshaped[:, 0].std():.4f} Hz")

    # Verificar variación temporal
    print("\n6. Verificando variación temporal:")
    u_variation = np.std(exc_pot_reshaped[:, 0])
    r_variation = np.std(exc_fir_reshaped[:, 0])
    print(f"   Desviación estándar u(t): {u_variation:.6f}")
    print(f"   Desviación estándar r(t): {r_variation:.6f}")

    if u_variation > 0.01:
        print("   ✓ Los potenciales de membrana SÍ varían con el tiempo")
    else:
        print("   ✗ ADVERTENCIA: Potenciales casi constantes")

    # Verificar relación u → r
    print("\n7. Verificando relación r = k * [u]_+^2:")
    # Calcular r esperado desde u
    k = 0.3  # scaling factor del modelo
    n = 2.0  # supralinear exponent
    u_sample = exc_pot_reshaped[100, 0]
    r_sample = exc_fir_reshaped[100, 0]
    r_expected = k * np.power(np.maximum(0, u_sample), n)
    print(f"   u(t=100) = {u_sample:.4f}")
    print(f"   r(t=100) = {r_sample:.4f} Hz (real)")
    print(f"   r_esperado = {r_expected:.4f} Hz (calculado)")
    print(f"   Diferencia: {abs(r_sample - r_expected):.6f}")

    # Crear visualizaciones
    print("\n8. Generando visualizaciones...")

    # Potenciales de membrana
    print("   a) Graficando potenciales de membrana u(t)...")
    fig1, _ = plot_membrane_potentials(
        result,
        neuron_indices=[0, 1, 2, 3, 4],
        population="both",
        title="Membrane potentials u(t) - Echeveste et al. (2020)",
    )

    # Firing rates
    print("   b) Graficando firing rates r(t)...")
    fig2, _ = plot_firing_rates(
        result,
        neuron_indices=[0, 1, 2, 3, 4],
        population="both",
        title="Firing rates r(t) - Echeveste et al. (2020)",
    )

    # Guardar figuras
    output_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs"
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(f"{output_dir}/test_membrane_potentials.png", dpi=150)
    print(f"      ✓ Guardado en {output_dir}/test_membrane_potentials.png")

    fig2.savefig(f"{output_dir}/test_firing_rates.png", dpi=150)
    print(f"      ✓ Guardado en {output_dir}/test_firing_rates.png")

    print("\n" + "=" * 70)
    print("✓ PRUEBA COMPLETADA")
    print("=" * 70)

    # Mostrar gráficos
    plt.show()


if __name__ == "__main__":
    main()
