#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ejemplo de cómo usar las funciones de visualización del módulo.

Este script muestra cómo importar y usar las funciones oficiales
de visualización que ahora están en el módulo skneuromsi.
"""

import sys

sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

# IMPORTANTE: Importar las funciones desde el módulo
from skneuromsi.neural import Echeveste2020  # noqa: E402
from skneuromsi.neural._echeveste_visualization import (  # noqa: E402
    plot_heatmap_activity,  # Heatmaps de todas las neuronas
    plot_spatial_sampling,  # Muestreo espacial
    plot_membrane_potentials,  # Gráfico tradicional (primeras neuronas)
    plot_firing_rates,  # Firing rates tradicional
)

print("=" * 70)
print("EJEMPLO: Uso de funciones de visualización del módulo")
print("=" * 70)

# 1. Crear modelo y ejecutar simulación
print("\n1. Creando modelo y ejecutando simulación...")
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
ssn.load_parameters()

result = ssn.run(
    stimulus_contrast=0.5,
    stimulus_orientation=0.0,
    noise_level=0.1,
    simulation_time=200.0,
)

output_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs"

# 2. NUEVO: Heatmap de TODAS las neuronas
print("\n2. Generando heatmap (TODAS las 50 neuronas)...")
fig1, axes1 = plot_heatmap_activity(
    result, population="excitatory", orientation_centered=True
)
fig1.savefig(f"{output_dir}/ejemplo_heatmap.png", dpi=150, bbox_inches="tight")
print("   ✓ Guardado: ejemplo_heatmap.png")

# 3. NUEVO: Muestreo espacial (neuronas distribuidas)
print("\n3. Generando muestreo espacial (10 neuronas distribuidas)...")
fig2, axes2 = plot_spatial_sampling(result, n_samples=10)
fig2.savefig(f"{output_dir}/ejemplo_spatial.png", dpi=150, bbox_inches="tight")
print("   ✓ Guardado: ejemplo_spatial.png")

# 4. Gráfico tradicional (primeras 5 neuronas)
print("\n4. Generando gráfico tradicional (primeras 5 neuronas)...")
fig3, axes3 = plot_membrane_potentials(result, neuron_indices=[0, 1, 2, 3, 4])
fig3.savefig(f"{output_dir}/ejemplo_tradicional.png", dpi=150, bbox_inches="tight")
print("   ✓ Guardado: ejemplo_tradicional.png")

print("\n" + "=" * 70)
print("✓ EJEMPLO COMPLETADO")
print("\nResumen de funciones disponibles:")
print("-" * 70)
print("• plot_heatmap_activity(result, population='excitatory')")
print("  → Muestra TODAS las neuronas como matriz de colores")
print("  → Incluye 4 paneles: u(t), r(t), tuning, población")
print("")
print("• plot_spatial_sampling(result, n_samples=10)")
print("  → Muestra neuronas distribuidas uniformemente en el ring")
print("  → Revela diversidad de respuestas según orientación")
print("")
print("• plot_membrane_potentials(result, neuron_indices=[...])")
print("  → Gráfico tradicional de líneas temporales")
print("  → Útil para ver dinámica detallada de pocas neuronas")
print("")
print("• plot_firing_rates(result, neuron_indices=[...])")
print("  → Similar a plot_membrane_potentials pero para r(t)")
print("=" * 70)
