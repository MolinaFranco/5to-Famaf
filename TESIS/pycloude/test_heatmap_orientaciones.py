#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test: Comparar heatmaps con diferentes convenciones de orientación.

Este script genera dos versiones del heatmap:
1. Con Neuron IDs (0-49) - versión por defecto
2. Con Orientaciones centradas (-90° a 90°) - como Fig. 2A del paper
"""

import sys
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

import matplotlib  # noqa: E402
matplotlib.use('Agg')

from skneuromsi.neural import Echeveste2020  # noqa: E402
from skneuromsi.neural._echeveste_visualization import (  # noqa: E402
    plot_heatmap_activity,
)

print("=" * 70)
print("TEST: Heatmaps con diferentes convenciones de orientación")
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

# 2. Versión por defecto: Neuron IDs
print("\n2. Generando heatmap con Neuron IDs (0-49)...")
fig1, _ = plot_heatmap_activity(
    result,
    population='excitatory',
    orientation_centered=False  # Explícito
)
fig1.savefig(f'{output_dir}/heatmap_neuron_ids.png', dpi=150,
             bbox_inches='tight')
print("   ✓ Guardado: heatmap_neuron_ids.png")
print("   - Eje Y: Neuron ID (0 a 49)")
print("   - Unidades u(t): arbitrary (a.u.)")

# 3. Versión estilo Fig. 2A: Orientaciones centradas
print("\n3. Generando heatmap con Orientaciones (-90° a 90°)...")
print("   (Similar a Echeveste et al. 2020, Fig. 2A)")
fig2, _ = plot_heatmap_activity(
    result,
    population='excitatory',
    orientation_centered=True  # ¡NUEVO!
)
fig2.savefig(f'{output_dir}/heatmap_orientations.png', dpi=150,
             bbox_inches='tight')
print("   ✓ Guardado: heatmap_orientations.png")
print("   - Eje Y: Orientation (-90° a 90°)")
print("   - Unidades u(t): arbitrary (a.u.)")

print("\n" + "=" * 70)
print("✓ TEST COMPLETADO")
print("\nArchivos generados:")
print("  - heatmap_neuron_ids.png: Convención por defecto (IDs)")
print("  - heatmap_orientations.png: Convención del paper (ángulos)")
print("\nNotas importantes:")
print("  • Ambos muestran LOS MISMOS DATOS")
print("  • Solo cambia la escala del eje Y")
print("  • orientation_centered=True es equivalente a Fig. 2A del paper")
print("\nSobre las unidades de u(t):")
print("  • El paper usa 'arbitrary units (a.u.)'")
print("  • NO son mV - este es un rate model, no conductance-based")
print("  • Los potenciales son variables de estado sin unidades físicas")
print("=" * 70)
