#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demostración: Dos convenciones de orientación en visualizaciones.

Este script genera figuras lado a lado mostrando:
1. Convención por defecto: Neuron IDs (0-49)
2. Convención del paper: Orientaciones centradas (-90° a 90°)

Ambas muestran LOS MISMOS DATOS, solo cambia la escala del eje Y.
"""

import sys
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

import matplotlib  # noqa: E402
matplotlib.use('Agg')

from skneuromsi.neural import Echeveste2020  # noqa: E402
from skneuromsi.neural._echeveste_visualization import (  # noqa: E402
    plot_heatmap_activity,
)
import matplotlib.pyplot as plt  # noqa: E402

print("=" * 70)
print("DEMO: Convenciones de Orientación en Visualizaciones")
print("=" * 70)

# 1. Crear modelo y ejecutar simulación
print("\n1. Ejecutando simulación...")
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
ssn.load_parameters()

result = ssn.run(
    stimulus_contrast=0.5,
    stimulus_orientation=0.0,  # Estímulo centrado en 0°
    noise_level=0.1,
    simulation_time=200.0,
)

output_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs"

print("   ✓ Simulación completada")
print(f"   Estímulo: orientación 0°, contraste 0.5")

# 2. Generar figura comparativa lado a lado
print("\n2. Generando figura comparativa...")

fig = plt.figure(figsize=(20, 8))

# Panel izquierdo: Neuron IDs (por defecto)
print("   Generando panel izquierdo (Neuron IDs)...")
ax_left = fig.add_subplot(1, 2, 1)
fig_left, axes_left = plot_heatmap_activity(
    result,
    population='excitatory',
    orientation_centered=False  # Neuron IDs
)

# Panel derecho: Orientaciones centradas
print("   Generando panel derecho (Orientaciones -90° a 90°)...")
ax_right = fig.add_subplot(1, 2, 2)
fig_right, axes_right = plot_heatmap_activity(
    result,
    population='excitatory',
    orientation_centered=True  # Centrado
)

# Guardar figuras individuales
fig_left.savefig(f'{output_dir}/demo_neuron_ids.png',
                 dpi=150, bbox_inches='tight')
fig_right.savefig(f'{output_dir}/demo_orientations.png',
                  dpi=150, bbox_inches='tight')

print("   ✓ Guardado: demo_neuron_ids.png")
print("   ✓ Guardado: demo_orientations.png")

# 3. Resumen de las diferencias
print("\n" + "=" * 70)
print("✓ DEMO COMPLETADA")
print("=" * 70)

print("\n📊 CONVENCIÓN 1: Neuron IDs (demo_neuron_ids.png)")
print("-" * 70)
print("  Eje Y: Neuron ID (0, 1, 2, ..., 49)")
print("  Interpretación directa:")
print("    • Fila 0 = Neurona 0 (orientación preferida: 0°)")
print("    • Fila 25 = Neurona 25 (orientación preferida: 90°)")
print("    • Fila 49 = Neurona 49 (orientación preferida: 176.4°)")
print("  Ventajas:")
print("    ✓ Mapeo directo neurona↔fila")
print("    ✓ Útil para indexar arrays o debuggear")
print("    ✓ Sin ambigüedades sobre qué neurona es cada fila")

print("\n📊 CONVENCIÓN 2: Orientaciones -90° a 90° (demo_orientations.png)")
print("-" * 70)
print("  Eje Y: Orientation (degrees) con ticks [-90°, -50°, 0°, 50°, 90°]")
print("  Interpretación física:")
print("    • -90° = Neurona 0 (0° - 90°)")
print("    • 0° = Neurona 25 (90° - 90°)")
print("    • 90° = Aprox. Neurona 50 (180° - 90°, real: 86.4°)")
print("  Ventajas:")
print("    ✓ Simétrico respecto a 0°")
print("    ✓ Consistente con Fig. 2A del paper (Echeveste et al. 2020)")
print("    ✓ Interpretación física más intuitiva")
print("    ✓ Facilita comparación con experimentos")

print("\n🔍 PUNTOS IMPORTANTES:")
print("-" * 70)
print("  1. MISMOS DATOS: Ambas figuras muestran exactamente los mismos")
print("     valores de u(t) y r(t). Solo cambia la escala del eje Y.")
print()
print("  2. MAPEO CONSISTENTE: La transformación es simplemente")
print("     orientation_centered = orientation_internal - 90°")
print()
print("  3. ENDPOINT=FALSE: El ring tiene 50 neuronas cubriendo 0° a 176.4°")
print("     (no llega a 180° para evitar duplicar 0°≡180°)")
print()
print("  4. TICKS EXPLÍCITOS: Cuando orientation_centered=True, los ticks")
print("     muestran claramente [-90°, -50°, 0°, 50°, 90°] para guiar")
print("     la lectura del gráfico.")
print()
print("  5. PAPER ECHEVESTE: Figure 2A usa la convención centrada.")
print("     Nuestro código ahora la reproduce fielmente.")

print("\n📂 ARCHIVOS GENERADOS:")
print("-" * 70)
print("  • demo_neuron_ids.png: Convención por defecto")
print("  • demo_orientations.png: Convención del paper")
print()
print("  Compara ambas imágenes para ver que son IDÉNTICAS excepto")
print("  por la escala del eje Y.")

print("\n📚 DOCUMENTACIÓN:")
print("-" * 70)
print("  Ver EXPLICACION_MAPEO_ORIENTACIONES.md para detalles completos")
print("  sobre el mapeo neurona→orientación y las dos convenciones.")

print("\n" + "=" * 70)
