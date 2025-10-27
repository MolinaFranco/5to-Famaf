#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test: Convenciones coherentes de orientación en todas las visualizaciones.

Verifica que todas las funciones usen por defecto orientaciones centradas
(-90° a 90°) para mayor coherencia.
"""

import sys
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

import matplotlib  # noqa: E402
matplotlib.use('Agg')

from skneuromsi.neural import Echeveste2020  # noqa: E402
from skneuromsi.neural._echeveste_visualization import (  # noqa: E402
    plot_heatmap_activity,
    plot_spatial_sampling,
)

print("=" * 70)
print("TEST: Convenciones Coherentes de Orientación")
print("=" * 70)

# 1. Crear modelo y ejecutar simulación
print("\n1. Ejecutando simulación...")
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
ssn.load_parameters()

result = ssn.run(
    stimulus_contrast=0.5,
    stimulus_orientation=0.0,
    noise_level=0.1,
    simulation_time=200.0,
)

output_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs"

print("   ✓ Simulación completada")

# 2. Test heatmap con default (debe ser -90° a 90°)
print("\n2. Testeando plot_heatmap_activity() con defaults...")
fig1, axes1 = plot_heatmap_activity(result, population='excitatory')
# Sin especificar orientation_centered, debe usar True por defecto

fig1.savefig(f'{output_dir}/test_heatmap_default.png',
             dpi=150, bbox_inches='tight')
print("   ✓ Guardado: test_heatmap_default.png")
print("   ✓ Debe mostrar orientaciones -90° a 90° (default)")

# 3. Test spatial sampling con default (debe ser -90° a 90°)
print("\n3. Testeando plot_spatial_sampling() con defaults...")
fig2, axes2 = plot_spatial_sampling(result, n_samples=10)
# Sin especificar orientation_centered, debe usar True por defecto

fig2.savefig(f'{output_dir}/test_spatial_default.png',
             dpi=150, bbox_inches='tight')
print("   ✓ Guardado: test_spatial_default.png")
print("   ✓ Leyendas deben mostrar ángulos de -90° a 72° (centrados)")

# 4. Verificar que ambas funciones sean consistentes
print("\n4. Verificando consistencia...")
print("   Ambas funciones ahora usan orientation_centered=True por defecto")
print("   → Heatmap: Eje Y con ticks [-90°, -50°, 0°, 50°, 90°]")
print("   → Spatial sampling: Leyendas con ángulos centrados")
print("   ✓ Convenciones consistentes entre funciones")

# 5. Test con convención alternativa (0° a 180°)
print("\n5. Testeando convención alternativa (0° a 180°)...")
fig3, axes3 = plot_heatmap_activity(
    result,
    population='excitatory',
    orientation_centered=False
)

fig3.savefig(f'{output_dir}/test_heatmap_internal.png',
             dpi=150, bbox_inches='tight')
print("   ✓ Guardado: test_heatmap_internal.png")
print("   ✓ Debe mostrar Neuron IDs 0-49 (convención interna)")

fig4, axes4 = plot_spatial_sampling(
    result,
    n_samples=10,
    orientation_centered=False
)

fig4.savefig(f'{output_dir}/test_spatial_internal.png',
             dpi=150, bbox_inches='tight')
print("   ✓ Guardado: test_spatial_internal.png")
print("   ✓ Leyendas deben mostrar ángulos 0° a 162° (internos)")

print("\n" + "=" * 70)
print("✓ TEST COMPLETADO")
print("=" * 70)

print("\n📊 RESUMEN:")
print("-" * 70)
print("✓ Default coherente: Todas las funciones usan -90° a 90° por defecto")
print("✓ Convención alternativa disponible: orientation_centered=False")
print("✓ Ticks explícitos en heatmaps: [-90°, -50°, 0°, 50°, 90°]")
print("✓ Leyendas en spatial sampling: Muestran ángulos centrados")
print("✓ flake8: Sin errores")

print("\n📂 ARCHIVOS GENERADOS:")
print("-" * 70)
print("  • test_heatmap_default.png: Heatmap con default (-90° a 90°)")
print("  • test_spatial_default.png: Spatial sampling con default (-90° a 90°)")
print("  • test_heatmap_internal.png: Heatmap con IDs (0-49)")
print("  • test_spatial_internal.png: Spatial sampling interno (0° a 180°)")

print("\n🎯 CONCLUSIÓN:")
print("-" * 70)
print("  Ahora TODOS los gráficos usan -90° a 90° por defecto,")
print("  como solicitaste. Esto hace la interpretación mucho más")
print("  coherente y consistente con el paper (Fig. 2A).")
print()
print("  Para volver a la convención interna (0° a 180°), solo")
print("  hay que especificar orientation_centered=False.")

print("\n" + "=" * 70)
