#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test rápido para verificar que graficos_multiples_estimulos.py funciona.

Solo ejecuta 2 contrastes con simulación corta.
"""

import sys
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from skneuromsi.neural import Echeveste2020  # noqa: E402

print("=" * 70)
print("TEST RÁPIDO: Gráficos múltiples estímulos")
print("=" * 70)

# Crear modelo
print("\n1. Creando modelo SSN...")
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)

# Cargar parámetros
print("2. Cargando parámetros entrenados...")
ssn.load_parameters()

# Probar 2 contrastes
contrast_levels = [
    (0.1, "Low contrast"),
    (0.5, "High contrast"),
]

results = {}

print("\n3. Ejecutando simulaciones...")
for contrast, name in contrast_levels:
    print(f"\n   Simulando {name} (c={contrast})...")
    result = ssn.run(
        stimulus_contrast=contrast,
        stimulus_orientation=0.0,
        noise_level=0.1,
        simulation_time=100.0,  # Solo 100ms
    )

    # Verificar que las claves son correctas
    df = result.get_modes()
    print(f"   Claves disponibles: {list(df.columns)}")

    # Extraer datos usando las claves correctas
    n_times = len(result.times_)
    u_e = df['excitatory_potential'].values.reshape(n_times, -1)
    r_e = df['excitatory_firing_rate'].values.reshape(n_times, -1)

    print(f"   Shape u_e: {u_e.shape}")
    print(f"   Shape r_e: {r_e.shape}")
    print(f"   u_e range: [{u_e.min():.3f}, {u_e.max():.3f}]")
    print(f"   r_e range: [{r_e.min():.3f}, {r_e.max():.3f}]")

    results[name] = result

print("\n4. Creando gráfico comparativo...")
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for name, result in results.items():
    df = result.get_modes()
    n_times = len(result.times_)
    r_e = df['excitatory_firing_rate'].values.reshape(n_times, -1)

    # Media poblacional
    r_mean = np.mean(r_e, axis=1)

    axes[0].plot(result.times_, r_e[:, 0], label=f'{name} - E0', alpha=0.7)
    axes[1].plot(result.times_, r_mean, label=name, alpha=0.8, lw=2)

axes[0].set_ylabel('Firing rate E0 (Hz)')
axes[0].set_title('Single neuron firing rate')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Mean firing rate (Hz)')
axes[1].set_title('Population mean')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

output_file = '/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs/'
output_file += 'test_graficos_rapido.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"   ✓ Guardado: {output_file}")

print("\n" + "=" * 70)
print("✓ TEST COMPLETADO - Las claves están correctas")
print("=" * 70)
