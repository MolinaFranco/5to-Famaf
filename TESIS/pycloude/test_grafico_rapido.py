#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script rápido para generar un gráfico de prueba."""

import sys
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")
from skneuromsi.neural import Echeveste2020  # noqa: E402

print("Creando modelo...")
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
ssn.load_parameters()

print("Ejecutando simulación corta (100 ms)...")
result = ssn.run(stimulus_contrast=0.019, simulation_time=100.0)

# Extraer datos
df = result.get_modes()
n_times = len(result.times_)

# Potenciales y firing rates
u_e = df['excitatory_potential'].values.reshape(n_times, -1)
r_e = df['excitatory'].values.reshape(n_times, -1)

# Crear figura
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Graficar potenciales de membrana
for i in range(5):
    ax1.plot(result.times_, u_e[:, i], label=f'E{i}', alpha=0.7)
ax1.set_ylabel('Membrane potential u (a.u.)')
ax1.set_title('Membrane potentials u(t) - First 5 excitatory neurons')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Graficar firing rates
for i in range(5):
    ax2.plot(result.times_, r_e[:, i], label=f'E{i}', alpha=0.7)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Firing rate (Hz)')
ax2.set_title('Firing rates r(t) - First 5 excitatory neurons')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs/test_rapido.png',
            dpi=150)
print("Gráfico guardado en pycloude/outputs/test_rapido.png")

# Mostrar estadísticas
print(f"\nEstadísticas neurona E0:")
print(f"  u(t): min={u_e[:, 0].min():.4f}, max={u_e[:, 0].max():.4f}, "
      f"std={u_e[:, 0].std():.6f}")
print(f"  r(t): min={r_e[:, 0].min():.4f}, max={r_e[:, 0].max():.4f}, "
      f"std={r_e[:, 0].std():.6f}")
