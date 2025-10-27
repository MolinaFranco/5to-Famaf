#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test rápido con UN nivel de estímulo."""

import sys
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from skneuromsi.neural import Echeveste2020  # noqa: E402
from skneuromsi.neural._echeveste_visualization import (  # noqa: E402
    plot_membrane_potentials,
    plot_firing_rates,
)

print("Creando modelo...")
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
ssn.load_parameters()

print("\nSimulando con contraste MEDIO (0.3)...")
result = ssn.run(stimulus_contrast=0.3, simulation_time=200.0)

print("\nGenerando gráficos...")
fig1, _ = plot_membrane_potentials(result, neuron_indices=[0, 1, 2, 3, 4])
fig1.savefig('/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs/test_c03_potentials.png', dpi=150)
print("✓ Potenciales guardados")

fig2, _ = plot_firing_rates(result, neuron_indices=[0, 1, 2, 3, 4])
fig2.savefig('/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs/test_c03_firing.png', dpi=150)
print("✓ Firing rates guardados")

print("\nLISTO!")
