import sys
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")
from skneuromsi.neural import Echeveste2020
import numpy as np

ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
ssn.load_parameters()
result = ssn.run(stimulus_contrast=0.5, simulation_time=200.0)

df = result.get_modes()
n_times = len(result.times_)
u_e = df['excitatory_potential'].values.reshape(n_times, -1)

print(f"Rango de u_e:")
print(f"  Min: {u_e.min():.3f}")
print(f"  Max: {u_e.max():.3f}")
print(f"  Mean: {u_e.mean():.3f}")
print(f"  Std: {u_e.std():.3f}")
