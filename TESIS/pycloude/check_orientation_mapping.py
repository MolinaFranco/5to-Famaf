import sys
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")
from skneuromsi.neural import Echeveste2020
import numpy as np

print("=" * 70)
print("INVESTIGACIÓN: Mapeo de Neuronas a Orientaciones")
print("=" * 70)

ssn = Echeveste2020(N_E=50, N_I=50, seed=42)

# Obtener configuración
N_E = ssn._N_E
position_range = ssn._position_range
position_res = ssn._position_res

print(f"\nConfiguración del modelo:")
print(f"  N_E = {N_E}")
print(f"  position_range = {position_range}")
print(f"  position_res = {position_res:.4f}°")

# Calcular orientaciones de cada neurona
# Según el modelo, las neuronas están espaciadas uniformemente
orientations = np.linspace(position_range[0], position_range[1], N_E, endpoint=False)

print(f"\nOrientaciones asignadas a cada neurona:")
print(f"  Neurona 0:  {orientations[0]:.2f}°")
print(f"  Neurona 1:  {orientations[1]:.2f}°")
print(f"  Neurona 10: {orientations[10]:.2f}°")
print(f"  Neurona 25: {orientations[25]:.2f}°")
print(f"  Neurona 40: {orientations[40]:.2f}°")
print(f"  Neurona 49: {orientations[49]:.2f}°")

# Convertir a -90 a 90
orientations_centered = orientations - 90.0

print(f"\nOrientaciones centradas (-90° a 90°):")
print(f"  Neurona 0:  {orientations_centered[0]:.2f}°")
print(f"  Neurona 1:  {orientations_centered[1]:.2f}°")
print(f"  Neurona 10: {orientations_centered[10]:.2f}°")
print(f"  Neurona 25: {orientations_centered[25]:.2f}°")
print(f"  Neurona 40: {orientations_centered[40]:.2f}°")
print(f"  Neurona 49: {orientations_centered[49]:.2f}°")

# En plot_spatial_sampling se usa:
# orientation = idx * 180.0 / n_e
print(f"\nFórmula usada en plot_spatial_sampling:")
print(f"  orientation = idx * 180.0 / {N_E}")
for idx in [0, 10, 20, 30, 40, 49]:
    ori = idx * 180.0 / N_E
    print(f"  Neurona {idx}: {ori:.2f}°")

print(f"\n⚠️  PROBLEMA DETECTADO:")
print(f"  - linspace(0, 180, 50, endpoint=False) va de 0° a 176.4°")
print(f"  - idx * 180 / 50 con idx=49 da 176.4°")
print(f"  - ¡Coinciden! ✓")
print(f"\n  Pero al centrar:")
print(f"  - Neurona 0: 0° → -90° ✓")
print(f"  - Neurona 25: 90° → 0° ✓")  
print(f"  - Neurona 49: 176.4° → 86.4° (NO es 90°!)")

print("\n" + "=" * 70)
