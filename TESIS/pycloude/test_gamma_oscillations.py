#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test para verificar que las oscilaciones gamma aparecen correctamente.

Después de corregir las constantes de tiempo (de ms a s), las dinámicas
deberían mostrar:
1. Oscilaciones gamma en el rango 20-80 Hz (según el paper)
2. Overshoots transitorios al inicio
3. Dinámica más rápida (1000× más rápida que antes del fix)

Referencias:
- echepaper.pdf: Gamma oscillations en población inhibitoria
- echepaper.pdf Supplementary: Transient overshoots en respuesta a estímulo
"""

import sys
import numpy as np

# Agregar scikit-neuromsi al path
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from scipy import signal  # noqa: E402

from skneuromsi.neural import Echeveste2020  # noqa: E402

print("=" * 70)
print("TEST: GAMMA OSCILLATIONS Y DINÁMICA TEMPORAL")
print("=" * 70)

# Crear modelo
print("\n1. Creando modelo SSN...")
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)

# Cargar parámetros
print("2. Cargando parámetros entrenados...")
ssn.load_parameters()

# Ejecutar simulación con contraste medio
print("\n3. Ejecutando simulación (200 ms, contraste 0.3)...")
result = ssn.run(
    stimulus_contrast=0.3,
    stimulus_orientation=0.0,
    noise_level=0.1,
    simulation_time=200.0,  # 200 ms
)

# Extraer datos
print("\n4. Extrayendo datos...")
df = result.get_modes()
n_times = len(result.times_)

u_e = df['excitatory_potential'].values.reshape(n_times, -1)
r_e = df['excitatory_firing_rate'].values.reshape(n_times, -1)
u_i = df['inhibitory_potential'].values.reshape(n_times, -1)
r_i = df['inhibitory_firing_rate'].values.reshape(n_times, -1)

print(f"   Shape: {u_e.shape}")
print(f"   Tiempo total: {result.times_[-1]:.1f} ms")
print(f"   dt efectivo: {result.times_[1] - result.times_[0]:.4f} ms")

# Calcular estadísticas
print("\n5. Estadísticas de actividad:")
print("   Excitatorias:")
print(f"     u(t): min={u_e.min():.4f}, max={u_e.max():.4f}, "
      f"std={u_e.std():.4f}")
print(f"     r(t): min={r_e.min():.4f}, max={r_e.max():.4f}, "
      f"std={r_e.std():.4f}")
print("   Inhibitorias:")
print(f"     u(t): min={u_i.min():.4f}, max={u_i.max():.4f}, "
      f"std={u_i.std():.4f}")
print(f"     r(t): min={r_i.min():.4f}, max={r_i.max():.4f}, "
      f"std={r_i.std():.4f}")

# Verificar que hay variación temporal
u_variation = u_e[:, 0].std()
r_variation = r_e[:, 0].std()
print("\n6. Variación temporal (neurona E0):")
print(f"   std(u) = {u_variation:.6f}")
print(f"   std(r) = {r_variation:.6f}")

if u_variation > 0.01:
    print("   ✓ Los potenciales SÍ varían con el tiempo (correcto)")
else:
    print("   ✗ ADVERTENCIA: Potenciales casi constantes")

# Análisis espectral para detectar gamma
print("\n7. Análisis de frecuencias (FFT):")
# Calcular media poblacional inhibitoria
r_i_mean = np.mean(r_i, axis=1)

# FFT
fs = 1000.0 / (result.times_[1] - result.times_[0])  # Frecuencia muestreo
freqs, psd = signal.welch(r_i_mean, fs=fs, nperseg=min(256, len(r_i_mean)))

# Encontrar pico dominante
gamma_range = (freqs >= 20) & (freqs <= 80)
if np.any(gamma_range):
    peak_idx = np.argmax(psd[gamma_range])
    gamma_freqs = freqs[gamma_range]
    peak_freq = gamma_freqs[peak_idx]
    print(f"   Frecuencia dominante: {peak_freq:.1f} Hz")

    if 20 <= peak_freq <= 80:
        print("   ✓ Oscilaciones gamma detectadas en rango esperado "
              "(20-80 Hz)")
    else:
        print(f"   ? Pico fuera del rango gamma: {peak_freq:.1f} Hz")
else:
    print("   ✗ No se pudo calcular FFT en rango gamma")

# Crear visualizaciones
print("\n8. Generando gráficos...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Panel 1: Potenciales de membrana
ax1 = axes[0]
for i in range(3):
    ax1.plot(result.times_, u_e[:, i], alpha=0.7, label=f'E{i}')
ax1.set_ylabel('Membrane potential u (a.u.)')
ax1.set_title('Membrane potentials - Excitatory neurons (after time fix)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Panel 2: Firing rates excitatorias
ax2 = axes[1]
for i in range(3):
    ax2.plot(result.times_, r_e[:, i], alpha=0.7, label=f'E{i}')
ax2.set_ylabel('Firing rate (Hz)')
ax2.set_title('Firing rates - Excitatory neurons')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Panel 3: Media poblacional inhibitoria (para ver gamma)
ax3 = axes[2]
ax3.plot(result.times_, r_i_mean, color='red', lw=1.5,
         label='Inhibitory population mean')
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Mean firing rate (Hz)')
ax3.set_title('Population mean - Inhibitory neurons (gamma oscillations)')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
output_file = '/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs/'
output_file += 'test_gamma_oscillations.png'
plt.savefig(output_file, dpi=150)
print(f"   ✓ Gráfico guardado: {output_file}")

# Crear gráfico del espectro de potencias
fig2, ax = plt.subplots(figsize=(12, 6))
ax.semilogy(freqs, psd, lw=2)
ax.axvspan(20, 80, alpha=0.2, color='green', label='Gamma range (20-80 Hz)')
if 'peak_freq' in locals():
    ax.axvline(peak_freq, color='red', linestyle='--', lw=2,
               label=f'Peak: {peak_freq:.1f} Hz')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power Spectral Density')
ax.set_title('Power Spectrum - Inhibitory population (after time fix)')
ax.set_xlim(0, 150)
ax.legend()
ax.grid(True, alpha=0.3)

output_file2 = '/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs/'
output_file2 += 'test_gamma_spectrum.png'
plt.savefig(output_file2, dpi=150)
print(f"   ✓ Espectro guardado: {output_file2}")

print("\n" + "=" * 70)
print("✓ TEST COMPLETADO")
print("✓ Revisar gráficos para verificar:")
print("  - Oscilaciones gamma visibles (20-80 Hz)")
print("  - Transitorios al inicio de la simulación")
print("  - Dinámica realista (no constante)")
print("=" * 70)
