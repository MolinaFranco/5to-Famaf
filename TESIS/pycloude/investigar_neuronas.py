#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Investigar por qué las neuronas tienen actividad similar.

Hipótesis a verificar:
1. ¿Las neuronas están organizadas espacialmente (ring topology)?
2. ¿Las primeras neuronas tienen orientaciones preferidas similares?
3. ¿La conectividad es homogénea o heterogénea?
4. ¿El estímulo es uniforme o tiene estructura espacial?
"""

import sys
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from skneuromsi.neural import Echeveste2020  # noqa: E402

print("=" * 70)
print("INVESTIGACIÓN: ¿Por qué las neuronas tienen actividad similar?")
print("=" * 70)

# Crear modelo
print("\n1. Creando modelo SSN...")
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)

# Cargar parámetros
print("2. Cargando parámetros entrenados...")
ssn.load_parameters()

# Investigar la topología espacial
print("\n3. Topología espacial:")
print(f"   N_E = {ssn._N_E} neuronas excitatorias")
print(f"   N_I = {ssn._N_I} neuronas inhibitorias")
print(f"   position_range = {ssn._position_range} (grados)")
print(f"   position_res = {ssn._position_res:.4f} (grados)")

# Calcular orientaciones preferidas
if hasattr(ssn, '_thetas_E'):
    thetas_E = ssn._thetas_E
    thetas_I = ssn._thetas_I
    print(f"\n4. Orientaciones preferidas:")
    print(f"   Excitatorias (primeras 10):")
    print(f"   {np.rad2deg(thetas_E[:10])}")
    print(f"   Inhibitorias (primeras 10):")
    print(f"   {np.rad2deg(thetas_I[:10])}")
else:
    print("\n4. Calculando orientaciones preferidas desde position_range...")
    # Ring topology: neuronas espaciadas uniformemente
    N_E = ssn._N_E
    N_I = ssn._N_I
    
    # Para excitatorias
    orientations_E = np.linspace(
        ssn._position_range[0],
        ssn._position_range[1],
        N_E,
        endpoint=False
    )
    
    # Para inhibitorias (asumiendo misma topología)
    orientations_I = np.linspace(
        ssn._position_range[0],
        ssn._position_range[1],
        N_I,
        endpoint=False
    )
    
    print(f"   Excitatorias (primeras 10 neuronas):")
    print(f"   Orientaciones: {orientations_E[:10]}")
    print(f"   Inhibitorias (primeras 10 neuronas):")
    print(f"   Orientaciones: {orientations_I[:10]}")

# Ejecutar simulación
print("\n5. Ejecutando simulación con estímulo orientado...")
result = ssn.run(
    stimulus_contrast=0.5,
    stimulus_orientation=0.0,  # Estímulo vertical (0 grados)
    noise_level=0.1,
    simulation_time=200.0,
)

# Extraer datos
df = result.get_modes()
n_times = len(result.times_)
r_e = df['excitatory_firing_rate'].values.reshape(n_times, -1)
r_i = df['inhibitory_firing_rate'].values.reshape(n_times, -1)

print("\n6. Análisis de actividad individual vs poblacional:")

# Media temporal de cada neurona
r_e_mean_time = np.mean(r_e, axis=0)  # Media en tiempo para cada neurona
r_i_mean_time = np.mean(r_i, axis=0)

print(f"\n   Excitatorias - firing rate promedio temporal:")
print(f"   Primeras 10: {r_e_mean_time[:10]}")
print(f"   Min: {r_e_mean_time.min():.3f}, Max: {r_e_mean_time.max():.3f}")
print(f"   Std entre neuronas: {r_e_mean_time.std():.3f}")
print(f"   Rango: {r_e_mean_time.max() - r_e_mean_time.min():.3f}")

print(f"\n   Inhibitorias - firing rate promedio temporal:")
print(f"   Primeras 10: {r_i_mean_time[:10]}")
print(f"   Min: {r_i_mean_time.min():.3f}, Max: {r_i_mean_time.max():.3f}")
print(f"   Std entre neuronas: {r_i_mean_time.std():.3f}")
print(f"   Rango: {r_i_mean_time.max() - r_i_mean_time.min():.3f}")

# Crear gráficos
print("\n7. Generando gráficos...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Panel 1: Actividad temporal de primeras 5 vs últimas 5 neuronas E
ax1 = axes[0, 0]
for i in range(5):
    ax1.plot(result.times_, r_e[:, i], alpha=0.7, label=f'E{i}')
for i in range(45, 50):
    ax1.plot(result.times_, r_e[:, i], alpha=0.7, linestyle='--',
             label=f'E{i}')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Firing rate (Hz)')
ax1.set_title('Primeras 5 vs Últimas 5 neuronas excitatorias')
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# Panel 2: Firing rate promedio de TODAS las neuronas E
ax2 = axes[0, 1]
neuron_ids = np.arange(ssn._N_E)
ax2.bar(neuron_ids, r_e_mean_time, alpha=0.7, color='blue')
ax2.set_xlabel('Neuron ID')
ax2.set_ylabel('Mean firing rate (Hz)')
ax2.set_title('Firing rate promedio (temporal) - Todas las neuronas E')
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Actividad temporal de primeras 5 vs últimas 5 neuronas I
ax3 = axes[1, 0]
for i in range(5):
    ax3.plot(result.times_, r_i[:, i], alpha=0.7, label=f'I{i}')
for i in range(45, 50):
    ax3.plot(result.times_, r_i[:, i], alpha=0.7, linestyle='--',
             label=f'I{i}')
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Firing rate (Hz)')
ax3.set_title('Primeras 5 vs Últimas 5 neuronas inhibitorias')
ax3.legend(fontsize=8, ncol=2)
ax3.grid(True, alpha=0.3)

# Panel 4: Firing rate promedio de TODAS las neuronas I
ax4 = axes[1, 1]
neuron_ids_i = np.arange(ssn._N_I)
ax4.bar(neuron_ids_i, r_i_mean_time, alpha=0.7, color='red')
ax4.set_xlabel('Neuron ID')
ax4.set_ylabel('Mean firing rate (Hz)')
ax4.set_title('Firing rate promedio (temporal) - Todas las neuronas I')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output = '/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs/'
output += 'investigacion_neuronas.png'
plt.savefig(output, dpi=150, bbox_inches='tight')
print(f"   ✓ Guardado: {output}")

print("\n" + "=" * 70)
print("CONCLUSIONES:")
print("=" * 70)

# Verificar si hay estructura espacial
if r_e_mean_time.std() > 1.0:
    print("✓ HAY VARIABILIDAD entre neuronas (std > 1.0)")
    print("  → Las neuronas NO tienen actividad idéntica")
    print("  → Puede haber estructura espacial/orientación preferida")
else:
    print("⚠ BAJA VARIABILIDAD entre neuronas (std < 1.0)")
    print("  → Las neuronas tienen actividad muy similar")
    print("  → Posible estímulo uniforme o conectividad muy homogénea")

if r_e_mean_time.max() - r_e_mean_time.min() > 5.0:
    print(f"\n✓ RANGO AMPLIO: {r_e_mean_time.max() - r_e_mean_time.min():.1f} Hz")
    print("  → Algunas neuronas responden más que otras")
else:
    print(f"\n⚠ RANGO PEQUEÑO: {r_e_mean_time.max() - r_e_mean_time.min():.1f} Hz")
    print("  → Respuesta homogénea en la población")

print("\nNOTA: Graficar solo las primeras 4-5 neuronas puede dar")
print("      impresión de homogeneidad si están espacialmente cercanas.")
print("      Ver gráfico de barras con TODAS las neuronas para verificar.")
print("=" * 70)
