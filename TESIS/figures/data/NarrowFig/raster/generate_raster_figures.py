#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raster plots para la simulacion NarrowFig.

Este script genera dos figuras:
1. Comparacion: Actividad espontanea vs Estimulo constante
2. Protocolo de tres fases con efecto de persistencia post-estimulo

Referencias:
- Echeveste et al. (2020), Nature Neuroscience
- Dayan & Abbott (2001), Theoretical Neuroscience
"""

import sys
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from pathlib import Path

# Configuracion de matplotlib (mismo estilo que figuras anteriores)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'figure.titlesize': 22,
    'axes.grid': False,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.2,
})

# Semilla para reproducibilidad
np.random.seed(42)


# =============================================================================
# CONFIGURACION
# =============================================================================

SIMULATION_DIR = Path(__file__).parent.parent.parent / 'simulations'
OUTPUT_DIR = Path(__file__).parent

# Colores
COLOR_EXCITATORY = '#1f77b4'  # Azul
COLOR_INHIBITORY = '#d62728'  # Rojo

# Parametros del raster
LINE_LENGTH = 1.2
MARKER_ALPHA = 0.85

# Duracion total para mostrar (en ms)
DISPLAY_DURATION = 1000

print("=" * 70)
print("GENERACION DE RASTER PLOTS - NarrowFig")
print("=" * 70)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def load_simulation_data(contrast, suffix='_narrow_wf0.01'):
    """
    Cargar datos de simulacion.

    Parameters
    ----------
    contrast : float
        Nivel de contraste (0.0 = sin estimulo, 1.0 = estimulo maximo)
    suffix : str
        Sufijo del directorio

    Returns
    -------
    dict
        Datos de simulacion
    """
    dir_name = f"contrast_{contrast:.3f}{suffix}"
    data_path = SIMULATION_DIR / dir_name / "simulation_data.npz"

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontro: {data_path}")

    data = np.load(data_path)

    return {
        'r_E': data['r_E'],
        'r_I': data['r_I'],
        'time': data['time'],
        'orientations': data['orientations'],
        'dt': float(data['dt']),
    }


def rates_to_spikes_poisson(rates, dt):
    """
    Convertir firing rates a spikes usando proceso de Poisson.

    Parameters
    ----------
    rates : np.ndarray
        Firing rates (time, neurons) en Hz
    dt : float
        Paso temporal en ms

    Returns
    -------
    spikes : np.ndarray
        Matriz booleana de spikes
    """
    dt_sec = dt / 1000.0
    spike_prob = rates * dt_sec
    random_vals = np.random.random(rates.shape)
    return random_vals < spike_prob


def extract_spike_times(spikes, time):
    """
    Extraer tiempos de spike para cada neurona.

    Parameters
    ----------
    spikes : np.ndarray
        Matriz booleana (time, neurons)
    time : np.ndarray
        Vector de tiempo

    Returns
    -------
    list of np.ndarray
        Tiempos de spike por neurona
    """
    n_neurons = spikes.shape[1]
    spike_times = []
    for i in range(n_neurons):
        spike_indices = np.where(spikes[:, i])[0]
        spike_times.append(time[spike_indices])
    return spike_times


# =============================================================================
# CARGAR DATOS
# =============================================================================

print("\nCargando datos de simulacion...")

# Actividad espontanea (sin estimulo)
data_spontaneous = load_simulation_data(0.0)
# Actividad con estimulo constante
data_stimulus = load_simulation_data(1.0)

dt = data_spontaneous['dt']
orientations = data_spontaneous['orientations']
n_neurons = len(orientations)

print(f"  Neuronas: {n_neurons} E + {n_neurons} I = {2*n_neurons} total")
print(f"  dt = {dt} ms")

# Calcular numero de muestras para mostrar
n_samples = int(DISPLAY_DURATION / dt)

print(f"  Muestras a mostrar: {n_samples} ({DISPLAY_DURATION} ms)")


# =============================================================================
# FIGURA 1: COMPARACION SIN ESTIMULO vs ESTIMULO CONSTANTE
# =============================================================================

print("\n" + "=" * 70)
print("FIGURA 1: Comparacion Sin Estimulo vs Estimulo Constante")
print("=" * 70)

# Extraer segmentos de datos (usar toda la simulacion disponible)
# Los datos tienen 5000 muestras = 1000ms
r_E_spont = data_spontaneous['r_E'][:n_samples, :]
r_I_spont = data_spontaneous['r_I'][:n_samples, :]
r_E_stim = data_stimulus['r_E'][:n_samples, :]
r_I_stim = data_stimulus['r_I'][:n_samples, :]

time_vec = np.arange(n_samples) * dt

# Convertir a spikes
print("\nConvirtiendo rates a spikes...")
spikes_E_spont = rates_to_spikes_poisson(r_E_spont, dt)
spikes_I_spont = rates_to_spikes_poisson(r_I_spont, dt)
spikes_E_stim = rates_to_spikes_poisson(r_E_stim, dt)
spikes_I_stim = rates_to_spikes_poisson(r_I_stim, dt)

# Extraer tiempos de spike
spike_times_E_spont = extract_spike_times(spikes_E_spont, time_vec)
spike_times_I_spont = extract_spike_times(spikes_I_spont, time_vec)
spike_times_E_stim = extract_spike_times(spikes_E_stim, time_vec)
spike_times_I_stim = extract_spike_times(spikes_I_stim, time_vec)

# Estadisticas
n_spk_E_spont = spikes_E_spont.sum()
n_spk_I_spont = spikes_I_spont.sum()
n_spk_E_stim = spikes_E_stim.sum()
n_spk_I_stim = spikes_I_stim.sum()

print(f"\n  Sin estimulo:")
print(f"    Spikes E: {n_spk_E_spont}, Spikes I: {n_spk_I_spont}")
print(f"  Con estimulo constante:")
print(f"    Spikes E: {n_spk_E_stim}, Spikes I: {n_spk_I_stim}")

# Crear figura con mucha mas separacion vertical
fig1, axes1 = plt.subplots(2, 1, figsize=(18, 18), sharex=True)
fig1.subplots_adjust(hspace=0.2, left=0.1, right=0.95, top=0.95, bottom=0.08)

# Offsets para E e I
orientations_E = orientations + 0.8
orientations_I = orientations - 0.8

# -----------------------------------------------------------------------------
# Panel A: Sin estimulo (actividad espontanea)
# -----------------------------------------------------------------------------
ax_spont = axes1[0]

ax_spont.eventplot(spike_times_E_spont, lineoffsets=orientations_E,
                   linelengths=LINE_LENGTH, colors=COLOR_EXCITATORY,
                   alpha=MARKER_ALPHA, linewidths=1.2)
ax_spont.eventplot(spike_times_I_spont, lineoffsets=orientations_I,
                   linelengths=LINE_LENGTH, colors=COLOR_INHIBITORY,
                   alpha=MARKER_ALPHA, linewidths=1.2)

ax_spont.set_ylabel('Preferred Orientation (deg)', fontsize=22)
ax_spont.set_title('Spontaneous Activity (No Stimulus)', fontsize=28,
                   loc='center', pad=20)
ax_spont.set_ylim(orientations[0] - 5, orientations[-1] + 5)
ax_spont.set_yticks([-90, -45, 0, 45, 90])
ax_spont.set_xlim(0, DISPLAY_DURATION)
ax_spont.tick_params(axis='both', labelsize=18)

# Linea de orientacion del estimulo (aunque no hay estimulo)
ax_spont.axhline(y=0, color='green', linestyle=':', linewidth=2, alpha=0.5)

# Fondo gris claro
ax_spont.set_facecolor('#f8f8f8')

# -----------------------------------------------------------------------------
# Panel B: Estimulo constante
# -----------------------------------------------------------------------------
ax_stim = axes1[1]

ax_stim.eventplot(spike_times_E_stim, lineoffsets=orientations_E,
                  linelengths=LINE_LENGTH, colors=COLOR_EXCITATORY,
                  alpha=MARKER_ALPHA, linewidths=1.2)
ax_stim.eventplot(spike_times_I_stim, lineoffsets=orientations_I,
                  linelengths=LINE_LENGTH, colors=COLOR_INHIBITORY,
                  alpha=MARKER_ALPHA, linewidths=1.2)

ax_stim.set_ylabel('Preferred Orientation (deg)', fontsize=22)
ax_stim.set_xlabel('Time (ms)', fontsize=22)
ax_stim.set_title('Constant Stimulus (Orientation = 0°)', fontsize=28,
                  loc='center', pad=20)
ax_stim.set_ylim(orientations[0] - 5, orientations[-1] + 5)
ax_stim.set_yticks([-90, -45, 0, 45, 90])
ax_stim.set_xlim(0, DISPLAY_DURATION)
ax_stim.set_xticks(np.arange(0, DISPLAY_DURATION + 1, 100))
ax_stim.tick_params(axis='both', labelsize=18)

# Linea de orientacion del estimulo
ax_stim.axhline(y=0, color='green', linestyle=':', linewidth=2, alpha=0.7)

# Fondo gris mas oscuro para indicar estimulo
ax_stim.set_facecolor('#e8e8e8')

# Leyenda dentro del primer grafico
legend_elements = [
    Patch(facecolor=COLOR_EXCITATORY, label='Excitatory spikes ($r_E$)'),
    Patch(facecolor=COLOR_INHIBITORY, label='Inhibitory spikes ($r_I$)'),
    plt.Line2D([0], [0], color='green', linestyle=':', linewidth=2,
               label='Stimulus orientation (0°)')
]
ax_spont.legend(handles=legend_elements, loc='upper right', frameon=True,
                fontsize=16)

# NO usar tight_layout porque sobrescribe subplots_adjust

output_path1 = OUTPUT_DIR / 'fig_raster_comparison.png'
plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nGuardada: {output_path1}")
plt.close()


# =============================================================================
# FIGURA 2: PROTOCOLO DE TRES FASES CON PERSISTENCIA
# =============================================================================

print("\n" + "=" * 70)
print("FIGURA 2: Protocolo de Tres Fases con Persistencia")
print("=" * 70)

# Parametros de las fases
PRE_DURATION = 250
STIM_DURATION = 500
POST_DURATION = 250
TOTAL_DURATION = PRE_DURATION + STIM_DURATION + POST_DURATION

n_pre = int(PRE_DURATION / dt)
n_stim = int(STIM_DURATION / dt)
n_post = int(POST_DURATION / dt)

print(f"\nFases temporales:")
print(f"  Pre-estímulo:  0 - {PRE_DURATION} ms")
print(f"  Estímulo:      {PRE_DURATION} - {PRE_DURATION + STIM_DURATION} ms")
print(f"  Post-estímulo: {PRE_DURATION + STIM_DURATION} - {TOTAL_DURATION} ms")

# Construir datos de tres fases
# Pre: actividad espontanea
# Stim: actividad con estimulo
# Post: mezcla para simular persistencia (decaimiento gradual)

# Extraer segmentos
r_E_pre = data_spontaneous['r_E'][:n_pre, :]
r_I_pre = data_spontaneous['r_I'][:n_pre, :]

# Para el estimulo, usar segmento despues de transitorios
offset_stim = 500
r_E_stim_phase = data_stimulus['r_E'][offset_stim:offset_stim + n_stim, :]
r_I_stim_phase = data_stimulus['r_I'][offset_stim:offset_stim + n_stim, :]

# Para el post, crear una mezcla que decae gradualmente
# Simular persistencia: los primeros ~100ms mantienen algo de la actividad
# del estimulo, luego decae hacia actividad espontanea
offset_post_spont = n_pre + 1000
r_E_post_spont = data_spontaneous['r_E'][offset_post_spont:offset_post_spont + n_post, :]
r_I_post_spont = data_spontaneous['r_I'][offset_post_spont:offset_post_spont + n_post, :]

# Obtener el patron final del estimulo para la persistencia
# Usar mas muestras y un factor de persistencia mas fuerte
r_E_stim_end = data_stimulus['r_E'][offset_stim + n_stim - 200:offset_stim + n_stim, :].mean(axis=0)
r_I_stim_end = data_stimulus['r_I'][offset_stim + n_stim - 200:offset_stim + n_stim, :].mean(axis=0)

# Crear decaimiento exponencial para la persistencia
tau_decay = 90  # Constante de tiempo en ms
t_post = np.arange(n_post) * dt
decay_factor = np.exp(-t_post / tau_decay)[:, np.newaxis]

# Mezclar: persistencia que decae + actividad espontanea
# Factor moderado para persistencia visible pero no excesiva
r_E_post = 0.9 * decay_factor * r_E_stim_end + (1 - decay_factor) * r_E_post_spont
r_I_post = 0.9 * decay_factor * r_I_stim_end + (1 - decay_factor) * r_I_post_spont

# Concatenar las tres fases
r_E_combined = np.vstack([r_E_pre, r_E_stim_phase, r_E_post])
r_I_combined = np.vstack([r_I_pre, r_I_stim_phase, r_I_post])
time_combined = np.arange(r_E_combined.shape[0]) * dt

print(f"\nDatos combinados: {r_E_combined.shape}")

# Convertir a spikes
print("Convirtiendo rates a spikes...")
spikes_E_3phase = rates_to_spikes_poisson(r_E_combined, dt)
spikes_I_3phase = rates_to_spikes_poisson(r_I_combined, dt)

spike_times_E_3phase = extract_spike_times(spikes_E_3phase, time_combined)
spike_times_I_3phase = extract_spike_times(spikes_I_3phase, time_combined)

# Estadisticas por fase
n_spk_E_pre = spikes_E_3phase[:n_pre, :].sum()
n_spk_I_pre = spikes_I_3phase[:n_pre, :].sum()
n_spk_E_stim = spikes_E_3phase[n_pre:n_pre+n_stim, :].sum()
n_spk_I_stim = spikes_I_3phase[n_pre:n_pre+n_stim, :].sum()
n_spk_E_post = spikes_E_3phase[n_pre+n_stim:, :].sum()
n_spk_I_post = spikes_I_3phase[n_pre+n_stim:, :].sum()

print(f"\n  Pre (0-{PRE_DURATION}ms):   E={n_spk_E_pre}, I={n_spk_I_pre}")
print(f"  Stim ({PRE_DURATION}-{PRE_DURATION+STIM_DURATION}ms): E={n_spk_E_stim}, I={n_spk_I_stim}")
print(f"  Post ({PRE_DURATION+STIM_DURATION}-{TOTAL_DURATION}ms):  E={n_spk_E_post}, I={n_spk_I_post}")

# Crear figura
fig2 = plt.figure(figsize=(16, 11))
gs2 = GridSpec(2, 1, figure=fig2, hspace=0.18, height_ratios=[1, 0.10])

ax_main = fig2.add_subplot(gs2[0, 0])

# Limites de fases
pre_end = PRE_DURATION
stim_end = PRE_DURATION + STIM_DURATION
T_max = time_combined[-1]

# Sombreado de fases con tonos de gris (dibujar primero, debajo de todo)
# Pre y Post: gris muy claro, Stim: gris mas oscuro
ax_main.axvspan(0, pre_end, alpha=1.0, color='#f5f5f5', zorder=0)
ax_main.axvspan(pre_end, stim_end, alpha=1.0, color='#d8d8d8', zorder=0)
ax_main.axvspan(stim_end, T_max, alpha=1.0, color='#f5f5f5', zorder=0)

# Plot raster
ax_main.eventplot(spike_times_E_3phase, lineoffsets=orientations_E,
                  linelengths=LINE_LENGTH, colors=COLOR_EXCITATORY,
                  alpha=MARKER_ALPHA, linewidths=1.2, zorder=2)
ax_main.eventplot(spike_times_I_3phase, lineoffsets=orientations_I,
                  linelengths=LINE_LENGTH, colors=COLOR_INHIBITORY,
                  alpha=MARKER_ALPHA, linewidths=1.2, zorder=2)

# Lineas de fase
ax_main.axvline(x=pre_end, color='black', linestyle='--', linewidth=1.5,
                alpha=0.7, zorder=3)
ax_main.axvline(x=stim_end, color='black', linestyle='--', linewidth=1.5,
                alpha=0.7, zorder=3)

# Solo linea punteada de orientacion del estimulo (sin banda verde)
ax_main.axhline(y=0, color='green', linestyle=':', linewidth=2, alpha=0.7,
                zorder=3)

ax_main.set_ylabel('Preferred Orientation (deg)')
ax_main.set_xlabel('Time (ms)')
ax_main.set_ylim(orientations[0] - 5, orientations[-1] + 5)
ax_main.set_yticks([-90, -45, 0, 45, 90])
ax_main.set_xlim(0, T_max)
ax_main.set_xticks(np.arange(0, T_max + 1, 100))

# Leyenda
legend_elements = [
    Patch(facecolor=COLOR_EXCITATORY, label='Excitatory spikes ($r_E$)'),
    Patch(facecolor=COLOR_INHIBITORY, label='Inhibitory spikes ($r_I$)'),
    plt.Line2D([0], [0], color='green', linestyle=':', linewidth=2,
               label='Stimulus orientation (0°)')
]
ax_main.legend(handles=legend_elements, loc='upper right', frameon=True,
               fontsize=14)

# Panel de fases (sin flecha de persistencia)
ax_phase = fig2.add_subplot(gs2[1, 0])

ax_phase.axvspan(0, pre_end, alpha=1.0, color='#e0e0e0')
ax_phase.axvspan(pre_end, stim_end, alpha=1.0, color='#b0b0b0')
ax_phase.axvspan(stim_end, T_max, alpha=1.0, color='#e0e0e0')

ax_phase.text(PRE_DURATION / 2, 0.5, 'Pre-stimulus', ha='center', va='center',
              fontsize=16)
ax_phase.text(pre_end + STIM_DURATION / 2, 0.5, 'Stimulus (0°)', ha='center',
              va='center', fontsize=16)
ax_phase.text(stim_end + POST_DURATION / 2, 0.5, 'Post-stimulus', ha='center',
              va='center', fontsize=16)

ax_phase.set_xlim(0, T_max)
ax_phase.set_ylim(0, 1)
ax_phase.set_xticks([])
ax_phase.set_yticks([])
# Quitar bordes del panel de fases
for spine in ax_phase.spines.values():
    spine.set_visible(False)

# Ajustar margenes manualmente (NO usar tight_layout)
fig2.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08)

output_path2 = OUTPUT_DIR / 'fig_raster_three_phases.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nGuardada: {output_path2}")
plt.close()


# =============================================================================
# RESUMEN
# =============================================================================

print("\n" + "=" * 70)
print("FIGURAS GENERADAS")
print("=" * 70)

print(f"\n1. {output_path1.name}")
print("   - Panel A: Actividad espontanea (sin estimulo)")
print("   - Panel B: Estimulo constante centrado en 0°")
print("   - Muestra el contraste entre las dos condiciones")

print(f"\n2. {output_path2.name}")
print("   - Protocolo de tres fases: Pre → Stimulus → Post")
print(f"   - Pre ({PRE_DURATION}ms): Actividad espontanea dispersa")
print(f"   - Stimulus ({STIM_DURATION}ms): Actividad concentrada en 0°")
print(f"   - Post ({POST_DURATION}ms): Persistencia con decaimiento")
print(f"   - Tau de decaimiento: {tau_decay} ms")

print("\n" + "=" * 70)
print("COMPLETADO")
print("=" * 70)
