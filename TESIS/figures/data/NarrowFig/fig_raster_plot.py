#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raster plot de actividad neuronal para la simulacion NarrowFig.

Este script genera un raster plot que muestra la actividad de disparo
de neuronas excitatorias (azul) e inhibitorias (rojo) durante el
protocolo de tres fases temporales.

Los firing rates continuos se convierten a spikes discretos usando
un proceso de Poisson, que es el metodo estandar en neurociencia
computacional.

Referencias:
- Echeveste et al. (2020), Nature Neuroscience
- Dayan & Abbott (2001), Theoretical Neuroscience, Cap. 1
"""

import sys
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Configuracion de matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.grid': False,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
})

# Semilla para reproducibilidad
np.random.seed(42)


# =============================================================================
# CONFIGURACION
# =============================================================================

# Directorios
SIMULATION_DIR = Path(__file__).parent.parent / 'simulations'
OUTPUT_DIR = Path(__file__).parent

# Parametros de las fases temporales (en ms)
PRE_DURATION = 250
STIM_DURATION = 500
POST_DURATION = 250
TOTAL_DURATION = PRE_DURATION + STIM_DURATION + POST_DURATION

# Colores para cada tipo de neurona
COLOR_EXCITATORY = '#1f77b4'  # Azul
COLOR_INHIBITORY = '#d62728'  # Rojo

# Parametros del raster plot
MARKER_SIZE = 3.0  # Tamano de los puntos (aumentado para mejor visibilidad)
MARKER_ALPHA = 0.85  # Transparencia
LINE_LENGTH = 0.8  # Longitud de las lineas en eventplot

print("=" * 70)
print("GENERACION DE RASTER PLOT")
print("=" * 70)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def load_simulation_data(contrast, suffix='_narrow_wf0.01'):
    """
    Cargar datos de simulacion con estimulos angostos.

    Parameters
    ----------
    contrast : float
        Nivel de contraste
    suffix : str
        Sufijo del directorio

    Returns
    -------
    dict
        Datos de simulacion incluyendo firing rates r_E y r_I
    """
    dir_name = f"contrast_{contrast:.3f}{suffix}"
    data_path = SIMULATION_DIR / dir_name / "simulation_data.npz"

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontro: {data_path}")

    data = np.load(data_path)

    return {
        'u_E': data['u_E'],
        'u_I': data['u_I'],
        'r_E': data['r_E'],
        'r_I': data['r_I'],
        'time': data['time'],
        'orientations': data['orientations'],
        'dt': float(data['dt']),
        'h_input': data['h_input'],
    }


def rates_to_spikes_poisson(rates, dt):
    """
    Convertir firing rates a spikes usando un proceso de Poisson.

    Para cada neurona y cada timestep, se genera un spike con
    probabilidad r(t) * dt, donde r(t) es el firing rate en Hz
    y dt esta en segundos.

    Parameters
    ----------
    rates : np.ndarray
        Firing rates de forma (time, neurons) en Hz
    dt : float
        Paso temporal en ms

    Returns
    -------
    spikes : np.ndarray
        Matriz booleana de spikes (time, neurons)

    References
    ----------
    Dayan & Abbott (2001), Theoretical Neuroscience, Ecuacion 1.35
    """
    # Convertir dt de ms a segundos
    dt_sec = dt / 1000.0

    # Probabilidad de spike en cada timestep
    # P(spike) = r * dt (aproximacion para dt pequeno)
    spike_prob = rates * dt_sec

    # Generar spikes aleatoriamente segun la probabilidad
    random_vals = np.random.random(rates.shape)
    spikes = random_vals < spike_prob

    return spikes


def create_three_phase_rates(data_spontaneous, data_stimulus, dt,
                              pre_dur, stim_dur, post_dur):
    """
    Crear datos de firing rates concatenados con tres fases temporales.

    Parameters
    ----------
    data_spontaneous : dict
        Datos de simulacion con contraste 0 (actividad espontanea)
    data_stimulus : dict
        Datos de simulacion con alto contraste (estimulo)
    dt : float
        Paso temporal en ms
    pre_dur, stim_dur, post_dur : float
        Duraciones de cada fase en ms

    Returns
    -------
    r_E_combined : np.ndarray
        Firing rates excitatorios (time, neurons)
    r_I_combined : np.ndarray
        Firing rates inhibitorios (time, neurons)
    time_combined : np.ndarray
        Vector de tiempo combinado
    """
    # Convertir duraciones a numero de muestras
    n_pre = int(pre_dur / dt)
    n_stim = int(stim_dur / dt)
    n_post = int(post_dur / dt)

    r_E_spont = data_spontaneous['r_E']
    r_I_spont = data_spontaneous['r_I']
    r_E_stim = data_stimulus['r_E']
    r_I_stim = data_stimulus['r_I']

    # Offset para evitar transitorios
    offset_stim = 500
    offset_post = n_pre + 500

    # Concatenar excitatorias
    r_E_combined = np.vstack([
        r_E_spont[:n_pre, :],
        r_E_stim[offset_stim:offset_stim + n_stim, :],
        r_E_spont[offset_post:offset_post + n_post, :]
    ])

    # Concatenar inhibitorias
    r_I_combined = np.vstack([
        r_I_spont[:n_pre, :],
        r_I_stim[offset_stim:offset_stim + n_stim, :],
        r_I_spont[offset_post:offset_post + n_post, :]
    ])

    # Vector de tiempo
    n_total = n_pre + n_stim + n_post
    time_combined = np.arange(n_total) * dt

    return r_E_combined, r_I_combined, time_combined


def extract_spike_times(spikes, time):
    """
    Extraer tiempos de spike para cada neurona.

    Parameters
    ----------
    spikes : np.ndarray
        Matriz booleana de spikes (time, neurons)
    time : np.ndarray
        Vector de tiempo

    Returns
    -------
    spike_times : list of np.ndarray
        Lista donde spike_times[i] contiene los tiempos de spike
        de la neurona i
    """
    n_neurons = spikes.shape[1]
    spike_times = []

    for neuron_idx in range(n_neurons):
        # Indices temporales donde hay spike
        spike_indices = np.where(spikes[:, neuron_idx])[0]
        # Convertir a tiempos
        spike_times.append(time[spike_indices])

    return spike_times


# =============================================================================
# CARGAR DATOS
# =============================================================================

print("\nCargando datos de simulacion...")

data_spontaneous = load_simulation_data(0.0)
data_stimulus = load_simulation_data(1.0)

dt = data_spontaneous['dt']
orientations = data_spontaneous['orientations']
n_neurons = len(orientations)

print(f"  Neuronas excitatorias: {n_neurons}")
print(f"  Neuronas inhibitorias: {n_neurons}")
print(f"  dt = {dt} ms")


# =============================================================================
# CREAR DATOS CON TRES FASES
# =============================================================================

print("\nCreando datos con tres fases temporales...")

r_E_combined, r_I_combined, time_combined = create_three_phase_rates(
    data_spontaneous, data_stimulus, dt,
    PRE_DURATION, STIM_DURATION, POST_DURATION
)

print(f"  Firing rates E: {r_E_combined.shape}")
print(f"  Firing rates I: {r_I_combined.shape}")
print(f"  Tiempo total: {time_combined[-1]:.1f} ms")


# =============================================================================
# CONVERTIR RATES A SPIKES
# =============================================================================

print("\nConvirtiendo firing rates a spikes (proceso de Poisson)...")

spikes_E = rates_to_spikes_poisson(r_E_combined, dt)
spikes_I = rates_to_spikes_poisson(r_I_combined, dt)

n_spikes_E = spikes_E.sum()
n_spikes_I = spikes_I.sum()

print(f"  Total spikes excitatorios: {n_spikes_E}")
print(f"  Total spikes inhibitorios: {n_spikes_I}")
print(f"  Tasa media E: {n_spikes_E / (n_neurons * TOTAL_DURATION / 1000):.1f} Hz")
print(f"  Tasa media I: {n_spikes_I / (n_neurons * TOTAL_DURATION / 1000):.1f} Hz")

# Extraer tiempos de spike
spike_times_E = extract_spike_times(spikes_E, time_combined)
spike_times_I = extract_spike_times(spikes_I, time_combined)


# =============================================================================
# GENERAR RASTER PLOT - VERSION 1: E e I separados (usando eventplot)
# =============================================================================

print("\nGenerando raster plot (version 1: E e I separados)...")

fig1 = plt.figure(figsize=(14, 10))
gs1 = GridSpec(3, 1, figure=fig1, hspace=0.15, height_ratios=[1, 1, 0.3])

# Limites de las fases
pre_end = PRE_DURATION
stim_end = PRE_DURATION + STIM_DURATION
T_max = time_combined[-1]

# -----------------------------------------------------------------------------
# Panel 1: Neuronas Excitatorias (usando eventplot)
# -----------------------------------------------------------------------------
ax_E = fig1.add_subplot(gs1[0, 0])

# eventplot requiere una lista de arrays de tiempos de spike
ax_E.eventplot(spike_times_E, lineoffsets=orientations, linelengths=LINE_LENGTH,
               colors=COLOR_EXCITATORY, alpha=MARKER_ALPHA, linewidths=1.2)

# Lineas de fase
ax_E.axvline(x=pre_end, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
ax_E.axvline(x=stim_end, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

ax_E.set_ylabel('Preferred Orientation (deg)')
ax_E.set_title('Excitatory Neurons (E)', fontsize=12, color=COLOR_EXCITATORY,
               fontweight='bold')
ax_E.set_ylim(orientations[0] - 5, orientations[-1] + 5)
ax_E.set_yticks([-90, -45, 0, 45, 90])
ax_E.set_xlim(0, T_max)
ax_E.set_xticks(np.arange(0, T_max + 1, 100))
ax_E.set_xticklabels([])

# Sombreado de fase de estimulo
ax_E.axvspan(pre_end, stim_end, alpha=0.12, color='orange', zorder=0)

# -----------------------------------------------------------------------------
# Panel 2: Neuronas Inhibitorias (usando eventplot)
# -----------------------------------------------------------------------------
ax_I = fig1.add_subplot(gs1[1, 0])

ax_I.eventplot(spike_times_I, lineoffsets=orientations, linelengths=LINE_LENGTH,
               colors=COLOR_INHIBITORY, alpha=MARKER_ALPHA, linewidths=1.2)

ax_I.axvline(x=pre_end, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
ax_I.axvline(x=stim_end, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

ax_I.set_ylabel('Preferred Orientation (deg)')
ax_I.set_title('Inhibitory Neurons (I)', fontsize=12, color=COLOR_INHIBITORY,
               fontweight='bold')
ax_I.set_ylim(orientations[0] - 5, orientations[-1] + 5)
ax_I.set_yticks([-90, -45, 0, 45, 90])
ax_I.set_xlim(0, T_max)
ax_I.set_xticks(np.arange(0, T_max + 1, 100))
ax_I.set_xticklabels([])

ax_I.axvspan(pre_end, stim_end, alpha=0.12, color='orange', zorder=0)

# -----------------------------------------------------------------------------
# Panel 3: Barra temporal con fases
# -----------------------------------------------------------------------------
ax_time = fig1.add_subplot(gs1[2, 0])

# Sombreado de fases
ax_time.axvspan(0, pre_end, alpha=0.3, color='gray', label='Pre-stimulus')
ax_time.axvspan(pre_end, stim_end, alpha=0.3, color='red', label='Stimulus')
ax_time.axvspan(stim_end, T_max, alpha=0.3, color='gray', label='Post-stimulus')

ax_time.set_xlim(0, T_max)
ax_time.set_ylim(0, 1)
ax_time.set_xlabel('Time (ms)')
ax_time.set_xticks(np.arange(0, T_max + 1, 100))
ax_time.set_yticks([])
ax_time.legend(loc='center', ncol=3, frameon=False)

# Etiquetas de fase
ax_time.text(PRE_DURATION / 2, 0.5, 'Pre', ha='center', va='center',
             fontsize=11, fontweight='bold')
ax_time.text(pre_end + STIM_DURATION / 2, 0.5, 'Stimulus', ha='center',
             va='center', fontsize=11, fontweight='bold')
ax_time.text(stim_end + POST_DURATION / 2, 0.5, 'Post', ha='center',
             va='center', fontsize=11, fontweight='bold')

fig1.suptitle('Raster Plot: Population Neural Activity', fontsize=14, y=0.98)
plt.tight_layout()

output_path1 = OUTPUT_DIR / 'fig_raster_separated.png'
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"  Guardada: {output_path1}")
plt.close()


# =============================================================================
# GENERAR RASTER PLOT - VERSION 2: E e I combinados
# =============================================================================

print("\nGenerando raster plot (version 2: E e I combinados)...")

fig2 = plt.figure(figsize=(14, 8))
gs2 = GridSpec(2, 1, figure=fig2, hspace=0.1, height_ratios=[1, 0.15])

ax_combined = fig2.add_subplot(gs2[0, 0])

# Organizar neuronas: E arriba (indices 50-99), I abajo (indices 0-49)
# Posiciones Y para cada neurona
offsets_E = np.arange(n_neurons) + n_neurons  # 50-99
offsets_I = np.arange(n_neurons)              # 0-49

# Usar eventplot para ambas poblaciones
ax_combined.eventplot(spike_times_E, lineoffsets=offsets_E,
                      linelengths=LINE_LENGTH, colors=COLOR_EXCITATORY,
                      alpha=MARKER_ALPHA, linewidths=1.2)
ax_combined.eventplot(spike_times_I, lineoffsets=offsets_I,
                      linelengths=LINE_LENGTH, colors=COLOR_INHIBITORY,
                      alpha=MARKER_ALPHA, linewidths=1.2)

# Linea separadora entre E e I
ax_combined.axhline(y=n_neurons - 0.5, color='black', linestyle='-',
                    linewidth=2, alpha=0.9)

# Lineas de fase
ax_combined.axvline(x=pre_end, color='gray', linestyle='--',
                    linewidth=1.5, alpha=0.8)
ax_combined.axvline(x=stim_end, color='gray', linestyle='--',
                    linewidth=1.5, alpha=0.8)

# Sombreado de fase de estimulo
ax_combined.axvspan(pre_end, stim_end, alpha=0.12, color='orange', zorder=0)

# Etiquetas de tipo de neurona en el margen izquierdo
ax_combined.text(-30, n_neurons + n_neurons / 2, 'E', ha='center', va='center',
                 fontsize=16, fontweight='bold', color=COLOR_EXCITATORY)
ax_combined.text(-30, n_neurons / 2, 'I', ha='center', va='center',
                 fontsize=16, fontweight='bold', color=COLOR_INHIBITORY)

ax_combined.set_ylabel('Neuron Index')
ax_combined.set_ylim(-2, 2 * n_neurons + 2)
ax_combined.set_xlim(0, T_max)
ax_combined.set_xticks(np.arange(0, T_max + 1, 100))
ax_combined.set_xticklabels([])

# Y-ticks personalizados
yticks = [0, n_neurons // 2, n_neurons, n_neurons + n_neurons // 2,
          2 * n_neurons]
yticklabels = ['0', '25', '50 (E/I)', '75', '100']
ax_combined.set_yticks(yticks)
ax_combined.set_yticklabels(yticklabels)

# Panel de tiempo
ax_time2 = fig2.add_subplot(gs2[1, 0])
ax_time2.axvspan(0, pre_end, alpha=0.3, color='gray')
ax_time2.axvspan(pre_end, stim_end, alpha=0.3, color='red')
ax_time2.axvspan(stim_end, T_max, alpha=0.3, color='gray')

ax_time2.text(PRE_DURATION / 2, 0.5, 'Pre', ha='center', va='center',
              fontsize=10, fontweight='bold')
ax_time2.text(pre_end + STIM_DURATION / 2, 0.5, 'Stimulus', ha='center',
              va='center', fontsize=10, fontweight='bold')
ax_time2.text(stim_end + POST_DURATION / 2, 0.5, 'Post', ha='center',
              va='center', fontsize=10, fontweight='bold')

ax_time2.set_xlim(0, T_max)
ax_time2.set_ylim(0, 1)
ax_time2.set_xlabel('Time (ms)')
ax_time2.set_xticks(np.arange(0, T_max + 1, 100))
ax_time2.set_yticks([])

# Leyenda
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLOR_EXCITATORY, label='Excitatory (E)'),
    Patch(facecolor=COLOR_INHIBITORY, label='Inhibitory (I)')
]
ax_combined.legend(handles=legend_elements, loc='upper right', frameon=True)

fig2.suptitle('Raster Plot: Combined Population Activity (100 neurons)',
              fontsize=14, y=0.98)
plt.tight_layout()

output_path2 = OUTPUT_DIR / 'fig_raster_combined.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"  Guardada: {output_path2}")
plt.close()


# =============================================================================
# GENERAR RASTER PLOT - VERSION 3: Ordenado por orientacion preferida
# =============================================================================

print("\nGenerando raster plot (version 3: ordenado por orientacion)...")

fig3 = plt.figure(figsize=(14, 10))
gs3 = GridSpec(2, 1, figure=fig3, hspace=0.15, height_ratios=[1, 0.15])

ax_orient = fig3.add_subplot(gs3[0, 0])

# Plot neuronas E e I intercaladas por orientacion usando eventplot
# E con offset positivo, I con offset negativo respecto a la orientacion
orientations_E = orientations + 0.8  # Offset leve arriba
orientations_I = orientations - 0.8  # Offset leve abajo

ax_orient.eventplot(spike_times_E, lineoffsets=orientations_E,
                    linelengths=LINE_LENGTH * 1.5, colors=COLOR_EXCITATORY,
                    alpha=MARKER_ALPHA, linewidths=1.2)
ax_orient.eventplot(spike_times_I, lineoffsets=orientations_I,
                    linelengths=LINE_LENGTH * 1.5, colors=COLOR_INHIBITORY,
                    alpha=MARKER_ALPHA, linewidths=1.2)

# Lineas de fase
ax_orient.axvline(x=pre_end, color='gray', linestyle='--',
                  linewidth=1, alpha=0.7)
ax_orient.axvline(x=stim_end, color='gray', linestyle='--',
                  linewidth=1, alpha=0.7)

# Sombreado
ax_orient.axvspan(pre_end, stim_end, alpha=0.08, color='red', zorder=0)

# Linea horizontal en orientacion 0 (estimulo)
ax_orient.axhline(y=0, color='green', linestyle=':', linewidth=1.5,
                  alpha=0.5, label='Stimulus orientation (0°)')

ax_orient.set_ylabel('Preferred Orientation (deg)')
ax_orient.set_ylim(orientations[0] - 5, orientations[-1] + 5)
ax_orient.set_yticks([-90, -45, 0, 45, 90])
ax_orient.set_xlim(0, T_max)
ax_orient.set_xticks(np.arange(0, T_max + 1, 100))
ax_orient.set_xticklabels([])

# Leyenda
legend_elements = [
    Patch(facecolor=COLOR_EXCITATORY, label='Excitatory (E)'),
    Patch(facecolor=COLOR_INHIBITORY, label='Inhibitory (I)'),
    plt.Line2D([0], [0], color='green', linestyle=':', linewidth=1.5,
               label='Stimulus orientation (0°)')
]
ax_orient.legend(handles=legend_elements, loc='upper right', frameon=True)

# Panel de tiempo
ax_time3 = fig3.add_subplot(gs3[1, 0])
ax_time3.axvspan(0, pre_end, alpha=0.3, color='gray')
ax_time3.axvspan(pre_end, stim_end, alpha=0.3, color='red')
ax_time3.axvspan(stim_end, T_max, alpha=0.3, color='gray')

ax_time3.text(PRE_DURATION / 2, 0.5, 'Pre', ha='center', va='center',
              fontsize=10, fontweight='bold')
ax_time3.text(pre_end + STIM_DURATION / 2, 0.5, 'Stimulus', ha='center',
              va='center', fontsize=10, fontweight='bold')
ax_time3.text(stim_end + POST_DURATION / 2, 0.5, 'Post', ha='center',
              va='center', fontsize=10, fontweight='bold')

ax_time3.set_xlim(0, T_max)
ax_time3.set_ylim(0, 1)
ax_time3.set_xlabel('Time (ms)')
ax_time3.set_xticks(np.arange(0, T_max + 1, 100))
ax_time3.set_yticks([])

fig3.suptitle('Raster Plot: Neurons Ordered by Preferred Orientation',
              fontsize=14, y=0.98)
plt.tight_layout()

output_path3 = OUTPUT_DIR / 'fig_raster_orientation.png'
plt.savefig(output_path3, dpi=300, bbox_inches='tight')
print(f"  Guardada: {output_path3}")
plt.close()


# =============================================================================
# RESUMEN
# =============================================================================

print("\n" + "=" * 70)
print("RASTER PLOTS GENERADOS")
print("=" * 70)

print(f"\n1. {output_path1.name}")
print("   - Paneles separados para E (azul) e I (rojo)")
print("   - Neuronas ordenadas por orientacion preferida")

print(f"\n2. {output_path2.name}")
print("   - Panel unico con E arriba e I abajo")
print("   - 100 neuronas totales (50 E + 50 I)")

print(f"\n3. {output_path3.name}")
print("   - E e I intercaladas por orientacion preferida")
print("   - Linea verde indica orientacion del estimulo (0 grados)")

print(f"\nEstadisticas:")
print(f"  Spikes excitatorios: {n_spikes_E}")
print(f"  Spikes inhibitorios: {n_spikes_I}")
print(f"  Ratio I/E: {n_spikes_I / n_spikes_E:.2f}")

print("\n" + "=" * 70)
print("COMPLETADO")
print("=" * 70)
