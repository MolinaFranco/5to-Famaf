#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate custom figures: SSN responses to stimuli with varying width_factor.

This script creates TWO separate figures, each with 4 panels (2 rows x 2 cols):
- Figure 1: width_factor = 0.15 and 0.10 (broad and medium tuning)
- Figure 2: width_factor = 0.05 and 0.01 (narrow and very narrow tuning)

Each figure has:
- Top row: 2 heatmaps showing SSN membrane potential (u_E) over time
- Bottom row: 2 input h curves showing the stimulus input for each simulation

The amplitudes are calibrated to maintain h_max = 58.95 (same as h_true_4 from
Echeveste) across all width_factors. This allows comparing network dynamics
with different input selectivity while keeping the peak drive constant.

The h_input is generated using GSM.generate_bump_stimulus() with:
- amplitude = calibrated per width_factor to maintain h_max ≈ 58.95
- dominant_orientation_idx = 25 (~0 degrees)
- normalize = False (raw bump, like Echeveste data files)

Then transformed to network input using GSM.generate_h_input_efficient().

References:
- Echeveste et al. (2020) Nature Neuroscience, Figure 2a
- parameters.md: transformation parameters (alpha_h, beta_h, gamma_h)
"""

import sys
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from skneuromsi.neural import Echeveste2020
from skneuromsi.generative import GSM
from skneuromsi.data import GSMDataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================

# Stimulus parameters
# Amplitudes calibradas para mantener h_max = 58.95 (igual que h_true_4)
# Estas amplitudes fueron encontradas por búsqueda binaria para cada width_factor
# de modo que el pico máximo de h sea idéntico al de Echeveste (h_true_4)
WIDTH_FACTORS = [0.15, 0.10, 0.05, 0.01]
AMPLITUDES = [6.00, 7.61, 13.26, 61.86]  # Calibradas para h_max ≈ 58.95

DOMINANT_IDX = 25  # ~0 degrees
CONTRAST = 1.0  # Contraste alto para ver bien el efecto

# Simulation parameters
SIMULATION_CONFIG = {
    'n_neurons_e': 50,
    'n_neurons_i': 50,
    'k': 0.3,
    'n': 2.0,
    'seed': 42,
    'simulation_time': 1000.0,  # ms
    'dt': 0.2,
    'burn_in_time': 500.0,  # ms
    'noise_level': 0.1,
}

# Output
OUTPUT_DIR = Path(__file__).parent

# Matplotlib configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.grid': False,
})


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate two figures with 4 panels each."""
    print("=" * 70)
    print("GENERATING MyFig2: SSN responses with varying width_factor")
    print("=" * 70)

    # Cargar GSM para generar h_inputs
    print("\nLoading GSM data loader...")
    loader = GSMDataLoader()
    params = loader.load_transformation_parameters()

    print(f"Transformation parameters:")
    print(f"  alpha_h = {params['alpha_h']:.6f}")
    print(f"  beta_h = {params['beta_h']:.6f}")
    print(f"  gamma_h = {params['gamma_h']:.6f}")

    # Crear GSM con parámetros exactos
    gsm = GSM(
        use_pretrained=True,
        alpha_h=params['alpha_h'],
        beta_h=params['beta_h'],
        gamma_h=params['gamma_h']
    )

    # Generar h_inputs para cada width_factor con amplitud escalada
    print("\nGenerating h_inputs (with scaled amplitudes)...")
    h_inputs = []

    for wf, amp in zip(WIDTH_FACTORS, AMPLITUDES):
        # Generar y (latent representation) con generate_bump_stimulus
        y = gsm.generate_bump_stimulus(
            dominant_orientation_idx=DOMINANT_IDX,
            amplitude=amp,
            width_factor=wf,
            normalize=False  # Como Echeveste
        )

        # Generar x (imagen) desde y
        x = gsm.generate_stimulus_from_y(y, contrast=CONTRAST, add_noise=False)

        # Generar h_input (transformado)
        h = gsm.generate_h_input_efficient(x)

        h_inputs.append(h)
        sigma = wf * gsm.n_orientations
        print(f"  wf={wf:.2f} (σ={sigma:.1f}), A={amp:.1f}: "
              f"h min={h.min():.2f}, max={h.max():.2f}")

    # Crear modelo SSN
    print("\nCreating SSN model...")
    ssn = Echeveste2020(
        N_E=SIMULATION_CONFIG['n_neurons_e'],
        N_I=SIMULATION_CONFIG['n_neurons_i'],
        k=SIMULATION_CONFIG['k'],
        n=SIMULATION_CONFIG['n'],
        seed=SIMULATION_CONFIG['seed']
    )

    # Cargar parámetros entrenados
    print("Loading trained parameters...")
    ssn.load_parameters()

    # Ejecutar simulaciones
    print("\nRunning simulations...")
    results = []

    for i, (wf, amp, h_input) in enumerate(zip(WIDTH_FACTORS, AMPLITUDES,
                                                h_inputs)):
        print(f"\n  Simulation {i+1}/4: width_factor={wf}, amplitude={amp:.1f}")

        result = ssn.run(
            noise_level=SIMULATION_CONFIG['noise_level'],
            simulation_time=SIMULATION_CONFIG['simulation_time'],
            burn_in_time=SIMULATION_CONFIG['burn_in_time'],
            use_three_phases=False,
            h_input=h_input  # Usar h_input personalizado
        )

        # Extraer datos
        df = result.get_modes()
        n_times = len(result.times_)
        u_E = df['excitatory_potential'].values.reshape(n_times, -1)
        time = result.times_ * SIMULATION_CONFIG['dt']

        results.append({
            'u_E': u_E,
            'time': time,
            'h_input': h_input,
            'width_factor': wf,
            'amplitude': amp,
        })

        print(f"    u_E: min={u_E.min():.2f}, max={u_E.max():.2f}, "
              f"mean={u_E.mean():.2f}")

    # ==========================================================================
    # CREAR FIGURAS
    # ==========================================================================
    print("\nCreating figures...")

    # Orientaciones
    orientations = np.linspace(-90, 90, SIMULATION_CONFIG['n_neurons_e'],
                               endpoint=False)

    # Determinar escalas globales (consistentes entre ambas figuras)
    # Para heatmaps (u_E): usar escala fija 0-10 como en Fig2 de Echeveste
    VMIN_U = 0.0
    VMAX_U = 10.0

    # Para h_inputs: determinar escala global
    h_min = min(r['h_input'].min() for r in results)
    h_max = max(r['h_input'].max() for r in results)
    h_margin = 0.05 * (h_max - h_min)
    VMIN_H = h_min - h_margin
    VMAX_H = h_max + h_margin

    T_max = results[0]['time'][-1]

    # -------------------------------------------------------------------------
    # FIGURA 1: width_factor = 0.15 y 0.10
    # -------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(14, 10))
    fig1.suptitle('SSN responses: variation in stimulus width',
                  fontsize=14, fontweight='bold')
    # Usar 3 columnas: 2 para heatmaps + 1 para colorbar
    gs1 = GridSpec(2, 3, figure=fig1, hspace=0.30, wspace=0.15,
                   height_ratios=[1.2, 1], width_ratios=[1, 1, 0.05], top=0.93)

    # Fila superior: heatmaps
    axes_heatmap1 = []
    for i, r in enumerate(results[:2]):
        ax = fig1.add_subplot(gs1[0, i])
        axes_heatmap1.append(ax)

        im = ax.imshow(
            r['u_E'].T,
            aspect='auto',
            cmap='coolwarm',
            extent=[0, T_max, orientations[0], orientations[-1]],
            origin='lower',
            vmin=VMIN_U,
            vmax=VMAX_U
        )

        ax.set_ylim(orientations[0], orientations[-1])
        ax.set_xlim(0, T_max)
        ax.set_xticks(np.arange(0, T_max + 1, 200))
        ax.set_xlabel('Time (ms)')

        if i == 0:
            ax.set_yticks([-90, -45, 0, 45, 90])
            ax.set_ylabel('Preferred Orientation (°)')
        else:
            ax.set_yticks([])  # Sin ticks en el eje Y para el gráfico derecho

    # Colorbar en columna dedicada (más delgado y a la derecha)
    cax1 = fig1.add_subplot(gs1[0, 2])
    cbar1 = fig1.colorbar(im, cax=cax1, label='$u_E$ (a.u.)')

    # Fila inferior: curvas de h_input
    for i, r in enumerate(results[:2]):
        ax = fig1.add_subplot(gs1[1, i])

        # Agregar "original" solo para wf=0.15
        if r['width_factor'] == 0.15:
            label = f'width_factor={r["width_factor"]:.2f} (original)'
        else:
            label = f'width_factor={r["width_factor"]:.2f}'
        ax.plot(orientations, r['h_input'], 'k-', linewidth=2.5, label=label)
        ax.fill_between(orientations, 0, r['h_input'], alpha=0.3, color='C0')

        ax.set_xlim(-90, 90)
        ax.set_ylim(VMIN_H, VMAX_H)
        ax.set_xticks([-90, -45, 0, 45, 90])

        ax.set_xlabel('Preferred Orientation (°)')
        if i == 0:
            ax.set_ylabel('h (a.u.)')

        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Guardar figura 1
    output_path1 = OUTPUT_DIR / 'myfig2_broad_tuning.png'
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure 1 saved: {output_path1}")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURA 2: width_factor = 0.05 y 0.01
    # -------------------------------------------------------------------------
    fig2 = plt.figure(figsize=(14, 10))
    fig2.suptitle('SSN responses: variation in stimulus width',
                  fontsize=14, fontweight='bold')
    # Usar 3 columnas: 2 para heatmaps + 1 para colorbar
    gs2 = GridSpec(2, 3, figure=fig2, hspace=0.30, wspace=0.15,
                   height_ratios=[1.2, 1], width_ratios=[1, 1, 0.05], top=0.93)

    # Fila superior: heatmaps
    axes_heatmap2 = []
    for i, r in enumerate(results[2:]):
        ax = fig2.add_subplot(gs2[0, i])
        axes_heatmap2.append(ax)

        im = ax.imshow(
            r['u_E'].T,
            aspect='auto',
            cmap='coolwarm',
            extent=[0, T_max, orientations[0], orientations[-1]],
            origin='lower',
            vmin=VMIN_U,
            vmax=VMAX_U
        )

        ax.set_ylim(orientations[0], orientations[-1])
        ax.set_xlim(0, T_max)
        ax.set_xticks(np.arange(0, T_max + 1, 200))
        ax.set_xlabel('Time (ms)')

        if i == 0:
            ax.set_yticks([-90, -45, 0, 45, 90])
            ax.set_ylabel('Preferred Orientation (°)')
        else:
            ax.set_yticks([])  # Sin ticks en el eje Y para el gráfico derecho

    # Colorbar en columna dedicada (más delgado y a la derecha)
    cax2 = fig2.add_subplot(gs2[0, 2])
    cbar2 = fig2.colorbar(im, cax=cax2, label='$u_E$ (a.u.)')

    # Fila inferior: curvas de h_input
    for i, r in enumerate(results[2:]):
        ax = fig2.add_subplot(gs2[1, i])

        label = f'width_factor={r["width_factor"]:.2f}'
        ax.plot(orientations, r['h_input'], 'k-', linewidth=2.5, label=label)
        ax.fill_between(orientations, 0, r['h_input'], alpha=0.3, color='C0')

        ax.set_xlim(-90, 90)
        ax.set_ylim(VMIN_H, VMAX_H)
        ax.set_xticks([-90, -45, 0, 45, 90])

        ax.set_xlabel('Preferred Orientation (°)')
        if i == 0:
            ax.set_ylabel('h (a.u.)')

        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Guardar figura 2
    output_path2 = OUTPUT_DIR / 'myfig2_narrow_tuning.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Figure 2 saved: {output_path2}")
    plt.close()

    # ==========================================================================
    # RESUMEN
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nGenerated 2 figures with {len(WIDTH_FACTORS)} width_factor values:")
    print("\nFigure 1 (Broad tuning):")
    for r in results[:2]:
        sigma = r['width_factor'] * SIMULATION_CONFIG['n_neurons_e']
        print(f"  σ={sigma:.1f}: wf={r['width_factor']:.2f}, "
              f"A={r['amplitude']:.1f}, h_max={r['h_input'].max():.1f}")
    print("\nFigure 2 (Narrow tuning):")
    for r in results[2:]:
        sigma = r['width_factor'] * SIMULATION_CONFIG['n_neurons_e']
        print(f"  σ={sigma:.1f}: wf={r['width_factor']:.2f}, "
              f"A={r['amplitude']:.1f}, h_max={r['h_input'].max():.1f}")
    print(f"\nStimulus parameters:")
    print(f"  Dominant index: {DOMINANT_IDX} "
          f"(~{-90 + DOMINANT_IDX * 180/49:.1f}°)")
    print(f"  Contrast: {CONTRAST}")
    print(f"  Amplitudes: calibrated to maintain h_max ≈ 58.95")
    print(f"\nSimulation parameters:")
    print(f"  Time: {SIMULATION_CONFIG['simulation_time']} ms")
    print(f"  Burn-in: {SIMULATION_CONFIG['burn_in_time']} ms")
    print(f"  Noise level: {SIMULATION_CONFIG['noise_level']}")
    print(f"\nOutputs:")
    print(f"  {output_path1}")
    print(f"  {output_path2}")
    print("=" * 70)


if __name__ == "__main__":
    main()
