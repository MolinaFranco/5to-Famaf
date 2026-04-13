#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate MyFig1: Visualization of the 5 narrow stimuli used for simulations.

This script creates a figure showing the 5 narrow input stimuli (h_input)
used in the narrow simulation experiments, with:
- width_factor = 0.01 (very narrow)
- Amplitudes calibrated to match h_true heights from Echeveste

The figure shows:
- All 5 stimuli overlaid on a single plot for comparison
- Individual subplots for each contrast level
- Comparison with h_true (original Echeveste) with consistent Y-scale

References:
- Echeveste et al. (2020), Nature Neuroscience
"""

import sys
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from skneuromsi.generative import GSM
from skneuromsi.data import GSMDataLoader

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
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path(__file__).parent

# Stimulus parameters
WIDTH_FACTOR = 0.01
DOMINANT_IDX = 25  # ~0 degrees

# Contrast levels and calibrated amplitudes (5 levels now)
# Amplitudes were found via binary search to match h_true max values
# h_max_target are the REAL h_true max values (verified from original data)
# Colores más distinguibles para h_narrow
CONTRAST_CONFIG = [
    {'contrast': 0.0, 'label': 'Spontaneous', 'amplitude': 0.000629,
     'h_max_target': 0.019280, 'color': '#E63946'},  # Rojo brillante
    {'contrast': 0.125, 'label': 'Low', 'amplitude': 7.7341,
     'h_max_target': 1.110615, 'color': '#2A9D8F'},  # Verde azulado
    {'contrast': 0.25, 'label': 'Medium', 'amplitude': 15.4641,
     'h_max_target': 3.940635, 'color': '#E9C46A'},  # Amarillo dorado
    {'contrast': 0.5, 'label': 'High', 'amplitude': 30.9287,
     'h_max_target': 14.963133, 'color': '#F4A261'},  # Naranja
    {'contrast': 1.0, 'label': 'Very High', 'amplitude': 61.8572,
     'h_max_target': 58.947728, 'color': '#264653'},  # Azul oscuro
]

N_NEURONS = 50


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate figure showing the 5 narrow stimuli."""
    print("=" * 70)
    print("GENERATING MyFig1: Narrow Stimuli Visualization")
    print("=" * 70)

    # Load GSM with transformation parameters
    print("\nLoading GSM...")
    loader = GSMDataLoader()
    params = loader.load_transformation_parameters()

    gsm = GSM(
        use_pretrained=True,
        alpha_h=params['alpha_h'],
        beta_h=params['beta_h'],
        gamma_h=params['gamma_h']
    )

    print(f"Transformation parameters:")
    print(f"  alpha_h = {params['alpha_h']:.4f}")
    print(f"  beta_h = {params['beta_h']:.4f}")
    print(f"  gamma_h = {params['gamma_h']:.4f}")

    # Orientations
    orientations = np.linspace(-90, 90, N_NEURONS, endpoint=False)

    # Generate h_inputs for each contrast
    print(f"\nGenerating h_inputs (width_factor={WIDTH_FACTOR})...")
    h_inputs = []

    for config in CONTRAST_CONFIG:
        y = gsm.generate_bump_stimulus(
            dominant_orientation_idx=DOMINANT_IDX,
            amplitude=config['amplitude'],
            width_factor=WIDTH_FACTOR,
            normalize=False
        )
        x = gsm.generate_stimulus_from_y(y, contrast=1.0, add_noise=False)
        h = gsm.generate_h_input_efficient(x)

        h_inputs.append(h)
        print(f"  {config['label']:12s} (c={config['contrast']:.3f}): "
              f"h_max={h.max():.4f} (target: {config['h_max_target']:.4f})")

    # =========================================================================
    # FIGURE 1: All stimuli overlaid
    # =========================================================================
    print("\nCreating combined figure...")

    fig1 = plt.figure(figsize=(10, 6))
    ax = fig1.add_subplot(111)

    for i, (config, h) in enumerate(zip(CONTRAST_CONFIG, h_inputs)):
        ax.plot(orientations, h, color=config['color'], linewidth=2.5,
                label=f"{config['label']} (c={config['contrast']})")

    ax.set_xlabel('Preferred Orientation (°)')
    ax.set_ylabel('h (a.u.)')
    ax.set_title(f'Narrow Stimuli (width_factor={WIDTH_FACTOR})\n'
                 f'Amplitudes calibrated to match h_true heights')
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add vertical line at dominant orientation
    dominant_ori = orientations[DOMINANT_IDX]
    ax.axvline(dominant_ori, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig1_path = OUTPUT_DIR / 'myfig1_narrow_stimuli_combined.png'
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig1_path}")
    plt.close()

    # =========================================================================
    # FIGURE 2: Individual subplots (3x2 grid for 5 stimuli)
    # =========================================================================
    print("\nCreating individual subplots figure...")

    fig2 = plt.figure(figsize=(12, 14))
    fig2.suptitle(f'Narrow Stimuli (width_factor={WIDTH_FACTOR})',
                  fontsize=14, fontweight='bold')
    gs = GridSpec(3, 2, figure=fig2, hspace=0.35, wspace=0.25)

    # Global y-scale
    h_max_global = max(h.max() for h in h_inputs)

    for i, (config, h) in enumerate(zip(CONTRAST_CONFIG, h_inputs)):
        row = i // 2
        col = i % 2
        ax = fig2.add_subplot(gs[row, col])

        ax.plot(orientations, h, color=config['color'], linewidth=2.5)
        ax.fill_between(orientations, 0, h, alpha=0.3, color=config['color'])

        ax.set_xlabel('Preferred Orientation (°)')
        ax.set_ylabel('h (a.u.)')
        ax.set_title(f"{config['label']} Contrast (c={config['contrast']})\n"
                     f"h_max={h.max():.4f}")
        ax.set_xlim(-90, 90)
        ax.set_ylim(0, h_max_global * 1.1)
        ax.set_xticks([-90, -45, 0, 45, 90])
        ax.grid(True, alpha=0.3)

        # Add info box
        info_text = (f"A = {config['amplitude']:.2f}\n"
                     f"σ = {WIDTH_FACTOR * N_NEURONS:.1f}")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig2_path = OUTPUT_DIR / 'myfig1_narrow_stimuli_individual.png'
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig2_path}")
    plt.close()

    # =========================================================================
    # FIGURE 3: Comparison with h_true (original Echeveste)
    # Con escala Y consistente y colores más distinguibles
    # =========================================================================
    print("\nCreating comparison with h_true figure...")

    fig3 = plt.figure(figsize=(15, 12))
    fig3.suptitle('Narrow Stimuli vs Original h_true (Echeveste)',
                  fontsize=14, fontweight='bold')
    gs = GridSpec(3, 2, figure=fig3, hspace=0.35, wspace=0.25)

    # h_true indices: contraste 0->h_true_0, 0.125->h_true_1, etc.
    h_true_indices = [0, 1, 2, 3, 4]

    # Calcular escala Y global (máximo de todos los h_true y h_narrow)
    all_h_true = []
    for idx in h_true_indices:
        h_true_full = loader.load_h_vec_transformed(idx)
        all_h_true.append(h_true_full[:50])

    y_max_global = max(
        max(h.max() for h in h_inputs),
        max(h.max() for h in all_h_true)
    ) * 1.1

    for i, (config, h_narrow) in enumerate(zip(CONTRAST_CONFIG, h_inputs)):
        row = i // 2
        col = i % 2
        ax = fig3.add_subplot(gs[row, col])

        # Load h_true
        h_true = all_h_true[i]

        # Graficar h_true primero (negro, más grueso)
        ax.plot(orientations, h_true, 'k-', linewidth=3, alpha=0.8,
                label='h_true (original)')

        # Graficar h_narrow con color brillante y línea punteada gruesa
        ax.plot(orientations, h_narrow, color=config['color'], linewidth=2.5,
                linestyle='--', label=f'h_narrow (wf={WIDTH_FACTOR})')

        ax.set_xlabel('Preferred Orientation (°)')
        ax.set_ylabel('h (a.u.)')
        ax.set_title(f"{config['label']} Contrast (c={config['contrast']})")
        ax.set_xlim(-90, 90)
        ax.set_ylim(0, y_max_global)  # Escala Y consistente
        ax.set_xticks([-90, -45, 0, 45, 90])
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add max values
        info_text = (f"h_true max: {h_true.max():.4f}\n"
                     f"h_narrow max: {h_narrow.max():.4f}")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig3_path = OUTPUT_DIR / 'myfig1_narrow_vs_htrue.png'
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig3_path}")
    plt.close()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nStimulus configuration:")
    print(f"  width_factor: {WIDTH_FACTOR}")
    print(f"  dominant_idx: {DOMINANT_IDX} ({orientations[DOMINANT_IDX]:.1f}°)")
    print(f"  σ (sigma): {WIDTH_FACTOR * N_NEURONS:.1f} neurons")

    print(f"\nAmplitudes and heights:")
    for config, h in zip(CONTRAST_CONFIG, h_inputs):
        print(f"  {config['label']:12s}: A={config['amplitude']:8.4f}, "
              f"h_max={h.max():8.4f} "
              f"(target: {config['h_max_target']:.4f})")

    print(f"\nOutputs:")
    print(f"  {fig1_path}")
    print(f"  {fig2_path}")
    print(f"  {fig3_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
