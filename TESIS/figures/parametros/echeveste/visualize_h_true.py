#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualización de todos los archivos h_true del GSM.

Este script visualiza los 5 niveles de contraste disponibles (h_true_0 al h_true_4)
con las MISMAS escalas en el eje Y para facilitar la comparación visual.

Siguiendo las convenciones del CLAUDE.md:
- Gráficos con mismas unidades deben tener escalas idénticas
- Minimizar barras de colores redundantes
- Ejes temporales/espaciales alineados
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Importar GSMDataLoader
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'scikit-neuromsi'))

from skneuromsi.data import GSMDataLoader


def main():
    """Visualizar todos los h_true del GSM."""

    # Cargar datos
    loader = GSMDataLoader()

    # Información de contrastes
    # Ref: parameters.md y div_norm_data_GSM.py
    contrasts_info = [
        {'idx': 0, 'label': 'Spontaneous (z=0.0)'},
        {'idx': 1, 'label': 'Low (z≈0.125)'},
        {'idx': 2, 'label': 'Medium (z≈0.25)'},
        {'idx': 3, 'label': 'High (z≈0.5)'},
        {'idx': 4, 'label': 'Very High (z≈1.0)'},
    ]

    # Cargar todos los h_true
    h_data = []
    for info in contrasts_info:
        h_full = loader.load_h_vec_transformed(info['idx'])
        # Solo tomar neuronas excitatorias (primeras 50)
        h_exc = h_full[:50]
        h_data.append(h_exc)
        print(f"h_true_{info['idx']}: min={h_exc.min():.4f}, "
              f"max={h_exc.max():.4f}, mean={h_exc.mean():.4f}")

    # Encontrar rango global para eje Y (MISMO para todos los gráficos)
    y_min = min(h.min() for h in h_data)
    y_max = max(h.max() for h in h_data)
    y_margin = 0.05 * (y_max - y_min)
    y_lims = (y_min - y_margin, y_max + y_margin)

    print(f"\nRango global Y: [{y_lims[0]:.4f}, {y_lims[1]:.4f}]")

    # Crear figura vertical (5 filas x 1 columna)
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    fig.suptitle('SSN Input h - preloaded',
                  fontsize=14, fontweight='bold')

    for ax, info, h in zip(axes, contrasts_info, h_data):
        orientations = np.linspace(-90, 90, 50)

        ax.plot(orientations, h, 'o-', linewidth=2, markersize=5,
                color='C0', alpha=0.8)
        ax.set_ylabel('h (a.u.)', fontsize=11)
        ax.set_title(info['label'], fontsize=11, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_lims)  # MISMO rango Y

        # Estadísticas
        stats_text = (f"min: {h.min():.3f} | max: {h.max():.3f} | "
                     f"mean: {h.mean():.3f} | std: {h.std():.3f}")
        ax.text(0.98, 0.95, stats_text,
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    axes[-1].set_xlabel('Orientation (degrees)', fontsize=11)
    plt.tight_layout()

    # Guardar figura
    output_dir = Path(__file__).parent
    output_path = output_dir / 'h_true_all_contrasts.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figura guardada en: {output_path}")


if __name__ == '__main__':
    main()
