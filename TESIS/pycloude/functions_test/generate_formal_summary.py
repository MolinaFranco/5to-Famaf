#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate formal summary figure for paper.

Genera una figura con tabla resumen en tonos de grises,
formato formal para publicacion academica.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

OUTPUT_DIR = Path(__file__).parent / 'validation_outputs'


def create_formal_summary_figure(output_path):
    """Genera figura formal con tabla resumen en escala de grises."""

    # Datos de las validaciones (copiados del resultado)
    data = [
        # Categoria 1: Distribucion Normal
        {'category': '1. Distribucion Normal', 'function': 'f1 (PDF)',
         'original': 'methods.py:13', 'ours': '_gsm.py', 'max_error': 0.0},
        {'category': '1. Distribucion Normal', 'function': 'f2_exact (CDF)',
         'original': 'methods.py:16', 'ours': '_gsm.py', 'max_error': 0.0},

        # Categoria 2: Activacion Neuronal
        {'category': '2. Activacion Neuronal', 'function': 'get_r (activacion)',
         'original': 'methods.py:43', 'ours': '_echeveste2020.py:63',
         'max_error': 0.0},
        {'category': '2. Activacion Neuronal', 'function': 'get_r_prime (derivada)',
         'original': 'methods.py:51', 'ours': '_echeveste2020.py',
         'max_error': 0.0},

        # Categoria 3: Momentos
        {'category': '3. Momentos (nu, gamma)', 'function': 'nu_recursive (n=2)',
         'original': 'methods.py:98', 'ours': '_echeveste2020.py',
         'max_error': 7.11e-15},
        {'category': '3. Momentos (nu, gamma)', 'function': 'get_gamma',
         'original': 'methods.py:146', 'ours': '_echeveste2020.py',
         'max_error': 0.0},
        {'category': '3. Momentos (nu, gamma)', 'function': 'nu_recursive (n=0)',
         'original': 'methods.py:99', 'ours': 'direct', 'max_error': 0.0},
        {'category': '3. Momentos (nu, gamma)', 'function': 'nu_recursive (n=1)',
         'original': 'methods.py:101', 'ours': 'direct', 'max_error': 0.0},

        # Categoria 4: Evolucion Temporal
        {'category': '4. Evolucion Temporal', 'function': 'new_u (Euler step)',
         'original': 'methods.py:289', 'ours': '_echeveste2020.py',
         'max_error': 0.0},
        {'category': '4. Evolucion Temporal', 'function': 'mu_dot (mean deriv)',
         'original': 'methods.py:322', 'ours': '_echeveste2020.py',
         'max_error': 0.0},
        {'category': '4. Evolucion Temporal',
         'function': 'u_dot_det (deterministic)',
         'original': 'methods.py:292', 'ours': '_echeveste2020.py',
         'max_error': 0.0},

        # Categoria 5: Conectividad Parametrica
        {'category': '5. Conectividad Parametrica',
         'function': 'parametric_kernel (Eq.10)',
         'original': 'Eq.10 paper', 'ours': '_echeveste2020.py:1404',
         'max_error': 0.0},
        {'category': '5. Conectividad Parametrica',
         'function': 'get_J (Jacobian)',
         'original': 'methods.py:128', 'ours': '_echeveste2020.py',
         'max_error': 0.0},

        # Categoria 6: Ruido O-U
        {'category': '6. Ruido (O-U Process)', 'function': 'eps_1 (O-U decay)',
         'original': 'methods.py:390', 'ours': '_echeveste2020.py',
         'max_error': 1.11e-16},
        {'category': '6. Ruido (O-U Process)', 'function': 'eps_2 (O-U noise)',
         'original': 'methods.py:391', 'ours': '_echeveste2020.py',
         'max_error': 1.11e-16},
        {'category': '6. Ruido (O-U Process)', 'function': 'O-U update step',
         'original': 'methods.py:396', 'ours': '_echeveste2020.py',
         'max_error': 0.0},

        # Categoria 7: GSM Posterior
        {'category': '7. GSM (Posterior)', 'function': 'get_Sigma_z (post cov)',
         'original': 'GSM.py:111', 'ours': '_gsm.py:789', 'max_error': 0.0},
        {'category': '7. GSM (Posterior)', 'function': 'get_mu_z (post mean)',
         'original': 'GSM.py:107', 'ours': '_gsm.py:789', 'max_error': 0.0},
        {'category': '7. GSM (Posterior)', 'function': 'get_x (stimulus)',
         'original': 'GSM.py:95', 'ours': '_gsm.py:303', 'max_error': 0.0},
        {'category': '7. GSM (Posterior)', 'function': 'P_z (gamma dist)',
         'original': 'GSM.py:204', 'ours': '_gsm.py', 'max_error': 0.0},

        # Categoria 8: Funciones Auxiliares
        {'category': '8. Funciones Auxiliares', 'function': 'sqr_norm',
         'original': 'methods.py:20', 'ours': 'direct', 'max_error': 0.0},
        {'category': '8. Funciones Auxiliares', 'function': 'rel_error',
         'original': 'methods.py:1116', 'ours': 'direct', 'max_error': 0.0},
        {'category': '8. Funciones Auxiliares', 'function': 'rescale',
         'original': 'methods.py:24', 'ours': 'direct', 'max_error': 0.0},
        {'category': '8. Funciones Auxiliares', 'function': 'rel_dist',
         'original': 'methods.py:27', 'ours': 'direct', 'max_error': 0.0},
        {'category': '8. Funciones Auxiliares', 'function': 'Mat_from_Chol',
         'original': 'methods.py:269', 'ours': 'direct', 'max_error': 0.0},
    ]

    # Crear figura
    fig = plt.figure(figsize=(14, 12))

    # Titulo sin fecha
    fig.suptitle(
        'Validation Summary: Implementation vs Original Code\n'
        '(N = 10,000 random samples per function)',
        fontsize=14, fontweight='bold', y=0.98
    )

    # Crear ejes para la tabla
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Columnas (sin N y Status)
    col_labels = ['Category', 'Function', 'Original', 'Implementation',
                  'Max Error']

    # Preparar datos de la tabla
    cell_text = []
    cell_colors = []

    # Colores en escala de grises alternando por categoria
    gray_light = '#f0f0f0'  # Gris muy claro
    gray_medium = '#e0e0e0'  # Gris claro
    white = '#ffffff'

    current_cat = None
    use_gray = False

    for item in data:
        # Alternar color por categoria
        if item['category'] != current_cat:
            current_cat = item['category']
            use_gray = not use_gray

        if use_gray:
            row_color = [gray_light] * 5
        else:
            row_color = [white] * 5

        # Formatear error
        if item['max_error'] == 0:
            error_str = '0.00'
        elif item['max_error'] < 1e-14:
            error_str = f"{item['max_error']:.2e}"
        else:
            error_str = f"{item['max_error']:.2e}"

        cell_text.append([
            item['category'],  # Siempre mostrar categoria
            item['function'],
            item['original'],
            item['ours'],
            error_str
        ])
        cell_colors.append(row_color)

    # Crear tabla
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=['#404040'] * 5,  # Gris oscuro para encabezado
        cellLoc='left',
        loc='center',
        bbox=[0.02, 0.08, 0.96, 0.88]
    )

    # Estilo de tabla
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Estilo de encabezados
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#404040')

    # Ajustar anchos de columna
    col_widths = [0.22, 0.22, 0.18, 0.22, 0.12]
    for i, width in enumerate(col_widths):
        for j in range(len(cell_text) + 1):
            table[(j, i)].set_width(width)

    # Centrar columna de error
    for j in range(len(cell_text) + 1):
        table[(j, 4)].set_text_props(ha='center')

    # Agregar lineas horizontales entre categorias
    # (esto se hace con el color alternado)

    # Nota al pie
    fig.text(
        0.5, 0.02,
        'All 25 functions passed validation. Maximum errors are within '
        'floating-point precision (~10$^{-15}$).',
        ha='center', fontsize=10, style='italic', color='#404040'
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Figura guardada en: {output_path}")


def main():
    """Genera la figura formal."""
    output_path = OUTPUT_DIR / 'comprehensive_validation_formal.png'
    create_formal_summary_figure(output_path)


if __name__ == '__main__':
    main()
