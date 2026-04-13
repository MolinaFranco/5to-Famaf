#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparación de h_true precomputed vs h generado por GSM.

Este script verifica que nuestro GSM genera los mismos valores de h_true
que tenemos precargados, usando los estímulos x guardados y los
parámetros de transformación EXACTOS.

Genera una figura con 4 subplots, cada uno mostrando:
- h_true precomputed (línea azul): valores precargados de archivos
- GSM loaded stimulus (línea roja): h generado desde x cargado del archivo
- GSM from y_latent (línea verde): h desde x generado con y latente recuperado
- GSM bump (línea morada): h desde y generado con generate_bump_stimulus(A, σ)

IMPORTANTE: Los parámetros de transformación deben ser los valores
EXACTOS de Echeveste, no los redondeados de parameters.md:
- α_h = 1.955517265245507 (no 1.96)
- β_h = 0.103232893380231 (no 0.10)
- γ_h = 2.034269617565710 (no 2.03)

References:
- Echeveste et al. (2020) Supplementary Table S1
- parameters.md: Parámetros de transformación (α_h, β_h, γ_h)
- Archivos: input_scaling, input_baseline, input_nl_pow
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Agregar scikit-neuromsi al path
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent / 'scikit-neuromsi')
)

from skneuromsi.data import GSMDataLoader
from skneuromsi.generative import GSM


def main():
    """Comparar h_true precomputed vs h generado por GSM."""

    # Cargar datos
    loader = GSMDataLoader()

    # Cargar parámetros EXACTOS de transformación
    params = loader.load_transformation_parameters()
    alpha_h = params['alpha_h']
    beta_h = params['beta_h']
    gamma_h = params['gamma_h']

    print("Parámetros de transformación EXACTOS:")
    print(f"  α_h = {alpha_h:.15f}")
    print(f"  β_h = {beta_h:.15f}")
    print(f"  γ_h = {gamma_h:.15f}")

    # Crear instancia GSM con parámetros EXACTOS de Echeveste
    gsm = GSM(
        use_pretrained=True,
        alpha_h=alpha_h,
        beta_h=beta_h,
        gamma_h=gamma_h
    )

    # Cargar niveles de contraste exactos
    z_array = loader.load_contrast_levels()

    # Información de contrastes a comparar (usamos 4: índices 1, 2, 3, 4)
    # El índice 0 es actividad espontánea (z=0), así que empezamos desde 1
    contrasts_info = [
        {'idx': 1, 'z': z_array[1], 'label': f'Low contrast (z={z_array[1]:.3f})'},
        {'idx': 2, 'z': z_array[2], 'label': f'Medium contrast (z={z_array[2]:.3f})'},
        {'idx': 3, 'z': z_array[3], 'label': f'High contrast (z={z_array[3]:.3f})'},
        {'idx': 4, 'z': z_array[4], 'label': f'Very High contrast (z={z_array[4]:.3f})'},
    ]

    # Recuperar el y latente desde los estímulos guardados
    # Los x guardados son el mismo estímulo base escalado por contraste:
    # x_i = z_i * A @ y
    # Entonces podemos recuperar y desde x_4 (z=1.0)
    x4, _ = loader.load_stimulus(4)
    Ay = x4 / z_array[4]  # A @ y = x / z
    y_latent = np.linalg.lstsq(gsm.A, Ay, rcond=None)[0]

    # Encontrar la orientación dominante del y latente recuperado
    dominant_idx = np.argmax(y_latent)
    dominant_deg = -90 + dominant_idx * 180 / 49

    print(f"\ny latente recuperado (desde x guardado):")
    print(f"  argmax: {dominant_idx} (orientación ~{dominant_deg:.1f}°)")

    # =========================================================================
    # GENERAR y DESDE CERO con generate_bump_stimulus
    # =========================================================================
    # El y de Echeveste es un bump gaussiano SIN CENTRAR (normalize=False):
    # - amplitude = 6.0 (altura del bump)
    # - width_factor = 0.15 (sigma = 0.15 * 50 = 7.5)
    # - dominant_orientation_idx = 25 (orientación central ~1.8°)
    #
    # Parámetros configurables:
    AMPLITUDE = 6.0
    WIDTH_FACTOR = 0.15

    y_from_scratch = gsm.generate_bump_stimulus(
        dominant_orientation_idx=dominant_idx,
        amplitude=AMPLITUDE,
        width_factor=WIDTH_FACTOR,
        normalize=False  # Sin centrar, como los datos de Echeveste
    )

    sigma = WIDTH_FACTOR * gsm.n_orientations
    print(f"\ny generado desde cero (generate_bump_stimulus, normalize=False):")
    print(f"  center_idx: {dominant_idx} (orientación ~{dominant_deg:.1f}°)")
    print(f"  amplitude: {AMPLITUDE}, width_factor: {WIDTH_FACTOR} (sigma={sigma})")
    print(f"  min={y_from_scratch.min():.4f}, max={y_from_scratch.max():.4f}")
    print(f"  mean={y_from_scratch.mean():.6f}")

    # Comparar con y_latent recuperado
    diff_y = np.abs(y_latent - y_from_scratch)
    print(f"  Max |diff| vs y_latent: {diff_y.max():.6f}")

    # Configurar semilla para reproducibilidad de estímulos generados desde cero
    gsm.set_random(42)

    # Crear figura 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    fig.suptitle(
        'Comparación: h_true precomputed vs h generado por GSM\n'
        f'Parámetros EXACTOS: α_h={gsm.alpha_h:.6f}, β_h={gsm.beta_h:.6f}, '
        f'γ_h={gsm.gamma_h:.6f}',
        fontsize=14, fontweight='bold'
    )

    # Eje de orientaciones
    orientations = np.linspace(-90, 90, 50)

    # Para determinar el rango Y global
    all_h_values = []

    # Almacenar datos para graficar después de conocer el rango
    plot_data = []

    for i, info in enumerate(contrasts_info):
        idx = info['idx']
        contrast = info['z']

        # Cargar h_true precomputed (ya tiene transformación aplicada)
        h_true_full = loader.load_h_vec_transformed(idx)
        h_true_exc = h_true_full[:50]  # Solo neuronas excitatorias

        # Cargar estímulo x correspondiente
        x_loaded, h_raw = loader.load_stimulus(idx)

        # Generar h usando el GSM con x cargado del archivo
        h_gsm_loaded = gsm.generate_h_input_efficient(x_loaded)

        # Generar estímulo con el GSM usando el y latente recuperado
        # Esto debería dar exactamente el mismo resultado que x_loaded
        x_from_y = gsm.generate_stimulus_from_y(
            y_latent, contrast=contrast, add_noise=False
        )
        h_gsm_from_y = gsm.generate_h_input_efficient(x_from_y)

        # Generar estímulo COMPLETAMENTE desde cero con el GSM
        # Usamos y_from_scratch generado con generate_bump_stimulus
        # (NO el y_latent recuperado del archivo)
        x_scratch = gsm.generate_stimulus_from_y(
            y_from_scratch, contrast=contrast, add_noise=False
        )
        h_gsm_scratch = gsm.generate_h_input_efficient(x_scratch)

        # Guardar para determinar rango global
        all_h_values.extend(h_true_exc)
        all_h_values.extend(h_gsm_loaded)
        all_h_values.extend(h_gsm_from_y)
        all_h_values.extend(h_gsm_scratch)

        # Calcular diferencia entre h_true y h_gsm_loaded (deben ser iguales)
        diff_loaded = h_true_exc - h_gsm_loaded
        max_abs_diff_loaded = np.max(np.abs(diff_loaded))

        # Calcular diferencia entre h_true y h_gsm_from_y (también deben ser iguales)
        diff_from_y = h_true_exc - h_gsm_from_y
        max_abs_diff_from_y = np.max(np.abs(diff_from_y))

        # Almacenar datos
        plot_data.append({
            'info': info,
            'h_true_exc': h_true_exc,
            'h_gsm_loaded': h_gsm_loaded,
            'h_gsm_from_y': h_gsm_from_y,
            'h_gsm_scratch': h_gsm_scratch,
            'max_abs_diff_loaded': max_abs_diff_loaded,
            'max_abs_diff_from_y': max_abs_diff_from_y,
        })

        print(f"\nContraste {idx} ({info['label']}):")
        print(f"  h_true:           min={h_true_exc.min():.4f}, "
              f"max={h_true_exc.max():.4f}, mean={h_true_exc.mean():.4f}")
        print(f"  h_gsm (loaded):   min={h_gsm_loaded.min():.4f}, "
              f"max={h_gsm_loaded.max():.4f}, mean={h_gsm_loaded.mean():.4f}")
        print(f"  h_gsm (from y):   min={h_gsm_from_y.min():.4f}, "
              f"max={h_gsm_from_y.max():.4f}, mean={h_gsm_from_y.mean():.4f}")
        print(f"  h_gsm (bump):     min={h_gsm_scratch.min():.4f}, "
              f"max={h_gsm_scratch.max():.4f}, mean={h_gsm_scratch.mean():.4f}")
        print(f"  h_true vs loaded  - Max |diff|: {max_abs_diff_loaded:.2e}")
        print(f"  h_true vs from_y  - Max |diff|: {max_abs_diff_from_y:.2e}")

    # Determinar rango Y global
    y_min = min(all_h_values)
    y_max = max(all_h_values)
    y_margin = 0.05 * (y_max - y_min)
    y_lims = (y_min - y_margin, y_max + y_margin)

    # Ahora graficar con escalas consistentes
    for i, data in enumerate(plot_data):
        ax = axes[i]
        info = data['info']

        # Plot h_true precomputed (azul)
        ax.plot(
            orientations, data['h_true_exc'],
            'o-', linewidth=2, markersize=4,
            color='C0', alpha=0.8,
            label='h_true (precomputed)'
        )

        # Plot h generado por GSM desde x cargado (rojo)
        ax.plot(
            orientations, data['h_gsm_loaded'],
            's--', linewidth=2, markersize=3,
            color='C3', alpha=0.8,
            label='GSM loaded stimulus'
        )

        # Plot h generado por GSM desde y latente recuperado (verde)
        ax.plot(
            orientations, data['h_gsm_from_y'],
            '^:', linewidth=1.5, markersize=3,
            color='C2', alpha=0.7,
            label='GSM from y_latent'
        )

        # Plot h generado con generate_bump_stimulus (morado)
        ax.plot(
            orientations, data['h_gsm_scratch'],
            'd-.', linewidth=1.5, markersize=3,
            color='C4', alpha=0.7,
            label=f'GSM bump (A={AMPLITUDE}, σ={WIDTH_FACTOR})'
        )

        ax.set_ylabel('h (a.u.)', fontsize=11)
        ax.set_title(info['label'], fontsize=11, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_lims)  # Rango Y consistente
        # Solo mostrar leyenda en el último subplot
        if i == len(plot_data) - 1:
            ax.legend(loc='upper right', fontsize=7)

        # Estadísticas de comparación
        stats_text = (
            f"vs loaded: {data['max_abs_diff_loaded']:.2e}\n"
            f"vs from_y: {data['max_abs_diff_from_y']:.2e}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
        )

    # Eje X solo en la fila inferior
    for ax in axes[2:]:
        ax.set_xlabel('Orientation (degrees)', fontsize=11)

    plt.tight_layout()

    # Guardar figura
    output_dir = Path(__file__).parent
    output_path = output_dir / 'compare_h_true_vs_gsm.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figura guardada en: {output_path}")

    # Verificación final
    print("\n" + "="*60)
    print("VERIFICACIÓN FINAL:")
    print("="*60)

    all_match = True
    for data in plot_data:
        # Verificar loaded
        if data['max_abs_diff_loaded'] < 1e-10:
            status_loaded = "✓ EXACTO"
        else:
            status_loaded = "✗ DIFERENTE"
            all_match = False

        # Verificar from_y
        if data['max_abs_diff_from_y'] < 1e-10:
            status_from_y = "✓ EXACTO"
        else:
            status_from_y = "✗ DIFERENTE"
            all_match = False

        print(f"{data['info']['label']}:")
        print(f"  vs loaded:  {status_loaded} "
              f"(max diff: {data['max_abs_diff_loaded']:.2e})")
        print(f"  vs from_y:  {status_from_y} "
              f"(max diff: {data['max_abs_diff_from_y']:.2e})")

    if all_match:
        print("\n✓ El GSM genera correctamente los h_true!")
        print("  - Desde x cargado del archivo (GSM loaded stimulus)")
        print("  - Desde x generado con y latente (GSM from y_latent)")
        print(f"  - Desde y generado con generate_bump_stimulus "
              f"(A={AMPLITUDE}, σ={WIDTH_FACTOR})")
    else:
        print("\n⚠ Hay diferencias en algunos casos.")


if __name__ == '__main__':
    main()
