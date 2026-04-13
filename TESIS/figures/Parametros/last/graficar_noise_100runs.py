#!/usr/bin/env python3
"""
Script para graficar parámetros de ruido de 100 corridas.

Genera una figura con 4 subplots (2x2) con:
- Escalas fijas para comparación directa
- Valores de referencia de Echeveste marcados en rojo punteado
- Bandas de tolerancia sombreadas ±10%

Lee los datos desde results_100runs_*.npy (generados por simular_100_corridas.py)

Basado en: Echeveste et al. (2020) - Tabla S1

Uso:
    python3 graficar_noise_100runs.py
    python3 graficar_noise_100runs.py --results_file results_100runs_XXX.npy
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Parámetros de Echeveste (Tabla S1 + repositorio original)
ECHEVESTE_REFERENCE = {
    # Noise covariance parameters (valores optimizados por Echeveste)
    "sigma_eta_width": 0.477952645032425183,
    "sigma_eta_std_e": 6.399186720550809504,
    "sigma_eta_std_i": 3.533514014946008697,
    "sigma_eta_rho": 0.992852418877574472,
}


def encontrar_ultimo_resultado(output_dir):
    """
    Find the most recent results file.

    Parameters
    ----------
    output_dir : Path
        Directory to search for results files

    Returns
    -------
    Path
        Path to the most recent results file
    """
    results_files = sorted(output_dir.glob("results_*runs*.npy"))

    if len(results_files) == 0:
        raise FileNotFoundError(
            f"No se encontraron archivos results_*runs*.npy en {output_dir}\n"
            f"Ejecutar primero: python3 simular_100_corridas.py"
        )

    return results_files[-1]


def graficar_noise(resultados, output_dir):
    """
    Generate noise covariance parameters figure with fixed scales.

    Creates a 2x2 figure:
    - width: spatial width of noise correlation
    - std_e: standard deviation of excitatory noise
    - std_i: standard deviation of inhibitory noise
    - rho: E-I correlation

    Parameters
    ----------
    resultados : list of dict
        Training results from each run
    output_dir : Path
        Output directory for the figure

    Based on: Echeveste et al. (2020) - Tabla S1
    """
    n_runs = len(resultados)
    runs = np.arange(1, n_runs + 1)

    # Colores para cada parámetro (evitando rojo)
    colors = {
        "width": "#1f77b4",   # Azul
        "std_e": "#2ca02c",   # Verde
        "std_i": "#ff7f0e",   # Naranja
        "rho": "#9467bd",     # Violeta
    }

    # Crear figura 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Noise Covariance Parameters ({n_runs} runs)",
        fontsize=16,
        fontweight="bold",
    )

    # Parámetros de ruido con sus escalas fijas
    # (param_key, label, row, col, color_key, y_max)
    # Escalas justificadas:
    # - Fila superior [0,1]: width (fracción normalizada) y ρ (correlación)
    # - Fila inferior [0,10]: σ_E y σ_I (desviaciones estándar en mV)
    noise_params = [
        ("sigma_eta_width", r"$d_{\eta}$ (spatial width)", 0, 0, "width", 1.0),
        ("sigma_eta_rho", r"$\rho$ (E-I correlation)", 0, 1, "rho", 1.0),
        ("sigma_eta_std_e", r"$\sigma_E$ (excitatory std)", 1, 0, "std_e", 10.0),
        ("sigma_eta_std_i", r"$\sigma_I$ (inhibitory std)", 1, 1, "std_i", 10.0),
    ]

    for param_key, label, row, col, color_key, y_max in noise_params:
        ax = axes[row, col]

        # Extraer valores de todas las corridas
        vals = np.array([r[param_key] for r in resultados])

        # Graficar valores entrenados
        ax.plot(
            runs,
            vals,
            "o-",
            linewidth=1.5,
            markersize=4,
            color=colors[color_key],
            alpha=0.8,
            zorder=3,
        )

        # Línea horizontal punteada de referencia de Echeveste (rojo)
        echeveste_val = ECHEVESTE_REFERENCE[param_key]
        ax.axhline(
            echeveste_val,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label=f"Echeveste: {echeveste_val:.3f}",
            zorder=2,
            alpha=0.9,
        )

        # Banda sombreada de tolerancia ±10% (rojo claro)
        # Usamos un grosor mínimo para que sea visible
        tolerance = 0.10  # 10%
        band_half_width = echeveste_val * tolerance
        # Grosor mínimo proporcional a la escala del eje
        min_half_width = y_max * 0.025
        band_half_width = max(band_half_width, min_half_width)

        ax.axhspan(
            echeveste_val - band_half_width,
            echeveste_val + band_half_width,
            alpha=0.25,
            color="red",
            zorder=1,
        )

        # Configuración del subplot (sin negrita en labels)
        ax.set_ylabel(label, fontsize=12)

        # Etiqueta X en fila inferior
        if row == 1:
            ax.set_xlabel("Training runs", fontsize=11)

        # ESCALA FIJA para cada parámetro
        ax.set_ylim(0, y_max)
        ax.set_xlim(0.5, n_runs + 0.5)

        # Grid y formato
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="upper right")

        # Ticks en X cada 20 corridas para 100 runs
        if n_runs > 20:
            tick_step = 20
            ax.set_xticks(np.arange(0, n_runs + 1, tick_step))
        else:
            ax.set_xticks(runs)

    plt.tight_layout()

    # Guardar figura
    output_file = output_dir / "noise_100runs.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Figura guardada: {output_file}")
    plt.close()

    # Mostrar estadísticas
    print(f"\n{'='*70}")
    print(f"NOISE COVARIANCE STATISTICS ({n_runs} runs)")
    print(f"{'='*70}\n")

    print(f"{'Parameter':<20} {'Mean':>10} {'Std':>10} "
          f"{'Echeveste':>12} {'Diff %':>10}")
    print("-" * 70)

    for param_key, label, _, _, _, _ in noise_params:
        vals = np.array([r[param_key] for r in resultados])
        ref = ECHEVESTE_REFERENCE[param_key]
        mean_val = vals.mean()
        std_val = vals.std()
        diff_percent = abs(mean_val - ref) / ref * 100

        print(f"{param_key:<20} {mean_val:>10.4f} {std_val:>10.4f} "
              f"{ref:>12.4f} {diff_percent:>9.1f}%")

    # Guardar caption
    caption = """Figure caption (noise_100runs.png):

Noise covariance parameters across 100 training runs compared to
Echeveste et al. (2020) reference values. Red dashed lines indicate
the original optimized values from Table S1, with shaded regions
representing ±10% tolerance bands.

Y-axis scales rationale:
- d_η (spatial width): Scale [0, 1] represents the normalized fraction
  of the orientation space over which noise correlations extend.
- σ_E and σ_I (excitatory/inhibitory standard deviations): Both use
  scale [0, 10] mV to enable direct visual comparison between
  excitatory and inhibitory noise magnitudes.
- ρ (E-I correlation): Scale [0, 1] as this is a dimensionless
  correlation coefficient bounded by definition.

Reference: Echeveste, R., Aitchison, L., Hennequin, G., & Lengyel, M.
(2020). Cortical-like dynamics in recurrent circuits optimized for
sampling-based probabilistic inference. Nature Neuroscience, 23(9),
1138-1149. Table S1.
"""

    caption_file = output_dir / "noise_100runs_caption.txt"
    with open(caption_file, "w") as f:
        f.write(caption)
    print(f"\n✓ Caption guardado: {caption_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Graficar parámetros de ruido de 100 corridas"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Archivo de resultados específico (default: más reciente)",
    )

    args = parser.parse_args()

    # Directorio de trabajo
    output_dir = Path(__file__).parent

    # Determinar archivo de resultados
    if args.results_file is None:
        results_file = encontrar_ultimo_resultado(output_dir)
        print(f"Usando archivo más reciente: {results_file.name}")
    else:
        results_file = output_dir / args.results_file
        if not results_file.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo: {results_file}"
            )

    print("\n" + "=" * 70)
    print("NOISE COVARIANCE PARAMETERS VISUALIZATION")
    print("=" * 70)
    print(f"\nArchivo de resultados: {results_file}")
    print(f"Directorio de salida: {output_dir}\n")

    # Cargar resultados
    resultados = np.load(results_file, allow_pickle=True)

    # Convertir a lista si es necesario
    if resultados.ndim == 0:
        resultados = [resultados.item()]
    else:
        resultados = list(resultados)

    print(f"✓ Cargadas {len(resultados)} corridas\n")

    # Generar figura
    print("Generando figura...")
    graficar_noise(resultados, output_dir)

    print("\n" + "=" * 70)
    print("✓ VISUALIZATION COMPLETED")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
