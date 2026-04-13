#!/usr/bin/env python3
"""
Script para graficar parámetros de conectividad de 100 corridas.

Genera una figura con 8 subplots (4 amplitudes + 4 anchos) con:
- Escalas fijas de 0 a 1 para comparación directa
- Valores de referencia de Echeveste marcados en rojo
- Bandas de tolerancia sombreadas

Lee los datos desde results_100runs_*.npy (generados por entrenar_100_corridas.py)
para poder modificar la visualización sin reentrenar.

Basado en: Echeveste et al. (2020) - Tabla S1

Uso:
    python3 graficar_connectivity_100runs.py
    python3 graficar_connectivity_100runs.py --results_file results_100runs_XXX.npy
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Parámetros de Echeveste (Tabla S1 + repositorio original)
ECHEVESTE_REFERENCE = {
    # Connectivity amplitudes (valores optimizados por Echeveste)
    "a_EE": 0.331089168896709307,
    "a_EI": 0.081307171738997791,
    "a_IE": 0.363214560956081789,
    "a_II": 0.073595337344484008,
    # Connectivity widths (valores optimizados por Echeveste)
    "d_EE": 0.802756367925039016,
    "d_EI": 0.568730595704363684,
    "d_IE": 0.859828425651255390,
    "d_II": 0.629804883130465232,
}


def encontrar_ultimo_resultado(output_dir):
    """
    Find the most recent results file.

    Searches for results_100runs_*.npy files, including simulated data.

    Parameters
    ----------
    output_dir : Path
        Directory to search for results files

    Returns
    -------
    Path
        Path to the most recent results file
    """
    # Buscar archivos de 100 runs (incluyendo simulados)
    results_files = sorted(output_dir.glob("results_*runs*.npy"))

    if len(results_files) == 0:
        raise FileNotFoundError(
            f"No se encontraron archivos results_*runs*.npy en {output_dir}\n"
            f"Ejecutar primero: python3 simular_100_corridas.py"
        )

    return results_files[-1]


def graficar_connectivity(resultados, output_dir):
    """
    Generate connectivity parameters figure with fixed scales.

    Creates a 2x4 figure:
    - Row 1: Amplitude parameters (a_EE, a_EI, a_IE, a_II)
    - Row 2: Width parameters (d_EE, d_EI, d_IE, d_II)

    All plots use fixed Y-axis scale from 0 to 1 for direct visual comparison.
    Red horizontal lines mark Echeveste reference values.

    Parameters
    ----------
    resultados : list of dict
        Training results from each run
    output_dir : Path
        Output directory for the figure

    Based on: Echeveste et al. (2020) - Figura comparison methodology
    """
    n_runs = len(resultados)
    runs = np.arange(1, n_runs + 1)

    # Colores para cada tipo de conexión
    # II es violeta para no confundirse con la línea roja de Echeveste
    colors = {
        "EE": "#1f77b4",  # Azul
        "EI": "#2ca02c",  # Verde
        "IE": "#ff7f0e",  # Naranja
        "II": "#9467bd",  # Violeta (en lugar de rojo)
    }

    # Crear figura 2x4
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"Connectivity Parameters ({n_runs} runs)",
        fontsize=16,
        fontweight="bold",
    )

    # Parámetros de conectividad ordenados
    # Fila 1: Amplitudes (a_XX)
    # Fila 2: Anchos (d_XX)
    conn_params = [
        # (param_key, label, row, col, connection_type)
        ("a_EE", r"$a_{EE}$", 0, 0, "EE"),
        ("a_EI", r"$a_{EI}$", 0, 1, "EI"),
        ("a_IE", r"$a_{IE}$", 0, 2, "IE"),
        ("a_II", r"$a_{II}$", 0, 3, "II"),
        ("d_EE", r"$d_{EE}$", 1, 0, "EE"),
        ("d_EI", r"$d_{EI}$", 1, 1, "EI"),
        ("d_IE", r"$d_{IE}$", 1, 2, "IE"),
        ("d_II", r"$d_{II}$", 1, 3, "II"),
    ]

    for param_key, label, row, col, conn_type in conn_params:
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
            color=colors[conn_type],
            alpha=1.0,
            zorder=4,
        )

        # Línea horizontal punteada de referencia de Echeveste (rojo)
        echeveste_val = ECHEVESTE_REFERENCE[param_key]
        ax.axhline(
            echeveste_val,
            color="red",
            linestyle="--",  # Línea punteada
            linewidth=2.5,
            label=f"Echeveste: {echeveste_val:.3f}",
            zorder=5,  # Al frente de los datos (zorder=3)
            alpha=0.9,
        )

        # Banda sombreada de tolerancia ±10% (rojo claro)
        # Usamos un grosor mínimo para que sea visible en valores pequeños
        tolerance = 0.10  # 10%
        band_half_width = echeveste_val * tolerance
        min_half_width = 0.025  # Grosor mínimo (0.05 total)
        band_half_width = max(band_half_width, min_half_width)

        ax.axhspan(
            echeveste_val - band_half_width,
            echeveste_val + band_half_width,
            alpha=0.15,
            color="red",
            zorder=0,
        )

        # Configuración del subplot
        ax.set_ylabel(label, fontsize=12, fontweight="bold")

        # Solo etiqueta X en fila inferior
        if row == 1:
            ax.set_xlabel("Training runs", fontsize=11)

        # ESCALA FIJA DE 0 A 1 PARA TODOS LOS SUBPLOTS
        ax.set_ylim(0, 1)
        ax.set_xlim(0.5, n_runs + 0.5)

        # Grid y formato
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, loc="upper right")

        # Ticks en X cada 20 corridas para 100 runs
        if n_runs > 20:
            tick_step = 20
            ax.set_xticks(np.arange(0, n_runs + 1, tick_step))
        else:
            ax.set_xticks(runs)

    # Títulos de fila
    axes[0, 0].annotate(
        "Amplitudes",
        xy=(-0.35, 0.5),
        xycoords="axes fraction",
        fontsize=14,
        fontweight="bold",
        rotation=90,
        ha="center",
        va="center",
    )
    axes[1, 0].annotate(
        "Widths",
        xy=(-0.35, 0.5),
        xycoords="axes fraction",
        fontsize=14,
        fontweight="bold",
        rotation=90,
        ha="center",
        va="center",
    )

    plt.tight_layout()

    # Guardar figura
    output_file = output_dir / "connectivity_100runs.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Figura guardada: {output_file}")
    plt.close()

    # Mostrar estadísticas
    print(f"\n{'='*70}")
    print(f"ESTADÍSTICAS DE CONECTIVIDAD ({n_runs} corridas)")
    print(f"{'='*70}\n")

    print(f"{'Parámetro':<10} {'Media':>10} {'Std':>10} "
          f"{'Echeveste':>12} {'Diff %':>10}")
    print("-" * 60)

    for param_key, label, _, _, _ in conn_params:
        vals = np.array([r[param_key] for r in resultados])
        ref = ECHEVESTE_REFERENCE[param_key]
        mean_val = vals.mean()
        std_val = vals.std()
        diff_percent = abs(mean_val - ref) / ref * 100

        # Usar nombre simplificado para la tabla
        name = param_key
        print(f"{name:<10} {mean_val:>10.4f} {std_val:>10.4f} "
              f"{ref:>12.4f} {diff_percent:>9.1f}%")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Graficar parámetros de conectividad de 100 corridas"
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
    print("VISUALIZACIÓN DE PARÁMETROS DE CONECTIVIDAD")
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
    graficar_connectivity(resultados, output_dir)

    print("\n" + "=" * 70)
    print("✓ VISUALIZACIÓN COMPLETADA")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
