#!/usr/bin/env python3
"""
Script para graficar resultados de entrenamientos del modelo SSN.

Carga resultados desde figuras/data/training/ y genera gráficos comparando
con los parámetros originales de Echeveste.

Características:
- Carga automáticamente el archivo de resultados más reciente
- Muestra líneas punteadas con valores de referencia de Echeveste
- Genera 3 figuras: Input Transformation, Connectivity, Noise
- Guarda figuras en figuras/Parametros/

Uso:
    python3 graficar_parametros.py
    python3 graficar_parametros.py --results_file results_20231114_103045.npy
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_latest_results(training_dir):
    """
    Encuentra el archivo de resultados más reciente.

    Parameters
    ----------
    training_dir : Path
        Directorio donde buscar archivos de resultados

    Returns
    -------
    Path or None
        Path al archivo más reciente, o None si no hay archivos
    """
    results_files = list(training_dir.glob("results_*.npy"))

    if not results_files:
        return None

    # Ordenar por fecha de modificación (más reciente primero)
    results_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return results_files[0]


def graficar_input_transformation(resultados, echeveste_ref, output_dir):
    """
    Genera gráfico de parámetros de transformación de input.

    Parameters
    ----------
    resultados : list
        Lista de diccionarios con resultados de entrenamientos
    echeveste_ref : dict
        Diccionario con valores de referencia de Echeveste
    output_dir : Path
        Directorio donde guardar la figura
    """
    if not resultados:
        print("❌ No hay resultados para graficar")
        return

    n_runs = len(resultados)
    runs = np.arange(1, n_runs + 1)

    # Extraer parámetros
    alpha = np.array([r["input_scaling"] for r in resultados])
    beta = np.array([r["input_baseline"] for r in resultados])
    gamma = np.array([r["input_nl_pow"] for r in resultados])

    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Input Transformation ({n_runs} corridas)",
        fontsize=14,
        fontweight="bold",
    )

    params_data = [
        (alpha, "input_scaling", "α_h (Input Scaling)"),
        (beta, "input_baseline", "β_h (Input Baseline)"),
        (gamma, "input_nl_pow", "γ_h (Input Power)"),
    ]

    for ax, (vals, param_key, ylabel) in zip(axes, params_data):
        # Graficar valores entrenados
        ax.plot(
            runs,
            vals,
            "o-",
            linewidth=2.5,
            markersize=10,
            label="Entrenado",
            color="steelblue",
            zorder=3,
        )

        # Línea punteada de referencia Echeveste
        echeveste_val = echeveste_ref[param_key]
        ax.axhline(
            echeveste_val,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label="Echeveste (original)",
            zorder=2,
            alpha=0.8,
        )

        # Banda de ±5% alrededor de Echeveste
        ax.axhspan(
            echeveste_val * 0.95,
            echeveste_val * 1.05,
            alpha=0.1,
            color="red",
            zorder=1,
        )

        # Configuración
        ax.set_xlabel("Corrida", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel.split("(")[0].strip(), fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_xticks(runs)

        # Mostrar valor de referencia en el eje
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        text_y = echeveste_val + y_range * 0.05
        ax.text(
            runs[-1] + 0.3,
            text_y,
            f"ref: {echeveste_val:.3f}",
            fontsize=9,
            color="red",
            va="bottom",
        )

    plt.tight_layout()
    output_file = output_dir / "input_transformation_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Guardado: {output_file}")
    plt.close()


def graficar_connectivity(resultados, echeveste_ref, output_dir):
    """
    Genera gráfico de parámetros de conectividad.

    Parameters
    ----------
    resultados : list
        Lista de diccionarios con resultados de entrenamientos
    echeveste_ref : dict
        Diccionario con valores de referencia de Echeveste
    output_dir : Path
        Directorio donde guardar la figura
    """
    if not resultados:
        print("❌ No hay resultados para graficar")
        return

    n_runs = len(resultados)
    runs = np.arange(1, n_runs + 1)

    # Extraer parámetros
    a_EE = np.array([r["a_EE"] for r in resultados])
    a_EI = np.array([r["a_EI"] for r in resultados])
    a_IE = np.array([r["a_IE"] for r in resultados])
    a_II = np.array([r["a_II"] for r in resultados])
    d_EE = np.array([r["d_EE"] for r in resultados])
    d_EI = np.array([r["d_EI"] for r in resultados])
    d_IE = np.array([r["d_IE"] for r in resultados])
    d_II = np.array([r["d_II"] for r in resultados])

    # Crear figura
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"Connectivity Parameters ({n_runs} corridas)",
        fontsize=14,
        fontweight="bold",
    )

    conn_params = [
        (a_EE, "a_EE", 0, 0, "Amplitude"),
        (a_EI, "a_EI", 0, 1, "Amplitude"),
        (a_IE, "a_IE", 0, 2, "Amplitude"),
        (a_II, "a_II", 0, 3, "Amplitude"),
        (d_EE, "d_EE", 1, 0, "Width"),
        (d_EI, "d_EI", 1, 1, "Width"),
        (d_IE, "d_IE", 1, 2, "Width"),
        (d_II, "d_II", 1, 3, "Width"),
    ]

    for vals, param_key, i, j, param_type in conn_params:
        ax = axes[i, j]

        # Graficar valores entrenados
        ax.plot(
            runs,
            vals,
            "o-",
            linewidth=2.5,
            markersize=10,
            label="Entrenado",
            color="steelblue",
            zorder=3,
        )

        # Línea punteada de referencia Echeveste
        echeveste_val = echeveste_ref[param_key]
        ax.axhline(
            echeveste_val,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label="Echeveste",
            zorder=2,
            alpha=0.8,
        )

        # Banda de ±10% para connectivity (tolerancia mayor)
        ax.axhspan(
            echeveste_val * 0.90,
            echeveste_val * 1.10,
            alpha=0.1,
            color="red",
            zorder=1,
        )

        # Configuración
        if i == 1:  # Fila inferior
            ax.set_xlabel("Corrida", fontsize=11)

        ax.set_ylabel(f"{param_key}\n({param_type})", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_xticks(runs)

        # Mostrar valor de referencia
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        text_y = echeveste_val + y_range * 0.05
        ax.text(
            runs[-1] + 0.3,
            text_y,
            f"{echeveste_val:.3f}",
            fontsize=8,
            color="red",
            va="bottom",
        )

    plt.tight_layout()
    output_file = output_dir / "connectivity_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Guardado: {output_file}")
    plt.close()


def graficar_noise(resultados, echeveste_ref, output_dir):
    """
    Genera gráfico de parámetros de covarianza de ruido.

    Parameters
    ----------
    resultados : list
        Lista de diccionarios con resultados de entrenamientos
    echeveste_ref : dict
        Diccionario con valores de referencia de Echeveste
    output_dir : Path
        Directorio donde guardar la figura
    """
    if not resultados:
        print("❌ No hay resultados para graficar")
        return

    n_runs = len(resultados)
    runs = np.arange(1, n_runs + 1)

    # Extraer parámetros
    width = np.array([r["sigma_eta_width"] for r in resultados])
    std_e = np.array([r["sigma_eta_std_e"] for r in resultados])
    std_i = np.array([r["sigma_eta_std_i"] for r in resultados])
    rho = np.array([r["sigma_eta_rho"] for r in resultados])

    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Noise Covariance Parameters ({n_runs} corridas)",
        fontsize=14,
        fontweight="bold",
    )

    noise_params = [
        (width, "sigma_eta_width", 0, 0, "Width"),
        (std_e, "sigma_eta_std_e", 0, 1, "σ_E (Std Excitatory)"),
        (std_i, "sigma_eta_std_i", 1, 0, "σ_I (Std Inhibitory)"),
        (rho, "sigma_eta_rho", 1, 1, "ρ (E-I Correlation)"),
    ]

    for vals, param_key, i, j, ylabel in noise_params:
        ax = axes[i, j]

        # Graficar valores entrenados
        ax.plot(
            runs,
            vals,
            "o-",
            linewidth=2.5,
            markersize=10,
            label="Entrenado",
            color="steelblue",
            zorder=3,
        )

        # Línea punteada de referencia Echeveste
        echeveste_val = echeveste_ref[param_key]
        ax.axhline(
            echeveste_val,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label="Echeveste",
            zorder=2,
            alpha=0.8,
        )

        # Banda de ±10%
        ax.axhspan(
            echeveste_val * 0.90,
            echeveste_val * 1.10,
            alpha=0.1,
            color="red",
            zorder=1,
        )

        # Configuración
        if i == 1:  # Fila inferior
            ax.set_xlabel("Corrida", fontsize=12)

        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_xticks(runs)

        # Mostrar valor de referencia
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        text_y = echeveste_val + y_range * 0.05
        ax.text(
            runs[-1] + 0.3,
            text_y,
            f"{echeveste_val:.3f}",
            fontsize=9,
            color="red",
            va="bottom",
        )

    plt.tight_layout()
    output_file = output_dir / "noise_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Guardado: {output_file}")
    plt.close()


def imprimir_estadisticas(resultados, echeveste_ref):
    """
    Imprime estadísticas comparativas de los resultados.

    Parameters
    ----------
    resultados : list
        Lista de diccionarios con resultados de entrenamientos
    echeveste_ref : dict
        Diccionario con valores de referencia de Echeveste
    """
    n_runs = len(resultados)

    print(f"\n{'='*70}")
    print(f"ESTADÍSTICAS COMPARATIVAS ({n_runs} corridas)")
    print(f"{'='*70}\n")

    # Grupos de parámetros
    param_groups = {
        "Input Transformation": [
            "input_scaling",
            "input_baseline",
            "input_nl_pow",
        ],
        "Connectivity Amplitudes": ["a_EE", "a_EI", "a_IE", "a_II"],
        "Connectivity Widths": ["d_EE", "d_EI", "d_IE", "d_II"],
        "Noise Covariance": [
            "sigma_eta_width",
            "sigma_eta_std_e",
            "sigma_eta_std_i",
            "sigma_eta_rho",
        ],
    }

    for group_name, params in param_groups.items():
        print(f"{group_name}:")
        for param in params:
            values = np.array([r[param] for r in resultados])
            mean_val = values.mean()
            std_val = values.std()
            ref_val = echeveste_ref[param]

            # Calcular diferencia relativa
            rel_diff = abs(mean_val - ref_val) / ref_val * 100

            # Indicador visual de qué tan cerca está
            if rel_diff < 5:
                indicator = "✓✓"  # Muy cerca
            elif rel_diff < 10:
                indicator = "✓"  # Cerca
            elif rel_diff < 20:
                indicator = "~"  # Moderado
            else:
                indicator = "!"  # Lejos

            print(
                f"  {param:20s}: {mean_val:.4f} ± {std_val:.4f}  |  "
                f"ref: {ref_val:.4f}  |  diff: {rel_diff:5.1f}% {indicator}"
            )
        print()

    # Estadísticas de convergencia
    print("Convergencia:")
    n_stage1 = sum(1 for r in resultados if r.get("stage1_success", False))
    n_stage2 = sum(1 for r in resultados if r.get("stage2_success", False))
    n_trained = sum(1 for r in resultados if r.get("is_trained", False))

    print(f"  Stage 1 exitoso:  {n_stage1}/{n_runs} "
          f"({n_stage1/n_runs*100:.0f}%)")
    print(f"  Stage 2 exitoso:  {n_stage2}/{n_runs} "
          f"({n_stage2/n_runs*100:.0f}%)")
    print(f"  Totalmente entrenado: {n_trained}/{n_runs} "
          f"({n_trained/n_runs*100:.0f}%)")

    # Costos finales promedio
    if any("stage1_final_cost" in r for r in resultados):
        costs_s1 = [
            r["stage1_final_cost"]
            for r in resultados
            if "stage1_final_cost" in r
        ]
        costs_s2 = [
            r["stage2_final_cost"]
            for r in resultados
            if "stage2_final_cost" in r
        ]

        if costs_s1:
            print(
                f"\n  Costo Stage 1: {np.mean(costs_s1):.6e} "
                f"± {np.std(costs_s1):.6e}"
            )
        if costs_s2:
            print(
                f"  Costo Stage 2: {np.mean(costs_s2):.6e} "
                f"± {np.std(costs_s2):.6e}"
            )


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Graficar resultados de entrenamientos SSN"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Archivo de resultados específico (default: más reciente)",
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        default=None,
        help="Directorio con datos de training "
        "(default: figuras/data/training/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directorio de salida (default: figuras/Parametros/)",
    )

    args = parser.parse_args()

    # Configurar directorios
    script_dir = Path(__file__).parent

    if args.training_dir is None:
        training_dir = script_dir.parent / "data" / "training"
    else:
        training_dir = Path(args.training_dir)

    if args.output_dir is None:
        output_dir = script_dir
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Encontrar archivo de resultados
    if args.results_file is None:
        results_file = find_latest_results(training_dir)
        if results_file is None:
            print(f"\n❌ No se encontraron archivos de resultados en: "
                  f"{training_dir}")
            print(f"\nPrimero ejecutar:")
            print(f"  python3 figuras/data/training/entrenar_parametros.py\n")
            return 1
        print(f"Usando archivo más reciente: {results_file.name}")
    else:
        results_file = training_dir / args.results_file
        if not results_file.exists():
            print(f"\n❌ Archivo no encontrado: {results_file}")
            return 1

    print(f"\n{'='*70}")
    print("GRAFICADOR DE PARÁMETROS SSN ECHEVESTE2020")
    print(f"{'='*70}\n")
    print(f"Archivo de entrada: {results_file}")
    print(f"Directorio de salida: {output_dir}\n")

    # Cargar resultados
    print("Cargando resultados...")
    resultados = np.load(results_file, allow_pickle=True)

    if len(resultados) == 0:
        print("❌ No hay resultados en el archivo")
        return 1

    print(f"✓ Cargados {len(resultados)} resultados\n")

    # Valores de referencia de Echeveste
    # (deben estar incluidos en el archivo, pero por si acaso los definimos)
    echeveste_ref = {
        "input_scaling": 1.955517265245506842,
        "input_baseline": 0.103232893380230673,
        "input_nl_pow": 2.034269617565710231,
        "a_EE": 0.331089168896709307,
        "a_EI": 0.081307171738997791,
        "a_IE": 0.363214560956081789,
        "a_II": 0.073595337344484008,
        "d_EE": 0.802756367925039016,
        "d_EI": 0.568730595704363684,
        "d_IE": 0.859828425651255390,
        "d_II": 0.629804883130465232,
        "sigma_eta_width": 0.477952645032425183,
        "sigma_eta_std_e": 6.399186720550809504,
        "sigma_eta_std_i": 3.533514014946008697,
        "sigma_eta_rho": 0.992852418877574472,
    }

    # Generar gráficos
    print("Generando gráficos...\n")

    graficar_input_transformation(resultados, echeveste_ref, output_dir)
    graficar_connectivity(resultados, echeveste_ref, output_dir)
    graficar_noise(resultados, echeveste_ref, output_dir)

    # Imprimir estadísticas
    imprimir_estadisticas(resultados, echeveste_ref)

    print(f"\n{'='*70}")
    print("✓ GRAFICADO COMPLETADO")
    print(f"{'='*70}\n")
    print(f"Figuras guardadas en: {output_dir}\n")

    return 0


if __name__ == "__main__":
    exit(main())
