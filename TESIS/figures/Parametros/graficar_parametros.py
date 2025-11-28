#!/usr/bin/env python3
"""
Script para graficar resultados de entrenamiento del modelo SSN.

Carga los archivos .npy generados por entrenar_parametros.py y genera
gráficos de comparación con los parámetros originales de Echeveste.

Uso:
    python3 graficar_parametros.py
    python3 graficar_parametros.py --results_file results_YYYYMMDD_HHMMSS.npy
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Directorio de datos de entrenamiento
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "training"


def encontrar_ultimo_resultado():
    """
    Encuentra el archivo de resultados más reciente.

    Returns
    -------
    Path
        Ruta al archivo de resultados más reciente
    """
    results_files = sorted(DATA_DIR.glob("results_*.npy"))

    if len(results_files) == 0:
        raise FileNotFoundError(
            f"No se encontraron archivos de resultados en {DATA_DIR}"
        )

    return results_files[-1]


def cargar_resultados(results_file):
    """
    Carga resultados de entrenamiento desde archivo .npy.

    Parameters
    ----------
    results_file : Path
        Ruta al archivo de resultados

    Returns
    -------
    list of dict
        Lista de diccionarios con resultados de cada corrida
    metadata : dict
        Metadatos del entrenamiento
    """
    # Cargar resultados
    results = np.load(results_file, allow_pickle=True)

    # Convertir a lista de diccionarios
    if results.ndim == 0:
        # Single dict saved as object array
        results_list = [results.item()]
    else:
        results_list = list(results)

    # Cargar metadatos
    timestamp = results_file.stem.replace("results_", "")
    metadata_file = results_file.parent / f"metadata_{timestamp}.json"

    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return results_list, metadata


def graficar_parametros(resultados, metadata, output_dir):
    """
    Genera gráficos de comparación de parámetros.

    Parameters
    ----------
    resultados : list of dict
        Resultados de entrenamiento
    metadata : dict
        Metadatos del entrenamiento
    output_dir : Path
        Directorio de salida para las figuras
    """
    # Obtener referencia de Echeveste
    echeveste_ref = metadata.get("echeveste_reference", {})

    # Timestamp para archivos de salida
    timestamp = metadata.get("timestamp", "unknown")

    # Número de corridas
    n_runs = len(resultados)
    runs = np.arange(1, n_runs + 1)

    # =================================================================
    # 1. Parámetros de transformación de entrada (Stage 2)
    # =================================================================
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle(
        f"Parámetros de Transformación de Entrada (n={n_runs} corridas)",
        fontsize=14,
        fontweight="bold",
    )

    input_params = [
        ("input_scaling", "α_h (Input Scaling)", 0.05),
        ("input_baseline", "β_h (Input Baseline)", 0.05),
        ("input_nl_pow", "γ_h (Input Nonlinearity)", 0.05),
    ]

    for ax, (param_key, param_label, tolerance) in zip(
        axes1, input_params
    ):
        # Extraer valores
        vals = np.array([r[param_key] for r in resultados])

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
        if param_key in echeveste_ref:
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

            # Banda de tolerancia
            ax.axhspan(
                echeveste_val * (1 - tolerance),
                echeveste_val * (1 + tolerance),
                alpha=0.1,
                color="red",
                zorder=1,
            )

        ax.set_xlabel("Corrida", fontsize=12)
        ax.set_ylabel(param_label, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(runs)

    plt.tight_layout()
    output_file1 = output_dir / f"input_transformation_{timestamp}.png"
    plt.savefig(output_file1, dpi=150, bbox_inches="tight")
    print(f"✓ Guardado: {output_file1}")
    plt.close()

    # =================================================================
    # 2. Parámetros de conectividad - Amplitudes (Stage 1)
    # =================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle(
        f"Amplitudes de Conectividad (n={n_runs} corridas)",
        fontsize=14,
        fontweight="bold",
    )

    connectivity_amp = [
        ("a_EE", "a_EE (E→E)", 0.1),
        ("a_EI", "a_EI (E→I)", 0.1),
        ("a_IE", "a_IE (I→E)", 0.1),
        ("a_II", "a_II (I→I)", 0.1),
    ]

    for ax, (param_key, param_label, tolerance) in zip(
        axes2.flat, connectivity_amp
    ):
        vals = np.array([r[param_key] for r in resultados])

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

        if param_key in echeveste_ref:
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
            ax.axhspan(
                echeveste_val * (1 - tolerance),
                echeveste_val * (1 + tolerance),
                alpha=0.1,
                color="red",
                zorder=1,
            )

        ax.set_xlabel("Corrida", fontsize=12)
        ax.set_ylabel(param_label, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(runs)

    plt.tight_layout()
    output_file2 = output_dir / f"connectivity_amplitudes_{timestamp}.png"
    plt.savefig(output_file2, dpi=150, bbox_inches="tight")
    print(f"✓ Guardado: {output_file2}")
    plt.close()

    # =================================================================
    # 3. Parámetros de conectividad - Anchos (Stage 1)
    # =================================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
    fig3.suptitle(
        f"Anchos de Conectividad (n={n_runs} corridas)",
        fontsize=14,
        fontweight="bold",
    )

    connectivity_width = [
        ("d_EE", "d_EE (E→E)", 0.1),
        ("d_EI", "d_EI (E→I)", 0.1),
        ("d_IE", "d_IE (I→E)", 0.1),
        ("d_II", "d_II (I→I)", 0.1),
    ]

    for ax, (param_key, param_label, tolerance) in zip(
        axes3.flat, connectivity_width
    ):
        vals = np.array([r[param_key] for r in resultados])

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

        if param_key in echeveste_ref:
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
            ax.axhspan(
                echeveste_val * (1 - tolerance),
                echeveste_val * (1 + tolerance),
                alpha=0.1,
                color="red",
                zorder=1,
            )

        ax.set_xlabel("Corrida", fontsize=12)
        ax.set_ylabel(param_label, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(runs)

    plt.tight_layout()
    output_file3 = output_dir / f"connectivity_widths_{timestamp}.png"
    plt.savefig(output_file3, dpi=150, bbox_inches="tight")
    print(f"✓ Guardado: {output_file3}")
    plt.close()

    # =================================================================
    # 4. Parámetros de covarianza de ruido (Stage 2)
    # =================================================================
    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10))
    fig4.suptitle(
        f"Parámetros de Covarianza de Ruido (n={n_runs} corridas)",
        fontsize=14,
        fontweight="bold",
    )

    noise_params = [
        ("sigma_eta_width", "width (Ancho espacial)", 0.1),
        ("sigma_eta_std_e", "σ_E (Desv. estándar E)", 0.1),
        ("sigma_eta_std_i", "σ_I (Desv. estándar I)", 0.1),
        ("sigma_eta_rho", "ρ (Correlación E-I)", 0.1),
    ]

    for ax, (param_key, param_label, tolerance) in zip(
        axes4.flat, noise_params
    ):
        vals = np.array([r[param_key] for r in resultados])

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

        if param_key in echeveste_ref:
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
            ax.axhspan(
                echeveste_val * (1 - tolerance),
                echeveste_val * (1 + tolerance),
                alpha=0.1,
                color="red",
                zorder=1,
            )

        ax.set_xlabel("Corrida", fontsize=12)
        ax.set_ylabel(param_label, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(runs)

    plt.tight_layout()
    output_file4 = output_dir / f"noise_covariance_{timestamp}.png"
    plt.savefig(output_file4, dpi=150, bbox_inches="tight")
    print(f"✓ Guardado: {output_file4}")
    plt.close()

    # =================================================================
    # 5. Convergencia de Stage 1 y Stage 2
    # =================================================================
    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
    fig5.suptitle(
        f"Convergencia de Optimización (n={n_runs} corridas)",
        fontsize=14,
        fontweight="bold",
    )

    # Stage 1 final cost
    stage1_costs = np.array([r["stage1_final_cost"] for r in resultados])
    axes5[0].plot(
        runs,
        stage1_costs,
        "o-",
        linewidth=2.5,
        markersize=10,
        color="steelblue",
    )
    axes5[0].set_xlabel("Corrida", fontsize=12)
    axes5[0].set_ylabel("Costo Final", fontsize=12)
    axes5[0].set_title("Stage 1 (ADAM)", fontsize=12, fontweight="bold")
    axes5[0].grid(True, alpha=0.3)
    axes5[0].set_xticks(runs)
    axes5[0].set_yscale("log")

    # Stage 2 final cost
    stage2_costs = np.array([r["stage2_final_cost"] for r in resultados])
    # Filtrar NaN values
    valid_mask = ~np.isnan(stage2_costs)
    if valid_mask.any():
        axes5[1].plot(
            runs[valid_mask],
            stage2_costs[valid_mask],
            "o-",
            linewidth=2.5,
            markersize=10,
            color="steelblue",
        )
        axes5[1].set_yscale("log")

    axes5[1].set_xlabel("Corrida", fontsize=12)
    axes5[1].set_ylabel("Costo Final", fontsize=12)
    axes5[1].set_title("Stage 2 (L-BFGS-B)", fontsize=12, fontweight="bold")
    axes5[1].grid(True, alpha=0.3)
    axes5[1].set_xticks(runs)

    plt.tight_layout()
    output_file5 = output_dir / f"convergence_{timestamp}.png"
    plt.savefig(output_file5, dpi=150, bbox_inches="tight")
    print(f"✓ Guardado: {output_file5}")
    plt.close()


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Graficar resultados de entrenamiento SSN"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Archivo de resultados específico (default: más reciente)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directorio de salida (default: figuras/Parametros/)",
    )

    args = parser.parse_args()

    # Determinar archivo de resultados
    if args.results_file is None:
        results_file = encontrar_ultimo_resultado()
        print(f"Usando archivo más reciente: {results_file.name}")
    else:
        results_file = DATA_DIR / args.results_file
        if not results_file.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo: {results_file}"
            )

    # Determinar directorio de salida
    if args.output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("VISUALIZACIÓN DE RESULTADOS DE ENTRENAMIENTO SSN")
    print("=" * 70)
    print(f"\nArchivo de resultados: {results_file}")
    print(f"Directorio de salida: {output_dir}\n")

    # Cargar resultados
    resultados, metadata = cargar_resultados(results_file)

    print(f"✓ Cargados {len(resultados)} resultados")
    print(f"  Stage 1 exitosos: {sum(r['stage1_success'] for r in resultados)}/{len(resultados)}")  # noqa: E501
    print(f"  Stage 2 exitosos: {sum(r['stage2_success'] for r in resultados)}/{len(resultados)}\n")  # noqa: E501

    # Generar gráficos
    print("Generando gráficos...")
    graficar_parametros(resultados, metadata, output_dir)

    print("\n" + "=" * 70)
    print("✓ VISUALIZACIÓN COMPLETADA")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
