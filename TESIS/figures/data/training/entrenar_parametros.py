#!/usr/bin/env python3
"""
Script para entrenar modelo SSN Echeveste2020 múltiples veces.

Guarda todos los resultados de entrenamiento en archivos .npy para que
puedan ser utilizados posteriormente por los scripts de visualización.

Estructura de salida:
    figuras/data/training/
        results_YYYYMMDD_HHMMSS.npy  # Resultados completos
        metadata_YYYYMMDD_HHMMSS.json  # Metadatos del entrenamiento

Uso:
    python3 entrenar_parametros.py --n_runs 5 --max_iter_stage1 250 \\
                                    --max_iter_stage2 250
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Agregar scikit-neuromsi al path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "scikit-neuromsi"))

from skneuromsi.generative import GSM
from skneuromsi.neural import Echeveste2020


# Parámetros de Echeveste (Tabla S1 + repositorio original)
# Estos son los valores de referencia contra los que comparamos
ECHEVESTE_REFERENCE = {
    # Input transformation (optimizados por Echeveste)
    "input_scaling": 1.955517265245506842,
    "input_baseline": 0.103232893380230673,
    "input_nl_pow": 2.034269617565710231,
    # Connectivity amplitudes (optimizados)
    "a_EE": 0.331089168896709307,
    "a_EI": 0.081307171738997791,
    "a_IE": 0.363214560956081789,
    "a_II": 0.073595337344484008,
    # Connectivity widths (optimizados)
    "d_EE": 0.802756367925039016,
    "d_EI": 0.568730595704363684,
    "d_IE": 0.859828425651255390,
    "d_II": 0.629804883130465232,
    # Noise covariance parameters (optimizados)
    "sigma_eta_width": 0.477952645032425183,
    "sigma_eta_std_e": 6.399186720550809504,
    "sigma_eta_std_i": 3.533514014946008697,
    "sigma_eta_rho": 0.992852418877574472,
}


def entrenar_una_corrida(
    gsm, run_number, n_runs, max_iter_stage1, max_iter_stage2
):
    """
    Ejecuta una corrida completa de entrenamiento.

    Parameters
    ----------
    gsm : GSM
        Modelo generativo GSM ya inicializado
    run_number : int
        Número de corrida (1-indexed para display)
    n_runs : int
        Total de corridas (para display)
    max_iter_stage1 : int
        Iteraciones máximas para Stage 1
    max_iter_stage2 : int
        Iteraciones máximas para Stage 2

    Returns
    -------
    dict
        Diccionario con todos los parámetros optimizados y metadatos
    """
    print(f"\n{'='*70}")
    print(f"CORRIDA {run_number}/{n_runs}")
    print(f"{'='*70}\n")

    # Crear red SSN (50E + 50I como en Echeveste)
    # Usamos seed diferente para cada corrida
    ssn = Echeveste2020(
        N_E=50,
        N_I=50,
        k=0.3,  # Tabla S1
        n=2.0,  # π=2 en Tabla S1
        tau_e=20.0,  # ms, Tabla S1
        tau_i=10.0,  # ms, Tabla S1
        tau_n=20.0,  # ms, Tabla S1
        seed=42 + run_number,  # Seed diferente por corrida
    )

    # Parámetros de entrenamiento (basados en Tabla S1)
    # λ_mean = 4.0e-5, λ_var = 8.0e-5, λ_cov = 8.0e-7
    stage1_params = {
        "max_iter": max_iter_stage1,
        "contrast": 0.5,  # Contraste usado en paper
        "n_samples_gsm": 50,  # N_trials en Tabla S1
        "lambda_mean": 4.0e-5,
        "lambda_var": 8.0e-5,
        "lambda_cov": 8.0e-7,
    }

    stage2_params = {
        "max_iter": max_iter_stage2,
        "contrast": 0.5,
        "n_samples_gsm": 50,
    }

    print(f"Parámetros Stage 1: max_iter={max_iter_stage1}, "
          f"n_samples={stage1_params['n_samples_gsm']}")
    print(f"Parámetros Stage 2: max_iter={max_iter_stage2}, "
          f"n_samples={stage2_params['n_samples_gsm']}\n")

    # Entrenar
    result = ssn.train(
        gsm, stage1_params=stage1_params, stage2_params=stage2_params
    )

    # Extraer parámetros optimizados
    stage1_result = result.get("stage1", {}) if result else {}
    stage2_result = result.get("stage2", {}) if result else {}

    # Construir diccionario de resultados
    # Extraer parámetros de input transformation desde stage2
    # Si stage2 falló, usar valores por defecto
    if stage2_result.get("success", False):
        input_scaling = stage2_result["optimized_params"].get(
            "input_scaling", 1.0
        )
        input_baseline = stage2_result["optimized_params"].get(
            "input_baseline", 0.0
        )
        input_nl_pow = stage2_result["optimized_params"].get(
            "input_nl_pow", 1.0
        )
    else:
        # Valores por defecto si Stage 2 falla
        input_scaling = 1.0
        input_baseline = 0.0
        input_nl_pow = 1.0

    # Convertir JAX arrays a float para guardar en numpy
    def to_float(val):
        """Convert JAX array or numpy value to Python float."""
        if hasattr(val, "item"):
            return float(val.item())
        return float(val)

    resultado_corrida = {
        # Metadatos
        "run_number": run_number,
        "seed": 42 + run_number,
        "max_iter_stage1": max_iter_stage1,
        "max_iter_stage2": max_iter_stage2,
        # Input transformation parameters (from Stage 2)
        "input_scaling": to_float(input_scaling),
        "input_baseline": to_float(input_baseline),
        "input_nl_pow": to_float(input_nl_pow),
        # Connectivity parameters (from Stage 1)
        "a_EE": to_float(stage1_result["optimized_params"]["a_EE"]),
        "a_EI": to_float(stage1_result["optimized_params"]["a_EI"]),
        "a_IE": to_float(stage1_result["optimized_params"]["a_IE"]),
        "a_II": to_float(stage1_result["optimized_params"]["a_II"]),
        "d_EE": to_float(stage1_result["optimized_params"]["d_EE"]),
        "d_EI": to_float(stage1_result["optimized_params"]["d_EI"]),
        "d_IE": to_float(stage1_result["optimized_params"]["d_IE"]),
        "d_II": to_float(stage1_result["optimized_params"]["d_II"]),
        # Noise covariance parameters (from Stage 2)
        "sigma_eta_width": to_float(
            stage2_result["optimized_params"].get("width", 0.8)
        ),
        "sigma_eta_std_e": to_float(
            stage2_result["optimized_params"].get("std_e", 2.0)
        ),
        "sigma_eta_std_i": to_float(
            stage2_result["optimized_params"].get("std_i", 2.0)
        ),
        "sigma_eta_rho": to_float(
            stage2_result["optimized_params"].get("rho", 0.8)
        ),
        # Convergence info
        "stage1_success": stage1_result.get("success", False),
        "stage1_final_cost": to_float(
            stage1_result.get("final_cost", np.nan)
        ),
        "stage1_iterations": stage1_result.get("n_iterations", 0),
        "stage2_success": stage2_result.get("success", False),
        "stage2_final_cost": to_float(
            stage2_result.get("final_cost", np.nan)
        ),
        "stage2_iterations": stage2_result.get("n_iterations", 0),
        "is_trained": result.get("is_trained", False),
    }

    # Mostrar resumen
    print(f"\n{'─'*70}")
    print(f"RESUMEN CORRIDA {run_number}")
    print(f"{'─'*70}\n")

    print(f"Stage 1: {'✓ CONVERGED' if stage1_result['success'] else '✗ FAILED'}")
    print(f"  Final cost: {stage1_result['final_cost']:.6e}")
    print(f"  Iterations: {stage1_result['n_iterations']}\n")

    print(f"Stage 2: {'✓ CONVERGED' if stage2_result['success'] else '✗ FAILED'}")
    print(f"  Final cost: {stage2_result['final_cost']:.6e}")
    print(f"  Iterations: {stage2_result['n_iterations']}\n")

    # Comparar con referencia de Echeveste
    print("Parámetros vs Echeveste:")
    print(f"  α_h: {resultado_corrida['input_scaling']:.4f} "
          f"(ref: {ECHEVESTE_REFERENCE['input_scaling']:.4f})")
    print(f"  β_h: {resultado_corrida['input_baseline']:.4f} "
          f"(ref: {ECHEVESTE_REFERENCE['input_baseline']:.4f})")
    print(f"  γ_h: {resultado_corrida['input_nl_pow']:.4f} "
          f"(ref: {ECHEVESTE_REFERENCE['input_nl_pow']:.4f})")
    print(f"  a_EE: {resultado_corrida['a_EE']:.4f} "
          f"(ref: {ECHEVESTE_REFERENCE['a_EE']:.4f})")

    return resultado_corrida


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Entrenar modelo SSN múltiples veces"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Número de corridas de entrenamiento (default: 5)",
    )
    parser.add_argument(
        "--max_iter_stage1",
        type=int,
        default=250,
        help="Iteraciones máximas Stage 1 (default: 250)",
    )
    parser.add_argument(
        "--max_iter_stage2",
        type=int,
        default=250,
        help="Iteraciones máximas Stage 2 (default: 250)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directorio de salida (default: figuras/data/training/)",
    )

    args = parser.parse_args()

    # Configurar directorio de salida
    if args.output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp para archivos de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO MÚLTIPLE DE MODELO SSN ECHEVESTE2020")
    print("=" * 70)
    print(f"\nConfiguración:")
    print(f"  Número de corridas: {args.n_runs}")
    print(f"  Stage 1 max_iter: {args.max_iter_stage1}")
    print(f"  Stage 2 max_iter: {args.max_iter_stage2}")
    print(f"  Directorio de salida: {output_dir}")
    print(f"  Timestamp: {timestamp}\n")

    # Crear modelo GSM (50 orientaciones como en Echeveste)
    print("Inicializando modelo GSM...")
    gsm = GSM(
        patch_size=16,  # 16x16 = 256 pixels (Tabla S1)
        n_orientations=50,  # N_y = 50 (Tabla S1)
        h_scale=1.0 / 15.0,  # W^ff = [AA]^T / 15.0 (Tabla S1)
        random_seed=42,
        use_pretrained=True,  # Usar filtros pre-entrenados
    )
    print("✓ GSM inicializado\n")

    # Ejecutar corridas
    resultados = []

    for i in range(1, args.n_runs + 1):
        try:
            resultado = entrenar_una_corrida(
                gsm,
                run_number=i,
                n_runs=args.n_runs,
                max_iter_stage1=args.max_iter_stage1,
                max_iter_stage2=args.max_iter_stage2,
            )
            resultados.append(resultado)

        except Exception as e:
            print(f"\n❌ ERROR en corrida {i}: {e}")
            print("Continuando con siguiente corrida...\n")
            continue

    # Verificar que tenemos al menos un resultado
    if len(resultados) == 0:
        print("\n❌ ERROR: No se completó ninguna corrida exitosamente")
        return 1

    # Convertir a array numpy
    resultados_array = np.array(resultados, dtype=object)

    # Guardar resultados
    results_file = output_dir / f"results_{timestamp}.npy"
    np.save(results_file, resultados_array)
    print(f"\n✓ Resultados guardados en: {results_file}")

    # Guardar metadatos
    metadata = {
        "timestamp": timestamp,
        "n_runs_completed": len(resultados),
        "n_runs_requested": args.n_runs,
        "max_iter_stage1": args.max_iter_stage1,
        "max_iter_stage2": args.max_iter_stage2,
        "gsm_config": {
            "patch_size": 16,
            "n_orientations": 50,
            "h_scale": 1.0 / 15.0,
        },
        "ssn_config": {
            "N_E": 50,
            "N_I": 50,
            "k": 0.3,
            "n": 2.0,
            "tau_e": 20.0,
            "tau_i": 10.0,
            "tau_n": 20.0,
        },
        "echeveste_reference": ECHEVESTE_REFERENCE,
    }

    metadata_file = output_dir / f"metadata_{timestamp}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadatos guardados en: {metadata_file}\n")

    # Mostrar estadísticas finales
    print("=" * 70)
    print(f"ESTADÍSTICAS FINALES ({len(resultados)} corridas)")
    print("=" * 70 + "\n")

    # Extraer parámetros
    params = [
        "input_scaling",
        "input_baseline",
        "input_nl_pow",
        "a_EE",
        "a_EI",
        "a_IE",
        "a_II",
        "d_EE",
        "d_EI",
        "d_IE",
        "d_II",
        "sigma_eta_width",
        "sigma_eta_std_e",
        "sigma_eta_std_i",
        "sigma_eta_rho",
    ]

    for param in params:
        values = np.array([r[param] for r in resultados])
        ref = ECHEVESTE_REFERENCE.get(
            param, None
        )  # Algunos pueden no estar en ref

        mean_val = values.mean()
        std_val = values.std()

        if ref is not None:
            diff_percent = abs(mean_val - ref) / ref * 100
            print(
                f"{param:20s}: {mean_val:.4f} ± {std_val:.4f} "
                f"(ref: {ref:.4f}, diff: {diff_percent:.1f}%)"
            )
        else:
            print(f"{param:20s}: {mean_val:.4f} ± {std_val:.4f}")

    # Mostrar convergencia
    n_stage1_success = sum(
        1 for r in resultados if r["stage1_success"]
    )
    n_stage2_success = sum(
        1 for r in resultados if r["stage2_success"]
    )

    print(f"\nConvergencia:")
    print(
        f"  Stage 1: {n_stage1_success}/{len(resultados)} "
        f"({n_stage1_success/len(resultados)*100:.0f}%)"
    )
    print(
        f"  Stage 2: {n_stage2_success}/{len(resultados)} "
        f"({n_stage2_success/len(resultados)*100:.0f}%)"
    )

    print("\n" + "=" * 70)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print("=" * 70 + "\n")

    print(f"Para visualizar resultados, ejecutar:")
    print(
        f"  python3 figuras/Parametros/graficar_parametros.py "
        f"--results_file {results_file.name}\n"
    )

    return 0


if __name__ == "__main__":
    exit(main())
