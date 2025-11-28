#!/usr/bin/env python3
"""
Script para graficar resultados de 5 corridas de entrenamiento.

Carga resultados_5runs.npy y genera 3 gráficos comparando con Echeveste.

Uso:
    python3 graficar_5_corridas.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Parámetros de Echeveste (Tabla S1 + repositorio original)
ECHEVESTE = {
    "input_scaling": 1.955517265245506842,
    "input_baseline": 0.103232893380230673,
    "input_nl_pow": 2.034269617565710231,
    "a_EE": 0.331089168896709307,
    "d_EE": 0.802756367925039016,
    "a_EI": 0.081307171738997791,
    "d_EI": 0.568730595704363684,
    "a_IE": 0.363214560956081789,
    "d_IE": 0.859828425651255390,
    "a_II": 0.073595337344484008,
    "d_II": 0.629804883130465232,
    "width": 0.477952645032425183,
    "std_e": 6.399186720550809504,
    "std_i": 3.533514014946008697,
    "rho": 0.992852418877574472,
}


def graficar(resultados, output_dir):
    """Genera 3 gráficos comparando con Echeveste."""
    if not resultados:
        print("❌ No hay resultados")
        return

    runs = np.arange(1, len(resultados) + 1)

    # Extraer parámetros
    alpha = np.array([r['input_scaling'] for r in resultados])
    beta = np.array([r['input_baseline'] for r in resultados])
    gamma = np.array([r['input_nl_pow'] for r in resultados])

    a_EE = np.array([r['a_EE'] for r in resultados])
    a_EI = np.array([r['a_EI'] for r in resultados])
    a_IE = np.array([r['a_IE'] for r in resultados])
    a_II = np.array([r['a_II'] for r in resultados])
    d_EE = np.array([r['d_EE'] for r in resultados])
    d_EI = np.array([r['d_EI'] for r in resultados])
    d_IE = np.array([r['d_IE'] for r in resultados])
    d_II = np.array([r['d_II'] for r in resultados])

    width = np.array([r['sigma_eta_width'] for r in resultados])
    std_e = np.array([r['sigma_eta_std_e'] for r in resultados])
    std_i = np.array([r['sigma_eta_std_i'] for r in resultados])
    rho = np.array([r['sigma_eta_rho'] for r in resultados])

    # ===== FIGURA 1: Input Transformation =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Input Transformation (5 corridas, 250 iter/stage)",
                 fontsize=14, fontweight="bold")

    for ax, vals, param, ylabel in zip(
        axes,
        [alpha, beta, gamma],
        ["input_scaling", "input_baseline", "input_nl_pow"],
        ["α_h", "β_h", "γ_h"]
    ):
        ax.plot(runs, vals, "o-", linewidth=2.5, markersize=10,
                label="Entrenado", zorder=2)
        ax.axhline(ECHEVESTE[param], color="red", linestyle="--",
                   linewidth=3, label="Echeveste", zorder=1)
        ax.set_xlabel("Corrida", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(runs)

    plt.tight_layout()
    fig.savefig(output_dir / "input_transformation_5runs.png", dpi=150)
    print(f"✓ {output_dir / 'input_transformation_5runs.png'}")
    plt.close()

    # ===== FIGURA 2: Connectivity =====
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Connectivity (5 corridas, 250 iter/stage)",
                 fontsize=14, fontweight="bold")

    conn_params = [
        (a_EE, "a_EE", 0, 0), (a_EI, "a_EI", 0, 1),
        (a_IE, "a_IE", 0, 2), (a_II, "a_II", 0, 3),
        (d_EE, "d_EE", 1, 0), (d_EI, "d_EI", 1, 1),
        (d_IE, "d_IE", 1, 2), (d_II, "d_II", 1, 3)
    ]

    for vals, name, i, j in conn_params:
        axes[i, j].plot(runs, vals, "o-", linewidth=2.5, markersize=10,
                        label="Entrenado", zorder=2)
        axes[i, j].axhline(ECHEVESTE[name], color="red", linestyle="--",
                           linewidth=3, label="Echeveste", zorder=1)
        if i == 1:
            axes[i, j].set_xlabel("Corrida", fontsize=12)
        axes[i, j].set_ylabel(name, fontsize=11, fontweight="bold")
        axes[i, j].legend(fontsize=9)
        axes[i, j].grid(True, alpha=0.3)
        axes[i, j].set_xticks(runs)

    plt.tight_layout()
    fig.savefig(output_dir / "connectivity_5runs.png", dpi=150)
    print(f"✓ {output_dir / 'connectivity_5runs.png'}")
    plt.close()

    # ===== FIGURA 3: Noise =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Noise Covariance (5 corridas, 250 iter/stage)",
                 fontsize=14, fontweight="bold")

    noise_params = [
        (width, "width", 0, 0), (std_e, "std_e", 0, 1),
        (std_i, "std_i", 1, 0), (rho, "rho", 1, 1)
    ]

    for vals, name, i, j in noise_params:
        axes[i, j].plot(runs, vals, "o-", linewidth=2.5, markersize=10,
                        label="Entrenado", zorder=2)
        axes[i, j].axhline(ECHEVESTE[name], color="red", linestyle="--",
                           linewidth=3, label="Echeveste", zorder=1)
        if i == 1:
            axes[i, j].set_xlabel("Corrida", fontsize=12)
        axes[i, j].set_ylabel(name, fontsize=11, fontweight="bold")
        axes[i, j].legend(fontsize=10)
        axes[i, j].grid(True, alpha=0.3)
        axes[i, j].set_xticks(runs)

    plt.tight_layout()
    fig.savefig(output_dir / "noise_5runs.png", dpi=150)
    print(f"✓ {output_dir / 'noise_5runs.png'}")
    plt.close()

    # ===== ESTADÍSTICAS =====
    print(f"\n{'='*70}")
    print("ESTADÍSTICAS (5 CORRIDAS)")
    print(f"{'='*70}\n")
    print(f"Input Transformation:")
    print(f"  α_h: {alpha.mean():.4f} ± {alpha.std():.4f} "
          f"(Tabla S1: {ECHEVESTE['input_scaling']:.4f})")
    print(f"  β_h: {beta.mean():.4f} ± {beta.std():.4f} "
          f"(Tabla S1: {ECHEVESTE['input_baseline']:.4f})")
    print(f"  γ_h: {gamma.mean():.4f} ± {gamma.std():.4f} "
          f"(Tabla S1: {ECHEVESTE['input_nl_pow']:.4f})")
    print(f"\nConnectivity:")
    print(f"  a_EE: {a_EE.mean():.4f} ± {a_EE.std():.4f}")
    print(f"  a_EI: {a_EI.mean():.4f} ± {a_EI.std():.4f}")
    print(f"  a_IE: {a_IE.mean():.4f} ± {a_IE.std():.4f}")
    print(f"  a_II: {a_II.mean():.4f} ± {a_II.std():.4f}")
    print(f"\nNoise:")
    print(f"  width: {width.mean():.4f} ± {width.std():.4f}")
    print(f"  σ_E: {std_e.mean():.4f} ± {std_e.std():.4f}")
    print(f"  σ_I: {std_i.mean():.4f} ± {std_i.std():.4f}")
    print(f"  ρ: {rho.mean():.4f} ± {rho.std():.4f}")


def main():
    output_dir = Path(__file__).parent
    input_file = output_dir / "resultados_5runs.npy"

    if not input_file.exists():
        print(f"❌ Error: No existe {input_file}")
        print(f"\nPrimero ejecutar:")
        print(f"  python3 entrenar_5_corridas.py\n")
        return 1

    print(f"Cargando resultados desde: {input_file}")
    resultados = np.load(input_file, allow_pickle=True)

    if len(resultados) == 0:
        print("❌ Error: No hay resultados en el archivo")
        return 1

    print(f"✓ Cargados {len(resultados)} resultados\n")
    print("Generando gráficos...\n")

    graficar(resultados, output_dir)

    print(f"\n✓ Completado! Figuras en: {output_dir}\n")
    return 0


if __name__ == "__main__":
    exit(main())
