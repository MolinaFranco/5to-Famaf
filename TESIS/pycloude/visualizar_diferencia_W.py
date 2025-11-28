#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para visualizar la diferencia entre W_exact y W_reconstruida.
"""

import sys
import os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'scikit-neuromsi')
)

import numpy as np
from skneuromsi.neural import Echeveste2020


def main():
    print("=" * 70)
    print("COMPARACIÓN: W_exact vs W_reconstruida")
    print("=" * 70)

    # Crear modelo y cargar parámetros
    model = Echeveste2020(N_E=50, N_I=50, seed=42)
    model.load_parameters()

    # W_exact: matriz cargada del archivo w_learn
    W_exact = model._W_exact

    # W_reconstruida: generada desde los 8 parámetros
    # Al llamar build_connectivity_matrix(), usa las matrices _W_XX que
    # fueron construidas en load_parameters() usando los 8 parámetros
    W_recon = model.build_connectivity_matrix()

    print("\n1. ¿QUÉ ES CADA MATRIZ?")
    print("-" * 70)
    print("\nW_exact:")
    print("  - Archivo: w_learn (del repositorio original)")
    print("  - Tamaño: 100×100 = 10,000 valores")
    print("  - Origen: Optimización completa del paper")
    print("  - Representa: Conexiones exactas del modelo entrenado")

    print("\nW_reconstruida:")
    print("  - Generada desde: 8 parámetros (a_EE, a_EI, ... d_II)")
    print("  - Fórmula: W_XY(θi,θj) = a_XY·exp[(cos(2(θi-θj))-1)/d_XY²]")
    print("  - Origen: Ecuación 10 del paper")
    print("  - Representa: Aproximación paramétrica de W_exact")

    print("\n" + "=" * 70)
    print("2. ESTADÍSTICAS BÁSICAS")
    print("=" * 70)

    print(f"\n{'Estadística':<20} {'W_exact':>15} {'W_reconstruida':>15} "
          f"{'Diferencia':>15}")
    print("-" * 70)

    stats = [
        ('Mínimo', W_exact.min(), W_recon.min()),
        ('Máximo', W_exact.max(), W_recon.max()),
        ('Media', W_exact.mean(), W_recon.mean()),
        ('Desv. Est.', W_exact.std(), W_recon.std()),
    ]

    for name, val_exact, val_recon in stats:
        diff = val_recon - val_exact
        print(f"{name:<20} {val_exact:>15.6f} {val_recon:>15.6f} "
              f"{diff:>15.6f}")

    print("\n" + "=" * 70)
    print("3. ANÁLISIS DE LA DIFERENCIA")
    print("=" * 70)

    diff = W_recon - W_exact

    print(f"\nDiferencia absoluta:")
    print(f"  Mínima:  {diff.min():>10.6f}")
    print(f"  Máxima:  {diff.max():>10.6f}")
    print(f"  Media:   {diff.mean():>10.6f}")
    print(f"  Desv.Est:{diff.std():>10.6f}")

    # Error relativo
    rel_error = np.abs(diff) / (np.abs(W_exact) + 1e-10)
    print(f"\nError relativo (%):")
    print(f"  Media:   {rel_error.mean() * 100:>10.2f}%")
    print(f"  Mediana: {np.median(rel_error) * 100:>10.2f}%")
    print(f"  Máximo:  {rel_error.max() * 100:>10.2f}%")

    print("\n" + "=" * 70)
    print("4. EJEMPLOS DE CONEXIONES ESPECÍFICAS")
    print("=" * 70)

    print("\nPrimeras 5 conexiones E→E (neurona 0 a neuronas 0-4):")
    print(f"{'Neurona j':<12} {'W_exact':>15} {'W_recon':>15} "
          f"{'Diff':>15}")
    print("-" * 60)
    for j in range(5):
        print(f"{j:<12} {W_exact[0, j]:>15.8f} {W_recon[0, j]:>15.8f} "
              f"{W_recon[0, j] - W_exact[0, j]:>15.8f}")

    print("\n" + "=" * 70)
    print("5. ANÁLISIS POR BLOQUES")
    print("=" * 70)

    N_E = 50

    # Extraer bloques
    blocks = {
        'E→E': (slice(0, N_E), slice(0, N_E)),
        'E→I': (slice(0, N_E), slice(N_E, 100)),
        'I→E': (slice(N_E, 100), slice(0, N_E)),
        'I→I': (slice(N_E, 100), slice(N_E, 100)),
    }

    print(f"\n{'Bloque':<8} {'Mean Exact':>12} {'Mean Recon':>12} "
          f"{'Diff Mean':>12} {'Diff Max':>12}")
    print("-" * 60)

    for name, (rows, cols) in blocks.items():
        exact_block = W_exact[rows, cols]
        recon_block = W_recon[rows, cols]
        diff_block = recon_block - exact_block

        print(f"{name:<8} {exact_block.mean():>12.6f} "
              f"{recon_block.mean():>12.6f} "
              f"{diff_block.mean():>12.6f} "
              f"{np.abs(diff_block).max():>12.6f}")

    print("\n" + "=" * 70)
    print("6. INTERPRETACIÓN")
    print("=" * 70)

    # Calcular si las diferencias son significativas
    max_abs_diff = np.abs(diff).max()
    mean_abs_diff = np.abs(diff).mean()
    max_val = np.abs(W_exact).max()

    rel_max_diff = (max_abs_diff / max_val) * 100

    print(f"\n¿Las diferencias son significativas?")
    print(f"  Diferencia máxima absoluta: {max_abs_diff:.6f}")
    print(f"  Como % del valor máximo de W: {rel_max_diff:.2f}%")

    if rel_max_diff < 5:
        print("\n  ✓ Las diferencias son PEQUEÑAS (<5%)")
        print("    Los 8 parámetros aproximan bien W_exact")
    elif rel_max_diff < 15:
        print("\n  ⚠ Las diferencias son MODERADAS (5-15%)")
        print("    Los 8 parámetros capturan la estructura general")
    else:
        print("\n  ✗ Las diferencias son GRANDES (>15%)")
        print("    Los 8 parámetros NO aproximan bien W_exact")

    print(f"\n¿Son matemáticamente equivalentes?")
    are_close = np.allclose(W_exact, W_recon, rtol=0.01, atol=0.01)
    print(f"  Con tolerancia del 1%: {are_close}")

    print("\n" + "=" * 70)
    print("CONCLUSIÓN")
    print("=" * 70)
    print("""
W_exact y W_reconstruida son DOS REPRESENTACIONES de la misma conectividad:

1. W_exact (10,000 valores):
   - Valores exactos del entrenamiento original
   - Usado para reproducir el paper exactamente

2. W_reconstruida (8 parámetros):
   - Aproximación paramétrica compacta
   - Usado para entender y adaptar el modelo

La diferencia es esperada y aceptable si es pequeña (<5-10%).
Ambas representaciones son válidas y útiles según el contexto.
    """)


if __name__ == "__main__":
    main()
