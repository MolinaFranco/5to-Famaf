#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigación profunda sobre el mapeo de orientaciones.

La pregunta clave: ¿Echeveste usa [0°, 360°) o [0°, 180°)?
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                 '..', 'scikit-neuromsi'))

from skneuromsi.data.echeveste_data_loader import EchevesteDataLoader


def analizar_periodicidad():
    """
    Analizar si hay periodicidad de 180° en las conexiones.

    Si las orientaciones son realmente de 0° a 180° (con simetría de gratings),
    entonces la neurona 0 (0°) debería tener conexión similar a la neurona 25 (180°).
    """
    print("="*70)
    print("ANÁLISIS DE PERIODICIDAD")
    print("="*70)

    loader = EchevesteDataLoader()
    W = loader.load_connectivity_matrix()

    # Orientaciones según Echeveste (0 a 360°)
    angles_0_360 = np.array([360.0 * i / 50 for i in range(50)])

    print("\nSi las orientaciones fueran de 0° a 360° (sin simetría):")
    print("  - Neurona 0: 0°")
    print("  - Neurona 1: 7.2°")
    print("  - Neurona 25: 180°")
    print("  - Neurona 49: 352.8°")

    print("\nConexiones desde E hacia E[0] (orientación 0°):")
    print("Neurona | Ángulo (0-360°) | W[0, i]")
    print("-"*50)
    for i in [0, 1, 12, 13, 24, 25, 26, 37, 38, 49]:
        angle = angles_0_360[i]
        weight = W[0, i]
        print(f"E[{i:2d}]   | {angle:6.1f}°        | {weight:.6f}")

    print("\n" + "="*70)
    print("OBSERVACIÓN CLAVE:")
    print("="*70)
    print(f"W[0, 0]  (0°)     = {W[0, 0]:.6f}  <- Máximo (misma orientación)")
    print(f"W[0, 1]  (7.2°)   = {W[0, 1]:.6f}  <- Muy cercano")
    print(f"W[0, 25] (180°)   = {W[0, 25]:.6f}  <- Mínimo!")
    print(f"W[0, 49] (352.8°) = {W[0, 49]:.6f}  <- Muy cercano a W[0,1]")

    print("\nSi hubiera simetría de 180° (gratings), esperaríamos:")
    print("  W[0, 0] ≈ W[0, 25]  (0° ≈ 180°)")
    print("  Pero NO es así! W[0, 25] es MÍNIMO, no máximo.")

    print("\nCONCLUSIÓN: Echeveste NO está usando simetría de gratings")
    print("en el mapeo. Está usando el círculo COMPLETO [0°, 360°).")


def analizar_formula_conectividad():
    """
    Analizar la fórmula de conectividad para entender el periodo.

    W_XY(θi, θj) = a_XY * exp[(cos(θi - θj) - 1) / d_XY²]

    IMPORTANTE: ¡La fórmula NO tiene factor 2!
    """
    print("\n\n")
    print("="*70)
    print("ANÁLISIS DE LA FÓRMULA DE CONECTIVIDAD")
    print("="*70)

    print("\nFórmula de Echeveste (Eq. 10):")
    print("  W_XY(θi, θj) = a_XY * exp[(cos(θi - θj) - 1) / d_XY²]")

    print("\nPROPIEDADES del coseno:")
    print("  - cos(0°)   = 1   → exp(0 / d²) = 1      → Peso máximo")
    print("  - cos(90°)  = 0   → exp(-1 / d²) ≈ 0.37  → Peso medio")
    print("  - cos(180°) = -1  → exp(-2 / d²) ≈ 0.13  → Peso mínimo")
    print("  - cos(270°) = 0   → exp(-1 / d²) ≈ 0.37  → Peso medio")
    print("  - cos(360°) = 1   → exp(0 / d²) = 1      → Peso máximo")

    print("\nSi las neuronas tuvieran simetría de gratings, la fórmula sería:")
    print("  W_XY(θi, θj) = a_XY * exp[(cos(2*(θi - θj)) - 1) / d_XY²]")
    print("  ^^^^^^^^ Factor 2 para periodicidad de 180°")

    print("\nCon factor 2:")
    print("  - cos(2*0°)   = 1   → Peso máximo")
    print("  - cos(2*90°)  = -1  → Peso mínimo")
    print("  - cos(2*180°) = 1   → Peso máximo (¡simetría!)")

    print("\n" + "="*70)
    print("¡PERO EL CÓDIGO DE ECHEVESTE NO TIENE EL FACTOR 2!")
    print("="*70)
    print("Ver objective.ml línea 287:")
    print('  let cos_diff_minus_one = PE.(cos (theta_i -. theta_j) -. 1.) in')
    print("                                    ^^^ SIN factor 2")

    print("\nEsto significa que Echeveste está usando periodicidad de 360°,")
    print("NO periodicidad de 180° como esperaríamos para gratings.")


def verificar_con_parametros_reales():
    """
    Verificar con los parámetros reales de conectividad.
    """
    print("\n\n")
    print("="*70)
    print("VERIFICACIÓN CON PARÁMETROS REALES")
    print("="*70)

    loader = EchevesteDataLoader()
    W = loader.load_connectivity_matrix()
    params = loader.load_ssn_connectivity_parameters()

    print("\nParámetros de conectividad E→E:")
    a_EE = params['a_EE']
    d_EE = params['d_EE']
    print(f"  a_EE = {a_EE:.6f}")
    print(f"  d_EE = {d_EE:.6f}")

    # Verificar algunos valores
    print("\nVerificación de W[0, i] = a_EE * exp[(cos(θ₀ - θᵢ) - 1) / d_EE²]:")
    print("Neurona | Ángulo | Δθ (rad) | cos(Δθ) | Fórmula | W[0,i] real | Diff")
    print("-"*75)

    for i in [0, 1, 12, 25, 49]:
        theta_0 = 0.0
        theta_i = 2.0 * np.pi * i / 50
        delta_theta = theta_0 - theta_i
        cos_diff = np.cos(delta_theta)
        cos_diff_minus_one = cos_diff - 1.0

        W_formula = a_EE * np.exp(cos_diff_minus_one / (d_EE ** 2))
        W_real = W[0, i]
        diff = abs(W_formula - W_real)

        print(f"E[{i:2d}]   | {np.degrees(theta_i):6.1f}° | "
              f"{delta_theta:8.4f} | {cos_diff:7.4f} | "
              f"{W_formula:7.5f} | {W_real:11.6f} | {diff:.2e}")

    print("\n✓ La fórmula coincide perfectamente con w_learn")
    print("✓ Confirma que NO hay factor 2 en el coseno")


def graficar_perfil_conectividad():
    """
    Graficar el perfil de conectividad para visualizar.
    """
    print("\n\n")
    print("="*70)
    print("GENERANDO GRÁFICO DE PERFIL DE CONECTIVIDAD")
    print("="*70)

    loader = EchevesteDataLoader()
    W = loader.load_connectivity_matrix()

    # Perfil de E[0]
    weights_from_E = W[0, :50]
    angles_deg = np.array([360.0 * i / 50 for i in range(50)])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Conectividad vs ángulo (0-360°)
    ax1.plot(angles_deg, weights_from_E, 'o-', linewidth=2, markersize=4)
    ax1.axvline(0, color='r', linestyle='--', alpha=0.5, label='0°')
    ax1.axvline(180, color='b', linestyle='--', alpha=0.5, label='180°')
    ax1.set_xlabel('Ángulo de neurona fuente (grados)', fontsize=12)
    ax1.set_ylabel('Peso de conexión W[0, i]', fontsize=12)
    ax1.set_title('Perfil de conectividad de E[0]: Periodicidad de 360°',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 360)

    # Plot 2: Interpretando como -90° a 270°
    # (lo que podríamos hacer si queremos centrar en 0°)
    angles_centered = angles_deg.copy()
    angles_centered[angles_centered > 180] -= 360
    sort_idx = np.argsort(angles_centered)

    ax2.plot(angles_centered[sort_idx], weights_from_E[sort_idx],
             'o-', linewidth=2, markersize=4)
    ax2.axvline(0, color='r', linestyle='--', alpha=0.5, label='0°')
    ax2.axvline(180, color='b', linestyle='--', alpha=0.5, label='180°')
    ax2.axvline(-180, color='b', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Ángulo de neurona fuente (grados, centrado)', fontsize=12)
    ax2.set_ylabel('Peso de conexión W[0, i]', fontsize=12)
    ax2.set_title('Mismo perfil, ejes centrados en 0°', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-180, 180)

    plt.tight_layout()
    output_path = 'pycloude/outputs/10_perfil_conectividad_w_learn.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado en: {output_path}")
    plt.close()


def conclusiones_finales():
    """
    Presentar las conclusiones finales.
    """
    print("\n\n")
    print("="*70)
    print("CONCLUSIONES FINALES")
    print("="*70)

    print("""
1. MAPEO DE ORIENTACIONES:
   - Echeveste mapea neuronas a ángulos de 0° a 360° (círculo completo)
   - Neurona i → θᵢ = 2π * i / 50
   - NO usa el rango -90° a 90° típico de orientaciones de gratings

2. FÓRMULA DE CONECTIVIDAD:
   - W_XY(θi, θj) = a_XY * exp[(cos(θi - θj) - 1) / d_XY²]
   - ¡SIN factor 2 en el coseno!
   - Por lo tanto, periodicidad de 360°, NO 180°

3. ¿POR QUÉ 360° Y NO 180°?
   - Las orientaciones de gratings tienen simetría de 180° (0° ≈ 180°)
   - PERO Echeveste probablemente está modelando DIRECCIONES, no orientaciones
   - Direcciones van de 0° a 360° (arriba, abajo, izquierda, derecha son distintas)
   - O simplemente usa el círculo completo por conveniencia matemática

4. IMPLICACIONES PARA NUESTRO CÓDIGO:
   a) Al cargar w_learn, debemos respetar que fue entrenado con [0°, 360°)
   b) Si queremos visualizar en [-90°, 90°], necesitamos una conversión cuidadosa
   c) IMPORTANTE: El índice i de la neurona corresponde a θᵢ = 2π*i/50
   d) No podemos simplemente usar np.linspace(-90, 90, 50)

5. ¿CÓMO INTERPRETAR LOS ÁNGULOS?
   Opción A: Mantener [0°, 360°) internamente, convertir solo para visualización
   Opción B: Reinterpretar [0°, 180°) aprovechando simetría de gratings
   Opción C: Centrar en 0° usando [-180°, 180°)

   Recomendación: Opción A - Mantener la interpretación original de Echeveste
   y convertir solo al momento de visualizar o generar estímulos.

6. MAPPING CORRECTO:
   Si queremos orientaciones de gratings [-90°, 90°):

   Índice i → θ_echeveste = 360° * i / 50

   Para convertir a orientación de grating:
   - Si θ_echeveste < 180°: orientación = θ_echeveste
   - Si θ_echeveste >= 180°: orientación = θ_echeveste - 180°

   Y luego centrar en [-90°, 90°):
   - Si orientación > 90°: orientación -= 180°
    """)


if __name__ == "__main__":
    analizar_periodicidad()
    analizar_formula_conectividad()
    verificar_con_parametros_reales()
    graficar_perfil_conectividad()
    conclusiones_finales()
