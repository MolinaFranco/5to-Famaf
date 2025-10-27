#!/usr/bin/env python3
"""
Análisis rápido de diferencias críticas con el código original.
"""

import sys
import numpy as np
import os

sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

def analyze_critical_differences():
    """Analizar solo las diferencias más críticas."""
    print("=" * 70)
    print("DIFERENCIAS CRÍTICAS CON CÓDIGO ORIGINAL")
    print("=" * 70)

    print("\n1. 🔍 MATRICES DE CONECTIVIDAD:")
    try:
        # Load matrices
        original_W = np.loadtxt('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/SSN/parameter_files/w_learn')
        our_W = np.loadtxt('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/w_learn')

        if np.allclose(original_W, our_W, rtol=1e-10):
            print("   ✅ IDÉNTICAS - Sin diferencias en conectividad")
        else:
            print(f"   ❌ DIFIEREN - Max diff: {np.max(np.abs(original_W - our_W)):.10f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n2. 📁 DATOS GSM:")
    gsm_original = '/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM/filters.npy'
    gsm_ours = '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/gsm/A'

    print(f"   Original GSM existe: {os.path.exists(gsm_original)}")
    print(f"   Nuestro GSM existe: {os.path.exists(gsm_ours)}")

    if not os.path.exists(gsm_ours):
        print("   ❌ PROBLEMA CRÍTICO: Falta datos GSM pre-entrenados")
        print("   ❌ Esto causa regeneración en cada ejecución")

    print("\n3. 🧪 PRUEBA SIMPLE DE VARIABILIDAD:")
    from skneuromsi.neural import Echeveste2020

    try:
        # Quick test with 3 runs
        activities = []
        for i in range(3):
            ssn = Echeveste2020(N_E=50, N_I=50, seed=None)
            ssn.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')
            response = ssn.run(stimulus_contrast=0.32, stimulus_orientation=45.0, simulation_time=100.0)

            df = response.get_modes()
            activity = df['excitatory'].values.reshape(500, 50)
            activities.append(np.mean(activity))
            print(f"   Ejecución {i+1}: media = {np.mean(activity):.6f}")

        variability = np.std(activities)
        print(f"   Variabilidad (std): {variability:.6f}")

        if variability > 0.01:
            print("   ✅ VARIABILIDAD ADECUADA")
        else:
            print("   ❌ VARIABILIDAD INSUFICIENTE")

    except Exception as e:
        print(f"   ❌ Error en prueba: {e}")

    print("\n4. 📊 PROBLEMAS IDENTIFICADOS:")
    print("   🔴 CRÍTICO: Datos GSM se regeneran en cada ejecución")
    print("   🔴 CRÍTICO: Posible diferencia en integración numérica")
    print("   🟡 MEDIO: Framework BrainPy vs original")
    print("   🟡 MEDIO: Estructura de salida diferente")

    print("\n5. 🎯 CAUSAS DE VARIACIÓN CON ORIGINAL:")
    print("   A. Regeneración de filtros Gabor en cada run")
    print("   B. Diferentes generadores de números aleatorios")
    print("   C. Implementación de ruido Ornstein-Uhlenbeck")
    print("   D. Resolución temporal fija vs adaptativa")

if __name__ == "__main__":
    analyze_critical_differences()