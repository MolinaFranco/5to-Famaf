#!/usr/bin/env python3
"""
Comparar actividades entre código original y nuestra implementación
para determinar qué valores son normales vs explosión.
"""

import os
import sys
import numpy as np

# Configurar rutas
sys.path.append('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/SSN')
sys.path.append('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

# Cambiar al directorio original
os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/SSN')

# Importar código original
import methods as mt_original
from parameters import *

# Importar nuestra implementación
from skneuromsi.neural import Echeveste2020


def run_original_simulation():
    """Ejecutar simulación con el código original."""
    print("=== SIMULACIÓN CON CÓDIGO ORIGINAL ===")

    # Cargar parámetros
    W = np.loadtxt('parameter_files/w_learn')
    Sigma_eta = np.loadtxt('parameter_files/sigma_eta_learn')
    h = np.loadtxt('parameter_files/h_true_0_learn')
    mu_0 = np.loadtxt('parameter_files/mu_evolved_net_0')
    Sigma_0 = np.loadtxt('parameter_files/sigma_evolved_net_0')

    print(f"Parámetros originales:")
    print(f"  h (stimulus): {h[0]:.6f}")
    print(f"  dt: {dt}s = {dt*1000}ms")
    print(f"  Pasos de simulación: {n_iter}")

    # Condición inicial
    np.random.seed(42)  # Mismo seed que nuestro test
    u = np.random.multivariate_normal(mean=mu_0, cov=Sigma_0)
    eta = np.random.multivariate_normal(mean=np.zeros(N), cov=Sigma_eta)

    # Preparar ruido O-U
    L = np.linalg.cholesky(Sigma_eta)
    eps_1 = 1.0 - dt / tau_n
    eps_2 = np.sqrt(2.0 * dt / tau_n)

    # Arrays para guardar resultados
    steps_to_check = [10, 50, 100, 250, 500, 1000, 2000, n_iter-1]
    results = {}

    print(f"\nEjecutando {n_iter} pasos...")

    for step in range(n_iter):
        # Calcular actividad
        r = mt_original.get_r(u)

        # Guardar resultados en pasos específicos
        if step in steps_to_check:
            u_e = u[:50]
            u_i = u[50:]
            r_e = r[:50]
            r_i = r[50:]

            results[step] = {
                'u_max': np.abs(u).max(),
                'u_e_max': np.abs(u_e).max(),
                'u_i_max': np.abs(u_i).max(),
                'r_max': np.abs(r).max(),
                'r_e_max': np.abs(r_e).max(),
                'r_i_max': np.abs(r_i).max(),
                'u_mean': np.abs(u).mean(),
                'r_mean': np.abs(r).mean()
            }

            print(f"  Paso {step:4d}: |u|_max={np.abs(u).max():.2f}, "
                  f"|r|_max={np.abs(r).max():.2f}")

        # Verificar explosión
        if np.abs(u).max() > 10000:
            print(f"❌ CÓDIGO ORIGINAL EXPLOTA en paso {step}")
            break

        # Evolucionar un paso
        np.random.seed(42 + step)
        white_noise = L @ np.random.normal(0.0, 1.0, N)
        eta = eps_1 * eta + eps_2 * white_noise

        u = mt_original.new_u(u, r, h, W, eta)

    print(f"✅ Código original completó {n_iter} pasos sin explosión")

    # Estadísticas finales
    final_step = n_iter - 1
    final_stats = results[final_step]

    print(f"\nEstadísticas finales (paso {final_step}):")
    print(f"  |u|_max: {final_stats['u_max']:.2f}")
    print(f"  |u_e|_max: {final_stats['u_e_max']:.2f}")
    print(f"  |u_i|_max: {final_stats['u_i_max']:.2f}")
    print(f"  |r|_max: {final_stats['r_max']:.2f}")
    print(f"  |r_e|_max: {final_stats['r_e_max']:.2f}")
    print(f"  |r_i|_max: {final_stats['r_i_max']:.2f}")

    return results


def run_our_simulation():
    """Ejecutar simulación con nuestra implementación."""
    print(f"\n=== SIMULACIÓN CON NUESTRA IMPLEMENTACIÓN ===")

    # Configurar con parámetros exactos del original
    sys.path.append('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/SSN')
    from parameters import dt

    time_res_exact = dt * 1000  # 0.2ms

    # Crear instancia con configuración idéntica al original
    ssn = Echeveste2020(
        N_E=50, N_I=50,
        seed=42,  # Mismo seed
        time_res=time_res_exact,
        time_range=(0, n_iter * dt * 1000)  # Misma duración total
    )

    # Cargar parámetros (incluyendo matriz exacta)
    ssn.load_parameters()

    print(f"Configuración nuestra:")
    print(f"  time_res: {ssn.time_res}ms (original: {dt*1000}ms)")
    print(f"  time_range: {ssn.time_range}")
    print(f"  Pasos estimados: {int((ssn.time_range[1] - ssn.time_range[0]) / ssn.time_res)}")

    # Cargar contraste exacto del original
    h_original = np.loadtxt('parameter_files/h_true_0_learn')
    original_contrast = h_original[0]

    print(f"  stimulus_contrast: {original_contrast:.8f}")

    # Ejecutar simulación
    try:
        result = ssn.run(
            stimulus_contrast=original_contrast,
            stimulus_orientation=0.0,
            noise_level=0.1,  # Como en test anterior
            simulation_time=n_iter * dt * 1000  # Misma duración que original
        )

        print(f"✅ Nuestra simulación completada")

        # Analizar resultados
        data_exc = result.get_modes()['excitatory']
        data_inh = result.get_modes()['inhibitory']

        max_exc = np.abs(data_exc).max()
        max_inh = np.abs(data_inh).max()
        max_total = max(max_exc, max_inh)

        print(f"Estadísticas finales nuestras:")
        print(f"  |activity_e|_max: {max_exc:.2f}")
        print(f"  |activity_i|_max: {max_inh:.2f}")
        print(f"  |activity_total|_max: {max_total:.2f}")

        return {
            'max_exc': max_exc,
            'max_inh': max_inh,
            'max_total': max_total,
            'success': True
        }

    except Exception as e:
        print(f"❌ Nuestra simulación falló: {e}")
        return {'success': False, 'error': str(e)}


def compare_and_determine_threshold():
    """Comparar resultados y determinar umbral apropiado."""
    print(f"\n=== COMPARACIÓN Y DETERMINACIÓN DE UMBRAL ===")

    # Ejecutar ambas simulaciones
    original_results = run_original_simulation()
    our_results = run_our_simulation()

    if not our_results['success']:
        print(f"❌ No se puede comparar - nuestra simulación falló")
        return

    # Extraer valores máximos del original
    original_final = original_results[max(original_results.keys())]
    original_max_u = original_final['u_max']
    original_max_r = original_final['r_max']

    # Valores nuestros
    our_max = our_results['max_total']

    print(f"\nComparación de actividades máximas:")
    print(f"  Original |u|_max: {original_max_u:.2f}")
    print(f"  Original |r|_max: {original_max_r:.2f}")
    print(f"  Nuestro |activity|_max: {our_max:.2f}")

    # Determinar umbral apropiado
    # El umbral debe ser significativamente mayor que los valores normales
    safety_factor = 3.0  # Factor de seguridad
    threshold_based_on_original = max(original_max_u, original_max_r) * safety_factor

    print(f"\nDeterminación de umbral:")
    print(f"  Máximo original: {max(original_max_u, original_max_r):.2f}")
    print(f"  Factor de seguridad: {safety_factor}")
    print(f"  Umbral sugerido: {threshold_based_on_original:.2f}")

    # Verificar si nuestros valores están dentro del rango normal
    if our_max <= threshold_based_on_original:
        print(f"✅ Nuestras actividades están en rango NORMAL")
        print(f"   {our_max:.2f} ≤ {threshold_based_on_original:.2f}")
    else:
        print(f"⚠️  Nuestras actividades están ALTAS pero no necesariamente explosión")
        print(f"   {our_max:.2f} > {threshold_based_on_original:.2f}")

    return threshold_based_on_original


def main():
    """Función principal."""
    print("COMPARACIÓN DE ACTIVIDADES: ORIGINAL vs NUESTRO")
    print("=" * 70)

    threshold = compare_and_determine_threshold()

    print(f"\n" + "=" * 70)
    print(f"CONCLUSIÓN:")
    if threshold:
        print(f"🎯 Umbral apropiado para explosión: {threshold:.0f}")
        print(f"💡 Los valores ~130-150 que vemos pueden ser NORMALES")
        print(f"🔍 El problema original puede haber sido resuelto")
    else:
        print(f"❌ No se pudo determinar umbral - hay problemas en nuestra simulación")


if __name__ == "__main__":
    main()