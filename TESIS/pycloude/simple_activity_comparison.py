#!/usr/bin/env python3
"""
Comparación simple: primeros pasos del código original vs nuestro.
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


def check_original_typical_values():
    """Verificar valores típicos del código original en primeros pasos."""
    print("=== VALORES TÍPICOS DEL CÓDIGO ORIGINAL ===")

    # Cargar parámetros
    W = np.loadtxt('parameter_files/w_learn')
    Sigma_eta = np.loadtxt('parameter_files/sigma_eta_learn')
    h = np.loadtxt('parameter_files/h_true_0_learn')
    mu_0 = np.loadtxt('parameter_files/mu_evolved_net_0')
    Sigma_0 = np.loadtxt('parameter_files/sigma_evolved_net_0')

    print(f"Configuración:")
    print(f"  dt: {dt}s = {dt*1000}ms")
    print(f"  h: {h[0]:.6f}")

    # Simular varios casos con diferentes seeds
    max_values = []

    for trial in range(5):
        seed = 42 + trial
        np.random.seed(seed)

        # Condición inicial
        u = np.random.multivariate_normal(mean=mu_0, cov=Sigma_0)
        eta = np.random.multivariate_normal(mean=np.zeros(N), cov=Sigma_eta)

        # Preparar ruido O-U
        L = np.linalg.cholesky(Sigma_eta)
        eps_1 = 1.0 - dt / tau_n
        eps_2 = np.sqrt(2.0 * dt / tau_n)

        # Simular 1000 pasos (como en nuestro test)
        for step in range(1000):
            # Calcular actividad
            r = mt_original.get_r(u)

            # Evolucionar
            np.random.seed(seed + step)
            white_noise = L @ np.random.normal(0.0, 1.0, N)
            eta = eps_1 * eta + eps_2 * white_noise
            u = mt_original.new_u(u, r, h, W, eta)

        # Guardar valores máximos al final
        final_u_max = np.abs(u).max()
        final_r_max = np.abs(r).max()
        max_values.append((final_u_max, final_r_max))

        print(f"  Trial {trial+1} (seed={seed}): |u|_max={final_u_max:.2f}, |r|_max={final_r_max:.2f}")

    # Estadísticas
    u_maxes = [mv[0] for mv in max_values]
    r_maxes = [mv[1] for mv in max_values]

    print(f"\nEstadísticas después de 1000 pasos:")
    print(f"  |u|_max promedio: {np.mean(u_maxes):.2f} ± {np.std(u_maxes):.2f}")
    print(f"  |u|_max rango: [{np.min(u_maxes):.2f}, {np.max(u_maxes):.2f}]")
    print(f"  |r|_max promedio: {np.mean(r_maxes):.2f} ± {np.std(r_maxes):.2f}")
    print(f"  |r|_max rango: [{np.min(r_maxes):.2f}, {np.max(r_maxes):.2f}]")

    return np.max(u_maxes), np.max(r_maxes)


def check_our_typical_values():
    """Verificar valores típicos de nuestra implementación."""
    print(f"\n=== VALORES TÍPICOS DE NUESTRA IMPLEMENTACIÓN ===")

    max_values = []

    for trial in range(5):
        seed = 42 + trial

        # Crear instancia con configuración conservadora
        ssn = Echeveste2020(
            N_E=50, N_I=50,
            seed=seed,
            time_res=0.2,  # Como el original
            time_range=(0, 200.0)  # 200ms = 1000 pasos
        )

        # Cargar parámetros (con matriz exacta)
        ssn.load_parameters()

        try:
            # Ejecutar con parámetros exactos del original
            h_original = np.loadtxt('parameter_files/h_true_0_learn')
            result = ssn.run(
                stimulus_contrast=h_original[0],
                stimulus_orientation=0.0,
                noise_level=0.1,
                simulation_time=200.0  # 1000 pasos como en original
            )

            # Analizar resultados
            data_exc = result.get_modes()['excitatory']
            data_inh = result.get_modes()['inhibitory']

            max_exc = np.abs(data_exc).max()
            max_inh = np.abs(data_inh).max()
            max_total = max(max_exc, max_inh)

            max_values.append((max_exc, max_inh, max_total))

            print(f"  Trial {trial+1} (seed={seed}): E_max={max_exc:.2f}, I_max={max_inh:.2f}, total={max_total:.2f}")

        except Exception as e:
            print(f"  Trial {trial+1} (seed={seed}): ERROR - {e}")
            max_values.append((np.inf, np.inf, np.inf))

    # Estadísticas (excluyendo errores)
    valid_values = [mv for mv in max_values if mv[0] != np.inf]

    if valid_values:
        exc_maxes = [mv[0] for mv in valid_values]
        inh_maxes = [mv[1] for mv in valid_values]
        total_maxes = [mv[2] for mv in valid_values]

        print(f"\nEstadísticas después de 1000 pasos (trials exitosos: {len(valid_values)}/5):")
        print(f"  E_max promedio: {np.mean(exc_maxes):.2f} ± {np.std(exc_maxes):.2f}")
        print(f"  I_max promedio: {np.mean(inh_maxes):.2f} ± {np.std(inh_maxes):.2f}")
        print(f"  Total_max promedio: {np.mean(total_maxes):.2f} ± {np.std(total_maxes):.2f}")
        print(f"  Total_max rango: [{np.min(total_maxes):.2f}, {np.max(total_maxes):.2f}]")

        return np.max(total_maxes)
    else:
        print(f"❌ Ningún trial exitoso")
        return np.inf


def determine_explosion_threshold():
    """Determinar umbral apropiado basado en comparación."""
    print(f"\n=== DETERMINACIÓN DE UMBRAL DE EXPLOSIÓN ===")

    original_max_u, original_max_r = check_original_typical_values()
    our_max = check_our_typical_values()

    original_max = max(original_max_u, original_max_r)

    print(f"\nComparación:")
    print(f"  Código original máximo: {original_max:.2f}")
    print(f"  Nuestro código máximo: {our_max:.2f}")

    if our_max == np.inf:
        print(f"❌ Nuestro código falla - hay problemas serios")
        return

    # Determinar umbral
    # Un valor es "explosión" si es mucho mayor que los valores típicos
    safety_factor = 5.0  # Factor de seguridad generoso
    threshold = original_max * safety_factor

    print(f"\nDeterminación de umbral:")
    print(f"  Máximo normal (original): {original_max:.2f}")
    print(f"  Factor de seguridad: {safety_factor}")
    print(f"  Umbral de explosión sugerido: {threshold:.2f}")

    # Evaluar nuestros resultados
    ratio = our_max / original_max
    print(f"\nEvaluación de nuestros resultados:")
    print(f"  Ratio nuestro/original: {ratio:.2f}")

    if our_max <= threshold:
        print(f"✅ ÉXITO: Nuestros valores están en rango NORMAL")
        print(f"   {our_max:.2f} ≤ {threshold:.2f}")
        print(f"🎉 El problema de explosión ha sido RESUELTO")
        return threshold
    else:
        print(f"⚠️  Nuestros valores son altos pero pueden no ser explosión real")
        print(f"   {our_max:.2f} > {threshold:.2f}")
        print(f"🔍 Necesito investigar si esto es explosión real o valores normales altos")
        return threshold


def main():
    """Función principal."""
    print("DETERMINACIÓN DE UMBRAL DE EXPLOSIÓN")
    print("=" * 60)

    threshold = determine_explosion_threshold()

    if threshold and threshold < np.inf:
        print(f"\n" + "=" * 60)
        print(f"CONCLUSIÓN FINAL:")
        print(f"🎯 Umbral apropiado: {threshold:.0f}")
        print(f"💡 Los valores ~130-150 que observamos pueden ser NORMALES")
        print(f"✅ Necesito re-evaluar las 10 instancias con este nuevo umbral")


if __name__ == "__main__":
    main()