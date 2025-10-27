#!/usr/bin/env python3
"""
Test: 10 ejecuciones con parámetros DISTINTOS entre sí.

Este test responde la pregunta específica:
"¿Las 10 ejecuciones con distintos parámetros entre sí dan los mismos
resultados numéricos y llegan a las mismas causas?"

Esperamos que parámetros DISTINTOS den resultados DISTINTOS.
"""

import sys
import numpy as np

sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

from skneuromsi.neural import Echeveste2020


def generate_distinct_parameters():
    """Generar 10 conjuntos de parámetros DISTINTOS entre sí."""

    parameter_sets = [
        {
            "name": "Contraste muy bajo",
            "stimulus_contrast": 0.05,
            "stimulus_orientation": 0.0,
            "noise_level": 0.1,
            "simulation_time": 1000.0
        },
        {
            "name": "Contraste bajo",
            "stimulus_contrast": 0.15,
            "stimulus_orientation": 45.0,
            "noise_level": 0.1,
            "simulation_time": 1000.0
        },
        {
            "name": "Contraste estándar",
            "stimulus_contrast": 0.32,
            "stimulus_orientation": 90.0,
            "noise_level": 0.1,
            "simulation_time": 1000.0
        },
        {
            "name": "Contraste alto",
            "stimulus_contrast": 0.6,
            "stimulus_orientation": 135.0,
            "noise_level": 0.1,
            "simulation_time": 1000.0
        },
        {
            "name": "Contraste muy alto",
            "stimulus_contrast": 0.9,
            "stimulus_orientation": 180.0,
            "noise_level": 0.1,
            "simulation_time": 1000.0
        },
        {
            "name": "Ruido bajo",
            "stimulus_contrast": 0.32,
            "stimulus_orientation": 0.0,
            "noise_level": 0.05,
            "simulation_time": 1000.0
        },
        {
            "name": "Ruido alto",
            "stimulus_contrast": 0.32,
            "stimulus_orientation": 0.0,
            "noise_level": 0.3,
            "simulation_time": 1000.0
        },
        {
            "name": "Simulación corta",
            "stimulus_contrast": 0.32,
            "stimulus_orientation": 0.0,
            "noise_level": 0.1,
            "simulation_time": 500.0
        },
        {
            "name": "Simulación larga",
            "stimulus_contrast": 0.32,
            "stimulus_orientation": 0.0,
            "noise_level": 0.1,
            "simulation_time": 2000.0
        },
        {
            "name": "Combinación extrema",
            "stimulus_contrast": 0.8,
            "stimulus_orientation": 270.0,
            "noise_level": 0.25,
            "simulation_time": 1500.0
        }
    ]

    return parameter_sets


def run_with_distinct_parameters():
    """Ejecutar 10 simulaciones con parámetros DISTINTOS."""
    print("🔬 EJECUTANDO 10 SIMULACIONES CON PARÁMETROS DISTINTOS")
    print("=" * 60)

    parameter_sets = generate_distinct_parameters()
    results = []

    for i, params in enumerate(parameter_sets):
        print(f"\n📊 Ejecución {i+1}/10: {params['name']}")
        print(f"   Parámetros: contrast={params['stimulus_contrast']}, "
              f"orientation={params['stimulus_orientation']}, "
              f"noise={params['noise_level']}")

        try:
            # Create fresh instance with fixed seed for reproducibility
            ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
            ssn.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')

            # Run simulation with distinct parameters
            response = ssn.run(
                stimulus_contrast=params['stimulus_contrast'],
                stimulus_orientation=params['stimulus_orientation'],
                simulation_time=params['simulation_time'],
                noise_level=params['noise_level']
            )

            # Extract activity
            df = response.get_modes()
            n_times = int(params['simulation_time'] / 0.2)  # time_res = 0.2
            excitatory_activity = df['excitatory'].values.reshape(n_times, 50)
            inhibitory_activity = df['inhibitory'].values.reshape(n_times, 50)

            # Use mean activity across time as network state for causal inference
            mean_exc_activity = np.mean(excitatory_activity, axis=0)  # Shape: (50,)
            mean_inh_activity = np.mean(inhibitory_activity, axis=0)  # Shape: (50,)

            # Combine E and I for full network activity vector (100,)
            network_state = np.concatenate([mean_exc_activity, mean_inh_activity])

            # Calculate causes using network state
            causes = ssn.calculate_causes(
                network_activity=network_state,
                confidence_threshold=0.95
            )

            # Calculate detailed statistics
            activity_mean = np.mean(excitatory_activity)
            activity_std = np.std(excitatory_activity)
            activity_max = np.max(excitatory_activity)
            final_state_mean = np.mean(excitatory_activity[-100:])  # Last 100 time points

            # Store results
            result = {
                'run': i + 1,
                'name': params['name'],
                'parameters': params,
                'activity_mean': activity_mean,
                'activity_std': activity_std,
                'activity_max': activity_max,
                'final_state_mean': final_state_mean,
                'n_causes': len(causes),
                'causes': causes,
                'success': True
            }

            results.append(result)

            print(f"   ✅ ÉXITO:")
            print(f"      Actividad media: {activity_mean:.4f}")
            print(f"      Actividad final: {final_state_mean:.4f}")
            print(f"      Causas detectadas: {len(causes)}")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({
                'run': i + 1,
                'name': params['name'],
                'parameters': params,
                'error': str(e),
                'success': False
            })

    return results


def analyze_numerical_differences(results):
    """Analizar diferencias numéricas entre ejecuciones con parámetros distintos."""
    print("\n" + "=" * 60)
    print("ANÁLISIS DE DIFERENCIAS NUMÉRICAS")
    print("=" * 60)

    successful_results = [r for r in results if r['success']]

    if len(successful_results) < 2:
        print("❌ Insuficientes resultados exitosos para comparar")
        return

    print(f"✅ Resultados exitosos: {len(successful_results)}/10")

    # Extract numerical metrics
    activity_means = [r['activity_mean'] for r in successful_results]
    activity_stds = [r['activity_std'] for r in successful_results]
    final_states = [r['final_state_mean'] for r in successful_results]

    print(f"\n📊 VARIABILIDAD NUMÉRICA CON PARÁMETROS DISTINTOS:")
    print(f"   Actividad media:")
    print(f"      Rango: [{np.min(activity_means):.4f}, {np.max(activity_means):.4f}]")
    print(f"      Desviación estándar: {np.std(activity_means):.4f}")
    print(f"      Coeficiente de variación: {np.std(activity_means)/np.mean(activity_means)*100:.2f}%")

    print(f"\n   Estados finales:")
    print(f"      Rango: [{np.min(final_states):.4f}, {np.max(final_states):.4f}]")
    print(f"      Desviación estándar: {np.std(final_states):.4f}")

    # Test if results are significantly different
    max_diff_means = np.max(activity_means) - np.min(activity_means)
    max_diff_finals = np.max(final_states) - np.min(final_states)

    print(f"\n🔍 ANÁLISIS DE DIFERENCIAS:")
    print(f"   Diferencia máxima en medias: {max_diff_means:.4f}")
    print(f"   Diferencia máxima en finales: {max_diff_finals:.4f}")

    if max_diff_means > 0.1:
        print(f"   ✅ Parámetros DISTINTOS → Resultados DISTINTOS (como esperado)")
    else:
        print(f"   ⚠️  Parámetros distintos → Resultados similares (inesperado)")

    return activity_means, final_states


def analyze_causal_differences(results):
    """Analizar diferencias en detección de causas."""
    print("\n" + "=" * 60)
    print("ANÁLISIS DE DETECCIÓN DE CAUSAS")
    print("=" * 60)

    successful_results = [r for r in results if r['success']]

    if len(successful_results) == 0:
        print("❌ No hay resultados exitosos para analizar causas")
        return

    # Extract causal information
    n_causes_list = [r['n_causes'] for r in successful_results]
    names = [r['name'] for r in successful_results]

    print(f"📈 DETECCIÓN DE CAUSAS POR CONDICIÓN:")
    print("Condición                 | Causas | Parámetros clave")
    print("-" * 55)

    for result in successful_results:
        params = result['parameters']
        key_params = f"c={params['stimulus_contrast']:.2f}, n={params['noise_level']:.2f}"
        print(f"{result['name']:25} | {result['n_causes']:6} | {key_params}")

    # Analyze patterns
    unique_cause_counts = set(n_causes_list)

    print(f"\n🔍 ANÁLISIS DE PATRONES:")
    print(f"   Número único de causas detectadas: {sorted(unique_cause_counts)}")
    print(f"   Variabilidad en detección: {len(unique_cause_counts)} niveles distintos")

    if len(unique_cause_counts) > 1:
        print(f"   ✅ Parámetros DISTINTOS → Causas DISTINTAS (comportamiento correcto)")

        # Find which parameters lead to more/fewer causes
        max_causes = max(n_causes_list)
        min_causes = min(n_causes_list)

        max_idx = n_causes_list.index(max_causes)
        min_idx = n_causes_list.index(min_causes)

        print(f"\n   📊 Extremos de detección:")
        print(f"      Más causas ({max_causes}): {names[max_idx]}")
        print(f"      Menos causas ({min_causes}): {names[min_idx]}")

    else:
        print(f"   ⚠️  Parámetros distintos → Mismas causas ({unique_cause_counts.pop()})")
        print(f"   Esto podría indicar baja sensibilidad o condiciones similares")

    return n_causes_list


def correlation_analysis(results):
    """Analizar correlaciones entre parámetros y resultados."""
    print("\n" + "=" * 60)
    print("ANÁLISIS DE CORRELACIONES PARÁMETRO-RESULTADO")
    print("=" * 60)

    successful_results = [r for r in results if r['success']]

    if len(successful_results) < 3:
        print("❌ Insuficientes datos para análisis de correlación")
        return

    # Extract parameter values and results
    contrasts = [r['parameters']['stimulus_contrast'] for r in successful_results]
    orientations = [r['parameters']['stimulus_orientation'] for r in successful_results]
    noise_levels = [r['parameters']['noise_level'] for r in successful_results]
    activity_means = [r['activity_mean'] for r in successful_results]
    n_causes = [r['n_causes'] for r in successful_results]

    print(f"🔍 CORRELACIONES PARÁMETRO → RESULTADO:")

    # Contrast vs Activity
    contrast_activity_corr = np.corrcoef(contrasts, activity_means)[0,1]
    print(f"   Contraste ↔ Actividad media: {contrast_activity_corr:.3f}")

    # Noise vs Activity variability
    noise_std_corr = np.corrcoef(noise_levels, [r['activity_std'] for r in successful_results])[0,1]
    print(f"   Ruido ↔ Variabilidad: {noise_std_corr:.3f}")

    # Contrast vs Causes
    if len(set(n_causes)) > 1:  # Only if there's variation in causes
        contrast_causes_corr = np.corrcoef(contrasts, n_causes)[0,1]
        print(f"   Contraste ↔ Número de causas: {contrast_causes_corr:.3f}")

    print(f"\n📊 INTERPRETACIÓN:")
    if abs(contrast_activity_corr) > 0.3:
        direction = "positiva" if contrast_activity_corr > 0 else "negativa"
        print(f"   ✅ Correlación {direction} fuerte entre contraste y actividad")
    else:
        print(f"   ⚠️  Correlación débil entre contraste y actividad")

    if abs(noise_std_corr) > 0.3:
        print(f"   ✅ El ruido afecta la variabilidad como esperado")
    else:
        print(f"   ⚠️  El ruido no afecta significativamente la variabilidad")


def final_verdict(results):
    """Veredicto final sobre la pregunta del usuario."""
    print("\n" + "=" * 60)
    print("RESPUESTA A LA PREGUNTA DEL USUARIO")
    print("=" * 60)

    successful_results = [r for r in results if r['success']]

    print(f"🎯 PREGUNTA:")
    print(f"   '¿Las 10 ejecuciones con distintos parámetros entre sí dan")
    print(f"    los mismos resultados numéricos y llegan a las mismas causas?'")
    print()

    if len(successful_results) < 5:
        print(f"❌ RESPUESTA: No se puede determinar")
        print(f"   Solo {len(successful_results)}/10 ejecuciones exitosas")
        print(f"   Insuficientes datos para conclusión robusta")
        return

    # Analyze numerical consistency
    activity_means = [r['activity_mean'] for r in successful_results]
    n_causes = [r['n_causes'] for r in successful_results]

    numerical_variation = np.std(activity_means) / np.mean(activity_means) * 100
    causal_variation = len(set(n_causes))

    print(f"📊 ANÁLISIS CUANTITATIVO:")
    print(f"   Variación numérica: {numerical_variation:.1f}%")
    print(f"   Variación en causas: {causal_variation} niveles distintos")
    print()

    if numerical_variation < 10 and causal_variation <= 2:
        print(f"❌ RESPUESTA: SÍ dan resultados similares")
        print(f"   (Esto sería INESPERADO - distintos parámetros deberían dar distintos resultados)")
        print()
        print(f"🔍 POSIBLES EXPLICACIONES:")
        print(f"   • Los parámetros no son suficientemente distintos")
        print(f"   • El modelo es poco sensible a las variaciones")
        print(f"   • Hay dominancia de un parámetro sobre otros")

    else:
        print(f"✅ RESPUESTA: NO dan los mismos resultados")
        print(f"   (Esto es CORRECTO - distintos parámetros → distintos resultados)")
        print()
        print(f"🎯 IMPLICACIONES:")
        print(f"   ✅ El modelo responde apropiadamente a cambios de parámetros")
        print(f"   ✅ Diferentes condiciones experimentales dan diferentes resultados")
        print(f"   ✅ La inferencia causal varía con las condiciones")
        print(f"   ✅ Comportamiento científico esperado")


def main():
    """Ejecutar test completo con parámetros distintos."""
    print("TEST: PARÁMETROS DISTINTOS → ¿RESULTADOS DISTINTOS?")
    print("Objetivo: Verificar si parámetros diferentes llevan a resultados diferentes")
    print()

    # Run with distinct parameters
    results = run_with_distinct_parameters()

    # Analyze numerical differences
    analyze_numerical_differences(results)

    # Analyze causal differences
    analyze_causal_differences(results)

    # Correlation analysis
    correlation_analysis(results)

    # Final verdict
    final_verdict(results)


if __name__ == "__main__":
    main()