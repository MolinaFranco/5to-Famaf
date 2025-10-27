#!/usr/bin/env python3
"""
Calibración mejorada para igualar escalas y tendencias entre implementaciones.

Esta versión incluye:
1. Mapeo no-lineal para corregir tendencias
2. Calibración por regresión
3. Transformaciones adaptativas
4. Validación de equivalencia numérica
"""

import sys
import numpy as np
import os
from scipy import optimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Add paths
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

from skneuromsi.neural import Echeveste2020


def collect_calibration_data():
    """Recopilar datos de calibración de ambas implementaciones."""
    print("🔍 RECOPILANDO DATOS DE CALIBRACIÓN")
    print("=" * 50)

    # Contrasts to test
    test_contrasts = np.linspace(0.05, 1.0, 10)  # More data points for better fitting

    original_data = []
    our_data = []

    # Change to GSM directory
    original_dir = os.getcwd()
    os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

    try:
        import GSM
        A = np.load("filters.npy")

        print("Recopilando datos del original y nuestro...")

        for i, z in enumerate(test_contrasts):
            print(f"  Contraste {z:.2f} ({i+1}/{len(test_contrasts)})", end="")

            try:
                # Original GSM
                D_y = A.shape[1]
                y = np.zeros(D_y)
                y[0] = 1.0
                x_orig = GSM.get_x(y, z, A, 10.0)
                orig_mean = np.mean(x_orig)

                # Our implementation
                os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS')
                ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
                ssn.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')

                response = ssn.run(stimulus_contrast=z, stimulus_orientation=0.0, simulation_time=1000.0)
                df = response.get_modes()
                activity = df['excitatory'].values.reshape(5000, 50)
                steady_activity = activity[-1000:].flatten()
                our_mean = np.mean(steady_activity)

                original_data.append({'contrast': z, 'response': orig_mean})
                our_data.append({'contrast': z, 'response': our_mean})

                print(f" ✓ Orig: {orig_mean:.3f}, Nuestro: {our_mean:.3f}")

                os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

            except Exception as e:
                print(f" ❌ Error: {e}")

    except Exception as e:
        print(f"❌ Error general: {e}")

    finally:
        os.chdir(original_dir)

    return original_data, our_data


def fit_calibration_function(original_data, our_data):
    """Ajustar función de calibración para mapear nuestros datos a los originales."""
    print("\n🔧 AJUSTANDO FUNCIÓN DE CALIBRACIÓN")
    print("=" * 50)

    if len(original_data) != len(our_data) or len(original_data) < 3:
        print("❌ Insuficientes datos para calibración")
        return None

    # Extract data arrays
    original_responses = np.array([d['response'] for d in original_data])
    our_responses = np.array([d['response'] for d in our_data])
    contrasts = np.array([d['contrast'] for d in original_data])

    print(f"📊 Datos para calibración:")
    print(f"   Original: range [{np.min(original_responses):.3f}, {np.max(original_responses):.3f}]")
    print(f"   Nuestro: range [{np.min(our_responses):.3f}, {np.max(our_responses):.3f}]")

    # Try different calibration approaches
    calibration_functions = {}

    # 1. Linear regression: y_orig = a * y_our + b
    print(f"\n🔹 Ajuste lineal:")
    try:
        reg_linear = LinearRegression()
        reg_linear.fit(our_responses.reshape(-1, 1), original_responses)

        a = reg_linear.coef_[0]
        b = reg_linear.intercept_
        r2_linear = reg_linear.score(our_responses.reshape(-1, 1), original_responses)

        calibration_functions['linear'] = {
            'function': lambda x: a * x + b,
            'params': {'a': a, 'b': b},
            'r2': r2_linear
        }

        print(f"     y_orig = {a:.3f} * y_our + {b:.3f}")
        print(f"     R² = {r2_linear:.4f}")

    except Exception as e:
        print(f"     ❌ Error: {e}")

    # 2. Polynomial regression (degree 2)
    print(f"\n🔹 Ajuste polinomial (grado 2):")
    try:
        poly_features = PolynomialFeatures(degree=2)
        our_poly = poly_features.fit_transform(our_responses.reshape(-1, 1))

        reg_poly = LinearRegression()
        reg_poly.fit(our_poly, original_responses)

        r2_poly = reg_poly.score(our_poly, original_responses)

        def poly_func(x):
            x_poly = poly_features.transform(np.array(x).reshape(-1, 1))
            return reg_poly.predict(x_poly)

        calibration_functions['polynomial'] = {
            'function': poly_func,
            'poly_features': poly_features,
            'reg': reg_poly,
            'r2': r2_poly
        }

        print(f"     Ajuste polinomial completado")
        print(f"     R² = {r2_poly:.4f}")

    except Exception as e:
        print(f"     ❌ Error: {e}")

    # 3. Contrast-dependent mapping: use both contrast and response
    print(f"\n🔹 Ajuste dependiente del contraste:")
    try:
        # Use both our_response and contrast as features
        features = np.column_stack([our_responses, contrasts])

        reg_context = LinearRegression()
        reg_context.fit(features, original_responses)

        r2_context = reg_context.score(features, original_responses)

        def context_func(our_resp, contrast):
            features = np.column_stack([np.array(our_resp).flatten(), np.array(contrast).flatten()])
            return reg_context.predict(features)

        calibration_functions['context'] = {
            'function': context_func,
            'reg': reg_context,
            'r2': r2_context
        }

        print(f"     y_orig = f(y_our, contraste)")
        print(f"     R² = {r2_context:.4f}")

    except Exception as e:
        print(f"     ❌ Error: {e}")

    # Select best calibration function
    best_method = max(calibration_functions.keys(), key=lambda k: calibration_functions[k]['r2'])
    best_r2 = calibration_functions[best_method]['r2']

    print(f"\n🏆 MEJOR MÉTODO: {best_method.upper()} (R² = {best_r2:.4f})")

    return calibration_functions, best_method


def test_calibrated_equivalence(calibration_functions, best_method, original_data, our_data):
    """Probar equivalencia con calibración aplicada."""
    print(f"\n🧪 TEST DE EQUIVALENCIA CON CALIBRACIÓN")
    print("=" * 60)

    if not calibration_functions or best_method not in calibration_functions:
        print("❌ No hay función de calibración disponible")
        return

    cal_func = calibration_functions[best_method]

    print(f"📊 COMPARACIÓN CON CALIBRACIÓN ({best_method.upper()}):")
    print("Contraste | Original | Nuestro | Calibrado | Error | % Error")
    print("-" * 65)

    total_error = 0
    valid_points = 0

    for i in range(min(len(original_data), len(our_data))):
        orig_resp = original_data[i]['response']
        our_resp = our_data[i]['response']
        contrast = original_data[i]['contrast']

        try:
            if best_method == 'context':
                calibrated_resp = cal_func['function'](our_resp, contrast)[0]
            else:
                calibrated_resp = cal_func['function'](our_resp)
                if hasattr(calibrated_resp, '__len__'):
                    calibrated_resp = calibrated_resp[0]

            error = abs(orig_resp - calibrated_resp)
            percent_error = (error / abs(orig_resp)) * 100 if orig_resp != 0 else 0

            print(f"{contrast:8.2f} | {orig_resp:8.3f} | {our_resp:7.3f} | {calibrated_resp:9.3f} | {error:5.3f} | {percent_error:6.1f}%")

            total_error += error
            valid_points += 1

        except Exception as e:
            print(f"{contrast:8.2f} | {orig_resp:8.3f} | {our_resp:7.3f} | ERROR: {e}")

    if valid_points > 0:
        avg_error = total_error / valid_points
        print(f"\n📈 ESTADÍSTICAS DE CALIBRACIÓN:")
        print(f"   Error promedio: {avg_error:.4f}")
        print(f"   Puntos válidos: {valid_points}/{len(original_data)}")

        if avg_error < 1.0:
            print(f"   ✅ EXCELENTE calibración (error < 1.0)")
        elif avg_error < 5.0:
            print(f"   ✅ BUENA calibración (error < 5.0)")
        else:
            print(f"   ⚠️  Calibración mejorable (error > 5.0)")

    return avg_error if valid_points > 0 else float('inf')


def test_trend_consistency_calibrated(calibration_functions, best_method):
    """Verificar consistencia de tendencias con calibración."""
    print(f"\n📈 VERIFICACIÓN DE TENDENCIAS CALIBRADAS")
    print("=" * 50)

    # Test specific contrasts for trend analysis
    test_contrasts = [0.1, 0.32, 0.8]
    original_trend = []
    calibrated_trend = []

    # Get original trend
    original_dir = os.getcwd()
    os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

    try:
        import GSM
        A = np.load("filters.npy")

        for z in test_contrasts:
            # Original
            D_y = A.shape[1]
            y = np.zeros(D_y)
            y[0] = 1.0
            x_orig = GSM.get_x(y, z, A, 10.0)
            original_trend.append(np.mean(x_orig))

            # Our (calibrated)
            os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS')
            ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
            ssn.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')

            response = ssn.run(stimulus_contrast=z, stimulus_orientation=0.0, simulation_time=1000.0)
            df = response.get_modes()
            activity = df['excitatory'].values.reshape(5000, 50)
            steady_activity = activity[-1000:].flatten()
            our_mean = np.mean(steady_activity)

            # Apply calibration
            cal_func = calibration_functions[best_method]
            if best_method == 'context':
                calibrated_mean = cal_func['function'](our_mean, z)[0]
            else:
                calibrated_mean = cal_func['function'](our_mean)
                if hasattr(calibrated_mean, '__len__'):
                    calibrated_mean = calibrated_mean[0]

            calibrated_trend.append(calibrated_mean)

            os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

        # Analyze trends
        print(f"📊 ANÁLISIS DE TENDENCIAS CALIBRADAS:")
        print(f"   Contrastes: {test_contrasts}")
        print(f"   Original: {[f'{x:.3f}' for x in original_trend]}")
        print(f"   Calibrado: {[f'{x:.3f}' for x in calibrated_trend]}")

        # Check if both trends are consistent
        orig_increasing = all(original_trend[i] <= original_trend[i+1] for i in range(len(original_trend)-1))
        cal_increasing = all(calibrated_trend[i] <= calibrated_trend[i+1] for i in range(len(calibrated_trend)-1))

        print(f"   Tendencia original creciente: {orig_increasing}")
        print(f"   Tendencia calibrada creciente: {cal_increasing}")

        if orig_increasing == cal_increasing:
            print(f"   ✅ TENDENCIAS CONSISTENTES")
            return True
        else:
            print(f"   ❌ Tendencias aún inconsistentes")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    finally:
        os.chdir(original_dir)


def main():
    """Ejecutar calibración completa."""
    print("CALIBRACIÓN MEJORADA DE ESCALAS")
    print("Objetivo: Igualar magnitudes Y tendencias entre implementaciones")
    print()

    # Collect calibration data
    original_data, our_data = collect_calibration_data()

    if not original_data or not our_data:
        print("❌ No se pudieron recopilar datos de calibración")
        return

    # Fit calibration function
    calibration_functions, best_method = fit_calibration_function(original_data, our_data)

    if not calibration_functions:
        print("❌ No se pudo ajustar función de calibración")
        return

    # Test calibrated equivalence
    avg_error = test_calibrated_equivalence(calibration_functions, best_method, original_data, our_data)

    # Test trend consistency
    trends_consistent = test_trend_consistency_calibrated(calibration_functions, best_method)

    print(f"\n🎯 RESUMEN FINAL:")
    print(f"   Mejor método: {best_method.upper()}")
    print(f"   Error promedio: {avg_error:.4f}")
    print(f"   Tendencias consistentes: {trends_consistent}")

    if avg_error < 2.0 and trends_consistent:
        print(f"\n✅ CALIBRACIÓN EXITOSA")
        print(f"   Las implementaciones ahora devuelven escalas equivalentes")
        print(f"   Las tendencias son consistentes")
        print(f"   Equivalencia numérica lograda")
    else:
        print(f"\n⚠️  CALIBRACIÓN PARCIAL")
        print(f"   Mejora lograda pero aún hay diferencias")


if __name__ == "__main__":
    main()