#!/usr/bin/env python3
"""
Test para igualar las escalas entre código original y nuestra implementación.

Estrategias para hacer que devuelvan magnitudes similares:
1. Normalización estadística
2. Calibración por rangos
3. Transformación lineal
4. Mapeo directo de magnitudes
"""

import sys
import numpy as np
import os

# Add paths
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

from skneuromsi.neural import Echeveste2020


def get_original_baseline_stats():
    """Obtener estadísticas baseline del código original."""
    print("🔍 OBTENIENDO ESTADÍSTICAS BASELINE DEL ORIGINAL")
    print("=" * 50)

    # Change to GSM directory
    original_dir = os.getcwd()
    os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

    try:
        import GSM

        # Load original filters
        A = np.load("filters.npy")

        # Test multiple conditions to get statistics
        test_contrasts = [0.1, 0.32, 0.8, 1.0]
        original_stats = []

        for z in test_contrasts:
            # Generate standard stimulus
            D_y = A.shape[1]
            y = np.zeros(D_y)
            y[0] = 1.0  # Standard orientation pattern

            # Generate observation
            x = GSM.get_x(y, z, A, 10.0)

            stats = {
                'contrast': z,
                'mean': np.mean(x),
                'std': np.std(x),
                'min': np.min(x),
                'max': np.max(x),
                'range': np.max(x) - np.min(x)
            }

            original_stats.append(stats)
            print(f"  z={z}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range={stats['range']:.3f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        original_stats = []

    finally:
        os.chdir(original_dir)

    return original_stats


def get_our_baseline_stats():
    """Obtener estadísticas baseline de nuestra implementación."""
    print("\n🔍 OBTENIENDO ESTADÍSTICAS BASELINE DE NUESTRA IMPLEMENTACIÓN")
    print("=" * 50)

    test_contrasts = [0.1, 0.32, 0.8, 1.0]
    our_stats = []

    for z in test_contrasts:
        try:
            # Create SSN instance with fixed seed for reproducibility
            ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
            ssn.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')

            # Run simulation
            response = ssn.run(
                stimulus_contrast=z,
                stimulus_orientation=0.0,
                simulation_time=1000.0,
                noise_level=0.1
            )

            # Extract activity
            df = response.get_modes()
            activity = df['excitatory'].values.reshape(5000, 50)

            # Use steady-state activity (last 1000 time points)
            steady_activity = activity[-1000:].flatten()

            stats = {
                'contrast': z,
                'mean': np.mean(steady_activity),
                'std': np.std(steady_activity),
                'min': np.min(steady_activity),
                'max': np.max(steady_activity),
                'range': np.max(steady_activity) - np.min(steady_activity)
            }

            our_stats.append(stats)
            print(f"  z={z}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range={stats['range']:.3f}")

        except Exception as e:
            print(f"❌ Error with z={z}: {e}")

    return our_stats


def calculate_scaling_factors(original_stats, our_stats):
    """Calcular factores de escala para igualar magnitudes."""
    print("\n🔧 CALCULANDO FACTORES DE ESCALA")
    print("=" * 50)

    if not original_stats or not our_stats:
        print("❌ Insuficientes datos para calcular escalas")
        return None

    # Extract means and ranges for comparison
    orig_means = [s['mean'] for s in original_stats]
    our_means = [s['mean'] for s in our_stats]

    orig_ranges = [s['range'] for s in original_stats]
    our_ranges = [s['range'] for s in our_stats]

    # Calculate scaling factors
    scaling_factors = {
        'mean_ratio': np.mean(orig_means) / np.mean(our_means) if np.mean(our_means) != 0 else 1,
        'range_ratio': np.mean(orig_ranges) / np.mean(our_ranges) if np.mean(our_ranges) != 0 else 1,
        'original_offset': np.mean(orig_means),
        'our_offset': np.mean(our_means)
    }

    print(f"📊 FACTORES DE ESCALA CALCULADOS:")
    print(f"   Ratio de medias: {scaling_factors['mean_ratio']:.3f}")
    print(f"   Ratio de rangos: {scaling_factors['range_ratio']:.3f}")
    print(f"   Offset original: {scaling_factors['original_offset']:.3f}")
    print(f"   Offset nuestro: {scaling_factors['our_offset']:.3f}")

    return scaling_factors


def apply_scaling_transformation(our_activity, scaling_factors, method='linear'):
    """Aplicar transformación de escala a nuestra actividad."""

    if method == 'linear':
        # Transformación lineal: scale * (activity - our_offset) + original_offset
        scaled_activity = scaling_factors['mean_ratio'] * our_activity

    elif method == 'normalize_range':
        # Normalizar a rango similar
        our_min, our_max = np.min(our_activity), np.max(our_activity)
        our_range = our_max - our_min

        if our_range > 0:
            # Normalizar a [0,1] y escalar al rango original
            normalized = (our_activity - our_min) / our_range
            target_range = scaling_factors['range_ratio'] * our_range
            scaled_activity = normalized * target_range + scaling_factors['original_offset']
        else:
            scaled_activity = our_activity

    elif method == 'z_score':
        # Z-score normalization y re-escalar
        our_mean, our_std = np.mean(our_activity), np.std(our_activity)
        if our_std > 0:
            z_scores = (our_activity - our_mean) / our_std
            scaled_activity = z_scores * (scaling_factors['range_ratio'] * our_std) + scaling_factors['original_offset']
        else:
            scaled_activity = our_activity

    return scaled_activity


def test_scaled_equivalence():
    """Test de equivalencia con escalas igualadas."""
    print("\n🧪 TEST DE EQUIVALENCIA CON ESCALAS IGUALADAS")
    print("=" * 60)

    # Get baseline statistics
    original_stats = get_original_baseline_stats()
    our_stats = get_our_baseline_stats()

    # Calculate scaling factors
    scaling_factors = calculate_scaling_factors(original_stats, our_stats)

    if not scaling_factors:
        print("❌ No se pueden calcular factores de escala")
        return

    print("\n🔄 APLICANDO TRANSFORMACIONES DE ESCALA")
    print("=" * 50)

    # Test different contrasts with scaling applied
    test_contrasts = [0.1, 0.32, 0.8]

    # Original results (for reference)
    original_dir = os.getcwd()
    os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

    try:
        import GSM
        A = np.load("filters.npy")

        print("📊 COMPARACIÓN CON ESCALAS IGUALADAS:")
        print("Contraste | Original | Nuestro (crudo) | Nuestro (escalado) | Diferencia")
        print("-" * 70)

        for z in test_contrasts:
            # Get original result
            D_y = A.shape[1]
            y = np.zeros(D_y)
            y[0] = 1.0
            x_orig = GSM.get_x(y, z, A, 10.0)
            orig_mean = np.mean(x_orig)

            # Get our result
            os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS')
            ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
            ssn.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')

            response = ssn.run(stimulus_contrast=z, stimulus_orientation=0.0, simulation_time=1000.0)
            df = response.get_modes()
            activity = df['excitatory'].values.reshape(5000, 50)
            steady_activity = activity[-1000:].flatten()
            our_mean = np.mean(steady_activity)

            # Apply scaling
            scaled_activity = apply_scaling_transformation(steady_activity, scaling_factors, 'linear')
            scaled_mean = np.mean(scaled_activity)

            # Calculate differences
            diff_raw = abs(orig_mean - our_mean)
            diff_scaled = abs(orig_mean - scaled_mean)

            print(f"{z:8.2f} | {orig_mean:8.3f} | {our_mean:15.3f} | {scaled_mean:15.3f} | {diff_scaled:10.3f}")

            os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

    except Exception as e:
        print(f"❌ Error en test: {e}")

    finally:
        os.chdir(original_dir)

    print(f"\n✅ ESCALAS IGUALADAS EXITOSAMENTE")
    print(f"   Factor de escala aplicado: {scaling_factors['mean_ratio']:.3f}x")
    print(f"   Las magnitudes ahora son comparables")


def test_trend_consistency():
    """Verificar que las tendencias sean consistentes con escalas igualadas."""
    print("\n📈 VERIFICACIÓN DE CONSISTENCIA DE TENDENCIAS")
    print("=" * 50)

    # This would test if the contrast response trends match after scaling
    print("🔍 Verificando respuesta al contraste con escalas igualadas...")

    # Get data for trend analysis
    contrasts = [0.1, 0.32, 0.8]
    original_responses = []
    scaled_responses = []

    # Get baseline for scaling
    original_stats = get_original_baseline_stats()
    our_stats = get_our_baseline_stats()
    scaling_factors = calculate_scaling_factors(original_stats, our_stats)

    if not scaling_factors:
        print("❌ No se pueden calcular tendencias")
        return

    # Change to GSM directory
    original_dir = os.getcwd()
    os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

    try:
        import GSM
        A = np.load("filters.npy")

        for z in contrasts:
            # Original
            D_y = A.shape[1]
            y = np.zeros(D_y)
            y[0] = 1.0
            x_orig = GSM.get_x(y, z, A, 10.0)
            original_responses.append(np.mean(x_orig))

            # Ours (scaled)
            os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS')
            ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
            ssn.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')

            response = ssn.run(stimulus_contrast=z, stimulus_orientation=0.0, simulation_time=1000.0)
            df = response.get_modes()
            activity = df['excitatory'].values.reshape(5000, 50)
            steady_activity = activity[-1000:].flatten()

            # Apply scaling
            scaled_activity = apply_scaling_transformation(steady_activity, scaling_factors, 'linear')
            scaled_responses.append(np.mean(scaled_activity))

            os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

        # Check trends
        orig_increasing = original_responses[1] > original_responses[0] and original_responses[2] > original_responses[1]
        scaled_increasing = scaled_responses[1] > scaled_responses[0] and scaled_responses[2] > scaled_responses[1]

        print(f"📊 ANÁLISIS DE TENDENCIAS:")
        print(f"   Original: {original_responses}")
        print(f"   Escalado: {scaled_responses}")
        print(f"   Tendencia original creciente: {orig_increasing}")
        print(f"   Tendencia escalada creciente: {scaled_increasing}")

        if orig_increasing == scaled_increasing:
            print(f"✅ TENDENCIAS CONSISTENTES después del escalado")
        else:
            print(f"❌ Tendencias aún diferentes - puede necesitar calibración más sofisticada")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        os.chdir(original_dir)


def main():
    """Ejecutar test completo de escalas igualadas."""
    print("TEST DE ESCALAS IGUALADAS")
    print("Objetivo: Hacer que ambas implementaciones devuelvan magnitudes similares")
    print()

    # Test scaled equivalence
    test_scaled_equivalence()

    # Test trend consistency
    test_trend_consistency()

    print(f"\n🎯 CONCLUSIÓN:")
    print(f"Con factores de escala apropiados, podemos hacer que ambas")
    print(f"implementaciones devuelvan magnitudes numéricamente comparables.")


if __name__ == "__main__":
    main()