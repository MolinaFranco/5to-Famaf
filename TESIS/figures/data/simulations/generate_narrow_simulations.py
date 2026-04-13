#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para generar simulaciones con estímulos angostos (width_factor=0.01).

Este script genera simulaciones para todos los niveles de contraste
usando estímulos angostos generados por nosotros (no h_true de Echeveste).
Las alturas de los estímulos coinciden con las de h_true originales,
pero el ancho es mucho menor (width_factor=0.01 vs ~0.15 original).

Configuración:
- noise_level: 0.05 (menor que las simulaciones originales con 0.1)
- width_factor: 0.01 (estímulos muy angostos)
- Amplitudes calibradas para mantener h_max igual a h_true

Los inputs son generados usando GSM.generate_bump_stimulus() con las
amplitudes calibradas y transformados con GSM.generate_h_input_efficient().

References:
- Echeveste et al. (2020) Nature Neuroscience
- parameters.md: transformation parameters (alpha_h, beta_h, gamma_h)
"""

import sys
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

import numpy as np
import json
from pathlib import Path
from datetime import datetime

from skneuromsi.neural import Echeveste2020
from skneuromsi.generative import GSM
from skneuromsi.data import GSMDataLoader


# =============================================================================
# CONFIGURACIÓN CENTRALIZADA
# =============================================================================

# Configuración de simulaciones
SIMULATION_CONFIG = {
    'n_neurons_e': 50,
    'n_neurons_i': 50,
    'k': 0.3,              # Ganancia
    'n': 2.0,              # Exponente supralineal
    'seed': 42,            # Semilla para reproducibilidad
    'simulation_time': 1000.0,  # ms
    'dt': 0.2,             # ms
    'burn_in': 200.0,      # ms (post-equilibrio, para estadísticas)
    'burn_in_time': 500.0,  # ms (pre-simulación, equilibra red)
    'noise_level': 0.05,   # Ruido reducido (era 0.1)
    'orientation': 0.0,    # Orientación preferida (no se usa con h_input)
    'use_three_phases': False,  # Estímulo constante
}

# Parámetros de estímulo
STIMULUS_CONFIG = {
    'width_factor': 0.01,  # Estímulo muy angosto
    'dominant_idx': 25,    # ~0 grados
    'contrast': 1.0,       # Contraste para generar x desde y
}

# Niveles de contraste y amplitudes calibradas
# Las amplitudes fueron encontradas por búsqueda binaria para que
# h_max coincida con h_true de Echeveste para cada contraste
# h_true_max son los valores REALES de h_true original (verificados)
CONTRAST_LEVELS = [
    {'contrast': 0.0, 'label': 'spontaneous', 'amplitude': 0.000629,
     'h_true_max': 0.019280},
    {'contrast': 0.125, 'label': 'low', 'amplitude': 7.7341,
     'h_true_max': 1.110615},
    {'contrast': 0.25, 'label': 'medium', 'amplitude': 15.4641,
     'h_true_max': 3.940635},
    {'contrast': 0.5, 'label': 'high', 'amplitude': 30.9287,
     'h_true_max': 14.963133},
    {'contrast': 1.0, 'label': 'very_high', 'amplitude': 61.8572,
     'h_true_max': 58.947728},
]


# =============================================================================
# FUNCIONES
# =============================================================================

def generate_narrow_h_input(gsm, amplitude, width_factor, dominant_idx,
                            contrast):
    """
    Generar h_input angosto usando GSM.

    Parameters
    ----------
    gsm : GSM
        Modelo GSM con parámetros de transformación
    amplitude : float
        Amplitud del bump (calibrada para h_max objetivo)
    width_factor : float
        Factor de ancho del bump (0.01 = muy angosto)
    dominant_idx : int
        Índice de orientación dominante (25 = ~0 grados)
    contrast : float
        Contraste para generar x desde y

    Returns
    -------
    h_input : np.ndarray, shape (50,)
        Input transformado para neuronas excitatorias
    """
    # Generar y (representación latente)
    y = gsm.generate_bump_stimulus(
        dominant_orientation_idx=dominant_idx,
        amplitude=amplitude,
        width_factor=width_factor,
        normalize=False  # Como Echeveste
    )

    # Generar x (imagen) desde y
    x = gsm.generate_stimulus_from_y(y, contrast=contrast, add_noise=False)

    # Generar h_input (transformado)
    h_input = gsm.generate_h_input_efficient(x)

    return h_input


def run_simulation(ssn, h_input, config):
    """
    Ejecutar simulación con h_input personalizado.

    Parameters
    ----------
    ssn : Echeveste2020
        Modelo SSN con parámetros cargados
    h_input : np.ndarray, shape (50,)
        Input para neuronas excitatorias
    config : dict
        Configuración de simulación

    Returns
    -------
    dict
        Datos de simulación y estadísticas
    """
    # Ejecutar simulación con h_input personalizado
    result = ssn.run(
        noise_level=config['noise_level'],
        simulation_time=config['simulation_time'],
        burn_in_time=config['burn_in_time'],
        use_three_phases=config['use_three_phases'],
        h_input=h_input
    )

    # Extraer datos
    df = result.get_modes()
    n_times = len(result.times_)

    u_E = df['excitatory_potential'].values.reshape(n_times, -1)
    u_I = df['inhibitory_potential'].values.reshape(n_times, -1)
    r_E = df['excitatory_firing_rate'].values.reshape(n_times, -1)
    r_I = df['inhibitory_firing_rate'].values.reshape(n_times, -1)
    time = result.times_ * config['dt']

    # Calcular estadísticas del estado estacionario
    burn_in_steps = int(config['burn_in'] / config['dt'])

    u_E_stat = u_E[burn_in_steps:]
    u_I_stat = u_I[burn_in_steps:]
    r_E_stat = r_E[burn_in_steps:]
    r_I_stat = r_I[burn_in_steps:]

    stats = {
        'u_E_mean': np.mean(u_E_stat, axis=0),
        'u_E_std': np.std(u_E_stat, axis=0),
        'u_E_cov': np.cov(u_E_stat.T),
        'u_E_corr': np.corrcoef(u_E_stat.T),
        'u_I_mean': np.mean(u_I_stat, axis=0),
        'u_I_std': np.std(u_I_stat, axis=0),
        'u_I_cov': np.cov(u_I_stat.T),
        'u_I_corr': np.corrcoef(u_I_stat.T),
        'r_E_mean': np.mean(r_E_stat, axis=0),
        'r_E_std': np.std(r_E_stat, axis=0),
        'r_E_cov': np.cov(r_E_stat.T),
        'r_E_corr': np.corrcoef(r_E_stat.T),
        'r_I_mean': np.mean(r_I_stat, axis=0),
        'r_I_std': np.std(r_I_stat, axis=0),
        'r_I_cov': np.cov(r_I_stat.T),
        'r_I_corr': np.corrcoef(r_I_stat.T),
    }

    return {
        'u_E': u_E,
        'u_I': u_I,
        'r_E': r_E,
        'r_I': r_I,
        'time': time,
        'stats': stats,
        'h_input': h_input,
    }


def save_simulation(contrast, data, h_input, output_dir, stim_config):
    """
    Guardar datos de simulación.

    Parameters
    ----------
    contrast : float
        Nivel de contraste
    data : dict
        Datos de simulación
    h_input : np.ndarray
        Input usado para la simulación
    output_dir : Path
        Directorio de salida
    stim_config : dict
        Configuración del estímulo
    """
    # Crear directorio con nombre descriptivo
    dir_name = f"contrast_{contrast:.3f}_narrow_wf{stim_config['width_factor']}"
    save_dir = output_dir / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGuardando en {save_dir}...")

    # Orientaciones
    orientations = np.linspace(-90, 90, SIMULATION_CONFIG['n_neurons_e'],
                               endpoint=False)

    # Guardar NPZ
    np.savez_compressed(
        save_dir / "simulation_data.npz",
        # Trayectorias temporales
        u_E=data['u_E'],
        u_I=data['u_I'],
        r_E=data['r_E'],
        r_I=data['r_I'],
        time=data['time'],
        # Estadísticas
        **data['stats'],
        # Input
        h_input=h_input,
        # Metadatos
        orientations=orientations,
        dt=SIMULATION_CONFIG['dt']
    )
    print("✓ Datos guardados (NPZ)")

    # Guardar parámetros JSON
    params = {
        'contrast': float(contrast),
        'noise_level': float(SIMULATION_CONFIG['noise_level']),
        'n_neurons_e': SIMULATION_CONFIG['n_neurons_e'],
        'n_neurons_i': SIMULATION_CONFIG['n_neurons_i'],
        'simulation_time': float(SIMULATION_CONFIG['simulation_time']),
        'dt': float(SIMULATION_CONFIG['dt']),
        'burn_in': float(SIMULATION_CONFIG['burn_in']),
        'use_three_phases': SIMULATION_CONFIG['use_three_phases'],
        # Parámetros de estímulo
        'stimulus_type': 'narrow_generated',
        'width_factor': stim_config['width_factor'],
        'dominant_idx': stim_config['dominant_idx'],
        'amplitude': stim_config['amplitude'],
        'h_max': float(h_input.max()),
        'h_min': float(h_input.min()),
        # Estadísticas
        'u_E_mean': float(data['stats']['u_E_mean'].mean()),
        'r_E_mean': float(data['stats']['r_E_mean'].mean()),
        'timestamp': datetime.now().isoformat(),
    }

    with open(save_dir / "parameters.json", 'w') as f:
        json.dump(params, f, indent=2)
    print("✓ Parámetros guardados (JSON)")

    # Guardar metadata
    with open(save_dir / "metadata.txt", 'w') as f:
        f.write(f"Simulación Echeveste2020 - Estímulos Angostos\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nParámetros de estímulo:\n")
        f.write(f"  Tipo: Generated (no h_true)\n")
        f.write(f"  width_factor: {stim_config['width_factor']}\n")
        f.write(f"  amplitude: {stim_config['amplitude']:.4f}\n")
        f.write(f"  dominant_idx: {stim_config['dominant_idx']}\n")
        f.write(f"  h_max: {h_input.max():.4f}\n")
        f.write(f"\nParámetros de simulación:\n")
        f.write(f"  Contraste: {contrast:.3f}\n")
        f.write(f"  Ruido: {SIMULATION_CONFIG['noise_level']}\n")
        f.write(f"  Neuronas: {SIMULATION_CONFIG['n_neurons_e']}E + "
                f"{SIMULATION_CONFIG['n_neurons_i']}I\n")
        f.write(f"  Tiempo: {SIMULATION_CONFIG['simulation_time']} ms\n")
        f.write(f"  dt: {SIMULATION_CONFIG['dt']} ms\n")
        f.write(f"  Burn-in: {SIMULATION_CONFIG['burn_in']} ms\n")
        f.write(f"\nEstadísticas (estado estacionario):\n")
        f.write(f"  u_E mean: {data['stats']['u_E_mean'].mean():.6f}\n")
        f.write(f"  r_E mean: {data['stats']['r_E_mean'].mean():.6f}\n")
        f.write(f"\nGenerado con estímulos angostos (width_factor=0.01)\n")
        f.write(f"Amplitudes calibradas para h_max = h_true original\n")

    print("✓ Metadata guardada (TXT)")

    return save_dir


def main():
    """Ejecutar todas las simulaciones."""
    print("=" * 70)
    print("GENERACIÓN DE SIMULACIONES - ESTÍMULOS ANGOSTOS")
    print("=" * 70)
    print("\nEste script genera simulaciones con estímulos angostos")
    print("(width_factor=0.01) y ruido reducido (0.05).")
    print("Las alturas de h coinciden con h_true de Echeveste.")
    print("=" * 70)

    # Mostrar configuración
    print("\nCONFIGURACIÓN DE SIMULACIÓN:")
    print(f"  Neuronas: {SIMULATION_CONFIG['n_neurons_e']}E + "
          f"{SIMULATION_CONFIG['n_neurons_i']}I")
    print(f"  Tiempo: {SIMULATION_CONFIG['simulation_time']} ms")
    print(f"  dt: {SIMULATION_CONFIG['dt']} ms")
    print(f"  Burn-in: {SIMULATION_CONFIG['burn_in']} ms")
    print(f"  Ruido: {SIMULATION_CONFIG['noise_level']}")

    print("\nCONFIGURACIÓN DE ESTÍMULO:")
    print(f"  width_factor: {STIMULUS_CONFIG['width_factor']}")
    print(f"  dominant_idx: {STIMULUS_CONFIG['dominant_idx']} (~0°)")

    print(f"\nNiveles de contraste: {len(CONTRAST_LEVELS)}")
    for level in CONTRAST_LEVELS:
        print(f"  - {level['contrast']:.3f} ({level['label']}) "
              f"→ amplitude={level['amplitude']:.4f}, "
              f"h_max_target={level['h_true_max']:.4f}")

    # Cargar GSM para generar h_inputs
    print("\n" + "=" * 70)
    print("PREPARACIÓN")
    print("=" * 70)
    print("\nCargando GSM data loader...")
    loader = GSMDataLoader()
    params = loader.load_transformation_parameters()

    print(f"Transformation parameters:")
    print(f"  alpha_h = {params['alpha_h']:.6f}")
    print(f"  beta_h = {params['beta_h']:.6f}")
    print(f"  gamma_h = {params['gamma_h']:.6f}")

    # Crear GSM
    gsm = GSM(
        use_pretrained=True,
        alpha_h=params['alpha_h'],
        beta_h=params['beta_h'],
        gamma_h=params['gamma_h']
    )

    # Crear modelo SSN
    print("\nCreando modelo SSN...")
    ssn = Echeveste2020(
        N_E=SIMULATION_CONFIG['n_neurons_e'],
        N_I=SIMULATION_CONFIG['n_neurons_i'],
        k=SIMULATION_CONFIG['k'],
        n=SIMULATION_CONFIG['n'],
        seed=SIMULATION_CONFIG['seed']
    )

    # Cargar parámetros entrenados
    print("Cargando parámetros entrenados...")
    ssn.load_parameters()
    print("✓ Parámetros cargados")

    # Directorio de salida
    output_dir = Path(__file__).parent

    # Ejecutar simulaciones
    results = []

    for level in CONTRAST_LEVELS:
        try:
            print(f"\n{'='*70}")
            print(f"SIMULACIÓN: Contraste {level['contrast']} ({level['label']})")
            print(f"{'='*70}")

            # Generar h_input angosto
            print("\nGenerando h_input angosto...")
            h_input = generate_narrow_h_input(
                gsm=gsm,
                amplitude=level['amplitude'],
                width_factor=STIMULUS_CONFIG['width_factor'],
                dominant_idx=STIMULUS_CONFIG['dominant_idx'],
                contrast=STIMULUS_CONFIG['contrast']
            )
            print(f"  h_input: min={h_input.min():.4f}, max={h_input.max():.4f}")
            print(f"  h_max target: {level['h_true_max']:.4f}")

            # Ejecutar simulación
            print("\nEjecutando simulación...")
            data = run_simulation(ssn, h_input, SIMULATION_CONFIG)
            print(f"✓ Simulación completada")
            print(f"  u_E mean: {data['stats']['u_E_mean'].mean():.6f}")
            print(f"  r_E mean: {data['stats']['r_E_mean'].mean():.6f}")

            # Guardar
            stim_config = {
                'width_factor': STIMULUS_CONFIG['width_factor'],
                'dominant_idx': STIMULUS_CONFIG['dominant_idx'],
                'amplitude': level['amplitude'],
            }
            save_dir = save_simulation(level['contrast'], data, h_input,
                                       output_dir, stim_config)

            results.append({
                'contrast': level['contrast'],
                'label': level['label'],
                'status': 'SUCCESS',
                'dir': save_dir,
                'h_max': h_input.max(),
                'u_E_mean': data['stats']['u_E_mean'].mean(),
                'r_E_mean': data['stats']['r_E_mean'].mean(),
            })

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'contrast': level['contrast'],
                'label': level['label'],
                'status': 'FAILED',
                'error': str(e)
            })

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN DE SIMULACIONES")
    print("=" * 70)

    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\nCompletadas: {success_count}/{len(CONTRAST_LEVELS)}")

    for result in results:
        status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
        print(f"\n{status_symbol} Contraste {result['contrast']:.3f} "
              f"({result['label']}): {result['status']}")
        if result['status'] == 'SUCCESS':
            print(f"    Directorio: {result['dir'].name}")
            print(f"    h_max: {result['h_max']:.4f}")
            print(f"    u_E mean: {result['u_E_mean']:.6f}")
            print(f"    r_E mean: {result['r_E_mean']:.6f}")

    print("\n" + "=" * 70)
    print("✓ TODAS LAS SIMULACIONES COMPLETADAS")
    print("=" * 70)


if __name__ == "__main__":
    main()
