#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script unificado para generar todas las simulaciones con h_true correcto.

Este script genera simulaciones para todos los niveles de contraste
disponibles en los archivos h_true del código original de Echeveste:
- h_true_0: contraste 0.0 (espontáneo)
- h_true_1: contraste ~0.125 (bajo)
- h_true_2: contraste ~0.25 (medio-alto)

Configuración centralizada para todas las simulaciones.
"""

import sys
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')

import numpy as np
import json
from pathlib import Path
from datetime import datetime

from skneuromsi.neural import Echeveste2020


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
    'simulation_time': 1000.0,  # ms (solo para use_three_phases=False)
    'dt': 0.2,             # ms
    'burn_in': 200.0,      # ms (post-equilibrio, para estadísticas)
    'burn_in_time': 500.0,  # ms (pre-simulación, equilibra red)
    'noise_level': 0.1,    # Ruido moderado
    'orientation': 0.0,    # Orientación preferida
    # Estructura temporal
    'use_three_phases': False,  # False: estímulo constante, True: transientes
    't_init': 225.0,       # ms: Fase 1 espontánea (solo si use_three_phases=True)
    't_stimulus': 100.0,   # ms: Fase 2 estímulo (solo si use_three_phases=True)
    't_final': 125.0,      # ms: Fase 3 espontánea (solo si use_three_phases=True)
}

# Niveles de contraste a simular
# NOTA: El modelo mapea automáticamente estos valores a h_true_0, h_true_1, h_true_2
CONTRAST_LEVELS = [
    # {'contrast': 0.0, 'label': 'spontaneous', 'h_true_file': 'h_true_0'},
    # {'contrast': 0.125, 'label': 'low', 'h_true_file': 'h_true_1'},
    # {'contrast': 0.25, 'label': 'medium', 'h_true_file': 'h_true_2'},
    # {'contrast': 0.5, 'label': 'high', 'h_true_file': 'h_true_3'},
    {'contrast': 1.0, 'label': 'very_high', 'h_true_file': 'h_true_4'},
]


# =============================================================================
# FUNCIONES
# =============================================================================

def run_simulation(contrast, config):
    """
    Ejecutar simulación para un nivel de contraste.

    Parameters
    ----------
    contrast : float
        Nivel de contraste
    config : dict
        Configuración de simulación

    Returns
    -------
    dict
        Datos de simulación y estadísticas
    """
    print(f"\n{'='*70}")
    print(f"SIMULACIÓN: Contraste {contrast}")
    print(f"{'='*70}")

    # Crear modelo
    print("\nCreando modelo...")
    ssn = Echeveste2020(
        N_E=config['n_neurons_e'],
        N_I=config['n_neurons_i'],
        k=config['k'],
        n=config['n'],
        seed=config['seed']
    )

    # Cargar parámetros entrenados
    print("Cargando parámetros entrenados...")
    ssn.load_parameters()
    print("✓ Parámetros cargados")

    # Ejecutar simulación
    print(f"\nEjecutando simulación...")

    # Parámetros base
    run_params = {
        'stimulus_contrast': contrast,
        'stimulus_orientation': config['orientation'],
        'noise_level': config['noise_level'],
        'burn_in_time': config['burn_in_time'],
        'use_three_phases': config['use_three_phases'],
    }

    # Agregar parámetros específicos según el modo
    if config['use_three_phases']:
        # Modo transientes: usar t_init, t_stimulus, t_final
        run_params['t_init'] = config['t_init']
        run_params['t_stimulus'] = config['t_stimulus']
        run_params['t_final'] = config['t_final']
    else:
        # Modo estímulo constante: usar simulation_time
        run_params['simulation_time'] = config['simulation_time']

    result = ssn.run(**run_params)

    print(f"✓ Simulación completada: {len(result.times_)} pasos temporales")

    # Extraer datos
    df = result.get_modes()
    n_times = len(result.times_)

    u_E = df['excitatory_potential'].values.reshape(n_times, -1)
    u_I = df['inhibitory_potential'].values.reshape(n_times, -1)
    r_E = df['excitatory_firing_rate'].values.reshape(n_times, -1)
    r_I = df['inhibitory_firing_rate'].values.reshape(n_times, -1)
    # Convertir pasos a tiempo en ms
    time = result.times_ * config['dt']

    # Calcular estadísticas del estado estacionario
    print("\nCalculando estadísticas del estado estacionario...")
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

    print(f"  u_E mean: {stats['u_E_mean'].mean():.6f}")
    print(f"  r_E mean: {stats['r_E_mean'].mean():.6f}")

    return {
        'u_E': u_E,
        'u_I': u_I,
        'r_E': r_E,
        'r_I': r_I,
        'time': time,
        'stats': stats,
        'config': config,
    }


def save_simulation(contrast, data, output_dir):
    """
    Guardar datos de simulación.

    Parameters
    ----------
    contrast : float
        Nivel de contraste
    data : dict
        Datos de simulación
    output_dir : Path
        Directorio de salida
    """
    # Crear directorio
    dir_name = f"contrast_{contrast:.3f}_h_true_correcto"
    save_dir = output_dir / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGuardando en {save_dir}...")

    # Orientaciones
    orientations = np.linspace(-90, 90, data['config']['n_neurons_e'],
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
        # Metadatos
        orientations=orientations,
        dt=data['config']['dt']
    )
    print("✓ Datos guardados (NPZ)")

    # Guardar parámetros JSON
    params = {
        'contrast': float(contrast),
        'noise_level': float(data['config']['noise_level']),
        'n_neurons_e': data['config']['n_neurons_e'],
        'n_neurons_i': data['config']['n_neurons_i'],
        'simulation_time': float(data['config']['simulation_time']),
        'dt': float(data['config']['dt']),
        'burn_in': float(data['config']['burn_in']),
        'use_three_phases': data['config']['use_three_phases'],
        'orientation': float(data['config']['orientation']),
        'u_E_mean': float(data['stats']['u_E_mean'].mean()),
        'r_E_mean': float(data['stats']['r_E_mean'].mean()),
        'timestamp': datetime.now().isoformat(),
    }

    # Agregar info de fases si use_three_phases=True
    if data['config']['use_three_phases']:
        params['t_init'] = float(data['config'].get('t_init', 225.0))
        params['t_stimulus'] = float(data['config'].get('t_stimulus', 100.0))
        params['t_final'] = float(data['config'].get('t_final', 125.0))

    with open(save_dir / "parameters.json", 'w') as f:
        json.dump(params, f, indent=2)
    print("✓ Parámetros guardados (JSON)")

    # Guardar metadata
    with open(save_dir / "metadata.txt", 'w') as f:
        f.write(f"Simulación Echeveste2020 - h_true correcto\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nParámetros:\n")
        f.write(f"  Contraste: {contrast:.3f}\n")
        f.write(f"  Ruido: {data['config']['noise_level']}\n")
        f.write(f"  Neuronas: {data['config']['n_neurons_e']}E + "
                f"{data['config']['n_neurons_i']}I\n")
        f.write(f"  Tiempo: {data['config']['simulation_time']} ms\n")
        f.write(f"  dt: {data['config']['dt']} ms\n")
        f.write(f"  Burn-in: {data['config']['burn_in']} ms\n")
        f.write(f"\nEstadísticas (estado estacionario):\n")
        f.write(f"  u_E mean: {data['stats']['u_E_mean'].mean():.6f}\n")
        f.write(f"  r_E mean: {data['stats']['r_E_mean'].mean():.6f}\n")
        f.write(f"\nUSA h_true correcto del código original Echeveste\n")

    print("✓ Metadata guardada (TXT)")

    return save_dir


def main():
    """Ejecutar todas las simulaciones."""
    print("="*70)
    print("GENERACIÓN DE SIMULACIONES - H_TRUE CORRECTO")
    print("="*70)
    print("\nEste script genera simulaciones para todos los niveles")
    print("de contraste usando h_true del código original de Echeveste.")
    print("="*70)

    # Mostrar configuración
    print("\nCONFIGURACIÓN:")
    print(f"  Neuronas: {SIMULATION_CONFIG['n_neurons_e']}E + "
          f"{SIMULATION_CONFIG['n_neurons_i']}I")
    print(f"  Tiempo: {SIMULATION_CONFIG['simulation_time']} ms")
    print(f"  dt: {SIMULATION_CONFIG['dt']} ms")
    print(f"  Burn-in: {SIMULATION_CONFIG['burn_in']} ms")
    print(f"  Ruido: {SIMULATION_CONFIG['noise_level']}")
    print(f"\nNiveles de contraste: {len(CONTRAST_LEVELS)}")
    for level in CONTRAST_LEVELS:
        print(f"  - {level['contrast']:.3f} ({level['label']}) "
              f"→ {level['h_true_file']}")

    # Directorio de salida
    output_dir = Path(__file__).parent

    # Ejecutar simulaciones
    results = []

    for level in CONTRAST_LEVELS:
        try:
            # Ejecutar simulación
            data = run_simulation(level['contrast'], SIMULATION_CONFIG)

            # Guardar
            save_dir = save_simulation(level['contrast'], data, output_dir)

            results.append({
                'contrast': level['contrast'],
                'label': level['label'],
                'status': 'SUCCESS',
                'dir': save_dir,
                'u_E_mean': data['stats']['u_E_mean'].mean(),
                'r_E_mean': data['stats']['r_E_mean'].mean(),
            })

            print(f"\n✓ Simulación exitosa: {level['label']}")

        except Exception as e:
            print(f"\n✗ Error en simulación (contraste={level['contrast']}): {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'contrast': level['contrast'],
                'label': level['label'],
                'status': 'FAILED',
                'error': str(e)
            })

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE SIMULACIONES")
    print("="*70)

    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\nCompletadas: {success_count}/{len(CONTRAST_LEVELS)}")

    for result in results:
        status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
        print(f"\n{status_symbol} Contraste {result['contrast']:.3f} "
              f"({result['label']}): {result['status']}")
        if result['status'] == 'SUCCESS':
            print(f"    Directorio: {result['dir'].name}")
            print(f"    u_E mean: {result['u_E_mean']:.6f}")
            print(f"    r_E mean: {result['r_E_mean']:.6f}")

    print("\n" + "="*70)
    print("✓ TODAS LAS SIMULACIONES COMPLETADAS")
    print("="*70)


if __name__ == "__main__":
    main()
