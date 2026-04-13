#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para generar simulación con tres fases temporales.

Genera una simulación con estructura temporal:
1. Pre-estímulo: actividad espontánea (sin estímulo)
2. Estímulo: respuesta a estímulo de alto contraste
3. Post-estímulo: retorno a actividad espontánea

La red evoluciona naturalmente mostrando transiciones realistas.

Referencias:
- Echeveste et al. (2020), Nature Neuroscience
- transients.py del repositorio original
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
# CONFIGURACIÓN
# =============================================================================

# Configuración de simulación
SIMULATION_CONFIG = {
    'n_neurons_e': 50,
    'n_neurons_i': 50,
    'k': 0.3,
    'n': 2.0,
    'seed': 42,
    'dt': 0.2,  # ms
    'burn_in_time': 500.0,  # ms (equilibra red antes de simulación)
    'noise_level': 0.05,  # Ruido reducido
}

# Configuración de las tres fases temporales (en ms)
PHASE_CONFIG = {
    't_init': 250.0,      # Pre-estímulo: actividad espontánea
    't_stimulus': 500.0,  # Estímulo presente
    't_final': 250.0,     # Post-estímulo: retorno a espontáneo
    'use_delays': False,  # Sin delays por neurona
}

# Configuración del estímulo (usando amplitud calibrada para contraste 1.0)
STIMULUS_CONFIG = {
    'width_factor': 0.01,  # Estímulo angosto
    'dominant_idx': 25,    # ~0 grados
    'contrast': 1.0,
    'amplitude': 61.8572,  # Calibrada para h_max = h_true de contraste 1.0
}

# Directorio de salida
OUTPUT_DIR = Path(__file__).parent


def generate_narrow_h_input(gsm, amplitude, width_factor, dominant_idx,
                            contrast):
    """
    Generar h_input angosto usando GSM.

    Parameters
    ----------
    gsm : GSM
        Modelo GSM con parámetros de transformación
    amplitude : float
        Amplitud del bump
    width_factor : float
        Factor de ancho del bump
    dominant_idx : int
        Índice de orientación dominante
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
        normalize=False
    )

    # Generar x (imagen) desde y
    x = gsm.generate_stimulus_from_y(y, contrast=contrast, add_noise=False)

    # Generar h_input (transformado)
    h_input = gsm.generate_h_input_efficient(x)

    return h_input


def run_three_phase_simulation(ssn, h_input, sim_config, phase_config):
    """
    Ejecutar simulación con tres fases temporales.

    Parameters
    ----------
    ssn : Echeveste2020
        Modelo SSN con parámetros cargados
    h_input : np.ndarray, shape (50,)
        Input para neuronas excitatorias (usado durante fase de estímulo)
    sim_config : dict
        Configuración de simulación
    phase_config : dict
        Configuración de fases temporales

    Returns
    -------
    dict
        Datos de simulación
    """
    # Ejecutar simulación con tres fases
    result = ssn.run(
        noise_level=sim_config['noise_level'],
        burn_in_time=sim_config['burn_in_time'],
        # Activar tres fases
        use_three_phases=True,
        t_init=phase_config['t_init'],
        t_stimulus=phase_config['t_stimulus'],
        t_final=phase_config['t_final'],
        use_delays=phase_config['use_delays'],
        # Input del estímulo
        h_input=h_input
    )

    # Extraer datos
    df = result.get_modes()
    n_times = len(result.times_)

    u_E = df['excitatory_potential'].values.reshape(n_times, -1)
    u_I = df['inhibitory_potential'].values.reshape(n_times, -1)
    r_E = df['excitatory_firing_rate'].values.reshape(n_times, -1)
    r_I = df['inhibitory_firing_rate'].values.reshape(n_times, -1)
    time = result.times_ * sim_config['dt']

    # Calcular índices de las fases
    dt = sim_config['dt']
    n_pre = int(phase_config['t_init'] / dt)
    n_stim = int(phase_config['t_stimulus'] / dt)
    n_post = int(phase_config['t_final'] / dt)

    # Crear vector temporal de h_input (para visualización)
    n_total = n_pre + n_stim + n_post
    n_neurons = len(h_input)
    h_temporal = np.zeros((n_neurons, n_total))
    h_temporal[:, n_pre:n_pre + n_stim] = np.tile(
        h_input[:, None], (1, n_stim)
    )

    return {
        'u_E': u_E,
        'u_I': u_I,
        'r_E': r_E,
        'r_I': r_I,
        'time': time,
        'h_input': h_input,
        'h_temporal': h_temporal,
        'phase_indices': {
            'n_pre': n_pre,
            'n_stim': n_stim,
            'n_post': n_post,
        }
    }


def save_simulation(data, h_input, output_dir, sim_config, phase_config,
                    stim_config):
    """
    Guardar datos de simulación.

    Parameters
    ----------
    data : dict
        Datos de simulación
    h_input : np.ndarray
        Input usado para la simulación
    output_dir : Path
        Directorio de salida
    sim_config : dict
        Configuración de simulación
    phase_config : dict
        Configuración de fases
    stim_config : dict
        Configuración del estímulo
    """
    # Crear directorio con nombre descriptivo
    total_time = (phase_config['t_init'] + phase_config['t_stimulus'] +
                  phase_config['t_final'])
    dir_name = f"three_phases_{total_time:.0f}ms_c{stim_config['contrast']:.1f}"
    save_dir = output_dir / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGuardando en {save_dir}...")

    # Orientaciones
    orientations = np.linspace(-90, 90, sim_config['n_neurons_e'],
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
        # Input
        h_input=h_input,
        h_temporal=data['h_temporal'],
        # Metadatos
        orientations=orientations,
        dt=sim_config['dt'],
        # Índices de fases
        n_pre=data['phase_indices']['n_pre'],
        n_stim=data['phase_indices']['n_stim'],
        n_post=data['phase_indices']['n_post'],
        # Duraciones de fases
        t_init=phase_config['t_init'],
        t_stimulus=phase_config['t_stimulus'],
        t_final=phase_config['t_final'],
    )
    print("✓ Datos guardados (NPZ)")

    # Guardar parámetros JSON
    params = {
        'contrast': float(stim_config['contrast']),
        'noise_level': float(sim_config['noise_level']),
        'n_neurons_e': sim_config['n_neurons_e'],
        'n_neurons_i': sim_config['n_neurons_i'],
        'dt': float(sim_config['dt']),
        'burn_in_time': float(sim_config['burn_in_time']),
        # Fases temporales
        'use_three_phases': True,
        't_init': float(phase_config['t_init']),
        't_stimulus': float(phase_config['t_stimulus']),
        't_final': float(phase_config['t_final']),
        'total_time': float(total_time),
        # Estímulo
        'stimulus_type': 'narrow_generated',
        'width_factor': stim_config['width_factor'],
        'dominant_idx': stim_config['dominant_idx'],
        'amplitude': stim_config['amplitude'],
        'h_max': float(h_input.max()),
        'h_min': float(h_input.min()),
        # Timestamp
        'timestamp': datetime.now().isoformat(),
    }

    with open(save_dir / "parameters.json", 'w') as f:
        json.dump(params, f, indent=2)
    print("✓ Parámetros guardados (JSON)")

    # Guardar metadata
    with open(save_dir / "metadata.txt", 'w') as f:
        f.write(f"Simulación Echeveste2020 - Tres Fases Temporales\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nEstructura temporal:\n")
        f.write(f"  Fase 1 (Pre-estímulo):  {phase_config['t_init']:.0f} ms\n")
        f.write(f"  Fase 2 (Estímulo):      {phase_config['t_stimulus']:.0f} "
                f"ms\n")
        f.write(f"  Fase 3 (Post-estímulo): {phase_config['t_final']:.0f} ms\n")
        f.write(f"  Total:                  {total_time:.0f} ms\n")
        f.write(f"\nParámetros de estímulo:\n")
        f.write(f"  Contraste: {stim_config['contrast']}\n")
        f.write(f"  width_factor: {stim_config['width_factor']}\n")
        f.write(f"  amplitude: {stim_config['amplitude']:.4f}\n")
        f.write(f"  h_max: {h_input.max():.4f}\n")
        f.write(f"\nParámetros de simulación:\n")
        f.write(f"  Ruido: {sim_config['noise_level']}\n")
        f.write(f"  Neuronas: {sim_config['n_neurons_e']}E + "
                f"{sim_config['n_neurons_i']}I\n")
        f.write(f"  dt: {sim_config['dt']} ms\n")
        f.write(f"  Burn-in: {sim_config['burn_in_time']} ms\n")

    print("✓ Metadata guardada (TXT)")

    return save_dir


def main():
    """Ejecutar simulación con tres fases."""
    print("=" * 70)
    print("GENERACIÓN DE SIMULACIÓN CON TRES FASES TEMPORALES")
    print("=" * 70)

    total_time = (PHASE_CONFIG['t_init'] + PHASE_CONFIG['t_stimulus'] +
                  PHASE_CONFIG['t_final'])

    print(f"\nEstructura temporal ({total_time:.0f} ms total):")
    print(f"  Pre-estímulo:  0 - {PHASE_CONFIG['t_init']:.0f} ms")
    print(f"  Estímulo:      {PHASE_CONFIG['t_init']:.0f} - "
          f"{PHASE_CONFIG['t_init'] + PHASE_CONFIG['t_stimulus']:.0f} ms")
    print(f"  Post-estímulo: "
          f"{PHASE_CONFIG['t_init'] + PHASE_CONFIG['t_stimulus']:.0f} - "
          f"{total_time:.0f} ms")

    print(f"\nConfiguración de estímulo:")
    print(f"  Contraste: {STIMULUS_CONFIG['contrast']}")
    print(f"  width_factor: {STIMULUS_CONFIG['width_factor']}")
    print(f"  amplitude: {STIMULUS_CONFIG['amplitude']}")

    # Cargar GSM
    print("\n" + "-" * 70)
    print("Cargando GSM...")
    loader = GSMDataLoader()
    params = loader.load_transformation_parameters()

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

    # Generar h_input
    print("\n" + "-" * 70)
    print("Generando h_input angosto...")
    h_input = generate_narrow_h_input(
        gsm=gsm,
        amplitude=STIMULUS_CONFIG['amplitude'],
        width_factor=STIMULUS_CONFIG['width_factor'],
        dominant_idx=STIMULUS_CONFIG['dominant_idx'],
        contrast=STIMULUS_CONFIG['contrast']
    )
    print(f"  h_input: min={h_input.min():.4f}, max={h_input.max():.4f}")

    # Ejecutar simulación
    print("\n" + "-" * 70)
    print("Ejecutando simulación con tres fases...")
    data = run_three_phase_simulation(
        ssn, h_input, SIMULATION_CONFIG, PHASE_CONFIG
    )
    print(f"✓ Simulación completada")
    print(f"  Forma de u_E: {data['u_E'].shape}")
    print(f"  Tiempo total: {data['time'][-1]:.1f} ms")

    # Guardar
    print("\n" + "-" * 70)
    save_dir = save_simulation(
        data, h_input, OUTPUT_DIR, SIMULATION_CONFIG,
        PHASE_CONFIG, STIMULUS_CONFIG
    )

    print("\n" + "=" * 70)
    print("✓ SIMULACIÓN COMPLETADA")
    print("=" * 70)
    print(f"\nDatos guardados en: {save_dir}")

    return save_dir


if __name__ == "__main__":
    main()
