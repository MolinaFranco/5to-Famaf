#!/usr/bin/env python3
"""
Análisis de oscilaciones gamma en la red SSN.

Basado en el análisis de Echeveste et al. (2020):
- Espectro de potencia del LFP (Local Field Potential)
- Detección de pico en rango gamma (30-80 Hz)
- Dependencia de la frecuencia pico con el contraste

Referencia: ssn_inference_numerical_experiments/SSN/autocorr_and_power_spectrum.py

Uso:
    python3 analizar_gamma.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def calcular_lfp(u_E):
    """
    Calculate Local Field Potential as average membrane potential.

    The LFP is computed as the mean across all excitatory neurons,
    following Echeveste et al. (2020).

    Parameters
    ----------
    u_E : ndarray, shape (n_time, n_neurons)
        Excitatory membrane potentials over time

    Returns
    -------
    lfp : ndarray, shape (n_time,)
        Local field potential
    """
    return np.mean(u_E, axis=1)


def calcular_espectro_potencia(signal_data, dt, nperseg=None):
    """
    Calculate power spectrum using Welch's method.

    Parameters
    ----------
    signal_data : ndarray
        Time series signal
    dt : float
        Time step in ms
    nperseg : int, optional
        Length of each segment for Welch method

    Returns
    -------
    freqs : ndarray
        Frequency array in Hz
    psd : ndarray
        Power spectral density
    """
    fs = 1000.0 / dt  # Frecuencia de muestreo en Hz

    if nperseg is None:
        # Usar ventana de ~500 ms para buena resolución en gamma
        nperseg = min(int(500 / dt), len(signal_data) // 4)

    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg)
    return freqs, psd


def encontrar_pico_gamma(freqs, psd, gamma_range=(30, 80)):
    """
    Find peak frequency in gamma range.

    Parameters
    ----------
    freqs : ndarray
        Frequency array in Hz
    psd : ndarray
        Power spectral density
    gamma_range : tuple
        (min_freq, max_freq) for gamma band in Hz

    Returns
    -------
    peak_freq : float
        Peak frequency in gamma range (Hz)
    peak_power : float
        Power at peak frequency
    """
    # Filtrar al rango gamma
    mask = (freqs >= gamma_range[0]) & (freqs <= gamma_range[1])
    freqs_gamma = freqs[mask]
    psd_gamma = psd[mask]

    if len(psd_gamma) == 0:
        return np.nan, np.nan

    # Encontrar pico
    idx_peak = np.argmax(psd_gamma)
    peak_freq = freqs_gamma[idx_peak]
    peak_power = psd_gamma[idx_peak]

    return peak_freq, peak_power


def cargar_simulaciones(data_dir):
    """
    Load all available simulations from data directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing simulation folders

    Returns
    -------
    dict
        Dictionary with contrast as key and simulation data as value
    """
    simulaciones = {}

    # Buscar carpetas de simulación
    for folder in sorted(data_dir.glob("contrast_*")):
        # Extraer contraste del nombre
        parts = folder.name.split("_")
        try:
            contrast = float(parts[1])
        except (IndexError, ValueError):
            continue

        # Cargar datos
        data_file = folder / "simulation_data.npz"
        if data_file.exists():
            data = np.load(data_file)
            simulaciones[contrast] = {
                "u_E": data["u_E"],
                "time": data["time"],
                "dt": float(data["dt"]),
            }
            print(f"✓ Cargado contraste {contrast}: {data['u_E'].shape}")

    return simulaciones


def analizar_gamma(simulaciones, output_dir):
    """
    Analyze gamma oscillations for all contrasts.

    Parameters
    ----------
    simulaciones : dict
        Dictionary with simulation data for each contrast
    output_dir : Path
        Output directory for figures
    """
    contrastes = sorted(simulaciones.keys())
    n_contrastes = len(contrastes)

    # Almacenar resultados
    resultados = {
        "contraste": [],
        "freq_pico_gamma": [],
        "potencia_pico": [],
        "potencia_total_gamma": [],
    }

    # Crear figura de espectros
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Power Spectrum of LFP - Gamma Oscillations Analysis",
                 fontsize=14, fontweight="bold")

    # Colores para cada contraste:
    # Tonos de azul oscuro (más oscuro = mayor contraste)
    colors = plt.cm.Blues(np.linspace(0.65, 1.0, n_contrastes))

    for idx, contrast in enumerate(contrastes):
        data = simulaciones[contrast]
        u_E = data["u_E"]
        dt = data["dt"]

        # Calcular LFP
        lfp = calcular_lfp(u_E)

        # Calcular espectro de potencia
        freqs, psd = calcular_espectro_potencia(lfp, dt)

        # Encontrar pico en gamma
        peak_freq, peak_power = encontrar_pico_gamma(freqs, psd)

        # Potencia total en gamma
        mask_gamma = (freqs >= 30) & (freqs <= 80)
        potencia_gamma = np.trapz(psd[mask_gamma], freqs[mask_gamma])

        # Guardar resultados
        resultados["contraste"].append(contrast)
        resultados["freq_pico_gamma"].append(peak_freq)
        resultados["potencia_pico"].append(peak_power)
        resultados["potencia_total_gamma"].append(potencia_gamma)

        # Graficar espectro
        if idx < 5:
            ax = axes.flat[idx]
            # La línea del espectro incluye el valor de contraste en la leyenda
            line_label = f"LFP Spectrum (c={contrast})"
            ax.semilogy(freqs, psd, color=colors[idx], linewidth=1.5,
                        label=line_label)
            ax.axvspan(30, 80, alpha=0.2, color="orange",
                       label="Gamma band (30-80 Hz)")
            if not np.isnan(peak_freq):
                ax.axvline(peak_freq, color="darkred", linestyle="--",
                           linewidth=1.5, label=f"Peak: {peak_freq:.1f} Hz")
            ax.set_xlim(0, 150)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power (a.u.)")
            ax.set_title(f"Contrast = {contrast}")
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

    # Último panel: frecuencia pico vs contraste
    ax = axes.flat[5]
    valid_mask = ~np.isnan(resultados["freq_pico_gamma"])
    ax.plot(np.array(resultados["contraste"])[valid_mask],
            np.array(resultados["freq_pico_gamma"])[valid_mask],
            "o-", color="darkred", linewidth=2, markersize=10,
            label="Peak frequency")
    ax.set_xlabel("Contrast")
    ax.set_ylabel("Peak Gamma Frequency (Hz)")
    ax.set_title("Gamma Frequency vs Contrast")
    ax.set_ylim(20, 100)
    ax.axhspan(30, 80, alpha=0.2, color="red", label="Gamma range")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    output_file = output_dir / "gamma_oscillations_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n✓ Figura guardada: {output_file}")
    plt.close()

    # Imprimir resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE OSCILACIONES GAMMA")
    print("=" * 60)
    print(f"\n{'Contraste':<12} {'Freq Pico (Hz)':<16} {'Potencia Gamma':<16}")
    print("-" * 50)
    for i, c in enumerate(resultados["contraste"]):
        freq = resultados["freq_pico_gamma"][i]
        pot = resultados["potencia_total_gamma"][i]
        freq_str = f"{freq:.1f}" if not np.isnan(freq) else "N/A"
        print(f"{c:<12.3f} {freq_str:<16} {pot:<16.4e}")

    return resultados


def main():
    """Main function."""
    # Directorio de datos
    data_dir = Path(__file__).resolve().parents[2] / "data" / "simulations"
    output_dir = Path(__file__).parent

    print("\n" + "=" * 60)
    print("ANÁLISIS DE OSCILACIONES GAMMA")
    print("=" * 60)
    print(f"\nDirectorio de datos: {data_dir}")
    print(f"Directorio de salida: {output_dir}\n")

    # Cargar simulaciones
    print("Cargando simulaciones...")
    simulaciones = cargar_simulaciones(data_dir)

    if not simulaciones:
        print("❌ No se encontraron simulaciones")
        return 1

    print(f"\n✓ Cargadas {len(simulaciones)} simulaciones\n")

    # Analizar gamma
    print("Analizando oscilaciones gamma...")
    resultados = analizar_gamma(simulaciones, output_dir)

    # Guardar resultados
    results_file = output_dir / "gamma_results.npz"
    np.savez(results_file, **resultados)
    print(f"✓ Resultados guardados: {results_file}")

    print("\n" + "=" * 60)
    print("✓ ANÁLISIS COMPLETADO")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
