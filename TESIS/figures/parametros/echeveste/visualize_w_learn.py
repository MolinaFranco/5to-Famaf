#!/usr/bin/env python3
"""
Visualización de la matriz de pesos aprendidos W_learn de Echeveste et al. 2020.

Este script genera 4 gráficos 2D (líneas) mostrando las conexiones:
- EE (Excitatory to Excitatory)
- EI (Excitatory to Inhibitory)
- IE (Inhibitory to Excitatory)
- II (Inhibitory to Inhibitory)

Cada línea de color representa una neurona PRESINÁPTICA (la que envía) y muestra
los pesos que envía hacia todas las neuronas postsinápticas (eje X).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# Ruta al archivo w_learn
DATA_PATH = Path(__file__).parent.parent.parent.parent / \
    "scikit-neuromsi/skneuromsi/data/echeveste2020/w_learn"
OUTPUT_PATH = Path(__file__).parent / "w_learn_visualization.png"


def load_w_learn(filepath):
    """
    Carga la matriz de pesos W_learn desde el archivo.

    Parameters
    ----------
    filepath : Path
        Ruta al archivo w_learn

    Returns
    -------
    W : ndarray
        Matriz de pesos de forma (100, 100)
        W[i,j] = peso de neurona j hacia neurona i
    """
    return np.loadtxt(filepath)


def neuron_index_to_orientation(neuron_idx, n_neurons=50, centered=True):
    """
    Convierte índice de neurona a su orientación preferida en grados.

    En Echeveste et al. 2020, las 50 neuronas E (y las 50 I) están distribuidas
    uniformemente en un ring topology de 0° a 180° (internamente).

    Mapeo con n_neurons=50:
        Neurona 0  → -90° (centrado) o 0° (interno)
        Neurona 25 → 0° (centrado) o 90° (interno)
        Neurona 49 → 86.4° (centrado) o 176.4° (interno)

    Parameters
    ----------
    neuron_idx : int or array
        Índice de neurona(s) dentro de su población (0-49)
    n_neurons : int
        Número de neuronas en la población (default: 50)
    centered : bool
        Si True, devuelve orientaciones de -90° a ~86° (convención de visualización)
        Si False, devuelve orientaciones de 0° a ~176° (convención interna)

    Returns
    -------
    orientation : float or array
        Orientación(es) en grados

    Notes
    -----
    Internamente el modelo usa linspace(0, pi, N_E, endpoint=False)
    Ver: _echeveste2020.py línea 1374
    """
    # orientations = np.linspace(0, np.pi, n_neurons, endpoint=False)
    orientation_rad = np.pi * neuron_idx / n_neurons
    orientation_deg = np.rad2deg(orientation_rad)

    if centered:
        # Recentrar: 0-180° → -90-90°
        orientation_deg = orientation_deg - 90.0

    return orientation_deg


def split_w_matrix(W, n_exc=50):
    """
    Divide la matriz W en sus 4 bloques: EE, EI, IE, II.

    Parameters
    ----------
    W : ndarray
        Matriz completa de pesos (N, N)
        W[i,j] = peso de neurona j hacia neurona i
    n_exc : int
        Número de neuronas excitatorias (default: 50)

    Returns
    -------
    W_EE, W_EI, W_IE, W_II : ndarray
        Los 4 bloques de la matriz de conectividad
        - W_EE[i,j]: peso de neurona E_j hacia neurona E_i
        - W_EI[i,j]: peso de neurona I_j hacia neurona E_i
        - W_IE[i,j]: peso de neurona E_j hacia neurona I_i
        - W_II[i,j]: peso de neurona I_j hacia neurona I_i
    """
    W_EE = W[:n_exc, :n_exc]
    W_EI = W[:n_exc, n_exc:]
    W_IE = W[n_exc:, :n_exc]
    W_II = W[n_exc:, n_exc:]

    return W_EE, W_EI, W_IE, W_II


def select_neurons_to_plot(n_neurons, n_to_show=10):
    """
    Selecciona índices de neuronas a mostrar, incluyendo obligatoriamente
    la neurona 0 y espaciando uniformemente el resto.

    Parameters
    ----------
    n_neurons : int
        Número total de neuronas en el bloque
    n_to_show : int
        Número de neuronas a mostrar (default: 10)

    Returns
    -------
    indices : ndarray
        Índices de las neuronas seleccionadas
    """
    if n_to_show >= n_neurons:
        return np.arange(n_neurons)

    # Espaciar uniformemente
    indices = np.linspace(0, n_neurons - 1, n_to_show).astype(int)

    # Asegurar que 0 esté incluido
    if 0 not in indices:
        indices[0] = 0
        indices = np.unique(indices)

    return np.sort(indices)


def plot_w_learn_blocks(W_EE, W_EI, W_IE, W_II, n_neurons_to_show=8):
    """
    Crea una figura con 4 gráficos 2D mostrando los bloques de W_learn.

    Cada gráfico muestra varias líneas de colores:
    - Cada línea representa una neurona PRESINÁPTICA (la que envía)
    - El eje X muestra las orientaciones de las neuronas POSTSINÁPTICAS
    - El eje Y muestra el peso de la conexión

    Parameters
    ----------
    W_EE, W_EI, W_IE, W_II : ndarray
        Los 4 bloques de la matriz de conectividad
    n_neurons_to_show : int
        Número de neuronas presinápticas a mostrar (default: 8)

    Returns
    -------
    fig : Figure
        Figura de matplotlib con los 4 gráficos
    """
    n_neurons_per_type = W_EE.shape[0]  # 50

    # Seleccionar neuronas PRESINÁPTICAS a visualizar (las líneas de colores)
    selected_pre = select_neurons_to_plot(
        n_neurons_per_type, n_neurons_to_show
    )

    # Orientaciones de TODAS las neuronas postsinápticas (eje X)
    # Las 50 neuronas cubren de -90° a ~86.4° (centrado)
    post_orientations = neuron_index_to_orientation(
        np.arange(n_neurons_per_type), n_neurons=n_neurons_per_type,
        centered=True
    )

    # Orientaciones de las neuronas presinápticas seleccionadas (para leyenda)
    pre_orientations = neuron_index_to_orientation(
        selected_pre, n_neurons=n_neurons_per_type, centered=True
    )

    # Crear figura con 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        'Learned Weight Matrix (W_learn) - Echeveste et al. 2020',
        fontsize=16, fontweight='bold'
    )

    # Bloques de conectividad
    blocks = [
        (W_EE, 'E → E', axes[0, 0]),
        (W_EI, 'I → E', axes[0, 1]),
        (W_IE, 'E → I', axes[1, 0]),
        (W_II, 'I → I', axes[1, 1])
    ]

    # Colores para las diferentes neuronas presinápticas
    # Usamos tab10 que tiene colores más distinguibles y agradables
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_pre)))

    # Calcular rango Y global para todos los gráficos
    all_values = np.concatenate([
        W_EE.flatten(), W_EI.flatten(),
        W_IE.flatten(), W_II.flatten()
    ])
    y_min, y_max = all_values.min(), all_values.max()
    y_margin = (y_max - y_min) * 0.1
    y_min -= y_margin
    y_max += y_margin

    for (block, title, ax) in blocks:
        # Para cada neurona PRESINÁPTICA seleccionada (cada línea)
        for idx, pre_idx in enumerate(selected_pre):
            # W[i,j] = peso de j hacia i
            # Queremos: pesos que ENVÍA la neurona pre_idx hacia todas las post
            # Eso es la COLUMNA pre_idx: block[:, pre_idx]
            weights = block[:, pre_idx]

            # Plotear
            label_str = f'{pre_orientations[idx]:.0f}°'
            ax.plot(
                post_orientations,
                weights,
                color=colors[idx],
                linewidth=2,
                label=label_str,
                alpha=0.8
            )

        # Configurar ejes
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Postsynaptic Orientation (deg)', fontsize=11)
        ax.set_ylabel('Connection Weight', fontsize=11)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-90, 90)
        ax.grid(True, alpha=0.3)

        # Agregar texto con sigma (ancho del perfil en grados)
        # Ajustar gaussiana a una columna central para obtener el ancho
        def gaussian(x, a, mu, sigma):
            return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        # Usar neurona central (índice 25, orientación ~0°)
        central_col = block[:, n_neurons_per_type // 2]
        try:
            popt, _ = curve_fit(
                gaussian, post_orientations, central_col,
                p0=[np.max(central_col), 0, 30],
                maxfev=5000
            )
            sigma_deg = abs(popt[2])
            ax.text(
                0.02, 0.98,
                f'σ: {sigma_deg:.1f}°',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        except Exception:
            pass  # Si falla el ajuste, no mostrar nada

        # Solo mostrar leyenda en el último gráfico (II)
        if title == 'I → I':
            ax.legend(
                title='Presynaptic',
                loc='upper right',
                fontsize=11,
                title_fontsize=12,
                ncol=2
            )

        # Línea horizontal en y=0 para referencia
        ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)

    plt.tight_layout()

    return fig


def main():
    """Función principal."""
    print("Cargando matriz W_learn...")
    W = load_w_learn(DATA_PATH)
    print(f"Matriz cargada: forma {W.shape}")
    print(f"  W[i,j] = peso de neurona j hacia neurona i")

    print("\nDividiendo en bloques EE, EI, IE, II...")
    W_EE, W_EI, W_IE, W_II = split_w_matrix(W)

    print(f"  W_EE: {W_EE.shape} - E → E")
    print(f"  W_EI: {W_EI.shape} - I → E")
    print(f"  W_IE: {W_IE.shape} - E → I")
    print(f"  W_II: {W_II.shape} - I → I")

    # Verificar mapeo de orientaciones
    print("\nMapeo neurona → orientación (50 neuronas, centrado):")
    print(f"  Neurona 0  → {neuron_index_to_orientation(0, 50, True):.1f}°")
    print(f"  Neurona 25 → {neuron_index_to_orientation(25, 50, True):.1f}°")
    print(f"  Neurona 49 → {neuron_index_to_orientation(49, 50, True):.1f}°")

    print("\nGenerando visualización...")
    fig = plot_w_learn_blocks(W_EE, W_EI, W_IE, W_II, n_neurons_to_show=8)

    print(f"Guardando figura en: {OUTPUT_PATH}")
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print("¡Listo!")

    # Mostrar estadísticas básicas
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE LOS BLOQUES:")
    print("="*60)

    blocks_info = [
        ("W_EE", W_EE),
        ("W_EI", W_EI),
        ("W_IE", W_IE),
        ("W_II", W_II)
    ]

    for name, block in blocks_info:
        print(f"\n{name}:")
        print(f"  Min:  {block.min():.6f}")
        print(f"  Max:  {block.max():.6f}")
        print(f"  Mean: {block.mean():.6f}")
        print(f"  Std:  {block.std():.6f}")


if __name__ == "__main__":
    main()
