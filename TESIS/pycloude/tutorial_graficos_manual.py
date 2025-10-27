#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TUTORIAL: Cómo hacer gráficos de redes neuronales MANUALMENTE.

Este script es un tutorial paso a paso que explica cómo crear cada tipo
de gráfico desde cero, sin usar funciones helper pre-construidas.

Aprenderás:
1. Cómo extraer datos de las simulaciones
2. Cómo crear diferentes tipos de gráficos con matplotlib
3. Cómo personalizar colores, etiquetas y estilos
4. Buenas prácticas para visualización científica

Cada sección está comentada en detalle explicando QUÉ hace cada línea
y POR QUÉ la hacemos así.
"""

import sys
sys.path.insert(0, "/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi")

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from skneuromsi.neural import Echeveste2020  # noqa: E402


def tutorial_paso_1_extraer_datos():
    """
    PASO 1: Cómo extraer datos de una simulación.

    Este es el primer paso para CUALQUIER gráfico: necesitamos obtener
    los datos de actividad neuronal de la simulación.
    """
    print("\n" + "="*70)
    print("PASO 1: EXTRAER DATOS DE LA SIMULACIÓN")
    print("="*70)

    # 1.1 Crear y configurar el modelo
    print("\n1.1. Crear modelo SSN")
    print("-" * 50)
    print("Código:")
    print("    ssn = Echeveste2020(N_E=50, N_I=50, seed=42)")
    print("\nExplicación:")
    print("  - N_E=50: 50 neuronas excitatorias")
    print("  - N_I=50: 50 neuronas inhibitorias")
    print("  - seed=42: Para reproducibilidad (mismo ruido cada vez)")

    ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
    ssn.load_parameters()

    # 1.2 Ejecutar simulación
    print("\n1.2. Ejecutar simulación")
    print("-" * 50)
    print("Código:")
    print("    result = ssn.run(")
    print("        stimulus_contrast=0.5,    # Contraste del estímulo")
    print("        stimulus_orientation=0.0, # Orientación en grados")
    print("        noise_level=0.1,          # Nivel de ruido")
    print("        simulation_time=100.0,    # Tiempo en ms")
    print("    )")
    print("\nExplicación:")
    print("  - stimulus_contrast: Qué tan fuerte es el estímulo (0-1)")
    print("  - stimulus_orientation: Orientación del Gabor patch (0-180°)")
    print("  - noise_level: Variabilidad estocástica")
    print("  - simulation_time: Cuánto tiempo simulamos (ms)")

    result = ssn.run(
        stimulus_contrast=0.5,
        stimulus_orientation=0.0,
        noise_level=0.1,
        simulation_time=100.0,
    )

    # 1.3 Extraer datos del resultado
    print("\n1.3. Extraer datos del objeto 'result'")
    print("-" * 50)
    print("Código:")
    print("    df = result.get_modes()")
    print("    n_times = len(result.times_)")
    print("\nExplicación:")
    print("  - get_modes(): Devuelve un DataFrame con todas las variables")
    print("  - result.times_: Array con los tiempos de simulación")

    df = result.get_modes()
    n_times = len(result.times_)

    print("\nDatos disponibles en df:")
    print(f"  Columnas: {list(df.columns)}")
    print(f"  Shape: {df.shape}")
    print(f"  Tiempos: {n_times} pasos temporales")

    # 1.4 Extraer arrays numpy específicos
    print("\n1.4. Convertir a arrays numpy para graficar")
    print("-" * 50)
    print("Código:")
    print("    u_e = df['excitatory_potential'].values.reshape(n_times, -1)")
    print("    r_e = df['excitatory_firing_rate'].values.reshape(n_times, -1)")
    print("\nExplicación:")
    print("  - .values: Convierte de pandas a numpy array")
    print("  - .reshape(n_times, -1): Organiza como matriz")
    print("    → Filas = tiempo, Columnas = neuronas")

    u_e = df['excitatory_potential'].values.reshape(n_times, -1)
    r_e = df['excitatory_firing_rate'].values.reshape(n_times, -1)
    u_i = df['inhibitory_potential'].values.reshape(n_times, -1)
    r_i = df['inhibitory_firing_rate'].values.reshape(n_times, -1)

    print("\nForma de los datos:")
    print(f"  u_e.shape = {u_e.shape} → ({n_times} tiempos, "
          f"{u_e.shape[1]} neuronas)")
    print(f"  r_e.shape = {r_e.shape}")
    print(f"  u_i.shape = {u_i.shape}")
    print(f"  r_i.shape = {r_i.shape}")

    print("\n✓ Ahora tenemos los datos listos para graficar!")

    return result, u_e, r_e, u_i, r_i


def tutorial_paso_2_grafico_basico(result, r_e):
    """
    PASO 2: Crear un gráfico básico de línea temporal.

    Aprenderás a graficar la actividad de una sola neurona vs tiempo.
    """
    print("\n" + "="*70)
    print("PASO 2: GRÁFICO BÁSICO - UNA NEURONA VS TIEMPO")
    print("="*70)

    print("\n2.1. Crear figura y eje")
    print("-" * 50)
    print("Código:")
    print("    fig, ax = plt.subplots(1, 1, figsize=(12, 6))")
    print("\nExplicación:")
    print("  - plt.subplots(): Crea figura y ejes")
    print("  - 1, 1: 1 fila, 1 columna (un solo panel)")
    print("  - figsize=(12, 6): Tamaño en pulgadas (ancho, alto)")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    print("\n2.2. Graficar la actividad de la neurona E0")
    print("-" * 50)
    print("Código:")
    print("    neuron_id = 0")
    print("    ax.plot(result.times_, r_e[:, neuron_id])")
    print("\nExplicación:")
    print("  - result.times_: Eje X (tiempo en ms)")
    print("  - r_e[:, neuron_id]: Eje Y (firing rate de neurona 0)")
    print("    → r_e[:, 0] significa 'todas las filas, columna 0'")

    neuron_id = 0
    ax.plot(result.times_, r_e[:, neuron_id], lw=2, color='blue',
            label=f'Neurona E{neuron_id}')

    print("\n2.3. Personalizar el gráfico")
    print("-" * 50)
    print("Código:")
    print("    ax.set_xlabel('Time (ms)')")
    print("    ax.set_ylabel('Firing rate (Hz)')")
    print("    ax.set_title('Actividad temporal de una neurona')")
    print("    ax.legend()")
    print("    ax.grid(True, alpha=0.3)")
    print("\nExplicación:")
    print("  - set_xlabel/ylabel: Etiquetas de ejes (¡siempre incluir unidades!)")  # noqa: E501
    print("  - set_title: Título descriptivo")
    print("  - legend(): Muestra la leyenda")
    print("  - grid(alpha=0.3): Grid semitransparente para leer valores")

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Firing rate (Hz)', fontsize=12)
    ax.set_title('Actividad temporal de una neurona excitatoria',
                 fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    print("\n2.4. Guardar el gráfico")
    print("-" * 50)
    print("Código:")
    print("    output = 'outputs/tutorial_paso2_basico.png'")
    print("    plt.savefig(output, dpi=150, bbox_inches='tight')")
    print("\nExplicación:")
    print("  - dpi=150: Resolución (dots per inch)")
    print("  - bbox_inches='tight': Recorta espacios blancos extra")

    output_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs"
    output = f'{output_dir}/tutorial_paso2_basico.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Gráfico guardado: {output}")


def tutorial_paso_3_multiples_neuronas(result, r_e):
    """
    PASO 3: Graficar múltiples neuronas en el mismo plot.

    Aprenderás a usar loops para graficar varias líneas.
    """
    print("\n" + "="*70)
    print("PASO 3: MÚLTIPLES NEURONAS EN UN PLOT")
    print("="*70)

    print("\n3.1. Crear el plot")
    print("-" * 50)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    print("\n3.2. Usar un loop para graficar varias neuronas")
    print("-" * 50)
    print("Código:")
    print("    for neuron_id in [0, 10, 20, 30, 40]:")
    print("        ax.plot(result.times_, r_e[:, neuron_id],")
    print("                label=f'E{neuron_id}', alpha=0.7)")
    print("\nExplicación:")
    print("  - Loop sobre IDs de neuronas espaciadas (cada 10)")
    print("  - alpha=0.7: Transparencia para ver líneas superpuestas")
    print("  - label=f'E{neuron_id}': f-string para nombres dinámicos")

    # Seleccionar neuronas espaciadas uniformemente
    neuron_ids = [0, 10, 20, 30, 40]
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for neuron_id, color in zip(neuron_ids, colors):
        ax.plot(result.times_, r_e[:, neuron_id],
                label=f'E{neuron_id}', alpha=0.7, lw=2, color=color)

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Firing rate (Hz)', fontsize=12)
    ax.set_title('Comparación de neuronas espaciadas uniformemente',
                 fontsize=14)
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    output_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs"
    output = f'{output_dir}/tutorial_paso3_multiples.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Gráfico guardado: {output}")


def tutorial_paso_4_heatmap(result, r_e):
    """
    PASO 4: Crear un HEATMAP (gráfico de calor).

    Este es el tipo de gráfico que pediste: todas las neuronas
    representadas como filas, con colores según actividad.
    """
    print("\n" + "="*70)
    print("PASO 4: HEATMAP - TODAS LAS NEURONAS × TIEMPO")
    print("="*70)

    print("\n4.1. Entender la estructura de datos para heatmap")
    print("-" * 50)
    print("Tenemos:")
    print(f"  r_e.shape = {r_e.shape}")
    print(f"  → {r_e.shape[0]} pasos de tiempo")
    print(f"  → {r_e.shape[1]} neuronas")
    print("\nPara el heatmap queremos:")
    print("  - Eje X: Tiempo")
    print("  - Eje Y: ID de neurona (0 a 49)")
    print("  - Color: Intensidad del firing rate")

    print("\n4.2. Crear el heatmap con imshow()")
    print("-" * 50)
    print("Código:")
    print("    fig, ax = plt.subplots(1, 1, figsize=(14, 8))")
    print("    im = ax.imshow(r_e.T,  # Transponemos!")
    print("                   aspect='auto',")
    print("                   cmap='viridis',")
    print("                   origin='lower')")
    print("\nExplicación:")
    print("  - r_e.T: TRANSPONER la matriz")
    print("    Original: (tiempo, neuronas)")
    print("    Transpuesta: (neuronas, tiempo)")
    print("    → imshow espera (filas=Y, columnas=X)")
    print("  - aspect='auto': Ajusta proporción automáticamente")
    print("  - cmap='viridis': Mapa de colores (viridis, plasma, etc.)")
    print("  - origin='lower': Y=0 abajo (como gráfico normal)")

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # IMPORTANTE: extent para mapear valores correctos en ejes
    print("\n4.3. Usar 'extent' para ejes correctos")
    print("-" * 50)
    print("Código:")
    print("    extent = [result.times_[0], result.times_[-1],")
    print("              0, r_e.shape[1]]")
    print("\nExplicación:")
    print("  - extent = [x_min, x_max, y_min, y_max]")
    print("  - Mapea los píxeles de la imagen a valores reales")

    im = ax.imshow(
        r_e.T,  # ¡Transponer!
        aspect='auto',
        cmap='viridis',
        extent=[result.times_[0], result.times_[-1],
                0, r_e.shape[1]],
        origin='lower',
        interpolation='nearest'  # Sin suavizado
    )

    print("\n4.4. Agregar colorbar")
    print("-" * 50)
    print("Código:")
    print("    cbar = plt.colorbar(im, ax=ax)")
    print("    cbar.set_label('Firing rate (Hz)', fontsize=12)")
    print("\nExplicación:")
    print("  - colorbar(): Barra de colores mostrando escala")
    print("  - set_label(): Etiqueta para la barra (¡con unidades!)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Firing rate (Hz)', fontsize=12)

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Neuron ID', fontsize=12)
    ax.set_title('Heatmap: Actividad de 50 neuronas excitatorias vs tiempo',
                 fontsize=14)

    output_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs"
    output = f'{output_dir}/tutorial_paso4_heatmap.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Gráfico guardado: {output}")

    print("\n4.5. Mapas de colores recomendados")
    print("-" * 50)
    print("  - 'viridis': Bueno para datos positivos (firing rates)")
    print("  - 'plasma': Alternativa a viridis")
    print("  - 'RdBu_r': Rojo-Blanco-Azul, para datos con cero central")
    print("              (potenciales de membrana)")
    print("  - 'coolwarm': Similar a RdBu_r")
    print("  - 'gray': Escala de grises (para publicaciones B&W)")


def tutorial_paso_5_subplots(result, u_e, r_e):
    """
    PASO 5: Múltiples paneles (subplots).

    Aprenderás a crear figuras con varios paneles.
    """
    print("\n" + "="*70)
    print("PASO 5: MÚLTIPLES PANELES (SUBPLOTS)")
    print("="*70)

    print("\n5.1. Crear grilla de subplots")
    print("-" * 50)
    print("Código:")
    print("    fig, axes = plt.subplots(2, 2, figsize=(14, 10))")
    print("\nExplicación:")
    print("  - 2, 2: Grilla de 2 filas × 2 columnas = 4 paneles")
    print("  - axes: Array 2D con los ejes")
    print("    → axes[0, 0] = panel superior izquierdo")
    print("    → axes[0, 1] = panel superior derecho")
    print("    → axes[1, 0] = panel inferior izquierdo")
    print("    → axes[1, 1] = panel inferior derecho")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    print("\n5.2. Graficar en cada panel")
    print("-" * 50)

    # Panel 1: Potenciales de varias neuronas
    ax1 = axes[0, 0]
    for i in [0, 10, 20, 30, 40]:
        ax1.plot(result.times_, u_e[:, i], label=f'E{i}', alpha=0.7)
    ax1.set_ylabel('Membrane potential u (a.u.)')
    ax1.set_title('Panel A: Potenciales de membrana')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Firing rates de las mismas neuronas
    ax2 = axes[0, 1]
    for i in [0, 10, 20, 30, 40]:
        ax2.plot(result.times_, r_e[:, i], label=f'E{i}', alpha=0.7)
    ax2.set_ylabel('Firing rate (Hz)')
    ax2.set_title('Panel B: Firing rates')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Heatmap de potenciales
    ax3 = axes[1, 0]
    im3 = ax3.imshow(u_e.T, aspect='auto', cmap='RdBu_r',
                     extent=[result.times_[0], result.times_[-1],
                             0, u_e.shape[1]],
                     origin='lower')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Neuron ID')
    ax3.set_title('Panel C: Heatmap potenciales')
    plt.colorbar(im3, ax=ax3, label='u (a.u.)')

    # Panel 4: Heatmap de firing rates
    ax4 = axes[1, 1]
    im4 = ax4.imshow(r_e.T, aspect='auto', cmap='viridis',
                     extent=[result.times_[0], result.times_[-1],
                             0, r_e.shape[1]],
                     origin='lower')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Neuron ID')
    ax4.set_title('Panel D: Heatmap firing rates')
    plt.colorbar(im4, ax=ax4, label='r (Hz)')

    print("\n5.3. Ajustar espaciado entre paneles")
    print("-" * 50)
    print("Código:")
    print("    plt.tight_layout()")
    print("\nExplicación:")
    print("  - Ajusta automáticamente para evitar solapamientos")

    plt.tight_layout()

    output_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs"
    output = f'{output_dir}/tutorial_paso5_subplots.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Gráfico guardado: {output}")


def tutorial_paso_6_estadisticas(result, r_e):
    """
    PASO 6: Gráficos con estadísticas (media, desviación estándar).

    Aprenderás a calcular y visualizar estadísticas poblacionales.
    """
    print("\n" + "="*70)
    print("PASO 6: GRÁFICOS CON ESTADÍSTICAS")
    print("="*70)

    print("\n6.1. Calcular media poblacional")
    print("-" * 50)
    print("Código:")
    print("    r_mean = np.mean(r_e, axis=1)")
    print("\nExplicación:")
    print(f"  - r_e.shape = {r_e.shape}")
    print("  - axis=1: Promediar sobre columnas (neuronas)")
    print("    → Resultado: array de tamaño (tiempo,)")
    print("  - axis=0 sería promediar sobre tiempo")

    r_mean = np.mean(r_e, axis=1)  # Media entre neuronas
    r_std = np.std(r_e, axis=1)    # Desviación estándar

    print(f"\n  Resultado: r_mean.shape = {r_mean.shape}")

    print("\n6.2. Graficar media ± desviación estándar")
    print("-" * 50)
    print("Código:")
    print("    fig, ax = plt.subplots(1, 1, figsize=(12, 6))")
    print("    ax.plot(result.times_, r_mean, lw=2, label='Media')")
    print("    ax.fill_between(result.times_,")
    print("                    r_mean - r_std,")
    print("                    r_mean + r_std,")
    print("                    alpha=0.3, label='± 1 std')")
    print("\nExplicación:")
    print("  - fill_between(): Rellena área entre dos curvas")
    print("  - alpha=0.3: Transparencia para ver la línea de media")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(result.times_, r_mean, lw=2.5, color='darkblue',
            label='Media poblacional')
    ax.fill_between(result.times_,
                    r_mean - r_std,
                    r_mean + r_std,
                    alpha=0.3, color='blue',
                    label='± 1 desviación estándar')

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Population mean firing rate (Hz)', fontsize=12)
    ax.set_title('Actividad poblacional: media ± std', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    output_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/pycloude/outputs"
    output = f'{output_dir}/tutorial_paso6_estadisticas.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Gráfico guardado: {output}")


def tutorial_resumen():
    """
    Resumen del tutorial con tips y buenas prácticas.
    """
    print("\n" + "="*70)
    print("RESUMEN Y BUENAS PRÁCTICAS")
    print("="*70)

    print("\n📊 TIPOS DE GRÁFICOS CUBIERTOS:")
    print("-" * 50)
    print("1. Gráfico básico de línea (1 neurona vs tiempo)")
    print("2. Múltiples líneas (varias neuronas)")
    print("3. Heatmap (todas las neuronas × tiempo)")
    print("4. Subplots (múltiples paneles)")
    print("5. Media ± desviación estándar")

    print("\n💡 CONCEPTOS CLAVE:")
    print("-" * 50)
    print("• Extracción de datos:")
    print("  - df = result.get_modes()")
    print("  - .values.reshape(n_times, -1)")
    print("\n• Indexación de numpy:")
    print("  - r_e[:, 0]: Todas las filas, columna 0 (neurona 0)")
    print("  - r_e[10, :]: Fila 10 (tiempo 10), todas las columnas")
    print("\n• Operaciones estadísticas:")
    print("  - np.mean(r_e, axis=1): Media sobre neuronas")
    print("  - np.mean(r_e, axis=0): Media sobre tiempo")

    print("\n✅ CHECKLIST PARA BUENOS GRÁFICOS:")
    print("-" * 50)
    print("□ Etiquetas de ejes CON UNIDADES (Time (ms), Rate (Hz))")
    print("□ Título descriptivo")
    print("□ Leyenda cuando hay múltiples líneas")
    print("□ Grid para facilitar lectura de valores")
    print("□ Tamaño de figura apropiado (figsize)")
    print("□ DPI suficiente para publicación (dpi=150 o 300)")
    print("□ Colores contrastantes y accesibles")
    print("□ Colorbar con etiqueta en heatmaps")

    print("\n🎨 MAPAS DE COLORES RECOMENDADOS:")
    print("-" * 50)
    print("Datos positivos (firing rates):")
    print("  - viridis, plasma, inferno, magma")
    print("\nDatos con cero central (potenciales):")
    print("  - RdBu_r, coolwarm, seismic")
    print("\nBlanco y negro:")
    print("  - gray, binary")

    print("\n📚 RECURSOS ADICIONALES:")
    print("-" * 50)
    print("• Matplotlib gallery: matplotlib.org/stable/gallery/")
    print("• Colormaps: matplotlib.org/stable/tutorials/colors/colormaps.html")  # noqa: E501
    print("• Seaborn (gráficos estadísticos): seaborn.pydata.org/")

    print("\n" + "="*70)


def main():
    """Ejecutar el tutorial completo."""
    print("="*70)
    print("TUTORIAL: CÓMO HACER GRÁFICOS DE REDES NEURONALES")
    print("="*70)
    print("\nEste tutorial te enseñará paso a paso cómo crear")
    print("visualizaciones profesionales de simulaciones neuronales.")
    print("\nCada paso está comentado en detalle. ¡Sigue leyendo!")

    # Ejecutar cada paso del tutorial
    result, u_e, r_e, u_i, r_i = tutorial_paso_1_extraer_datos()
    tutorial_paso_2_grafico_basico(result, r_e)
    tutorial_paso_3_multiples_neuronas(result, r_e)
    tutorial_paso_4_heatmap(result, r_e)
    tutorial_paso_5_subplots(result, u_e, r_e)
    tutorial_paso_6_estadisticas(result, r_e)
    tutorial_resumen()

    print("\n" + "="*70)
    print("✓ TUTORIAL COMPLETADO")
    print("="*70)
    print("\nRevisa los archivos generados en outputs/ para ver")
    print("los resultados de cada paso.")
    print("\n¡Ahora ya sabes cómo hacer estos gráficos manualmente!")
    print("="*70)


if __name__ == "__main__":
    main()
