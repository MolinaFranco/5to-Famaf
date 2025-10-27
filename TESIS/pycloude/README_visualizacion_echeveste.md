# Módulo de Visualización - Modelo Echeveste et al. (2020)

## Descripción

Este módulo proporciona herramientas de visualización para analizar los estados neuronales durante la simulación del modelo SSN (Stabilized Supralinear Network) de Echeveste et al. (2020).

Es el primer modelo en este proyecto donde los estados de las neuronas durante la simulación son importantes (no solo el estado final), por lo que se requieren visualizaciones temporales detalladas.

## Archivos

### Módulo principal
- **[scikit-neuromsi/skneuromsi/neural/_echeveste_visualization.py](../scikit-neuromsi/skneuromsi/neural/_echeveste_visualization.py)**
  - Módulo de visualización con 8 funciones especializadas
  - Basado en las figuras del paper de Echeveste (echepaper.pdf)
  - Todas las funciones documentadas en inglés con referencias a papers

### Script de ejemplo
- **[pycloude/ejemplo_visualizacion_echeveste.py](ejemplo_visualizacion_echeveste.py)**
  - Script completo que demuestra cómo usar el módulo
  - Ejecuta una simulación y genera todas las visualizaciones
  - Guarda los gráficos en `pycloude/outputs/`

## Funciones de visualización disponibles

### 1. `plot_membrane_potentials()`
Grafica los potenciales de membrana u(t) de neuronas individuales a través del tiempo.
- **Basado en**: Main paper Eq. 8, Figuras 2, 4, 5
- **Uso**: Analizar la evolución temporal de neuronas específicas
- **Parámetros**: índices de neuronas, población (E/I/ambas)

### 2. `plot_population_activity()`
Muestra la actividad de toda la población como heatmap (estilo raster).
- **Basado en**: Main paper Fig. 2a
- **Uso**: Ver patrones de actividad en toda la red
- **Visualización**: Heatmap con tiempo en x, neuronas en y, color = firing rate

### 3. `plot_mean_firing_rates()`
Grafica las tasas de disparo promedio ± desviación estándar.
- **Basado en**: Main paper Fig. 2b
- **Uso**: Analizar actividad poblacional promedio y su variabilidad
- **Características**: Suavizado opcional con ventana móvil

### 4. `plot_autocorrelation()`
Análisis de autocorrelación temporal de la actividad neuronal.
- **Basado en**: Main paper Fig. 4a, Eq. 11
- **Uso**: Revelar oscilaciones en la actividad
- **Detecta**: Oscilaciones gamma (20-80 Hz) emergentes de interacciones E-I

### 5. `plot_power_spectrum()`
Espectro de potencia para analizar componentes frecuenciales.
- **Basado en**: Main paper Fig. 5c, Fig. 6
- **Uso**: Identificar oscilaciones gamma
- **Método**: Welch's periodogram (scipy.signal.welch)
- **Marca**: Banda gamma (20-80 Hz) automáticamente

### 6. `plot_transient_analysis()`
Análisis de transitorios (overshoots) en respuestas neuronales.
- **Basado en**: Main paper Fig. 7, Supplementary Sec. S2.3
- **Uso**: Medir overshoots al inicio del estímulo
- **Parámetros**: ventanas de baseline y transitorio personalizables

### 7. `plot_fano_factor()`
Factor de Fano (varianza/media) a través del tiempo.
- **Basado en**: Main paper Fig. 5d
- **Uso**: Analizar variabilidad trial-to-trial
- **Interpretación**: Modulación por intensidad de estímulo

### 8. `plot_neural_dynamics_summary()`
Resumen multi-panel con 6 análisis principales.
- **Basado en**: Combinación de Figuras 2-7 del paper
- **Uso**: Vista panorámica de todas las dinámicas
- **Paneles**: Actividad media, raster, autocorrelación, espectro, varianza, Fano

## Uso básico

```python
from skneuromsi.neural import Echeveste2020
from skneuromsi.neural._echeveste_visualization import (
    plot_membrane_potentials,
    plot_neural_dynamics_summary,
)

# Crear y entrenar modelo
ssn = Echeveste2020(N_E=50, N_I=50)
ssn.load_parameters()  # o ssn.train()

# Ejecutar simulación
result = ssn.run(
    stimulus_contrast=0.019,
    stimulus_orientation=0.0,
    noise_level=0.1,
    simulation_time=1000.0
)

# Visualizar
fig1, ax1 = plot_membrane_potentials(result, neuron_indices=[0,1,2,3,4])
fig2, ax2 = plot_neural_dynamics_summary(result)
```

## Ejecutar el ejemplo completo

```bash
cd /home/molina/FAMAF/5to-Famaf/TESIS/pycloude
python ejemplo_visualizacion_echeveste.py
```

Esto generará 8 gráficos guardados en `pycloude/outputs/`:
1. `echeveste_membrane_potentials.png` - Potenciales de membrana
2. `echeveste_population_activity.png` - Actividad poblacional (heatmap)
3. `echeveste_mean_firing_rates.png` - Firing rates promedio
4. `echeveste_autocorrelation.png` - Autocorrelación temporal
5. `echeveste_power_spectrum.png` - Espectro de potencia (gamma)
6. `echeveste_transients.png` - Análisis de transitorios
7. `echeveste_fano_factor.png` - Factor de Fano
8. `echeveste_dynamics_summary.png` - Resumen multi-panel

## Referencias matemáticas

Todas las funciones están basadas en:

- **Paper principal**: Echeveste et al. (2020), Nature Neuroscience
  - "Cortical-like dynamics in recurrent circuits optimized for sampling-based probabilistic inference"
  - Archivo: `echepaper.pdf` (42 páginas)

- **Material suplementario**: Supplementary Information
  - Derivaciones matemáticas detalladas
  - Archivo: `Supplementary_information_echerpaper(1).pdf` (11 páginas)

### Ecuaciones clave

- **Eq. 8**: Dinámica de potenciales de membrana
  ```
  τ_α du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α
  ```

- **Eq. 9**: Activación supralineal (firing rates)
  ```
  r_α(t) = k[u_α(t)]_+^n
  ```
  donde k=0.3, n=2.0, [x]_+ = max(0, x)

- **Eq. 11**: Correlaciones del ruido
  ```
  ⟨η(t)η(t+s)^T⟩ = Σ_η exp(-s/τ_η)
  ```

## Estándares de código

✓ Todo el código pasa flake8 (PEP 8)
✓ Máximo 79 caracteres por línea
✓ Docstrings completos en inglés
✓ Comentarios en español (inline con #)
✓ Referencias a papers en todos los docstrings

## Próximos pasos sugeridos

1. Ejecutar `ejemplo_visualizacion_echeveste.py` para generar visualizaciones
2. Experimentar con diferentes parámetros de simulación
3. Comparar resultados con figuras del paper original
4. Usar estas visualizaciones para validar el modelo

## Autor

Implementado como parte del proyecto scikit-neuromsi
Basado en el modelo de Echeveste et al. (2020)
