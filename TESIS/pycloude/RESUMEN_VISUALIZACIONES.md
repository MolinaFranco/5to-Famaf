# ✅ Funciones de Visualización - AHORA EN EL MÓDULO

## 🎉 Cambios Importantes

### 1. Funciones en el módulo
Las funciones de **heatmap** y **muestreo espacial** ahora están disponibles **directamente en el módulo** `skneuromsi.neural._echeveste_visualization`!

Ya no necesitas copiar código - simplemente importa y usa.

### 2. Convención coherente de orientaciones
**TODAS** las funciones de visualización ahora usan **-90° a 90°** por defecto (como Fig. 2A del paper).

- ✓ Coherencia entre todos los gráficos
- ✓ Consistente con el paper (Echeveste et al. 2020)
- ✓ Menos confusión al interpretar resultados
- ✓ Convención alternativa disponible con `orientation_centered=False`

Ver [`CAMBIO_CONVENCIONES_COHERENTES.md`](CAMBIO_CONVENCIONES_COHERENTES.md) para detalles.

---

## 📦 Funciones Disponibles en el Módulo

### 1. `plot_heatmap_activity()` - ¡NUEVO!

**Muestra TODAS las 50 neuronas simultáneamente como matriz de colores**

```python
from skneuromsi.neural._echeveste_visualization import plot_heatmap_activity

# Usar la función
fig, axes = plot_heatmap_activity(result, population='excitatory')
fig.savefig('heatmap.png', dpi=150, bbox_inches='tight')
```

**Qué genera:**
- Panel 1: Heatmap de potenciales u(t) - Todas las neuronas × tiempo
- Panel 2: Heatmap de firing rates r(t) - Todas las neuronas × tiempo
- Panel 3: Gráfico de barras - Promedio temporal por neurona (tuning espacial)
- Panel 4: Serie temporal - Media poblacional ± desviación estándar

**Parámetros:**
- `result`: Resultado de `ssn.run()`
- `population`: `'excitatory'` o `'inhibitory'` (default: `'excitatory'`)
- `orientation_centered`: `True` para mostrar -90° a 90° (como Fig. 2A) (default: `True`), `False` para Neuron IDs
- `figsize`: Tamaño de figura (default: `(16, 10)`)

**Retorna:**
- `fig`: Objeto Figure
- `axes`: Tupla con 4 axes `(ax1, ax2, ax3, ax4)`

---

### 2. `plot_spatial_sampling()` - ¡NUEVO!

**Muestra neuronas distribuidas uniformemente en el ring topology**

```python
from skneuromsi.neural._echeveste_visualization import plot_spatial_sampling

# Usar la función
fig, axes = plot_spatial_sampling(result, n_samples=10)
fig.savefig('spatial.png', dpi=150, bbox_inches='tight')
```

**Qué genera:**
- 4 paneles (2×2): u_e, r_e, u_i, r_i
- En cada panel: n_samples neuronas espaciadas uniformemente
- Por defecto (orientation_centered=True): neuronas en -90°, -70°, -50°, ..., 70°
- Con orientation_centered=False: neuronas en 0°, 20°, 40°, ..., 160°

**Parámetros:**
- `result`: Resultado de `ssn.run()`
- `n_samples`: Número de neuronas a muestrear (default: 10)
- `orientation_centered`: `True` para -90° a 90° (default: `True`), `False` para 0° a 180°
- `figsize`: Tamaño de figura (default: `(16, 10)`)

**Retorna:**
- `fig`: Objeto Figure
- `axes`: Array 2×2 de axes

---

### 3. Funciones Existentes

Estas ya estaban en el módulo:

```python
from skneuromsi.neural._echeveste_visualization import (
    plot_membrane_potentials,   # Gráfico tradicional de u(t)
    plot_firing_rates,          # Gráfico tradicional de r(t)
    plot_population_activity,   # Media poblacional
    plot_mean_firing_rates,     # Media temporal
    plot_autocorrelation,       # Autocorrelación temporal
    plot_power_spectrum,        # Espectro de potencias (FFT)
    plot_cross_correlation,     # Correlación cruzada E-I
    plot_cv_over_time,          # Coeficiente de variación
    plot_fano_factor,           # Factor de Fano
    plot_neural_dynamics_summary,  # Resumen multi-panel
)
```

---

## 🚀 Ejemplo Completo de Uso

```python
import sys
sys.path.insert(0, "/path/to/scikit-neuromsi")

from skneuromsi.neural import Echeveste2020
from skneuromsi.neural._echeveste_visualization import (
    plot_heatmap_activity,
    plot_spatial_sampling,
    plot_membrane_potentials,
)

# Crear modelo y ejecutar
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
ssn.load_parameters()

result = ssn.run(
    stimulus_contrast=0.5,
    simulation_time=200.0,
)

# 1. Heatmap de TODAS las neuronas
fig1, _ = plot_heatmap_activity(result, population='excitatory')
fig1.savefig('heatmap_E.png', dpi=150, bbox_inches='tight')

# 2. Muestreo espacial (10 neuronas distribuidas)
fig2, _ = plot_spatial_sampling(result, n_samples=10)
fig2.savefig('spatial_sampling.png', dpi=150, bbox_inches='tight')

# 3. Gráfico tradicional (primeras 5 neuronas)
fig3, _ = plot_membrane_potentials(result, neuron_indices=[0, 1, 2, 3, 4])
fig3.savefig('traditional.png', dpi=150, bbox_inches='tight')
```

---

## 📂 Archivos en pycloude/

### Scripts que usan el módulo

**`ejemplo_uso_visualizaciones.py`** ✅ RECOMENDADO
- Ejemplo simple mostrando cómo usar las funciones del módulo
- Importa desde `skneuromsi.neural._echeveste_visualization`
- Genera 3 tipos de gráficos

**`visualizaciones_completas_refactorizado.py`** ✅ RECOMENDADO
- Versión actualizada que usa funciones del módulo
- Incluye comparación de contrastes
- Genera 4 tipos de visualizaciones

### Scripts educativos

**`tutorial_graficos_manual.py`**
- Tutorial paso a paso
- Explica cómo hacer cada gráfico MANUALMENTE
- Para aprender matplotlib desde cero

**`GUIA_VISUALIZACIONES.md`**
- Guía completa con documentación
- Mapas de colores, buenas prácticas, tips

### Scripts legacy (opcionales)

**`visualizaciones_completas.py`** (versión anterior)
- Define funciones localmente (no importa del módulo)
- Mantener por compatibilidad pero usar la versión refactorizada

---

## ❓ Preguntas Frecuentes

### ¿Por qué solo 4-5 neuronas y no las 50?

**Antes:** Se graficaban solo las primeras 5 neuronas con `plot_membrane_potentials()`

**Ahora:** Usa `plot_heatmap_activity()` para ver **TODAS** las 50 neuronas simultáneamente!

### ¿Por qué las primeras neuronas tienen actividad similar?

**Ring topology:** Las neuronas están organizadas por orientación preferida
- E0: 0.0°, E1: 3.6°, E2: 7.2°, E3: 10.8°, E4: 14.4°
- Solo 3.6° de separación → responden de forma parecida

**Solución 1:** Usa `plot_spatial_sampling()` para ver neuronas distribuidas

**Solución 2:** Usa `plot_heatmap_activity()` para ver la estructura completa

### ¿Cómo funcionan las orientaciones centradas (-90° a 90°)?

**Convenciones de visualización:**
- `orientation_centered=False`: Eje Y muestra Neuron IDs (0 a 49)
- `orientation_centered=True`: Eje Y muestra orientaciones (-90° a 90°)

**Mapeo fundamental:**
- Ring topology: 50 neuronas cubren 0° a 176.4° (con endpoint=False)
- Al centrar: restamos 90° → rango de -90° a 86.4°
- Ticks explícitos: [-90°, -50°, 0°, 50°, 90°] para claridad visual

**Ejemplos:**
- Neurona 0: 0° → (centrada) -90°
- Neurona 25: 90° → (centrada) 0°
- Neurona 49: 176.4° → (centrada) 86.4°

Ver `EXPLICACION_MAPEO_ORIENTACIONES.md` para detalles completos.

### ¿Cómo hacer un heatmap manualmente?

Ver el tutorial `tutorial_graficos_manual.py` - Paso 4

**Concepto clave:**
```python
# ¡IMPORTANTE: Transponer la matriz!
im = ax.imshow(
    r_e.T,  # .T transpone (tiempo, neuronas) → (neuronas, tiempo)
    aspect='auto',
    cmap='viridis',
    extent=[t_min, t_max, 0, n_neurons],
    origin='lower'
)
plt.colorbar(im, label='Firing rate (Hz)')
```

---

## 📊 Comparación: Antes vs Ahora

### Antes (scripts locales)

```python
# Tenías que copiar funciones o ejecutar scripts completos
# No podías importar y usar directamente
```

### Ahora (módulo oficial)

```python
# Simplemente importa y usa
from skneuromsi.neural._echeveste_visualization import plot_heatmap_activity

fig, axes = plot_heatmap_activity(result)
# ¡Listo!
```

**Ventajas:**
- ✅ Funciones disponibles en cualquier script
- ✅ Mantenidas y testeadas
- ✅ Documentadas con docstrings
- ✅ Consistentes con el resto del módulo
- ✅ Pasan flake8

---

## 🎨 Tips de Visualización

### Mapas de colores

**Para firing rates (positivos):**
- `viridis` (excitatorias)
- `plasma` (inhibitorias)

**Para potenciales (con cero central):**
- `RdBu_r` (Rojo-Blanco-Azul invertido)

### Personalizar heatmaps

```python
fig, axes = plot_heatmap_activity(result)

# Personalizar título del panel 1
axes[0].set_title('Mi Título Custom')

# Cambiar límites de colorbar
# (Se hace antes de generar el heatmap, modificando el código)
```

### Guardar figuras

```python
# Alta resolución para papers
fig.savefig('figure.png', dpi=300, bbox_inches='tight')

# Resolución normal para presentaciones
fig.savefig('figure.png', dpi=150, bbox_inches='tight')

# PDF vectorial
fig.savefig('figure.pdf', bbox_inches='tight')
```

---

## ✅ Resumen

**Ahora tienes:**
1. ✅ Funciones de heatmap en el módulo oficial
2. ✅ Funciones de muestreo espacial en el módulo
3. ✅ Scripts de ejemplo mostrando cómo usarlas
4. ✅ Tutorial completo para aprender matplotlib
5. ✅ Guía de visualizaciones con buenas prácticas

**Para empezar:**
```bash
cd /home/molina/FAMAF/5to-Famaf/TESIS/pycloude
python3 ejemplo_uso_visualizaciones.py
```

**Para aprender:**
```bash
python3 tutorial_graficos_manual.py
less GUIA_VISUALIZACIONES.md
```

**Para gráficos avanzados:**
```bash
python3 visualizaciones_completas_refactorizado.py
```

¡Listo para crear visualizaciones profesionales! 🎉
