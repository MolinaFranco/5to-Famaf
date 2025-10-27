# Guía de Visualizaciones para SSN Echeveste2020

## 📊 Scripts Disponibles

### 1. `visualizaciones_completas.py`
Script profesional con visualizaciones avanzadas listas para usar.

**Tipos de gráficos que genera:**

#### A. Heatmaps de Actividad Completa
- **Qué muestra**: Actividad de TODAS las 50 neuronas vs tiempo
- **Formato**: Matriz de colores (neuronas en Y, tiempo en X)
- **Incluye**:
  - Heatmap de potenciales de membrana u(t)
  - Heatmap de firing rates r(t)
  - Gráfico de barras: promedio temporal por neurona
  - Serie temporal: media poblacional ± std

**Archivos generados:**
- `heatmap_excitatory.png`: Población excitatoria completa
- `heatmap_inhibitory.png`: Población inhibitoria completa

#### B. Muestreo Espacial
- **Qué muestra**: Neuronas espaciadas uniformemente a lo largo del ring
- **Por defecto**: 10 neuronas distribuidas en orientaciones 0°, 20°, 40°, ..., 160°
- **Propósito**: Ver diversidad de respuestas según orientación preferida

**Archivo generado:**
- `spatial_sampling.png`: 4 paneles (u_e, r_e, u_i, r_i)

#### C. Comparación de Contrastes
- **Qué muestra**: Cómo cambia la actividad con diferentes niveles de estímulo
- **Formato**: Heatmaps lado a lado + series temporales
- **Contrasts por defecto**: 0.1 (bajo), 0.3 (medio), 0.5 (alto)

**Archivo generado:**
- `contrast_comparison.png`: Grid de 3×3 (3 contrastes × 3 tipos de visualización)

### 2. `tutorial_graficos_manual.py`
Tutorial interactivo paso a paso que EXPLICA cómo hacer cada gráfico.

**Qué aprenderás:**

#### Paso 1: Extraer Datos
```python
# Cómo obtener datos de la simulación
df = result.get_modes()
n_times = len(result.times_)
u_e = df['excitatory_potential'].values.reshape(n_times, -1)
r_e = df['excitatory_firing_rate'].values.reshape(n_times, -1)
```

**Conceptos clave:**
- `.get_modes()`: Devuelve DataFrame con todas las variables
- `.values`: Convierte pandas → numpy
- `.reshape(n_times, -1)`: Organiza como matriz (tiempo × neuronas)

#### Paso 2: Gráfico Básico de Línea
```python
# Una neurona vs tiempo
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(result.times_, r_e[:, 0], label='E0')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Firing rate (Hz)')
ax.legend()
ax.grid(True, alpha=0.3)
```

**Archivo generado:** `tutorial_paso2_basico.png`

#### Paso 3: Múltiples Neuronas
```python
# Loop para graficar varias neuronas
for neuron_id in [0, 10, 20, 30, 40]:
    ax.plot(result.times_, r_e[:, neuron_id],
            label=f'E{neuron_id}', alpha=0.7)
```

**Archivo generado:** `tutorial_paso3_multiples.png`

#### Paso 4: Heatmap (¡IMPORTANTE!)
```python
# Gráfico de calor: TODAS las neuronas × tiempo
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# ¡TRANSPONER! imshow espera (filas=Y, columnas=X)
im = ax.imshow(
    r_e.T,  # ← IMPORTANTE: .T transpone la matriz
    aspect='auto',
    cmap='viridis',
    extent=[result.times_[0], result.times_[-1], 0, r_e.shape[1]],
    origin='lower'
)

# Colorbar con etiqueta
plt.colorbar(im, ax=ax, label='Firing rate (Hz)')
```

**Conceptos clave:**
- `r_e.T`: Transponer matriz (tiempo, neuronas) → (neuronas, tiempo)
- `extent`: Mapear píxeles a valores reales [x_min, x_max, y_min, y_max]
- `cmap`: Mapa de colores (viridis, plasma, RdBu_r, etc.)
- `origin='lower'`: Y=0 en la parte inferior

**Archivo generado:** `tutorial_paso4_heatmap.png`

#### Paso 5: Subplots (Múltiples Paneles)
```python
# Grilla de 2×2 = 4 paneles
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Acceder a cada panel
ax1 = axes[0, 0]  # Superior izquierdo
ax2 = axes[0, 1]  # Superior derecho
ax3 = axes[1, 0]  # Inferior izquierdo
ax4 = axes[1, 1]  # Inferior derecho
```

**Archivo generado:** `tutorial_paso5_subplots.png`

#### Paso 6: Estadísticas Poblacionales
```python
# Media y desviación estándar
r_mean = np.mean(r_e, axis=1)  # Promedio sobre neuronas
r_std = np.std(r_e, axis=1)

# Graficar media ± std
ax.plot(result.times_, r_mean, lw=2, label='Media')
ax.fill_between(result.times_,
                r_mean - r_std,
                r_mean + r_std,
                alpha=0.3, label='± 1 std')
```

**Conceptos clave:**
- `axis=1`: Promedio sobre columnas (neuronas)
- `axis=0`: Promedio sobre filas (tiempo)
- `fill_between()`: Rellena área entre dos curvas

**Archivo generado:** `tutorial_paso6_estadisticas.png`

---

## 🎨 Mapas de Colores Recomendados

### Para Firing Rates (valores positivos)
- **viridis**: Amarillo-Verde-Azul (perceptualmente uniforme)
- **plasma**: Amarillo-Naranja-Morado
- **inferno**: Negro-Rojo-Amarillo
- **magma**: Negro-Morado-Blanco

### Para Potenciales de Membrana (cero central)
- **RdBu_r**: Rojo-Blanco-Azul (r = reversed)
- **coolwarm**: Azul-Blanco-Rojo
- **seismic**: Azul-Blanco-Rojo

### Para Publicaciones B&N
- **gray**: Escala de grises
- **binary**: Blanco y negro

---

## ✅ Checklist para Gráficos Científicos

**Siempre incluir:**
- ✅ Etiquetas de ejes **CON UNIDADES** (`Time (ms)`, `Rate (Hz)`)
- ✅ Título descriptivo
- ✅ Leyenda cuando hay múltiples líneas
- ✅ Grid semitransparente (`alpha=0.3`) para lectura de valores
- ✅ Colorbar con etiqueta en heatmaps
- ✅ Tamaño apropiado (`figsize=(12, 6)` para figuras individuales)
- ✅ DPI suficiente (`dpi=150` para presentaciones, `dpi=300` para papers)

**Buenas prácticas:**
- ✅ Usar `bbox_inches='tight'` al guardar para recortar espacios
- ✅ Colores contrastantes y accesibles (evitar solo rojo/verde)
- ✅ Transparencia (`alpha=0.7`) cuando hay líneas superpuestas
- ✅ `tight_layout()` para subplots sin solapamientos

---

## 🔍 Respuesta a tu Pregunta Original

### ¿Por qué solo 4-5 neuronas y no las 50?

**Razón práctica:** Graficar 50 líneas en un plot sería ilegible.

**Solución 1:** Usar **heatmaps** para visualizar TODAS las neuronas simultáneamente
```python
# Esto muestra las 50 neuronas como filas de colores
im = ax.imshow(r_e.T, aspect='auto', cmap='viridis')
```

**Solución 2:** Graficar **muestras espaciadas**
```python
# En lugar de [0, 1, 2, 3, 4] (neuronas cercanas)
# Usar [0, 10, 20, 30, 40] (neuronas distribuidas)
neuron_indices = [0, 10, 20, 30, 40]
```

### ¿Por qué las primeras neuronas tienen actividad similar?

**Ring topology:** Las neuronas están organizadas en un ring con orientaciones preferidas:
- Neurona E0: 0.0°
- Neurona E1: 3.6°
- Neurona E2: 7.2°
- ...
- Neurona E49: 176.4°

**Separación:** Solo 3.6° entre neuronas consecutivas → **responden de forma parecida**

**Evidencia de diversidad:**
```
Min: 1.44 Hz, Max: 6.58 Hz  → Rango de 5.1 Hz (¡SÍ hay diversidad!)
Std: 1.37 Hz                → Variabilidad significativa
```

**Solución:** Graficar neuronas espaciadas o usar heatmaps para ver la estructura completa.

---

## 🚀 Cómo Usar los Scripts

### Ejecutar visualizaciones completas
```bash
cd /home/molina/FAMAF/5to-Famaf/TESIS/pycloude
python3 visualizaciones_completas.py
```

**Output:**
- `outputs/heatmap_excitatory.png`
- `outputs/heatmap_inhibitory.png`
- `outputs/spatial_sampling.png`
- `outputs/contrast_comparison.png`

### Ejecutar tutorial interactivo
```bash
python3 tutorial_graficos_manual.py
```

**Output:**
- Imprime explicaciones paso a paso en la terminal
- Genera ejemplos: `tutorial_paso2_basico.png`, etc.

### Personalizar visualizaciones

**Cambiar neuronas a graficar:**
```python
plot_spatial_sampling(result, n_samples=15)  # 15 neuronas en lugar de 10
```

**Cambiar contrastes:**
```python
plot_contrast_comparison(ssn, contrasts=[0.05, 0.2, 0.5, 0.8])
```

**Cambiar tiempo de simulación:**
```python
result = ssn.run(simulation_time=500.0)  # 500 ms en lugar de 200 ms
```

---

## 📚 Archivos Relacionados

### Scripts principales
- `visualizaciones_completas.py`: Visualizaciones profesionales
- `tutorial_graficos_manual.py`: Tutorial paso a paso
- `investigar_neuronas.py`: Análisis de diversidad neuronal
- `test_gamma_oscillations.py`: Verificación de oscilaciones gamma

### Otros
- `graficos_multiples_estimulos.py`: Comparación de múltiples patrones
- `TIME_CONSTANTS_FIX_SUMMARY.md`: Documentación del fix de unidades

---

## 💡 Tips y Trucos

### Indexación de numpy
```python
r_e[:, 0]      # Neurona 0, todos los tiempos
r_e[10, :]     # Tiempo 10, todas las neuronas
r_e[:, 0:5]    # Primeras 5 neuronas, todos los tiempos
r_e[0:100, :]  # Primeros 100 tiempos, todas las neuronas
```

### Operaciones útiles
```python
np.mean(r_e, axis=0)  # Promedio temporal de cada neurona
np.mean(r_e, axis=1)  # Promedio poblacional en cada tiempo
np.max(r_e, axis=0)   # Máximo temporal de cada neurona
r_e.T                 # Transponer matriz
```

### Colores personalizados
```python
colors = ['blue', 'green', 'red', 'purple', 'orange']
for i, color in zip([0, 10, 20, 30, 40], colors):
    ax.plot(times, r_e[:, i], color=color, label=f'E{i}')
```

---

## 📖 Referencias

- **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/
- **Colormaps**: https://matplotlib.org/stable/tutorials/colors/colormaps.html
- **Seaborn**: https://seaborn.pydata.org/ (gráficos estadísticos avanzados)
- **Scientific Python**: https://scipy-lectures.org/intro/matplotlib/

---

¡Ahora tienes todo lo necesario para crear visualizaciones profesionales de tus simulaciones SSN! 🎉
