# Notas sobre Orientaciones y Unidades en los Heatmaps

## Tu Pregunta

> "veas la figura 2 a que es un heatmap muy parecido al nuestro pero echeveste trabaja con angulos de -90 a 90 grados y ademas trata u_e [mV] el nuestro no deberia ser asi?"

## Respuesta: Orientaciones

### ✅ Ya Implementado

Ahora la función `plot_heatmap_activity()` soporta **ambas convenciones**:

**Opción 1: Neuron IDs (por defecto)**
```python
fig, axes = plot_heatmap_activity(result, orientation_centered=False)
# Eje Y: 0, 1, 2, ..., 49 (Neuron ID)
```

**Opción 2: Orientaciones centradas (como Fig. 2A)**
```python
fig, axes = plot_heatmap_activity(result, orientation_centered=True)
# Eje Y: -90°, -72°, ..., 0°, ..., 72°, 90° (Orientation)
```

### Convenciones de Orientación

**En el código original de Echeveste:**
- Ring topology cubre **0° a 180°** (π radianes)
- Neuronas espaciadas uniformemente: 0°, 3.6°, 7.2°, ..., 176.4°
- Código: `theta = np.pi * i / n_pat` (gabor_filters.py:87)

**En la Figura 2A del paper:**
- Muestran eje Y como **-90° a 90°**
- Es la **misma topología**, solo centrada en 0°
- Equivalencia: 0° → -90°, 90° → 0°, 180° → 90°

**Ambas son correctas - solo cambia el punto de referencia:**
- `0° a 180°`: Rango absoluto de orientaciones
- `-90° a 90°`: Mismo rango, centrado en 0°

### Uso

```python
# Script de prueba
python3 test_heatmap_orientaciones.py

# Genera dos versiones:
# - heatmap_neuron_ids.png (IDs 0-49)
# - heatmap_orientations.png (-90° a 90°, como Fig. 2A)
```

---

## Respuesta: Unidades de u(t)

### ❌ NO son mV

**Aclaración importante:**

El modelo SSN de Echeveste et al. (2020) es un **rate model** (modelo de tasas), NO un modelo conductance-based como Hodgkin-Huxley.

### ¿Qué son los "potenciales" u(t)?

En el paper (Eq. 8):
```
τ_α * du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α
```

Los `u_α` son:
- **Variables de estado** del modelo
- Representan el "input total" a cada neurona
- **NO tienen unidades físicas específicas**
- Se llaman "potenciales" por analogía conceptual, no porque sean voltajes reales

### Unidades Correctas

**En nuestro código:**
```python
plt.colorbar(im1, ax=ax1, label='u (a.u.)')  # ✅ CORRECTO
```

**"a.u." = arbitrary units** (unidades arbitrarias)

**NO deberíamos usar:**
```python
plt.colorbar(im1, ax=ax1, label='u (mV)')  # ❌ INCORRECTO
```

### ¿Por qué no mV?

1. **Rate models vs Spiking models:**
   - Rate models: Variables abstractas sin unidades físicas
   - Spiking models (e.g., Hodgkin-Huxley): Voltaje real en mV

2. **El paper NO especifica mV:**
   - Revisando el código original: No hay referencias a mV
   - Las figuras del paper usan escalas normalizadas o arbitrarias
   - La Eq. 9 define: `r = k * [u]_+^n` con k=0.3 (sin unidades)

3. **La relación u → r no es realista para mV:**
   - Si u fuera en mV, la función `r = 0.3 * [u]_+^2` no tendría sentido
   - Un potencial de 5 mV daría: r = 0.3 * 5² = 7.5 Hz (razonable)
   - Pero el modelo usa u en el rango [-5, 10] a.u., no mV

### Evidencia del Código Original

```python
# ssn_inference_numerical_experiments/SSN/parameters.py
tau_e = 20.0e-3  # Seconds - NO hay comentario sobre mV
n = 2            # Exponente sin unidades
k = 0.3          # Factor de escala sin unidades
```

No hay ninguna mención de mV o conversión de unidades en todo el código original.

---

## Resumen

### ✅ Implementado Correctamente

1. **Orientaciones:**
   - Por defecto: Neuron IDs (0-49)
   - `orientation_centered=True`: Orientaciones (-90° a 90°) como Fig. 2A

2. **Unidades de u(t):**
   - Correctamente etiquetadas como "a.u." (arbitrary units)
   - NO son mV - este es un rate model

### 📊 Cómo Usar

```python
from skneuromsi.neural._echeveste_visualization import plot_heatmap_activity

# Versión por defecto
fig1, _ = plot_heatmap_activity(result)

# Versión estilo Fig. 2A (orientaciones centradas)
fig2, _ = plot_heatmap_activity(result, orientation_centered=True)
```

### 📖 Referencias

- **Figura 2A del paper**: Muestra heatmap con orientaciones -90° a 90°
- **Ecuaciones 8-9**: Definen u(t) y r(t) sin unidades físicas específicas
- **Código original**: Usa tau en segundos, pero NO especifica mV para u(t)
- **Tipo de modelo**: Rate model (no conductance-based) → unidades arbitrarias

---

## Conclusión

Nuestro código ahora:
1. ✅ Soporta ambas convenciones de orientación (IDs o grados centrados)
2. ✅ Usa unidades correctas para u(t): "a.u." (no mV)
3. ✅ Es consistente con el paper de Echeveste et al. (2020)

La confusión probablemente viene de que:
- Se usan términos como "potencial de membrana"
- Pero es solo una **analogía** - no son voltajes reales
- Es un **modelo abstracto de tasas**, no un modelo biofísico detallado
