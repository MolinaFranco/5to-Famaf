# Cambio a Convenciones Coherentes de Orientación

## 🎯 Cambio Implementado

**Fecha:** 2025-01-24

**Solicitud del usuario:**
> "necesito que seamos coherentes con las orientaciones de las neuronas, quiero que siempre usemos de -90 a 90 en todos los graficos, asi es menos confuso"

**Solución:**
- Cambiado el default de `orientation_centered` de `False` a `True` en todas las funciones de visualización
- Ahora **TODOS** los gráficos muestran orientaciones de **-90° a 90°** por defecto
- Mayor coherencia y consistencia con el paper (Echeveste et al. 2020, Fig. 2A)

---

## ✅ Funciones Modificadas

### 1. `plot_heatmap_activity()`

**Antes:**
```python
def plot_heatmap_activity(result, population='excitatory', figsize=(16, 10),
                          orientation_centered=False):  # Default: IDs
```

**Ahora:**
```python
def plot_heatmap_activity(result, population='excitatory', figsize=(16, 10),
                          orientation_centered=True):  # Default: -90° a 90°
```

**Comportamiento por defecto:**
- Eje Y: Orientaciones de -90° a 90°
- Ticks explícitos: [-90°, -50°, 0°, 50°, 90°]
- Consistente con Figure 2A del paper

---

### 2. `plot_spatial_sampling()`

**Antes:**
```python
def plot_spatial_sampling(result, n_samples=10, figsize=(16, 10)):
    # No tenía parámetro orientation_centered
    # Leyendas mostraban 0° a 162° (interno)
```

**Ahora:**
```python
def plot_spatial_sampling(result, n_samples=10, figsize=(16, 10),
                          orientation_centered=True):  # Nuevo parámetro
```

**Comportamiento por defecto:**
- Leyendas muestran orientaciones centradas: -90° a 72°
- Ejemplo: `E0 (θ=-90.0°)`, `E25 (θ=0.0°)`, `E45 (θ=72.0°)`
- Consistente con heatmaps y paper

---

## 📊 Comparación Visual

### Antes (orientation_centered=False por defecto)

**Heatmap:**
```
Eje Y: 0, 10, 20, 30, 40, 49  (Neuron IDs)
```

**Spatial sampling:**
```
Leyendas: E0 (θ=0.0°), E10 (θ=36.0°), ..., E45 (θ=162.0°)
```

❌ **Problema:** Confuso - cada gráfico usa una convención diferente

---

### Ahora (orientation_centered=True por defecto)

**Heatmap:**
```
Eje Y: -90°, -50°, 0°, 50°, 90°  (Orientaciones centradas)
```

**Spatial sampling:**
```
Leyendas: E0 (θ=-90.0°), E10 (θ=-54.0°), ..., E45 (θ=72.0°)
```

✅ **Ventaja:** Coherente - todos usan la misma convención

---

## 🔄 Convención Alternativa

Si necesitas la convención interna (Neuron IDs / 0° a 180°), simplemente especifica:

```python
# Heatmap con Neuron IDs
fig, axes = plot_heatmap_activity(result, orientation_centered=False)

# Spatial sampling con orientaciones 0° a 180°
fig, axes = plot_spatial_sampling(result, n_samples=10,
                                  orientation_centered=False)
```

---

## 📚 Documentación Actualizada

### Docstrings

**En ambas funciones:**

```python
Parameters
----------
orientation_centered : bool, optional
    If True, show orientations as -90° to 90° (like Fig. 2A) (default)
    If False, show neuron IDs 0-49 (plot_heatmap_activity)
              or orientations 0° to 180° (plot_spatial_sampling)
```

### Ejemplos actualizados

```python
>>> ssn = Echeveste2020(N_E=50, N_I=50)
>>> result = ssn.run(stimulus_contrast=0.5, simulation_time=200.0)
>>> # Default: Centered orientations -90° to 90° (like Fig. 2A)
>>> fig, axes = plot_heatmap_activity(result, population='excitatory')
>>> # Alternative: Show neuron IDs 0-49
>>> fig, axes = plot_heatmap_activity(result,
...                                   orientation_centered=False)
```

---

## ✅ Verificación de Calidad

### flake8

```bash
flake8 skneuromsi/neural/_echeveste_visualization.py
# ✓ 0 errores
```

### Tests

```bash
python3 test_convenciones_coherentes.py
# ✓ Todas las funciones usan -90° a 90° por defecto
# ✓ Convención alternativa funciona correctamente
# ✓ 4 gráficos generados exitosamente
```

### Archivos de prueba generados

```
pycloude/outputs/
├── test_heatmap_default.png         # Default: -90° a 90° ✓
├── test_spatial_default.png         # Default: -90° a 90° ✓
├── test_heatmap_internal.png        # Alt: Neuron IDs ✓
└── test_spatial_internal.png        # Alt: 0° a 180° ✓
```

---

## 🎓 Justificación Técnica

### ¿Por qué -90° a 90° como default?

1. **Coherencia:** Todas las visualizaciones usan la misma convención
2. **Paper:** Consistente con Echeveste et al. 2020, Figure 2A
3. **Interpretación:** Simétrico respecto a 0° (más intuitivo)
4. **Comparación:** Facilita comparar resultados con experimentos
5. **Menos confusión:** El usuario no necesita recordar qué convención usa cada función

### Mapeo matemático

```
Orientación interna → Orientación centrada
0°                  → -90°
90°                 → 0°
180°                → 90°

Fórmula: orientation_centered = orientation_internal - 90°
```

### Ring topology

```
N_E = 50 neuronas
position_range = (0, 180)
linspace(0, 180, 50, endpoint=False)
→ [0.0°, 3.6°, 7.2°, ..., 172.8°, 176.4°]

Al centrar:
→ [-90.0°, -86.4°, -82.8°, ..., 82.8°, 86.4°]
```

---

## 📝 Archivos Modificados

### Código principal

- [`skneuromsi/neural/_echeveste_visualization.py`](../scikit-neuromsi/skneuromsi/neural/_echeveste_visualization.py)
  - `plot_heatmap_activity()`: línea 992 (default cambiado)
  - `plot_spatial_sampling()`: líneas 1169-1170 (parámetro agregado)
  - Líneas 1239-1296: Lógica para orientaciones centradas en leyendas

### Scripts de prueba

- [`test_convenciones_coherentes.py`](test_convenciones_coherentes.py): Test completo del cambio

### Documentación

- Este archivo: `CAMBIO_CONVENCIONES_COHERENTES.md`
- [`EXPLICACION_MAPEO_ORIENTACIONES.md`](EXPLICACION_MAPEO_ORIENTACIONES.md): Ya documentaba ambas convenciones
- [`RESUMEN_VISUALIZACIONES.md`](RESUMEN_VISUALIZACIONES.md): Actualizar para reflejar nuevo default

---

## 🚀 Impacto en Scripts Existentes

### Scripts que usan defaults (sin especificar orientation_centered)

**Antes del cambio:**
```python
fig, axes = plot_heatmap_activity(result)
# Mostraba Neuron IDs 0-49
```

**Después del cambio:**
```python
fig, axes = plot_heatmap_activity(result)
# Ahora muestra -90° a 90° ✓
```

### Scripts que especifican explícitamente

```python
# Estos NO cambian:
fig, axes = plot_heatmap_activity(result, orientation_centered=True)
fig, axes = plot_heatmap_activity(result, orientation_centered=False)
# Siguen funcionando exactamente igual
```

### Migración recomendada

**Si tu script dependía del comportamiento anterior:**

```python
# Antes:
fig, axes = plot_heatmap_activity(result)  # IDs por defecto

# Ahora, para mantener IDs:
fig, axes = plot_heatmap_activity(result, orientation_centered=False)
```

**Pero se recomienda usar el nuevo default coherente:**

```python
# Recomendado:
fig, axes = plot_heatmap_activity(result)  # -90° a 90° ✓
```

---

## 🎯 Beneficios

### ✅ Para el usuario

1. **Menos confusión:** Una sola convención en todos los gráficos
2. **Más intuitivo:** -90° a 90° es simétrico y natural
3. **Coherencia:** No necesita especificar parámetros extra
4. **Paper-compatible:** Directamente comparable con Fig. 2A

### ✅ Para el código

1. **Consistencia:** Todas las funciones usan el mismo default
2. **Documentado:** Docstrings claros sobre ambas opciones
3. **Retrocompatible:** La convención antigua sigue disponible
4. **Testeado:** Scripts de verificación confirman funcionamiento

---

## 📚 Referencias

### Código relevante

- `skneuromsi/neural/_echeveste_visualization.py`:
  - Líneas 991-1166: `plot_heatmap_activity()`
  - Líneas 1169-1308: `plot_spatial_sampling()`

### Paper

- **Echeveste et al. (2020):** Figure 2A usa -90° a 90°
- **Archivo:** `echepaper.pdf` en directorio raíz

### Documentación relacionada

- `EXPLICACION_MAPEO_ORIENTACIONES.md`: Detalles técnicos
- `RESUMEN_CAMBIOS_ORIENTACIONES.md`: Cambio anterior (ticks explícitos)
- `RESUMEN_VISUALIZACIONES.md`: Overview de todas las funciones

---

## ✅ Checklist de Verificación

- [x] Default cambiado en `plot_heatmap_activity()`
- [x] Parámetro agregado en `plot_spatial_sampling()`
- [x] Lógica implementada para orientaciones centradas en leyendas
- [x] Docstrings actualizados
- [x] Ejemplos actualizados
- [x] flake8 pasa sin errores
- [x] Script de prueba ejecutado exitosamente
- [x] Documentación completa creada

---

**Conclusión:**
Todas las funciones de visualización ahora usan **-90° a 90°** por defecto, como solicitaste. Esto hace el código mucho más coherente y fácil de usar. La convención alternativa (Neuron IDs / 0° a 180°) sigue disponible con `orientation_centered=False`.
