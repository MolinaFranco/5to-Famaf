# Mapeo de Neuronas a Orientaciones - Explicación Completa

## 🎯 Resumen Ejecutivo

El modelo Echeveste2020 organiza las neuronas en un **ring topology** que cubre orientaciones de **0° a 180°**. Debido a que usa `endpoint=False`, la última neurona está en **176.4°** (no exactamente 180°).

**Mapeo fundamental:**
- Neurona 0 → 0°
- Neurona 25 → 90°
- Neurona 49 → 176.4° (último punto antes de completar el círculo)

---

## 📐 Configuración del Ring Topology

### Parámetros del modelo

```python
N_E = 50  # Número de neuronas excitatorias
position_range = (0, 180)  # Rango de orientaciones
position_res = 3.6°  # Resolución angular (180° / 50)
```

### Cálculo del mapeo interno

```python
orientations = np.linspace(0, 180, 50, endpoint=False)
# Resultado: [0.0, 3.6, 7.2, ..., 172.8, 176.4]
```

**¿Por qué endpoint=False?**
- El ring topology es **periódico**: 0° y 180° representan la misma orientación física
- endpoint=False evita duplicar el punto de inicio/fin
- Las 50 neuronas cubren uniformemente el espacio sin solapamiento

---

## 🔄 Dos Convenciones de Visualización

### Convención 1: Neuron IDs (por defecto)

**Rango:** 0 a 49

```python
plot_heatmap_activity(result, orientation_centered=False)
```

**Eje Y:** Neuron ID
- 0 → Neurona 0
- 25 → Neurona 25
- 49 → Neurona 49

**Ventajas:**
- Directo: el índice coincide con el ID de la neurona
- Útil para debuggear o indexar arrays

---

### Convención 2: Orientaciones Centradas (Fig. 2A del paper)

**Rango:** -90° a 90°

```python
plot_heatmap_activity(result, orientation_centered=True)
```

**Transformación:** `orientation_centered = orientation - 90°`

**Eje Y:** Orientation (degrees)
- -90° → Neurona 0 (originalmente en 0°)
- 0° → Neurona 25 (originalmente en 90°)
- 90° → Aproximadamente neurona 50 (originalmente en ~180°)

**Nota importante:**
- El extent se configura como [-90, 90] para visualización clara
- Pero matemáticamente la neurona 49 está en 86.4° (no exactamente 90°)
- Los ticks muestran explícitamente: **-90°, -50°, 0°, 50°, 90°**

**Ventajas:**
- Simétrico respecto a 0° (más intuitivo)
- Consistente con la notación del paper (Echeveste et al. 2020, Fig. 2A)
- Facilita comparación con experimentos donde 0° es orientación "base"

---

## 📊 Mapeo Detallado por Neurona

### Neuronas clave (ejemplos)

| Neurona | Orientación (0-180) | Orientación Centrada (-90 a 90) |
|---------|---------------------|----------------------------------|
| **0**   | 0.0°                | **-90.0°** (límite inferior)     |
| 5       | 18.0°               | -72.0°                           |
| 12      | 43.2°               | -46.8°                           |
| **25**  | **90.0°**           | **0.0°** (centro)                |
| 37      | 133.2°              | 43.2°                            |
| 45      | 162.0°              | 72.0°                            |
| **49**  | **176.4°**          | **86.4°** (casi el límite sup.)  |

### Distribución uniforme

```
Espaciado: 3.6° entre neuronas consecutivas

E0   E1   E2   E3   ...   E47  E48  E49
↓    ↓    ↓    ↓    ...   ↓    ↓    ↓
0°   3.6° 7.2° 10.8° ...  169.2° 172.8° 176.4°
```

**Importante:** El espacio entre E49 y E0 NO está representado (sería 3.6° también, completando el círculo).

---

## 🧪 Verificación de Consistencia

### En `plot_heatmap_activity()`

**Con `orientation_centered=False`:**
```python
extent = [t_min, t_max, 0, 49]  # Neuron IDs
yticks = [0, 10, 20, 30, 40, 49]  # (automáticos)
```

**Con `orientation_centered=True`:**
```python
extent = [t_min, t_max, -90, 90]  # Orientaciones centradas
yticks = [-90, -50, 0, 50, 90]  # Explícitos
yticklabels = ['-90°', '-50°', '0°', '50°', '90°']
```

### En `plot_spatial_sampling()`

**Fórmula de mapeo:**
```python
orientation = idx * 180.0 / n_e
# Con n_e = 50:
# idx=0  → 0.0°
# idx=10 → 36.0°
# idx=25 → 90.0°
# idx=49 → 176.4°
```

**Consistencia verificada:**
- Esta fórmula produce **exactamente** los mismos valores que `linspace(0, 180, 50, endpoint=False)`
- No hay discrepancia entre funciones ✓

---

## ❓ Preguntas Frecuentes

### ¿Por qué la neurona 49 está en 86.4° y no en 90° (centrado)?

**Matemáticamente correcto:**
- Neurona 49 en sistema original: 176.4°
- Al centrar: 176.4° - 90° = **86.4°**
- Falta 3.6° para llegar a 180° (que al centrar sería 90°)

**Visualmente aproximado:**
- El extent del heatmap se configura como [-90, 90] para que el gráfico sea simétrico
- Esto significa que la neurona 49 aparece MUY cerca del borde superior (90°)
- La diferencia de 3.6° es visualmente insignificante en un heatmap de 50 neuronas

### ¿Por qué no usar endpoint=True?

Si usáramos `endpoint=True`:
```python
orientations = np.linspace(0, 180, 50, endpoint=True)
# [0.0, 3.673, 7.347, ..., 176.327, 180.0]
```

**Problema:** 0° y 180° representan **la misma orientación física** en un ring topology. Tendríamos:
- Neurona 0 en 0°
- Neurona 49 en 180° ≡ 0° (duplicado!)
- Solo 49 orientaciones únicas en vez de 50

**Solución:** `endpoint=False` garantiza 50 orientaciones únicas uniformemente espaciadas.

### ¿Los ticks [-90°, -50°, 0°, 50°, 90°] son correctos?

**Sí, son informativos y claros:**
- Muestran el límite inferior: **-90°** (neurona 0)
- Muestran el centro: **0°** (neurona 25)
- Muestran el límite superior **aproximado**: **90°** (neurona 49 está en 86.4°)

La pequeña diferencia (3.6°) es:
- Matemáticamente inevitable con 50 neuronas y endpoint=False
- Visualmente imperceptible en el heatmap
- Documentada claramente en el código y comentarios

---

## 🎨 Interpretación Visual

### En heatmaps con orientation_centered=True:

```
     90° ├─────────────────────────────┤ ← Neurona ~49 (en realidad 86.4°)
     50° ├─────────────────────────────┤ ← Neurona ~39
      0° ├─────────────────────────────┤ ← Neurona 25 (centro exacto)
    -50° ├─────────────────────────────┤ ← Neurona ~11
    -90° ├─────────────────────────────┤ ← Neurona 0 (límite inferior)
         └─────────────────────────────┘
       0 ms                       200 ms
```

### En spatial sampling (n_samples=10):

Muestra neuronas distribuidas uniformemente:
```
E0 (θ=0.0°)   ─┐
E5 (θ=18.0°)   │
E10 (θ=36.0°)  │ 10 neuronas
E15 (θ=54.0°)  │ espaciadas
E20 (θ=72.0°)  │ uniformemente
E25 (θ=90.0°)  │ de 0° a 162°
E30 (θ=108.0°) │
E35 (θ=126.0°) │
E40 (θ=144.0°) │
E45 (θ=162.0°) ─┘
```

**Nota:** plot_spatial_sampling NO usa orientaciones centradas en las leyendas.
Muestra los ángulos originales (0° a 162°).

---

## ✅ Checklist de Verificación

Para confirmar que el mapeo es consistente, verifica:

- [ ] `check_orientation_mapping.py`: Muestra el mapeo neurona por neurona
- [ ] `verificar_consistencia_mapeo.py`: Compara fórmulas entre funciones
- [ ] `outputs/verificacion_heatmap.png`: Heatmap con ticks explícitos (-90°, 0°, 90°)
- [ ] `outputs/verificacion_spatial.png`: Spatial sampling con orientaciones en leyendas

**Resultado esperado:** Todas las funciones usan `idx * 180.0 / 50` o equivalente.

---

## 📚 Referencias

**Código relevante:**
- `skneuromsi/neural/echeveste2020.py`: Líneas donde se define `_position_range` y `_position_res`
- `skneuromsi/neural/_echeveste_visualization.py`:
  - `plot_heatmap_activity()`: Líneas 1010-1165 (extent y yticks)
  - `plot_spatial_sampling()`: Líneas 1168-1290 (fórmula de orientación)

**Paper:**
- Echeveste et al. (2020): Figure 2A usa convención centrada (-90° a 90°)
- Ver `echepaper.pdf` en el directorio raíz

---

## 🎓 Conclusión

El mapeo neurona→orientación es **matemáticamente consistente** entre todas las funciones de visualización:

1. **Ring topology:** 50 neuronas cubren 0° a 176.4° (con endpoint=False)
2. **Heatmap centrado:** Mapea a -90° a 86.4°, visualizado como -90° a 90° (aproximación)
3. **Ticks explícitos:** Muestran claramente -90°, 0°, 90° para orientación del usuario
4. **Spatial sampling:** Usa la misma fórmula `idx * 180.0 / n_e`

**No hay inconsistencia** - solo dos convenciones de visualización (IDs vs ángulos centrados) aplicadas consistentemente.
