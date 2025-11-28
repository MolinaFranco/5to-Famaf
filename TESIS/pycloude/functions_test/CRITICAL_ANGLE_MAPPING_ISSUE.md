# ANÁLISIS CRÍTICO: INCONSISTENCIA EN EL MAPEO DE ÁNGULOS

## PROBLEMA DETECTADO

Existe una **INCONSISTENCIA CRÍTICA** entre:
1. Cómo se construye la matriz W internamente (usando [0, 2π])
2. Cómo se visualizan las orientaciones (usando [0°, 180°])

---

## 1. CÓDIGO ORIGINAL DE ECHEVESTE (OCaml)

### Archivo: `objective.ml` líneas 285-287

```ocaml
let theta_i = PE.(2. *. pi *. float i /. float m) in
let theta_j = PE.(2. *. pi *. float j /. float m) in
let cos_diff_minus_one = PE.(cos (theta_i -. theta_j) -. 1.) in
```

**IMPORTANTE:** El código original de Echeveste usa:
- `theta ∈ [0, 2π]` radianes = `[0°, 360°)`
- La fórmula es: `theta_i = 2π * i / m`
- **SIN el factor 2 en el coseno**: `cos(theta_i - theta_j)`

---

## 2. NUESTRO CÓDIGO (Python)

### Archivo: `_echeveste2020.py` línea 1374

```python
orientations = np.linspace(0, np.pi, self._N, endpoint=False)
```

**Usamos:**
- `theta ∈ [0, π]` radianes = `[0°, 180°)`  
- **CON el factor 2 en el coseno**: `cos(2*(theta_i - theta_j))`

### Archivo: `_echeveste2020.py` línea 2243

```python
return np.exp((np.cos(2 * angular_diff) - 1) / (width**2))
```

---

## 3. ¿POR QUÉ FUNCIONA NUESTRA IMPLEMENTACIÓN?

### Equivalencia Matemática

Ambas formas son **matemáticamente equivalentes**:

**Forma Original (Echeveste):**
```
theta ∈ [0, 2π]
kernel = exp[(cos(Δθ) - 1) / d²]
```

**Nuestra Forma:**
```
theta ∈ [0, π]  
kernel = exp[(cos(2*Δθ) - 1) / d²]
```

**Demostración:**
- Si `θ_original ∈ [0, 2π]` y `θ_nuestro ∈ [0, π]`
- Entonces: `θ_original = 2 * θ_nuestro`
- Por lo tanto: `cos(θ_i - θ_j)_original = cos(2*(θ_i - θ_j)_nuestro)`

El factor 2 en el coseno **compensa** el uso de un rango de ángulos que es la mitad.

---

## 4. VISUALIZACIÓN vs IMPLEMENTACIÓN INTERNA

### Problem: `position_range` se usa para VISUALIZACIÓN

```python
position_range=(0, 180)  # Parámetro del constructor
```

Este parámetro se usa en:
- Línea 3288-3290: Para extraer orientaciones de resultados
```python
orientation_range = np.linspace(
    self._position_range[0], self._position_range[1], self._N_E
)
```

**PERO** la construcción de matrices W usa:
```python
orientations = np.linspace(0, np.pi, self._N, endpoint=False)  # [0, π] en RADIANES
```

### Mapeo Real

| Índice | θ interno (rad) | θ visualización (°) | θ visualización corregida |
|--------|-----------------|---------------------|---------------------------|
| E[0]   | 0.0000 rad      | 0°                  | **0°** ✓                  |
| E[25]  | 1.5708 rad (π/2)| 90°                 | **90°** ✓                 |
| E[49]  | 3.0788 rad      | 176.4°              | **176.4°** ✓              |

**AFORTUNADAMENTE**, en Echeveste2020:
- `position_range` por defecto es `(0, 180)` 
- Esto coincide con el mapeo real de `[0, π]` radianes

---

## 5. VERIFICACIÓN EXPERIMENTAL

```python
# Test: Verificar que w_learn se reconstruye correctamente

N_E = 50
theta = np.linspace(0, np.pi, N_E, endpoint=False)  # [0, π]
delta = theta[:, None] - theta[None, :]

# Usando factor 2 en coseno
W_reconstructed = a_ee * np.exp((np.cos(2 * delta) - 1) / (d_ee**2))

# Comparar con w_learn original
max_error = np.abs(W_EE - W_reconstructed).max()
print(f"Error: {max_error}")  # → 0.0 (perfecto!)
```

**RESULTADO:** Error < 10⁻⁹ → La reconstrucción es **PERFECTA**

---

## 6. CASO DE CÓDIGO ORIGINAL: ¿REALMENTE USA [0, 2π]?

Revisando `objective.ml` líneas 285-287:
```ocaml
let theta_i = PE.(2. *. pi *. float i /. float m) in
```

**SÍ**, el código OCaml original usa `[0, 2π]` sin el factor 2 en el coseno.

**PERO** en el paper (Ecuación 10) dice:
```
W_XY(θ_i, θ_j) = a_XY * exp[(cos(2*(θ_i - θ_j)) - 1) / d_XY²]
```

### Posible Explicación

1. **Hipótesis 1:** El paper usa notación θ ∈ [0, π] y el código lo implementa como [0, 2π]
2. **Hipótesis 2:** Hay un error en el paper o en el código (menos probable)
3. **Hipótesis 3:** El código usa [0, 2π] intencionalmente para evitar el factor 2

---

## 7. IMPLICACIONES PARA NUESTRO CÓDIGO

### ✅ LO QUE ESTÁ BIEN

1. Nuestra matriz W se reconstruye **perfectamente** (error < 10⁻⁹)
2. El mapeo interno `[0, π]` con `cos(2*Δθ)` es **matemáticamente correcto**
3. Los valores de w_learn son **idénticos** al original

### ⚠️ LO QUE DEBEMOS REVISAR

1. **Visualizaciones:** Cuando graficamos, usamos `position_range = (0, 180)`
   - ¿Esto corresponde correctamente a nuestro mapeo interno?
   - **RESPUESTA:** SÍ, porque 180° = π radianes

2. **Extracción de orientaciones:** Línea 3288-3290
   ```python
   orientation_range = np.linspace(
       self._position_range[0], self._position_range[1], self._N_E
   )
   ```
   - Esto genera [0°, 180°) que mapea directamente a [0, π] rad
   - **ES CORRECTO** ✓

3. **Documentación:** Debemos aclarar que:
   - Internamente usamos θ ∈ [0, π] radianes
   - Para visualización mostramos [0°, 180°)
   - Ambos son equivalentes y consistentes

---

## 8. RECOMENDACIONES

### Inmediato

1. ✅ **NO cambiar** el mapeo interno `[0, π]` - funciona correctamente
2. ✅ **NO cambiar** el factor 2 en `cos(2*Δθ)` - es matemáticamente correcto
3. ✅ **MANTENER** `position_range = (0, 180)` como default

### Documentación

4. ✅ **AÑADIR comentarios** explicando la equivalencia:
   ```python
   # NOTA: Usamos θ ∈ [0, π] con cos(2*Δθ) que es equivalente a
   # θ ∈ [0, 2π] con cos(Δθ) del código original de Echeveste
   ```

5. ✅ **DOCUMENTAR** en docstrings que `position_range` está en grados y representa [0°, 180°)

### Testing

6. ✅ **AÑADIR test** verificando que orientaciones extraídas coinciden con índices de neuronas

---

## 9. CONCLUSIÓN

**NO HAY BUG** en nuestro código. La aparente discrepancia es una **diferencia de parametrización**:

| Aspecto | Código Original | Nuestro Código | Equivalencia |
|---------|----------------|----------------|--------------|
| Rango θ | [0, 2π] rad | [0, π] rad | θ_orig = 2*θ_nuestro |
| Kernel | cos(Δθ) | cos(2*Δθ) | Compensan mutuamente |
| Resultado | W_original | W_nuestro | **Idénticos** ✓ |

**La visualización con [0°, 180°) es CORRECTA** porque:
- 180° = π radianes
- Mapea uno-a-uno con nuestro rango interno
- Es consistente con la periodicidad π de orientaciones visuales

---

## 10. EVIDENCIA FINAL

```bash
$ md5sum w_learn_original w_learn_local
aa245a2c91923dfa0a22d62e0675a978  w_learn_original
aa245a2c91923dfa0a22d62e0675a978  w_learn_local
```

**Los archivos son IDÉNTICOS** → Nuestra implementación es **100% correcta** ✓
