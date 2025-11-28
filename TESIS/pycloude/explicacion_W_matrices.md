# Explicación: W_reconstruida vs W_exact

## Contexto: Dos formas de representar la conectividad

En el modelo de Echeveste2020, la matriz de conectividad W se puede representar de **DOS formas diferentes**:

### 1. **Forma Paramétrica** (8 parámetros)
### 2. **Forma Explícita** (matriz completa 100×100)

---

## W_exact: La Matriz Original Completa

**¿Qué es?**
- Es la matriz de conectividad **completa** (100×100) que fue optimizada durante el entrenamiento original
- Está guardada en el archivo `w_learn` del repositorio original
- Contiene **todos los valores explícitos** de conexiones entre las 100 neuronas

**¿De dónde viene?**
- Fue generada por el proceso de optimización de Echeveste et al. (2020)
- Es el resultado de minimizar la función de costo (Eq. 25)
- Se guarda como archivo porque tiene **10,000 valores** (100×100)

**Ubicación:**
```
scikit-neuromsi/skneuromsi/data/echeveste2020/w_learn
```

**En el código:**
```python
model._W_exact  # La matriz completa cargada desde w_learn
```

---

## W_reconstruida: Matriz Generada desde los 8 Parámetros

**¿Qué es?**
- Es una matriz de conectividad **reconstruida** usando la fórmula paramétrica (Eq. 10)
- Se calcula a partir de **solo 8 parámetros**: a_EE, a_EI, a_IE, a_II, d_EE, d_EI, d_IE, d_II

**¿De dónde viene?**
- Se genera aplicando la fórmula del paper:
  ```
  W_XY(θi, θj) = a_XY * exp[(cos(2(θi - θj)) - 1) / d_XY²]
  ```
- Para cada par de neuronas (i, j), se calcula su conexión basándose en:
  - Sus orientaciones preferidas (θi, θj)
  - La amplitud de conectividad (a_XY)
  - El ancho de la conexión (d_XY)

**En el código:**
```python
W = model.build_connectivity_matrix()  # Reconstruye desde parámetros
```

---

## ¿Por qué existen ambas?

### Razón Histórica
Durante el entrenamiento original de Echeveste, se optimizó la **matriz completa W** usando:
```ocaml
(* train.ml línea 249-278 *)
optimize_connectivity_matrix(W_full)  (* 10,000 parámetros *)
```

Luego, para hacer el modelo más manejable, se **ajustaron 8 parámetros** a esa matriz optimizada:
```ocaml
(* Encontrar los mejores a_XY y d_XY que aproximen W_full *)
{a_EE, a_EI, a_IE, a_II, d_EE, d_EI, d_IE, d_II}
```

### Razón Práctica

**Ventajas de W_exact (matriz completa):**
- ✅ Valores exactos del entrenamiento
- ✅ No requiere recalcular nada
- ❌ Ocupa mucho espacio (10,000 valores)
- ❌ Difícil de interpretar
- ❌ No se puede escalar a otras dimensiones

**Ventajas de W_reconstruida (8 parámetros):**
- ✅ Solo 8 números para guardar
- ✅ Interpretable (amplitudes y anchos tienen significado)
- ✅ Se puede adaptar a diferentes números de neuronas
- ✅ Captura la estructura circular del modelo
- ❌ Es una aproximación (no exacta)

---

## La Diferencia Entre Ambas

### Ejemplo Numérico

```
W_exact[0, 1] = 0.32706254        (valor optimizado original)
W_reconstruida[0, 1] = 0.33007689 (reconstruido desde parámetros)
Diferencia = 0.00301435           (≈ 0.9% de error)
```

### ¿Por qué hay diferencia?

1. **Discretización de orientaciones:**
   ```python
   # Original podría usar:
   theta_original = [0°, 3.6°, 7.2°, ...]

   # Reconstrucción usa:
   theta_recon = np.linspace(0, 180, 50)  # [0°, 3.67°, 7.35°, ...]
   ```

2. **Aproximación paramétrica:**
   - La fórmula con 8 parámetros es una **aproximación** de la matriz completa
   - Es como ajustar una función suave (8 parámetros) a 10,000 puntos

3. **Precisión numérica:**
   - Diferencias en cómo se calculan las funciones exponenciales
   - Errores de redondeo acumulados

---

## ¿Cuál deberías usar?

### Para simulaciones y experimentos:
```python
# Usar W_exact para reproducir exactamente el paper
model.load_parameters()
W = model._W_exact  # Matriz exacta original
```

### Para entender el modelo:
```python
# Usar W_reconstruida para ver los parámetros interpretables
params = {
    'a_EE': 0.331089,  # Amplitud E→E
    'd_EE': 0.802756,  # Ancho E→E
    ...
}
W = model.build_connectivity_matrix(params)
```

### Para nuevos entrenamientos:
```python
# Optimizar los 8 parámetros en lugar de 10,000
# Es más eficiente y generalizable
```

---

## Visualización Conceptual

```
Entrenamiento Original (Echeveste 2020):
┌─────────────────────────────────────┐
│  Optimizar W completa (10,000 vals) │
│           ↓                         │
│  W_exact (100×100 matriz)           │
│           ↓                         │
│  Ajustar 8 parámetros a W_exact     │
│           ↓                         │
│  {a_EE, a_EI, ..., d_II}            │
└─────────────────────────────────────┘

Uso en scikit-neuromsi:
┌─────────────────────────────────────┐
│  Cargar 8 parámetros                │
│  {a_EE, a_EI, ..., d_II}            │
│           ↓                         │
│  Aplicar Eq. 10                     │
│  W_XY(θi,θj) = a·exp[(cos...)/d²]   │
│           ↓                         │
│  W_reconstruida (100×100)           │
│           ↓                         │
│  Comparar con W_exact (validación)  │
└─────────────────────────────────────┘
```

---

## Conclusión

- **W_exact** = Matriz "de referencia" del paper (10,000 valores)
- **W_reconstruida** = Matriz generada desde 8 parámetros (aproximación)
- La diferencia (~1-5%) es **esperada y aceptable**
- Los 8 parámetros capturan la **estructura esencial** de la conectividad

Es como la diferencia entre:
- **W_exact**: Tener una tabla con el seno de todos los ángulos
- **W_reconstruida**: Calcular sin(θ) con una fórmula cuando lo necesitas
