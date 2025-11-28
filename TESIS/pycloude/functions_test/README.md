# Documentación de Comparación: Implementación vs Código Original

Este directorio contiene documentación exhaustiva comparando nuestra implementación en `scikit-neuromsi` con el código original de Echeveste et al. (2020).

---

## 📚 Documentos Disponibles

### 1. **Comparación Detallada 1-a-1**
**Archivo:** [`comparison_implementation_vs_original.md`](comparison_implementation_vs_original.md)

**Propósito:** Documentación exhaustiva de cada función de simulación

**Contenido:**
- ✅ Activación supralineal y derivadas
- ✅ Ecuaciones de evolución (dinámicas SSN)
- ✅ Cálculo de momentos (media y covarianza)
- ✅ Ruido correlacionado (proceso Ornstein-Uhlenbeck)
- ✅ Conectividad paramétrica (Eq. 10 del paper)
- ✅ Modelo GSM (Gaussian Scale Mixture)
- ✅ Inferencia posterior (MAP y full Bayesian)
- ✅ Transformación no-lineal del input
- ✅ Integración numérica (Euler, RK4)
- ✅ Resumen de equivalencias y validación

**Audiencia:** Investigadores que necesitan verificar fidelidad científica

---

### 2. **Tabla de Referencia Rápida**
**Archivo:** [`quick_reference_table.md`](quick_reference_table.md)

**Propósito:** Navegación rápida entre funciones equivalentes

**Contenido:**
- 📊 Tablas comparativas con ubicaciones exactas
- 🔍 Símbolos visuales de estado (✅≡⚡⚠️)
- 🗺️ Mapa de arquitectura (Original vs Nuestra)
- ⚖️ Resumen de validación numérica
- 🎯 Workflow de uso comparado

**Audiencia:** Desarrolladores que necesitan encontrar funciones específicas rápidamente

---

### 3. **Ejemplos de Código Lado a Lado**
**Archivo:** [`code_examples_side_by_side.md`](code_examples_side_by_side.md)

**Propósito:** Comparación visual directa del código fuente

**Contenido:**
- 💻 Fragmentos de código reales (Original vs Nuestra)
- 🧮 Equivalencias matemáticas explicadas
- 📈 Ejemplos numéricos ejecutables
- 🔬 Casos de uso completos
- ✨ Mejoras implementadas destacadas

**Audiencia:** Programadores que quieren entender las diferencias de implementación

---

## 🎯 Guía de Uso por Necesidad

### "¿Esta función está implementada?"
→ **Usa:** [`quick_reference_table.md`](quick_reference_table.md)
- Busca la función en las tablas
- Verifica el símbolo de estado
- Encuentra la ubicación exacta en ambos códigos

### "¿Cómo se implementó exactamente esta función?"
→ **Usa:** [`code_examples_side_by_side.md`](code_examples_side_by_side.md)
- Busca la sección correspondiente
- Compara el código original vs nuestra implementación
- Lee las notas de equivalencia matemática

### "¿Puedo confiar en que nuestra implementación es correcta?"
→ **Usa:** [`comparison_implementation_vs_original.md`](comparison_implementation_vs_original.md)
- Lee la sección 10 (Resumen de Equivalencias)
- Revisa las validaciones numéricas
- Consulta las referencias al paper

### "¿Qué funciones NO necesitamos implementar?"
→ **Usa:** [`quick_reference_table.md`](quick_reference_table.md)
- Sección "Casos Especiales Implementados"
- Lista de funciones originales no necesarias (con justificación)

### "¿Qué mejoras agregamos?"
→ **Usa:** Cualquiera de los tres documentos
- Busca el símbolo ⚡ en quick_reference_table.md
- Lee "Ventajas de Nuestra Implementación" en comparison_*.md
- Revisa ejemplos mejorados en code_examples_*.md

---

## 🔗 Navegación por Componente

| Componente | Comparación Detallada | Tabla Rápida | Código Lado a Lado |
|------------|----------------------|--------------|-------------------|
| **Activación supralineal** | [Sec. 1.1](comparison_implementation_vs_original.md#11-activación-supralineal) | [Tabla: Core SSN](quick_reference_table.md#funciones-core-ssn) | [Ejemplo 1](code_examples_side_by_side.md#1-activación-supralineal-r--ku_n) |
| **Ecuación membrana** | [Sec. 2.1](comparison_implementation_vs_original.md#21-ecuación-principal-de-membrana) | [Tabla: Core SSN](quick_reference_table.md#funciones-core-ssn) | [Ejemplo 2](code_examples_side_by_side.md#2-ecuación-de-membrana-dudt) |
| **Ruido O-U (η)** | [Sec. 4.1](comparison_implementation_vs_original.md#41-proceso-de-ornstein-uhlenbeck) | [Tabla: Core SSN](quick_reference_table.md#funciones-core-ssn) | [Ejemplo 3](code_examples_side_by_side.md#3-proceso-ornstein-uhlenbeck-ruido-η) |
| **Momentos ν** | [Sec. 3.1](comparison_implementation_vs_original.md#31-media-de-las-tasas-ν) | [Tabla: Momentos](quick_reference_table.md#momentos-y-estadísticas) | [Ejemplo 4](code_examples_side_by_side.md#4-cálculo-de-momentos-media-ν) |
| **Conectividad W** | [Sec. 5](comparison_implementation_vs_original.md#5-conectividad-paramétrica) | [Tabla: Conectividad](quick_reference_table.md#conectividad-y-ruido) | [Ejemplo 5](code_examples_side_by_side.md#5-conectividad-paramétrica-w_xy) |
| **Ruido Σ_η** | [Sec. 4.2](comparison_implementation_vs_original.md#42-matriz-de-covarianza-del-ruido-σ_η) | [Tabla: Conectividad](quick_reference_table.md#conectividad-y-ruido) | [Ejemplo 9](code_examples_side_by_side.md#9-construcción-de-covarianza-del-ruido-σ_η) |
| **GSM filtros** | [Sec. 6.1](comparison_implementation_vs_original.md#61-filtros-de-gabor-matriz-a) | [Tabla: GSM](quick_reference_table.md#modelo-gsm) | Ver comparación |
| **GSM posterior** | [Sec. 6.3-6.4](comparison_implementation_vs_original.md#63-posterior-gsm-media-μ_post) | [Tabla: GSM](quick_reference_table.md#modelo-gsm) | [Ejemplo 6](code_examples_side_by_side.md#6-gsm-posterior-μ_post-σ_post) |
| **Inferencia z** | [Sec. 6.5](comparison_implementation_vs_original.md#65-inferencia-de-contraste-z_map) | [Tabla: GSM](quick_reference_table.md#modelo-gsm) | [Ejemplo 7](code_examples_side_by_side.md#7-inferencia-de-contraste-z_map) |
| **Input h** | [Sec. 8](comparison_implementation_vs_original.md#8-transformación-no-lineal-del-input) | [Tabla: Input](quick_reference_table.md#input-y-transformaciones) | [Ejemplo 8](code_examples_side_by_side.md#8-transformación-no-lineal-del-input-h) |
| **Integración** | [Sec. 9](comparison_implementation_vs_original.md#9-integración-numérica) | [Tabla: Integración](quick_reference_table.md#integración-numérica) | Ver notas |
| **Entrenamiento** | Ver comparison (Stage 1/2) | [Tabla: Training](quick_reference_table.md#entrenamiento-training) | N/A (no en original) |

---

## 📊 Validación Numérica (Resumen Ejecutivo)

Todos los tests de validación confirman equivalencia:

| Test | Archivo | Status |
|------|---------|--------|
| Equivalencia parámetros | `test_equivalencia_parametros.py` | ✅ Error < 1e-10 |
| Modelo funcionando | `test_modelo_funcionando.py` | ✅ Correlación > 0.99 |
| Variabilidad ruido | `test_variabilidad_10_ejecuciones.py` | ✅ KS p > 0.05 |
| GSM posterior | `REPORTE_COMPARACION_GSM.md` | ✅ Error < 1e-8 |
| Entrenamiento completo | `entrenamiento_completo_5runs.log` | ✅ 5/5 success |

**Conclusión:** Nuestra implementación es **numéricamente equivalente** al código original.

---

## 🔍 Diferencias Principales (Arquitectónicas, NO Matemáticas)

### 1. **Construcción vs Carga**
- **Original:** Carga W y Σ_η desde archivos pre-calculados
- **Nuestra:** Construye W y Σ_η desde parámetros (permite training)

### 2. **Backend Computacional**
- **Original:** NumPy puro (solo CPU)
- **Nuestra:** BrainPy + JAX (GPU/TPU support + auto-diff)

### 3. **Estructura de Código**
- **Original:** Scripts separados (SSN/, GSM/, GP/)
- **Nuestra:** Clase unificada `Echeveste2020`

### 4. **Entrenamiento**
- **Original:** Código OCaml separado (`ssn_inference_optimizer`)
- **Nuestra:** Implementación Python+JAX integrada (Stage 1 + Stage 2)

### 5. **API**
- **Original:** Funciones libres con parámetros globales
- **Nuestra:** API scikit-learn-style (fit/predict pattern)

**Todas estas diferencias preservan la equivalencia matemática.**

---

## 📖 Referencias Externas

### Código Original
- **Repositorio:** `ssn_inference_numerical_experiments/`
- **Archivos clave:**
  - `SSN/methods.py` - Funciones dinámicas
  - `SSN/parameters.py` - Parámetros optimizados
  - `GSM/GSM.py` - Modelo generativo

### Paper
- **Echeveste, R., Aitchison, L., Hennequin, G., & Lengyel, M. (2020).**
  *Nature Neuroscience*, 23(12), 1138-1149.
  - Main paper: Ecuaciones 8-10 (SSN dynamics)
  - Supplementary: Table S1, Eq. S16-S25

### Nuestra Implementación
- **Archivo:** `scikit-neuromsi/skneuromsi/neural/_echeveste2020.py`
- **Tests:** `pycloude/test_*.py`

### Documentación Complementaria
- `parameters.md` - Parámetros optimizados de Table S1
- `REPORTE_COMPARACION_GSM.md` - Validación GSM
- `ANALISIS_PARAMETROS_TRANSFORMACION_NOLINEAL.md` - Input h optimizado
- `entrenamiento_completo_5runs.log` - Logs de training

---

## 🚀 Inicio Rápido

### Para investigadores:
1. Lee [`comparison_implementation_vs_original.md`](comparison_implementation_vs_original.md) - Sección 10 (Resumen)
2. Revisa validación numérica
3. Consulta paper references

### Para desarrolladores:
1. Abre [`quick_reference_table.md`](quick_reference_table.md)
2. Busca la función que necesitas
3. Ve a [`code_examples_side_by_side.md`](code_examples_side_by_side.md) para ver código

### Para reproducir validaciones:
```bash
cd pycloude

# Test de equivalencia básica
python test_equivalencia_parametros.py

# Test de dinámica completa
python test_modelo_funcionando.py

# Test de variabilidad con ruido
python test_variabilidad_10_ejecuciones.py
```

---

## 📝 Contribuciones Futuras

Si encuentras discrepancias o quieres agregar más comparaciones:

1. **Nueva función:** Agrega entrada en las 3 tablas:
   - Comparación detallada (con matemáticas)
   - Tabla rápida (con ubicaciones)
   - Ejemplo código (con fragmentos)

2. **Nueva validación:** Documenta en sección 10 de comparison_*.md

3. **Nuevo test:** Agrega a `pycloude/test_*.py` y actualiza README

---

## ✅ Checklist de Equivalencia

Para cada nueva función implementada, verificar:

- [ ] Matemáticas coinciden con paper (ecuaciones citadas)
- [ ] Código produce resultados numéricos equivalentes (error < tol)
- [ ] Documentación referencia código original
- [ ] Test de validación agregado
- [ ] Entrada en las 3 tablas de comparación
- [ ] Ejemplo de código side-by-side (si aplica)

---

## 📧 Contacto

Para preguntas sobre esta documentación o la implementación:
- Ver `CLAUDE.md` en directorio raíz del proyecto
- Consultar issues en repositorio

---

**Última actualización:** 2025-11-22

**Esta documentación garantiza trazabilidad completa entre nuestra implementación y el código original de Echeveste et al. (2020).**
