# Extended Data Figure E1 - Echeveste et al. (2020)

## Descripción General

Este directorio contiene las visualizaciones de los parámetros del modelo GSM (Gaussian Scale Mixture) y de la red SSN (Stabilized Supralinear Network) optimizada para inferencia probabilística basada en muestreo.

**Referencia**: Echeveste et al. (2020), "Cortical-like dynamics in recurrent circuits optimized for sampling-based probabilistic inference", Nature Neuroscience.

---

## Archivos Generados

- `FigE1b_prior_covariance_GSM.png`: Covarianza previa del GSM
- `FigE1d_recurrent_weights.png`: Pesos recurrentes optimizados (3 subplots)
- `FigE1e_input_nonlinearity.png`: Transformación no-lineal de entrada
- `generar_extended_data_fig1.py`: Script para generar las figuras
- `README.md`: Este archivo

---

## FigE1b: Prior Covariance in the GSM

### ¿Qué muestra?
La matriz de covarianza previa **C** (50×50) del modelo generativo GSM.

### Interpretación
- **Ejes**: Orientaciones preferidas de -90° a 90° (grados)
  - Eje X: Orientación preferida de la neurona j
  - Eje Y: Orientación preferida de la neurona i (-90° abajo, 90° arriba)
- **Colorbar**: Valores de covarianza de -4 a 4
  - Rojo: Covarianza positiva (variables correlacionadas)
  - Azul: Covarianza negativa (variables anti-correlacionadas)
  - Blanco: Covarianza cercana a cero

### Significado en el modelo
Esta matriz define cómo se correlacionan las variables latentes del GSM antes de observar cualquier dato (prior). En el contexto de orientaciones visuales:
- La diagonal principal (rojo intenso) muestra alta varianza
- Las bandas rojas cercanas a la diagonal indican que neuronas con orientaciones preferidas similares están correlacionadas
- La estructura captura las estadísticas naturales de las imágenes

---

## FigE1d: Recurrent Weights After Training

### ¿Qué muestra?
Los pesos recurrentes optimizados de la red SSN, mostrados como función de la **diferencia** de orientación preferida entre neuronas.

### Concepto de "diferencias de orientación"

En una matriz de conectividad circulante, los pesos dependen solo de la **diferencia relativa** entre orientaciones preferidas:

```
W[i,j] = función(θ_i - θ_j)
```

**Por qué graficamos vs diferencias:**
- La matriz W es circulante: la estructura se repite para todas las neuronas
- Los pesos dependen de la diferencia Δθ = θ_i - θ_j, no de θ_i o θ_j individualmente
- Simetría traslacional: misma estructura de conectividad, solo rotada

**En el gráfico:**
- Eje X = diferencia de orientación (0° = hacia neuronas con misma orientación)
- El pico en 0° muestra conexiones más fuertes con neuronas similares

### Estructura de 3 subplots

#### 1. Raw recurrent weights (×10⁻¹)
- EE (azul): Conexiones Excitatorias → Excitatorias
- EI (magenta): Conexiones Excitatorias → Inhibitorias (negativo)
- IE (naranja): Conexiones Inhibitorias → Excitatorias
- II (rojo): Conexiones Inhibitorias → Inhibitorias (negativo)

#### 2. Normalized absolute recurrent weights
- Compara amplitudes de sintonización entre tipos de conexiones
- EE y IE tienen perfiles similares

#### 3. Process noise covariance (mV²)
- EE: Covarianza entre pares E
- EI=IE: Covarianza entre pares E-I (simétrica)
- II: Covarianza entre pares I
- Pico en 0° indica alta correlación de ruido entre neuronas similares

---

## FigE1e: Input Nonlinearity

### ¿Qué muestra?
La transformación no-lineal que convierte activaciones de campo receptivo (W_ff·x)_i en inputs a la red h_i.

### Elementos del gráfico

**Curva negra** - Transformación optimizada:
```
h_i = 1.96 * [(W_ff·x)_i - 0.10]_+^2.03
```

**Distribución gris**:
- Distribución de (W_ff·x)_i del training set
- Muestra cómo se distribuyen las activaciones de campo receptivo
- Centrada mayormente en valores positivos (parches de imágenes naturales)

**Línea roja punteada** - Threshold β_h = 0.10:
- Valores menores → h_i = 0
- Implementa rectificación selectiva

### Interpretación

**Eje X: (W_ff·x)_i** (-2 a 2)
- Activaciones del campo receptivo (productos de filtros de Gabor con patches)
- Pueden ser negativas o positivas

**Eje Y: h_i**
- Input efectivo a la red SSN
- Después de threshold y no-linealidad

### Correcciones aplicadas

**Problema original**: La distribución gris tenía valores positivos para (W_ff·x)_i < 0.

**Solución**: Usar distribución chi-cuadrado para simular patches de imágenes naturales, que genera activaciones sesgadas hacia valores positivos (más realista que gaussiana).

**Resultado**: La distribución gris ahora está correctamente centrada en valores positivos, reflejando el comportamiento real del GSM con imágenes naturales.

---

## Relación entre las Figuras

### Pipeline completo:

1. **Estímulo visual x** (256 píxeles)
2. **Filtros de Gabor** → (W_ff·x)_i
3. **FigE1e: Transformación no-lineal** → h_i
4. **FigE1d: Red recurrente SSN** → dinámica de sampling
5. **FigE1b: Prior GSM (C)** → distribución objetivo

La red implementa inferencia bayesiana combinando evidencia sensorial (FigE1e) con prior estructurado (FigE1b) mediante dinámicas recurrentes optimizadas (FigE1d).

---

## Notas Técnicas

### Cambios de nombre
- Todos los archivos usan prefijo **FigE1** para indicar Extended Data Figure 1
- FigE1b, FigE1d, FigE1e corresponden a los paneles del paper

### Parámetros optimizados
- 8 parámetros de conectividad (a_XY, d_XY)
- 4 parámetros de ruido (σ_E, σ_I, ρ, d_σ)
- 3 parámetros de transformación de entrada (α_h, β_h, γ_h)
- Total: 15 parámetros libres

### Referencias
- Paper: Echeveste et al. (2020), Nature Neuroscience
- Parámetros: `parametros.md`
- Código original: `ssn_inference_numerical_experiments`
