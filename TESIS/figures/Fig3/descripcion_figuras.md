# Figura 3: Generalización en la Red Optimizada

## Descripción General

La Figura 3 del paper de Echeveste et al. (2020) demuestra la capacidad de generalización de la red SSN entrenada. Muestra que la red no solo puede aproximar el posterior bayesiano para los estímulos de entrenamiento, sino que también **generaliza correctamente a estímulos no vistos durante el entrenamiento**.

Este es un resultado fundamental porque demuestra que la red ha aprendido la estructura estadística del modelo generativo GSM, no solo ha memorizado las respuestas para los estímulos de entrenamiento.

---

## Fig. 3a: Generalización a Través de Niveles de Contraste

### ¿Qué muestra?

Gráfico de media poblacional vs. contraste, comparando:
- **Línea verde**: Media y desviación estándar de las variables latentes bajo el posterior GSM
- **Línea roja**: Media y desviación estándar de las respuestas estacionarias de la red SSN
- **Círculos**: Niveles de contraste usados durante el entrenamiento (0.05, 0.10, 0.20, 0.40)
- **Segmentos entre círculos**: Generalización a niveles de contraste **no entrenados**

### ¿Qué representa para el modelo?

1. **Capacidad de interpolación**: La red puede predecir correctamente las respuestas para contrastes intermedios que nunca vio durante el entrenamiento.

2. **Concordancia posterior**: Las respuestas de la red (rojo) siguen muy de cerca el posterior bayesiano ideal (verde), incluso para estímulos no entrenados.

3. **Validación del aprendizaje**: Si la red solo hubiera memorizado los estímulos de entrenamiento, las respuestas entre círculos serían erróneas. La concordancia demuestra que la red aprendió la regla general.

### Interpretación

- **Media creciente con contraste**: Tanto el posterior GSM como la red muestran que la actividad promedio aumenta con el contraste del estímulo.
- **Variabilidad decreciente**: La desviación estándar disminuye con el contraste, reflejando mayor certeza en la inferencia para estímulos de alto contraste.

---

## Fig. 3b: Momentos Estacionarios (Scatter Plots)

### ¿Qué muestra?

Dos scatter plots comparando momentos estadísticos:

**Panel superior**: Media estacionaria
- Eje X: Media del posterior GSM (mV)
- Eje Y: Media de la respuesta de red (mV)
- Puntos lavanda: Estímulos de **entrenamiento**
- Puntos naranja: Estímulos de **test** (no entrenados)

**Panel inferior**: Covarianza estacionaria
- Eje X: Covarianza del posterior GSM (mV²)
- Eje Y: Covarianza de la respuesta de red (mV²)
- Mismo código de colores

### ¿Qué representa para el modelo?

1. **Alineación perfecta**: Los puntos se alinean sobre la diagonal (línea negra punteada), indicando que la red aproxima correctamente tanto medias como covarianzas.

2. **Generalización a test set**: Los puntos naranja (test) se alinean igual de bien que los lavanda (entrenamiento), demostrando generalización.

3. **Momentos de primer y segundo orden**: La red captura correctamente:
   - **Medias** (primer momento): Respuesta promedio de cada neurona
   - **Covarianzas** (segundo momento): Correlaciones entre pares de neuronas

### Interpretación

- **Cada punto**: Representa la respuesta de una neurona individual (panel superior) o un par de neuronas (panel inferior) a un estímulo específico.
- **Dispersión**: Pequeña dispersión alrededor de la diagonal indica aproximación precisa.
- **Sin separación train/test**: No hay diferencia visible entre estímulos de entrenamiento y test, confirmando generalización robusta.

---

## Fig. 3c: Ejemplos de Generalización

### ¿Qué muestra?

6 filas de ejemplos (1 entrenamiento + 5 test), cada una con 3 columnas:

**Columna 1: Estímulo Visual**
- Gabor patch con orientación θ y contraste c específicos
- Primera fila: Estímulo de entrenamiento
- Filas 2-6: Estímulos de test (nuevos)

**Columna 2: Medias y Componentes Principales**
- **Línea verde continua**: Media del posterior GSM
- **Línea roja continua**: Media de la respuesta de red
- **Líneas punteadas/continuas coloreadas**: Primeros 3 componentes principales (PCs)
  - Verde: PCs de la covarianza GSM
  - Roja: PCs de la covarianza de red
  - Escalados por √(varianza explicada)

**Columna 3: Matrices de Correlación**
- **Izquierda (verde)**: Correlaciones de Pearson del posterior GSM
- **Derecha (roja)**: Correlaciones de Pearson de la red
- Colormap: Azul (correlación negativa) → Rojo (correlación positiva)

### ¿Qué representa para el modelo?

1. **Tuning curves**: Las medias (col. 2) muestran cómo cada neurona responde según su orientación preferida:
   - Pico máximo cerca de la orientación del estímulo
   - Forma de campana gaussiana (tuning selectivo)

2. **Estructura de covarianza**: Los PCs revelan los patrones dominantes de correlación:
   - PC1: Modo principal de variabilidad
   - PC2, PC3: Modos secundarios
   - La red reproduce estos modos fielmente (líneas verdes y rojas superpuestas)

3. **Correlaciones neuronales**: Las matrices (col. 3) muestran:
   - Estructura de correlación espacial entre neuronas
   - Neuronas cercanas en preferencia de orientación están más correlacionadas
   - La red (derecha) replica la estructura GSM (izquierda)

### Interpretación

- **Fila 1 (training)**: Demuestra que la red aprendió correctamente un estímulo de entrenamiento.
- **Filas 2-6 (test)**: Demuestran generalización a:
  - Nuevos contrastes (0.08, 0.15, 0.25, 0.35, 0.45)
  - Nuevas orientaciones (-60°, -15°, 30°, 60°, 80°)
- **Superposición perfecta**: Las líneas verdes (GSM) y rojas (red) prácticamente coinciden en todos los casos.

---

## Conclusión

La Figura 3 completa demuestra que:

1. **La red generaliza** a estímulos no vistos (contrastes y orientaciones nuevas)
2. **Captura todos los momentos estadísticos** del posterior GSM (media, std, covarianza, correlaciones)
3. **Aprende la estructura del problema**, no solo memoriza ejemplos
4. **Implementa inferencia bayesiana aproximada** a través de su dinámica de muestreo

Esto valida que el proceso de optimización de dos etapas (Stage 1: conectividad, Stage 2: ruido correlacionado) produjo una red que realmente implementa inferencia probabilística en un modelo generativo visual realista.

---

## Referencias

- Echeveste et al. (2020), Nature Neuroscience, Figure 3
- Paper: "Cortical-like dynamics in recurrent circuits optimized for sampling-based probabilistic inference"
- Métodos: "Response moments were estimated from n=20,000 independent samples (taken 200 ms apart)"
