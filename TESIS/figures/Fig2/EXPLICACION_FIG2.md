# Explicación de la Figura 2: Inferencia y Respuestas en la Red Optimizada

## Contexto General

La Figura 2 del paper Echeveste et al. (2020) demuestra que la red neuronal optimizada (SSN)
implementa **inferencia probabilística mediante muestreo** de la distribución posterior del
modelo generativo (GSM - Gaussian Scale Mixture). Esta es una validación clave de que el
modelo neuronal aprende a realizar inferencia bayesiana.

## Panel a: Actividad Poblacional de Potenciales de Membrana

### Qué se muestra:
- **Dos heatmaps** de potenciales de membrana de neuronas excitatorias ($u_E$)
- **Eje vertical**: Orientación preferida de cada neurona (de -90° a 90°)
- **Eje horizontal**: Tiempo (últimos 100ms de actividad estacionaria)
- **Panel superior**: Contraste cero (c = 0.0)
- **Panel inferior**: Contraste alto (c = 1.0) con estímulo orientado a 0° (flecha)

### Qué representa para el modelo:
1. **Estado del sistema**: Los potenciales de membrana $u_E$ son las variables internas
   de la red que representan las variables latentes del GSM (orientaciones en el estímulo).

2. **Sintonización por orientación**:
   - Con **contraste cero**, la actividad es relativamente uniforme y fluctúa aleatoriamente
     (ruido puro, sin información del estímulo).
   - Con **contraste alto**, las neuronas cercanas a 0° (orientación del estímulo)
     muestran mayor actividad, reflejando la inferencia de la orientación presente.

3. **Muestreo estocástico**: Las fluctuaciones temporales representan **muestras** de la
   distribución posterior. La red no converge a un valor fijo, sino que genera muestras
   que reflejan la incertidumbre.

## Panel b: Elipses de Covarianza

### Qué se muestra:
- **Panel izquierdo**: Estímulos del training set (cajas de colores representan diferentes
  niveles de contraste)
- **Panel derecho**: Proyección 2D de las distribuciones de respuestas para dos neuronas
  seleccionadas (orientaciones preferidas ~42° y ~16°)
- **Elipses verdes** (líneas punteadas): Covarianza del posterior ideal (GSM, 2 desviaciones estándar)
- **Elipses rojas** (líneas continuas): Covarianza de las respuestas de la red (2 desviaciones estándar)
- **Trayectorias coloreadas**: Secuencias de 500ms de actividad de la red

### Qué representa para el modelo:
1. **Coincidencia de distribuciones**: La superposición de elipses verdes y rojas demuestra
   que la red **muestrea de la distribución posterior correcta**. Esto es evidencia directa
   de que implementa inferencia probabilística.

2. **Estructura de covarianza**: Las elipses capturan la correlación entre neuronas. La forma
   y orientación de las elipses revelan cómo las orientaciones se correlacionan en el posterior.

3. **Trayectorias de muestreo**: Las líneas coloreadas muestran cómo la red explora el espacio
   de estados a lo largo del tiempo, generando muestras independientes de la distribución posterior.

4. **Dependencia del contraste**: Elipses más pequeñas con mayor contraste indican menor
   incertidumbre (posterior más concentrado cuando hay más evidencia sensorial).

## Panel c: Media y Desviación Estándar

### Qué se muestra:
- **4 subpanels** organizados en 2x2:
  - **Fila superior**: Medias
  - **Fila inferior**: Desviaciones estándar
  - **Columna izquierda**: Posterior del ideal observer (GSM)
  - **Columna derecha**: Respuestas de la red (SSN)
- **Eje X**: Orientación preferida de las neuronas (-90° a 90°)
- **Líneas de colores**: Diferentes niveles de contraste (de azul oscuro = 0.0 a rojo = 1.0)

### Qué representa para el modelo:
1. **Sintonización de la media**:
   - **Ideal observer (izquierda)**: Media posterior centrada en la orientación del estímulo (0°)
   - **Red (derecha)**: Media de potenciales de membrana debe coincidir con el posterior
   - La **amplitud** de la curva aumenta con el contraste (más evidencia → respuesta más fuerte)

2. **Quenching de variabilidad** (panel inferior):
   - **Observación clave**: La desviación estándar **disminuye** con el contraste, especialmente
     en la orientación preferida (0°).
   - **Interpretación**: Con más evidencia (mayor contraste), la **incertidumbre posterior**
     disminuye, y la red refleja esto con menor variabilidad en sus respuestas.
   - Este fenómeno se llama **"quenching"** y es una predicción distintiva de la teoría de
     inferencia probabilística que ha sido observada experimentalmente en corteza visual.

3. **Modulación por contraste**:
   - Los colores representan cómo cambian los momentos con diferentes niveles de contraste
   - El comportamiento debe ser similar entre ideal observer y red para validar el modelo

## Panel d: Matrices de Correlación

### Qué se muestra:
- **Dos matrices de correlación** (50x50, una para cada población de neuronas excitatorias)
- **Panel izquierdo**: Correlaciones del posterior del ideal observer (verde)
- **Panel derecho**: Correlaciones de las respuestas de la red (rojo)
- **Ejes**: Orientación preferida de las neuronas (-90° a 90°)
- **Colormap**: De -1 (anti-correlación) a +1 (correlación perfecta)

### Qué representa para el modelo:
1. **Estructura de correlaciones del ruido**:
   - La diagonal (correlación = 1) representa la auto-correlación de cada neurona
   - **Patrón de bandas**: Neuronas con orientaciones similares están correlacionadas
   - **Decaimiento con distancia**: La correlación disminuye a medida que las orientaciones
     preferidas se alejan

2. **Matching de correlaciones**:
   - La similitud entre ambas matrices demuestra que la red **reproduce la estructura de
     covarianza** del posterior
   - Esto es un **sello distintivo del muestreo**: las correlaciones en las muestras deben
     coincidir con las correlaciones de la distribución objetivo

3. **Circulante por bloques**:
   - La matriz muestra simetría circular porque la topología de la red es un "anillo" de
     orientaciones
   - Esta estructura refleja la geometría del espacio de orientaciones (periodicidad de 180°)

4. **Ruido correlacionado**:
   - Las correlaciones no-diagonales indican que el ruido en la red está estructurado
     (no es independiente entre neuronas)
   - Esto es necesario para implementar muestreo de una distribución con covarianza no-trivial

## Implicaciones para el Modelo

### Validación de Inferencia Probabilística:
La Figura 2 en conjunto demuestra que:

1. **La red implementa sampling-based inference**: Las respuestas fluctúan alrededor de la
   media posterior y exploran el espacio con la varianza correcta.

2. **Los momentos coinciden**: Media, varianza y correlaciones de la red coinciden con el
   posterior ideal, confirmando que la red aprendió la transformación correcta.

3. **Modulación por estímulo**: La red adapta sus respuestas apropiadamente a diferentes
   contrastes y orientaciones.

4. **Quenching de variabilidad**: La red reproduce el fenómeno de reducción de variabilidad
   con mayor contraste, una predicción clave de la teoría de inferencia probabilística.

## Conexión con el Paper

Esta figura corresponde a la sección de resultados que demuestra:

> "The optimized network performs probabilistic inference by sampling from the posterior
> distribution of the GSM. Network responses match the posterior moments (mean, variance,
> covariance) across all contrast levels."

Los 4 paneles proveen evidencia complementaria:
- **Panel a**: Visualización cualitativa de la actividad
- **Panel b**: Validación de covarianzas en proyección 2D
- **Panel c**: Validación cuantitativa de momentos primer y segundo orden
- **Panel d**: Validación de estructura de correlaciones completa

## Referencias del Paper

- Echeveste et al. (2020). "Cortical-like dynamics in recurrent circuits optimized for
  sampling-based probabilistic inference." Nature Neuroscience 23(9): 1138-1149.
- Fig. 2: "Inference and responses in the optimized network"
- Sección de resultados: "The network implements sampling-based inference" (p. 1140-1141)
