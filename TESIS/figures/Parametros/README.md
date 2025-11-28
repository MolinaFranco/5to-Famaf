# Visualización de Parámetros de Entrenamiento SSN

Este directorio contiene scripts para generar gráficos de comparación entre los parámetros entrenados de nuestro modelo SSN y los parámetros originales reportados en Echeveste et al. (2020).

## Estructura del Sistema

```
figuras/
├── data/
│   └── training/                    # Datos de entrenamiento
│       ├── entrenar_parametros.py   # Script de entrenamiento
│       ├── results_YYYYMMDD_HHMMSS.npy      # Resultados (.npy)
│       └── metadata_YYYYMMDD_HHMMSS.json    # Metadatos (.json)
└── Parametros/                      # Visualizaciones
    ├── graficar_parametros.py       # Script de visualización
    ├── README.md                    # Esta documentación
    └── *.png                        # Gráficos generados
```

## Workflow

### 1. Ejecutar Entrenamiento

El script `figuras/data/training/entrenar_parametros.py` ejecuta múltiples corridas de entrenamiento y guarda todos los resultados.

```bash
# Activar entorno virtual
source ~/.virtualenvs/tesis/bin/activate

# Ejecutar entrenamiento (ejemplo: 5 corridas, 250 iter/stage)
cd figuras/data/training
python3 entrenar_parametros.py --n_runs 5 --max_iter_stage1 250 --max_iter_stage2 250
```

**Parámetros:**
- `--n_runs`: Número de corridas de entrenamiento (default: 5)
- `--max_iter_stage1`: Iteraciones máximas para Stage 1 (default: 250)
- `--max_iter_stage2`: Iteraciones máximas para Stage 2 (default: 250)
- `--output_dir`: Directorio de salida (default: figuras/data/training/)

**Salida:**
- `results_YYYYMMDD_HHMMSS.npy`: Array numpy con todos los parámetros optimizados
- `metadata_YYYYMMDD_HHMMSS.json`: Metadatos del entrenamiento (configuración, referencias)

### 2. Generar Gráficos

El script `graficar_parametros.py` carga los resultados y genera gráficos de comparación.

```bash
# Desde el directorio raíz del proyecto
cd figuras/Parametros

# Usar el archivo más reciente (automático)
python3 graficar_parametros.py

# O especificar un archivo de resultados
python3 graficar_parametros.py --results_file results_20251114_080921.npy
```

**Parámetros:**
- `--results_file`: Archivo de resultados específico (default: más reciente)
- `--output_dir`: Directorio de salida para gráficos (default: figuras/Parametros/)

**Gráficos generados:**

1. **`input_transformation_TIMESTAMP.png`**
   - Parámetros de transformación de entrada (Stage 2)
   - α_h (input_scaling), β_h (input_baseline), γ_h (input_nl_pow)
   - Líneas punteadas rojas: valores de Echeveste
   - Bandas de tolerancia: ±5%

2. **`connectivity_amplitudes_TIMESTAMP.png`**
   - Amplitudes de conectividad (Stage 1)
   - a_EE, a_EI, a_IE, a_II
   - Líneas punteadas rojas: valores de Echeveste
   - Bandas de tolerancia: ±10%

3. **`connectivity_widths_TIMESTAMP.png`**
   - Anchos de conectividad (Stage 1)
   - d_EE, d_EI, d_IE, d_II
   - Líneas punteadas rojas: valores de Echeveste
   - Bandas de tolerancia: ±10%

4. **`noise_covariance_TIMESTAMP.png`**
   - Parámetros de covarianza de ruido (Stage 2)
   - width, σ_E, σ_I, ρ
   - Líneas punteadas rojas: valores de Echeveste
   - Bandas de tolerancia: ±10%

5. **`convergence_TIMESTAMP.png`**
   - Costos finales de optimización
   - Stage 1 (ADAM) y Stage 2 (L-BFGS-B)
   - Escala logarítmica

## Parámetros de Referencia de Echeveste

Los valores originales reportados en Echeveste et al. (2020), Tabla S1 y repositorio original:

### Transformación de Entrada (Stage 2)
- **α_h** (input_scaling): 1.9555
- **β_h** (input_baseline): 0.1032
- **γ_h** (input_nl_pow): 2.0343

### Conectividad (Stage 1)
**Amplitudes:**
- **a_EE**: 0.3311
- **a_EI**: 0.0813
- **a_IE**: 0.3632
- **a_II**: 0.0736

**Anchos:**
- **d_EE**: 0.8028
- **d_EI**: 0.5687
- **d_IE**: 0.8598
- **d_II**: 0.6298

### Covarianza de Ruido (Stage 2)
- **width**: 0.4780
- **σ_E**: 6.3992
- **σ_I**: 3.5335
- **ρ**: 0.9929

## Interpretación de Resultados

### Líneas Punteadas Rojas
Indican los valores originales de Echeveste. Idealmente, nuestros valores entrenados deberían converger cerca de estas líneas.

### Bandas de Tolerancia
- Regiones sombreadas en rojo alrededor de las líneas de referencia
- ±5% para parámetros de entrada
- ±10% para conectividad y ruido
- Valores dentro de estas bandas se consideran aceptables

### Variabilidad entre Corridas
- Baja variabilidad: optimización robusta
- Alta variabilidad: posibles problemas de convergencia o múltiples mínimos locales

### Convergencia
- Stage 1 debe converger en ~200-300 iteraciones
- Stage 2 más sensible, puede requerir más iteraciones o ajuste de hiperparámetros

## Ejemplo Completo

```bash
# 1. Activar entorno
source ~/.virtualenvs/tesis/bin/activate

# 2. Entrenar modelo (5 corridas, 250 iter/stage)
cd figuras/data/training
python3 entrenar_parametros.py --n_runs 5 --max_iter_stage1 250 --max_iter_stage2 250

# 3. Generar gráficos
cd ../Parametros
python3 graficar_parametros.py

# 4. Ver archivos generados
ls -lh *.png
```

## Notas Técnicas

### Formato de Datos
- Los resultados se guardan como arrays numpy (`.npy`) para preservar tipos de datos
- JAX arrays se convierten a Python floats antes de guardar
- Los metadatos se guardan en JSON para fácil lectura

### Acceso a Parámetros Post-Entrenamiento
Los parámetros se extraen del diccionario retornado por `ssn.train()`:

```python
result = ssn.train(gsm, stage1_params=..., stage2_params=...)

# Stage 1: conectividad
stage1_params = result["stage1"]["optimized_params"]
a_EE = stage1_params["a_EE"]

# Stage 2: transformación de entrada y ruido
stage2_params = result["stage2"]["optimized_params"]
input_scaling = stage2_params["input_scaling"]
width = stage2_params["width"]
```

### Manejo de Errores
- Si Stage 2 falla, se usan valores por defecto para evitar crashes
- Se registran flags de éxito para cada stage
- Los gráficos filtran valores NaN automáticamente

## Referencias

- Echeveste, R., Aitchison, L., Hennequin, G., & Lengyel, M. (2020). Cortical-like dynamics in recurrent circuits optimized for sampling-based probabilistic inference. *Nature Neuroscience*, 23, 1138-1149.
- Repositorio original: [ssn_inference_optimizer](https://github.com/rubencoencagli/ssn-inference-numerical-experiments)
