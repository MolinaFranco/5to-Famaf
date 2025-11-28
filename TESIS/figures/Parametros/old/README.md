# Figuras de Parámetros - Modelo SSN Echeveste2020

Este directorio contiene scripts para entrenar y visualizar parámetros del modelo SSN (Stabilized Supralinear Network) siguiendo la metodología de Echeveste et al. (2020).

## 📁 Estructura de Archivos

```
figuras/
├── Parametros/                    # Scripts de visualización
│   ├── graficar_parametros.py    # Script principal de graficado
│   ├── README.md                  # Esta documentación
│   └── *.png                      # Figuras generadas
│
└── data/
    └── training/                  # Datos de entrenamiento
        ├── entrenar_parametros.py # Script de entrenamiento
        ├── results_*.npy          # Resultados de entrenamientos
        └── metadata_*.json        # Metadatos de entrenamientos
```

## 🎯 Workflow

### 1. Entrenar el Modelo

El script `entrenar_parametros.py` ejecuta múltiples corridas de entrenamiento y guarda todos los resultados para análisis posterior:

```bash
# Desde el directorio raíz del proyecto
cd figuras/data/training

# Entrenamiento con configuración por defecto (5 corridas, 250 iter/stage)
workon tesis
python3 entrenar_parametros.py

# Entrenamiento personalizado
python3 entrenar_parametros.py --n_runs 10 \
                               --max_iter_stage1 500 \
                               --max_iter_stage2 500
```

**Parámetros disponibles:**
- `--n_runs`: Número de corridas independientes (default: 5)
- `--max_iter_stage1`: Iteraciones máximas Stage 1 (connectivity) (default: 250)
- `--max_iter_stage2`: Iteraciones máximas Stage 2 (noise covariance) (default: 250)
- `--output_dir`: Directorio de salida (default: figuras/data/training/)

**Salida:**
- `results_YYYYMMDD_HHMMSS.npy`: Array numpy con todos los parámetros optimizados
- `metadata_YYYYMMDD_HHMMSS.json`: Metadatos del entrenamiento (configuración, timestamps, etc.)

**Duración estimada:**
- 5 corridas con 250 iter/stage: ~15-30 minutos (depende del hardware)
- 10 corridas con 500 iter/stage: ~1-2 horas

### 2. Visualizar Resultados

El script `graficar_parametros.py` carga los resultados y genera gráficos comparativos con los valores de referencia de Echeveste:

```bash
# Desde el directorio raíz del proyecto
cd figuras/Parametros

# Usar el archivo de resultados más reciente (automático)
workon tesis
python3 graficar_parametros.py

# Especificar un archivo de resultados específico
python3 graficar_parametros.py --results_file results_20231114_103045.npy
```

**Parámetros disponibles:**
- `--results_file`: Archivo específico a graficar (default: más reciente)
- `--training_dir`: Directorio con datos de training (default: ../data/training/)
- `--output_dir`: Directorio para guardar figuras (default: figuras/Parametros/)

**Figuras generadas:**
1. `input_transformation_comparison.png`: Parámetros α_h, β_h, γ_h
2. `connectivity_comparison.png`: Amplitudes y widths de conectividad (a_XY, d_XY)
3. `noise_comparison.png`: Parámetros de covarianza de ruido (width, σ_E, σ_I, ρ)

## 📊 Interpretación de Gráficos

Cada gráfico muestra:

- **Puntos azules conectados**: Valores obtenidos en cada corrida de entrenamiento
- **Línea roja punteada**: Valor de referencia de Echeveste et al. (2020)
- **Banda roja clara**: Tolerancia aceptable (±5% para input, ±10% para connectivity/noise)
- **Texto rojo**: Valor exacto de referencia

### Indicadores de Convergencia

En las estadísticas impresas por `graficar_parametros.py`:

- `✓✓`: Diferencia < 5% (excelente)
- `✓`: Diferencia 5-10% (buena)
- `~`: Diferencia 10-20% (moderada)
- `!`: Diferencia > 20% (revisar)

## 🔧 Parámetros de Referencia (Echeveste et al. 2020)

### Input Transformation
- **α_h** (input_scaling): 1.9555
- **β_h** (input_baseline): 0.1032
- **γ_h** (input_nl_pow): 2.0343

### Connectivity
- **a_EE**: 0.3311, **d_EE**: 0.8028
- **a_EI**: 0.0813, **d_EI**: 0.5687
- **a_IE**: 0.3632, **d_IE**: 0.8598
- **a_II**: 0.0736, **d_II**: 0.6298

### Noise Covariance
- **width**: 0.4780
- **σ_E**: 6.3992
- **σ_I**: 3.5335
- **ρ**: 0.9929

## 💡 Notas Importantes

1. **Duración razonable de entrenamientos**:
   - Los entrenamientos deben ser suficientemente largos para convergencia significativa
   - Pero no tan largos que desperdicien tiempo computacional
   - 250 iteraciones por stage es un buen balance para pruebas
   - 500+ iteraciones para entrenamientos finales

2. **Variabilidad entre corridas**:
   - Es esperado que haya variación entre corridas (diferentes seeds)
   - La desviación estándar indica robustez del método
   - Valores consistentemente alejados de referencia pueden indicar problemas

3. **Convergencia**:
   - Stage 1 optimiza W (connectivity matrix)
   - Stage 2 optimiza Σ_η (noise covariance)
   - Ambos stages deben converger para un entrenamiento exitoso

4. **Reutilización de datos**:
   - Los archivos .npy pueden ser reutilizados para diferentes análisis
   - No es necesario re-entrenar para cambiar visualizaciones
   - Facilita comparaciones entre diferentes configuraciones

## 🔗 Referencias

- Paper: Echeveste, R., Aitchison, L., Hennequin, G., & Lengyel, M. (2020).
  "Cortical-like dynamics in recurrent circuits optimized for sampling-based probabilistic inference"
  Nature Neuroscience, 23(9), 1138-1149.

- Código original: https://github.com/roxana-zeraati/SSN-inference

## 📝 Ejemplo de Uso Completo

```bash
# 1. Entrenar (desde figuras/data/training/)
workon tesis
python3 entrenar_parametros.py --n_runs 5 --max_iter_stage1 250 --max_iter_stage2 250

# 2. Visualizar (desde figuras/Parametros/)
cd ../../Parametros
python3 graficar_parametros.py

# 3. Ver figuras generadas
ls -lh *.png
```

## ❓ Troubleshooting

**Problema:** Script de entrenamiento falla con "No module named 'skneuromsi'"
- **Solución:** Asegurarse de que scikit-neuromsi está en el PYTHONPATH o ejecutar desde el entorno virtual correcto

**Problema:** "No se encontraron archivos de resultados"
- **Solución:** Verificar que los archivos .npy están en figuras/data/training/

**Problema:** Stage 1 o Stage 2 no converge
- **Solución:** Aumentar max_iter_stage1/stage2 o revisar parámetros del modelo GSM

**Problema:** Parámetros muy alejados de referencia
- **Solución:**
  - Verificar que GSM usa filtros pre-entrenados (use_pretrained=True)
  - Revisar configuración de lambdas en stage1_params
  - Considerar más iteraciones de optimización
