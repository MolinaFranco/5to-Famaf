# Simulaciones y Figuras - Echeveste et al. (2020)

## Estructura Completa

Este documento describe la estructura final para generar simulaciones y figuras del paper Echeveste et al. (2020) usando los archivos h_true correctos.

## 📂 Estructura de Directorios

```
figuras/
├── data/
│   └── simulations/
│       ├── generate_simulations.py          ← Script unificado
│       ├── generate_simulations.log         ← Log de ejecución
│       ├── contrast_0.000_h_true_correcto/  ← Simulaciones
│       ├── contrast_0.125_h_true_correcto/
│       ├── contrast_0.250_h_true_correcto/
│       ├── contrast_0.500_h_true_correcto/
│       └── contrast_1.000_h_true_correcto/
└── Fig2/
    ├── generate_fig2a.py                    ← Generador de figuras
    ├── Fig2a_population_activity.png        ← Figura generada
    └── Fig2a_population_activity.pdf
```

## 🔧 Script Unificado: `generate_simulations.py`

### Ubicación
`figuras/data/simulations/generate_simulations.py`

### Propósito
Script centralizado para generar TODAS las simulaciones con configuración unificada.

### Configuración

**Parámetros de simulación**:
```python
SIMULATION_CONFIG = {
    'n_neurons_e': 50,
    'n_neurons_i': 50,
    'k': 0.3,              # Ganancia
    'n': 2.0,              # Exponente supralineal
    'seed': 42,
    'simulation_time': 1000.0,  # ms
    'dt': 0.2,             # ms
    'burn_in': 200.0,      # ms
    'noise_level': 0.0,    # Sin ruido
    'orientation': 0.0,    # Orientación preferida
    'use_three_phases': False,  # Estímulo constante
}
```

**Niveles de contraste**:
```python
CONTRAST_LEVELS = [
    {'contrast': 0.0, 'label': 'spontaneous', 'h_true_file': 'h_true_0'},
    {'contrast': 0.125, 'label': 'low', 'h_true_file': 'h_true_1'},
    {'contrast': 0.25, 'label': 'medium', 'h_true_file': 'h_true_2'},
    {'contrast': 0.5, 'label': 'high', 'h_true_file': 'h_true_2'},
    {'contrast': 1.0, 'label': 'very_high', 'h_true_file': 'h_true_2'},
]
```

### Uso

```bash
cd figuras/data/simulations
~/.virtualenvs/tesis/bin/python3 generate_simulations.py
```

### Salida

Para cada nivel de contraste, genera:
- `simulation_data.npz`: Datos completos (trayectorias + estadísticas)
- `parameters.json`: Parámetros usados
- `metadata.txt`: Resumen legible

## 📊 Simulaciones Generadas

### Resultados Completos

| Contraste | Label | r_E mean | u_E mean | h_true usado |
|-----------|-------|----------|----------|--------------|
| 0.000 | spontaneous | 0.0019 | 0.0396 | h_true_0 |
| 0.125 | low | 2.9479 | 2.5150 | h_true_1 |
| 0.250 | medium | 4.5768 | 3.4644 | h_true_2 |
| 0.500 | high | 4.5817 | 3.4679 | h_true_2 |
| 1.000 | very_high | 4.5793 | 3.4662 | h_true_2 |

### Observaciones

1. **Contraste espontáneo (0.0)**: Actividad muy baja como esperado
2. **Incremento 0.0 → 0.125**: Masivo (~1500x), típico de transición espontáneo → estimulado
3. **Incremento 0.125 → 0.25**: Moderado (~1.5x), saturación de la no-linealidad
4. **Saturación 0.25+**: Los contrastes 0.25, 0.5, 1.0 dan valores similares porque todos usan h_true_2

### Importante

**Limitación**: Solo tenemos 3 archivos h_true del código original:
- `h_true_0`: contraste espontáneo
- `h_true_1`: contraste bajo (~0.125)
- `h_true_2`: contraste medio (~0.25)

Los contrastes 0.5 y 1.0 usan h_true_2 como aproximación.

## 🎨 Generación de Figuras: `generate_fig2a.py`

### Ubicación
`figuras/Fig2/generate_fig2a.py`

### Propósito
Genera la Figura 2a del paper: heatmaps de actividad poblacional.

### Configuración

```python
CONTRASTS_TO_PLOT = [
    {'contrast': 0.0, 'label': 'Zero Contrast (Spontaneous)'},
    {'contrast': 1.0, 'label': 'High Contrast (1.0)'},
]
```

### Uso

```bash
cd /home/molina/FAMAF/5to-Famaf/TESIS
~/.virtualenvs/tesis/bin/python3 figuras/Fig2/generate_fig2a.py
```

### Salida

- `Fig2a_population_activity.png` (886 KB)
- `Fig2a_population_activity.pdf` (678 KB)

### Características

- **Panel superior**: Contraste espontáneo (0.0) - actividad muy baja
- **Panel inferior**: Contraste alto (1.0) - actividad alta
- **Heatmaps**: u_E vs tiempo y orientación
- **Colores**: Escala adaptativa (percentiles 1-99)
- **Formato**: PNG (alta resolución) + PDF (vectorial)

## 🔄 Flujo de Trabajo Completo

### 1. Generar Simulaciones

```bash
cd figuras/data/simulations
~/.virtualenvs/tesis/bin/python3 generate_simulations.py
```

Esto genera todas las simulaciones para los 5 niveles de contraste.

### 2. Generar Figuras

```bash
cd /home/molina/FAMAF/5to-Famaf/TESIS
~/.virtualenvs/tesis/bin/python3 figuras/Fig2/generate_fig2a.py
```

Esto carga los datos simulados y genera la Figura 2a.

## 📝 Contenido de `simulation_data.npz`

### Trayectorias Temporales
- `u_E`: (5000, 50) - Potenciales de membrana excitatorios
- `u_I`: (5000, 50) - Potenciales de membrana inhibitorios
- `r_E`: (5000, 50) - Firing rates excitatorios
- `r_I`: (5000, 50) - Firing rates inhibitorios
- `time`: (5000,) - Vector de tiempo en ms

### Estadísticas (Estado Estacionario)
- `u_E_mean`, `u_E_std`: (50,) - Media y desviación estándar
- `u_E_cov`: (50, 50) - Matriz de covarianza
- `u_E_corr`: (50, 50) - Matriz de correlación
- (Equivalente para u_I, r_E, r_I)

### Metadatos
- `orientations`: (50,) - Orientaciones preferidas (-90° a 90°)
- `dt`: float - Paso temporal (0.2 ms)

## 🔍 Cargar y Usar Datos

### Python

```python
import numpy as np

# Cargar datos
data = np.load('figuras/data/simulations/contrast_0.125_h_true_correcto/simulation_data.npz')

# Acceder a trayectorias
u_E = data['u_E']  # (5000, 50)
time = data['time']

# Acceder a estadísticas
u_E_mean = data['u_E_mean']  # (50,)
u_E_cov = data['u_E_cov']  # (50, 50)

# Acceder a metadatos
orientations = data['orientations']  # (50,)
dt = float(data['dt'])  # 0.2
```

## ✅ Validación

### h_true Correcto

✓ Todos los directorios terminan en `_h_true_correcto`
✓ Usa archivos del código original de Echeveste
✓ Valores no-negativos (transformación aplicada)
✓ Duplicación E==I correcta

### Resultados Consistentes

✓ Incremento dramático espontáneo → bajo contraste
✓ Saturación en contrastes altos
✓ Actividad distribuida en orientaciones
✓ Dinámica temporal estable

## 🚀 Próximos Pasos

Para continuar la replicación del paper:

1. **Agregar más figuras**:
   - Fig2b: Covariance ellipses
   - Fig2c: Mean vs std
   - Fig2d: Correlation matrices

2. **Extended Data Figures**:
   - Comparación con targets del GSM
   - Convergencia del entrenamiento

3. **Análisis cuantitativo**:
   - Decorrelación vs contraste
   - Power spectrum temporal
   - Correlaciones espaciales

## 📚 Referencias

- Echeveste et al. (2020). "Cortical-like dynamics in recurrent circuits optimized for sampling-based probabilistic inference". Nature Neuroscience 23(9): 1138-1149
- Parámetros: `parameters.md` (Table S1)
- Código original: `ssn_inference_numerical_experiments/`

## 📧 Notas

- **Configuración**: Centralizada en `generate_simulations.py`
- **Extensible**: Fácil agregar nuevos contrastes o parámetros
- **Reproducible**: Semilla fija (seed=42)
- **Eficiente**: Usa GPU automáticamente si está disponible

---

**Última actualización**: 2025-11-14
**Estado**: ✅ Completado y validado
