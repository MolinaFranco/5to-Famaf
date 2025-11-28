# Simulaciones Base - Echeveste et al. (2020)

## Descripción

Este directorio contiene simulaciones pre-computadas con los parámetros exactos de Echeveste et al. (2020).
Las simulaciones pueden ser reutilizadas para generar múltiples figuras sin necesidad de volver a ejecutar.

## Estructura

```
simulations/
├── contrast_0.000/
│   ├── simulation_data.npz  # Datos completos
│   ├── parameters.json      # Parámetros usados
│   └── metadata.txt         # Resumen legible
├── contrast_1.000/
│   ├── simulation_data.npz  # Datos completos
│   ├── parameters.json      # Parámetros usados
│   └── metadata.txt         # Resumen legible
└── README.md                # Este archivo
```

## Contenido de simulation_data.npz

### Trayectorias temporales
- `u_E`: (n_times, 50) - Potenciales de membrana excitatorios
- `u_I`: (n_times, 50) - Potenciales de membrana inhibitorios
- `r_E`: (n_times, 50) - Firing rates excitatorios
- `r_I`: (n_times, 50) - Firing rates inhibitorios
- `time`: (n_times,) - Vector de tiempo en ms

### Estadísticas (período estacionario)
- `u_E_mean`, `u_E_std`: (50,) - Media y desviación estándar
- `u_E_cov`: (50, 50) - Matriz de covarianza
- `u_E_corr`: (50, 50) - Matriz de correlación
- (Equivalente para u_I, r_E, r_I)

### Metadatos
- `orientations`: (50,) - Orientaciones preferidas (-90 a 90°)
- `dt`: float - Paso temporal en ms

## Parámetros de Simulación

- Neuronas: 50E + 50I
- Tiempo de simulación: 1000.0 ms
- Paso temporal: 0.2 ms
- Nivel de ruido: 0.0
- Burn-in: 200.0 ms
- Estímulo: Constante (use_three_phases=False)
- Orientación: 0.0°

## Uso

```python
import numpy as np

# Cargar simulación
data = np.load('contrast_0.000/simulation_data.npz')

# Acceder a datos
u_E = data['u_E']  # (n_times, 50)
mean_u_E = data['u_E_mean']  # (50,)
cov_u_E = data['u_E_cov']  # (50, 50)
```

## Referencias

- Echeveste et al. (2020). Nature Neuroscience 23(9): 1138-1149
- Parámetros: Table S1 (parameters.md)
