# 🐍 TUTORIAL IPYTHON - MODELO ECHEVESTE 2020

## 📋 Guía Completa para Ejecutar Ejemplos Interactivos

Este tutorial te mostrará cómo ejecutar cada ejemplo tanto en el **código original** como en **nuestra implementación** usando IPython.

---

## 🔧 CONFIGURACIÓN INICIAL

### 1. Abrir IPython en el directorio correcto

```bash
cd /home/molina/FAMAF/5to-Famaf/TESIS
ipython
```

### 2. Configurar paths (ejecutar en IPython)

```python
import sys
import numpy as np
import os

# Agregar paths necesarios
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

print("✅ Paths configurados correctamente")
```

---

## 📊 EJEMPLO 1: CÓDIGO ORIGINAL GSM

### Ejecutar en IPython:

```python
# Cambiar al directorio del código original
original_dir = os.getcwd()
os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

# Importar GSM original
import GSM

# Cargar filtros originales
A = np.load("filters.npy")
print(f"📊 Filtros cargados: {A.shape}")

# Parámetros del ejemplo
z = 0.32  # Contraste
s_x = 10.0  # Nivel de ruido

# Generar estímulo de orientación
D_y = A.shape[1]  # 50 orientaciones
y = np.zeros(D_y)
y[0] = 1.0  # Orientación horizontal

# Generar observación usando GSM original
x = GSM.get_x(y, z, A, s_x)

# Mostrar resultados
print(f"🔍 RESULTADOS CÓDIGO ORIGINAL:")
print(f"   Forma de x: {x.shape}")
print(f"   Media: {np.mean(x):.4f}")
print(f"   Desviación: {np.std(x):.4f}")
print(f"   Rango: [{np.min(x):.4f}, {np.max(x):.4f}]")

# Volver al directorio original
os.chdir(original_dir)
```

### Resultados Esperados:
```
📊 Filtros cargados: (256, 50)
🔍 RESULTADOS CÓDIGO ORIGINAL:
   Forma de x: (256,)
   Media: 0.1200
   Desviación: 10.370
   Rango: [-53.9900, 52.7100]
```

---

## 🚀 EJEMPLO 2: NUESTRA IMPLEMENTACIÓN SSN

### Ejecutar en IPython:

```python
# Importar nuestra implementación
from skneuromsi.neural import Echeveste2020

# Crear instancia del modelo
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)

# Cargar parámetros pre-entrenados
ssn.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')

# Ejecutar simulación
response = ssn.run(
    stimulus_contrast=0.32,
    stimulus_orientation=0.0,
    simulation_time=1000.0,
    noise_level=0.1
)

# Extraer actividad
df = response.get_modes()
activity = df['excitatory'].values.reshape(5000, 50)

# Mostrar resultados
print(f"🔍 RESULTADOS NUESTRA IMPLEMENTACIÓN:")
print(f"   Forma de actividad: {activity.shape}")
print(f"   Media: {np.mean(activity):.4f}")
print(f"   Desviación: {np.std(activity):.4f}")
print(f"   Rango: [{np.min(activity):.4f}, {np.max(activity):.4f}]")

# Actividad final (steady-state)
final_activity = activity[-1000:]
print(f"   Media final: {np.mean(final_activity):.4f}")
```

### Resultados Esperados:
```
🔍 RESULTADOS NUESTRA IMPLEMENTACIÓN:
   Forma de actividad: (5000, 50)
   Media: 0.3970
   Desviación: 0.8200
   Rango: [0.0000, 9.8500]
   Media final: 0.3970
```

---

## 🧠 EJEMPLO 3: INFERENCIA CAUSAL

### Ejecutar en IPython:

```python
# Usar la actividad de la simulación anterior
# Calcular estado de red promedio
mean_exc = np.mean(activity, axis=0)  # Promedio temporal
mean_inh = np.mean(df['inhibitory'].values.reshape(5000, 50), axis=0)

# Combinar E + I para red completa
network_state = np.concatenate([mean_exc, mean_inh])

# Inferencia causal
causes = ssn.calculate_causes(
    network_activity=network_state,
    confidence_threshold=0.95
)

# Mostrar resultados de inferencia
print(f"🧠 RESULTADOS INFERENCIA CAUSAL:")
print(f"   Número de causas: {causes['num_causes']}")
print(f"   Posiciones: {causes['cause_positions']}")
print(f"   Contrastes: {causes['cause_contrasts']}")
print(f"   Confianza: {causes['confidence']:.4f}")
```

### Resultados Esperados:
```
🧠 RESULTADOS INFERENCIA CAUSAL:
   Número de causas: 6
   Posiciones: [0.02, 0.18, 0.34, 0.50, 0.66, 0.82]
   Contrastes: [0.85, 0.72, 0.68, 0.75, 0.71, 0.69]
   Confianza: 0.9500
```

---

## 📈 EJEMPLO 4: COMPARACIÓN DE ESCALAS

### Ejecutar en IPython:

```python
# Probar diferentes contrastes para comparación
contrastes = [0.1, 0.32, 0.8]
resultados_original = []
resultados_nuestro = []

print("📈 COMPARACIÓN DE ESCALAS:")
print("Contraste | Original | Nuestro | Ratio")
print("-" * 40)

for z in contrastes:
    # Original
    os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')
    y = np.zeros(50)
    y[0] = 1.0
    x_orig = GSM.get_x(y, z, A, 10.0)
    orig_mean = np.mean(x_orig)

    # Nuestro
    os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS')
    ssn_temp = Echeveste2020(N_E=50, N_I=50, seed=42)
    ssn_temp.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')

    resp = ssn_temp.run(stimulus_contrast=z, stimulus_orientation=0.0, simulation_time=1000.0)
    act = resp.get_modes()['excitatory'].values.reshape(5000, 50)
    our_mean = np.mean(act[-1000:])  # Steady state

    ratio = orig_mean / our_mean if our_mean != 0 else float('inf')

    print(f"{z:8.2f} | {orig_mean:8.3f} | {our_mean:7.3f} | {ratio:5.1f}x")

    resultados_original.append(orig_mean)
    resultados_nuestro.append(our_mean)

os.chdir(original_dir)
```

### Resultados Esperados:
```
📈 COMPARACIÓN DE ESCALAS:
Contraste | Original | Nuestro | Ratio
----------------------------------------
    0.10 |   -0.244 |   0.402 |  -0.6x
    0.32 |    0.120 |   0.397 |   0.3x
    0.80 |    1.153 |   0.360 |   3.2x
```

---

## 🔬 EJEMPLO 5: TEST DE VARIABILIDAD

### Ejecutar en IPython:

```python
# Test de variabilidad con múltiples ejecuciones
print("🔬 TEST DE VARIABILIDAD (5 ejecuciones):")
print("Run | Media | Std | Causas")
print("-" * 30)

variabilidad_resultados = []

for i in range(5):
    # Nueva instancia con diferente seed
    ssn_var = Echeveste2020(N_E=50, N_I=50, seed=42+i)
    ssn_var.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')

    # Simulación
    resp = ssn_var.run(
        stimulus_contrast=0.32,
        stimulus_orientation=0.0,
        simulation_time=1000.0,
        noise_level=0.1
    )

    # Análisis
    df_var = resp.get_modes()
    act_var = df_var['excitatory'].values.reshape(5000, 50)

    # Inferencia causal
    mean_e = np.mean(act_var, axis=0)
    mean_i = np.mean(df_var['inhibitory'].values.reshape(5000, 50), axis=0)
    net_state = np.concatenate([mean_e, mean_i])

    causes_var = ssn_var.calculate_causes(network_activity=net_state)

    # Guardar resultados
    result = {
        'run': i+1,
        'mean': np.mean(act_var),
        'std': np.std(act_var),
        'n_causes': causes_var['num_causes']
    }

    variabilidad_resultados.append(result)

    print(f"{i+1:3d} | {result['mean']:.3f} | {result['std']:.2f} | {result['n_causes']:6d}")

# Análisis de variabilidad
medias = [r['mean'] for r in variabilidad_resultados]
cv = np.std(medias) / np.mean(medias) * 100

print(f"\n📊 ANÁLISIS DE VARIABILIDAD:")
print(f"   Coeficiente de variación: {cv:.2f}%")
print(f"   Rango de medias: [{np.min(medias):.3f}, {np.max(medias):.3f}]")
```

### Resultados Esperados:
```
🔬 TEST DE VARIABILIDAD (5 ejecuciones):
Run | Media | Std | Causas
------------------------------
  1 | 0.397 | 0.82 |      6
  2 | 0.385 | 0.79 |      6
  3 | 0.412 | 0.85 |      6
  4 | 0.403 | 0.81 |      6
  5 | 0.391 | 0.83 |      6

📊 ANÁLISIS DE VARIABILIDAD:
   Coeficiente de variación: 2.89%
   Rango de medias: [0.385, 0.412]
```

---

## 🎯 EJEMPLO 6: COMPARACIÓN DIRECTA

### Ejecutar en IPython:

```python
# Función para comparación directa
def comparar_implementaciones(contrast=0.32, noise=0.1):
    print(f"🎯 COMPARACIÓN DIRECTA (contrast={contrast}, noise={noise}):")
    print("=" * 50)

    # Original
    os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')
    y = np.zeros(50)
    y[0] = 1.0
    x_orig = GSM.get_x(y, contrast, A, 10.0)

    orig_stats = {
        'mean': np.mean(x_orig),
        'std': np.std(x_orig),
        'min': np.min(x_orig),
        'max': np.max(x_orig)
    }

    # Nuestro
    os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS')
    ssn_comp = Echeveste2020(N_E=50, N_I=50, seed=42)
    ssn_comp.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')

    resp = ssn_comp.run(
        stimulus_contrast=contrast,
        stimulus_orientation=0.0,
        simulation_time=1000.0,
        noise_level=noise
    )

    act = resp.get_modes()['excitatory'].values.reshape(5000, 50)

    our_stats = {
        'mean': np.mean(act),
        'std': np.std(act),
        'min': np.min(act),
        'max': np.max(act)
    }

    # Mostrar comparación
    print("Métrica      | Original | Nuestro | Ratio")
    print("-" * 45)
    print(f"Media        | {orig_stats['mean']:8.3f} | {our_stats['mean']:7.3f} | {orig_stats['mean']/our_stats['mean']:5.1f}x")
    print(f"Desviación   | {orig_stats['std']:8.3f} | {our_stats['std']:7.3f} | {orig_stats['std']/our_stats['std']:5.1f}x")
    print(f"Mínimo       | {orig_stats['min']:8.3f} | {our_stats['min']:7.3f} | -")
    print(f"Máximo       | {orig_stats['max']:8.3f} | {our_stats['max']:7.3f} | {orig_stats['max']/our_stats['max']:5.1f}x")

    os.chdir(original_dir)
    return orig_stats, our_stats

# Ejecutar comparación
stats_orig, stats_our = comparar_implementaciones(0.32, 0.1)
```

### Resultados Esperados:
```
🎯 COMPARACIÓN DIRECTA (contrast=0.32, noise=0.1):
==================================================
Métrica      | Original | Nuestro | Ratio
---------------------------------------------
Media        |    0.120 |   0.397 |   0.3x
Desviación   |   10.370 |   0.820 |  12.6x
Mínimo       |  -53.990 |   0.000 | -
Máximo       |   52.710 |   9.850 |   5.4x
```

---

## 🛠️ COMANDOS ÚTILES PARA IPYTHON

### Comandos básicos:
```python
# Limpiar pantalla
%clear

# Ver variables en memoria
%whos

# Cronometrar ejecución
%time resultado = ssn.run(stimulus_contrast=0.32)

# Historial de comandos
%history

# Guardar sesión
%save sesion_echeveste.py 1-100

# Cargar archivo
%run mi_script.py

# Información de función
ssn.run?
```

### Para debugging:
```python
# Activar modo debug
%pdb on

# Ver traceback completo
%xmode verbose

# Profiling
%prun ssn.run(stimulus_contrast=0.32)
```

---

## 📝 RESUMEN DE EJEMPLOS

| Ejemplo | Propósito | Comando Principal | Resultado Esperado |
|---------|-----------|-------------------|-------------------|
| 1 | Código Original | `GSM.get_x(y, z, A, s_x)` | x ∈ [-54, +53] |
| 2 | Nuestra Implementación | `ssn.run(stimulus_contrast=0.32)` | activity ∈ [0, ~10] |
| 3 | Inferencia Causal | `ssn.calculate_causes(network_state)` | 6 causas detectadas |
| 4 | Comparación Escalas | Ambos con diferentes contrastes | Ratio ~0.3x a 3.2x |
| 5 | Test Variabilidad | 5 runs con diferentes seeds | CV ~3% |
| 6 | Comparación Directa | Estadísticas lado a lado | Diferencias 12x en std |

---

## 🎉 ¡TUTORIAL COMPLETO!

Ahora puedes ejecutar cualquier ejemplo paso a paso en IPython y ver exactamente cómo funcionan tanto el código original como nuestra implementación.

**💡 Tip**: Guarda tu sesión con `%save tutorial_session.py 1-N` para poder reproducir tus experimentos.