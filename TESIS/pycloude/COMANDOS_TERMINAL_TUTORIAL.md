# 💻 COMANDOS TERMINAL - TUTORIAL ECHEVESTE 2020

## 🚀 Guía Completa de Comandos para Terminal y IPython

Esta guía te muestra todos los comandos necesarios para ejecutar los ejemplos desde terminal e IPython.

---

## 🔧 CONFIGURACIÓN INICIAL

### 1. Navegación básica

```bash
# Ir al directorio principal
cd /home/molina/FAMAF/5to-Famaf/TESIS

# Ver contenido del directorio
ls -la

# Ver estructura del proyecto
tree -L 2
```

### 2. Verificar instalaciones

```bash
# Verificar Python
python3 --version

# Verificar IPython
ipython --version

# Verificar Jupyter
jupyter --version

# Verificar numpy
python3 -c "import numpy; print(numpy.__version__)"
```

---

## 🐍 COMANDOS IPYTHON

### Inicio rápido

```bash
# Iniciar IPython en el directorio correcto
cd /home/molina/FAMAF/5to-Famaf/TESIS
ipython
```

### En IPython - Carga de ejemplos rápidos

```python
# Cargar script de ejemplos rápidos
%run ejemplos_rapidos_ipython.py

# Ver ayuda
mostrar_ayuda()

# Ejemplo original
orig = ejemplo_original()

# Ejemplo nuestro
nues = ejemplo_nuestro()

# Comparación
comp = comparacion_rapida()
```

### En IPython - Configuración manual

```python
# Configurar paths
import sys
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

# Imports básicos
import numpy as np
import os
from skneuromsi.neural import Echeveste2020
```

### Comandos útiles IPython

```python
# Limpiar pantalla
%clear

# Ver variables
%whos

# Cronometrar
%time resultado = ejemplo_nuestro()

# Historial
%history

# Guardar sesión
%save mi_sesion.py 1-50

# Cargar script
%load mi_script.py

# Ayuda de función
ejemplo_original?

# Código fuente de función
ejemplo_original??
```

---

## 🖥️ COMANDOS TERMINAL DIRECTOS

### Ejecutar ejemplos desde terminal

```bash
# Ejemplo completo del sistema
python3 ejemplo_sistema_completo_echeveste.py

# Test de parámetros distintos
python3 test_parametros_distintos.py

# Test de variabilidad
python3 test_variabilidad_simple.py

# Calibración de escalas
python3 calibracion_escalas_mejorada.py

# Test de equivalencia
python3 test_equivalencia_parametros.py
```

### Ejecutar con timeouts (para simulaciones largas)

```bash
# Con timeout de 5 minutos
timeout 300 python3 test_parametros_distintos.py

# Con timeout de 10 minutos
timeout 600 python3 calibracion_escalas_mejorada.py

# Sin buffer de salida (ver resultados en tiempo real)
python3 -u test_variabilidad_simple.py
```

### Ejecutar con variables de entorno

```bash
# Configurar PYTHONPATH
export PYTHONPATH="/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi:$PYTHONPATH"

# Ejecutar con PYTHONPATH específico
PYTHONPATH=/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi python3 test_real_case.py

# Configurar variables para BrainPy (si es necesario)
export BRAINPY_DEVICE=cpu
```

---

## 📓 COMANDOS JUPYTER NOTEBOOK

### Iniciar Jupyter

```bash
# Iniciar Jupyter en el directorio
cd /home/molina/FAMAF/5to-Famaf/TESIS
jupyter notebook

# O Jupyter Lab
jupyter lab

# Convertir notebook a script
jupyter nbconvert --to script tutorial_echeveste_interactivo.ipynb
```

### Ejecutar notebook desde terminal

```bash
# Ejecutar notebook completo
jupyter nbconvert --execute --to notebook tutorial_echeveste_interactivo.ipynb

# Ejecutar y guardar como HTML
jupyter nbconvert --execute --to html tutorial_echeveste_interactivo.ipynb
```

---

## 🔬 EJEMPLOS ESPECÍFICOS

### 1. Ejecutar código original

```bash
# En terminal
cd /home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM
python3 -c "
import GSM
import numpy as np
A = np.load('filters.npy')
y = np.zeros(50); y[0] = 1.0
x = GSM.get_x(y, 0.32, A, 10.0)
print(f'Media: {np.mean(x):.4f}')
print(f'Std: {np.std(x):.4f}')
"
```

### 2. Ejecutar nuestra implementación

```bash
# En terminal
cd /home/molina/FAMAF/5to-Famaf/TESIS
python3 -c "
import sys
sys.path.insert(0, 'scikit-neuromsi')
from skneuromsi.neural import Echeveste2020
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
ssn.load_parameters('scikit-neuromsi/skneuromsi/data/echeveste2020/')
response = ssn.run(stimulus_contrast=0.32)
df = response.get_modes()
activity = df['excitatory'].values.reshape(5000, 50)
print(f'Media: {activity.mean():.4f}')
print(f'Std: {activity.std():.4f}')
"
```

### 3. Test rápido de inferencia causal

```bash
# Una línea completa
python3 -c "
import sys, numpy as np
sys.path.insert(0, 'scikit-neuromsi')
from skneuromsi.neural import Echeveste2020
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
ssn.load_parameters('scikit-neuromsi/skneuromsi/data/echeveste2020/')
resp = ssn.run(stimulus_contrast=0.32)
df = resp.get_modes()
exc = df['excitatory'].values.reshape(5000, 50)
inh = df['inhibitory'].values.reshape(5000, 50)
state = np.concatenate([exc.mean(0), inh.mean(0)])
causes = ssn.calculate_causes(network_activity=state)
print(f'Causas: {causes[\"num_causes\"]}')
"
```

---

## 🛠️ COMANDOS DE DEBUGGING

### Verificar instalaciones

```bash
# Verificar scikit-neuromsi
python3 -c "from skneuromsi.neural import Echeveste2020; print('✅ scikit-neuromsi OK')"

# Verificar GSM original
cd /home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM
python3 -c "import GSM; print('✅ GSM original OK')"

# Verificar filtros
python3 -c "import numpy as np; A=np.load('filters.npy'); print(f'Filtros: {A.shape}')"
```

### Debug de errores comunes

```bash
# Error de paths
python3 -c "
import sys
print('Python paths:')
for p in sys.path: print(f'  {p}')
"

# Error de imports
python3 -c "
try:
    from skneuromsi.neural import Echeveste2020
    print('✅ Import OK')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# Error de datos
python3 -c "
import os
data_path = 'scikit-neuromsi/skneuromsi/data/echeveste2020/'
if os.path.exists(data_path):
    files = os.listdir(data_path)
    print(f'✅ Data files: {files}')
else:
    print(f'❌ Data path not found: {data_path}')
"
```

---

## ⚡ SCRIPTS DE AUTOMATIZACIÓN

### Script de test completo

```bash
# Crear script de test
cat > test_completo.sh << 'EOF'
#!/bin/bash
echo "🧪 TEST COMPLETO ECHEVESTE 2020"
echo "================================"

cd /home/molina/FAMAF/5to-Famaf/TESIS

echo "1. Test código original..."
timeout 60 python3 -c "
import sys, os
os.chdir('ssn_inference_numerical_experiments/GSM')
import GSM, numpy as np
A = np.load('filters.npy')
y = np.zeros(50); y[0] = 1.0
x = GSM.get_x(y, 0.32, A, 10.0)
print(f'✅ Original: media={np.mean(x):.4f}')
"

echo "2. Test nuestra implementación..."
timeout 120 python3 -c "
import sys
sys.path.insert(0, 'scikit-neuromsi')
from skneuromsi.neural import Echeveste2020
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
ssn.load_parameters('scikit-neuromsi/skneuromsi/data/echeveste2020/')
resp = ssn.run(stimulus_contrast=0.32)
df = resp.get_modes()
activity = df['excitatory'].values.reshape(5000, 50)
print(f'✅ Nuestro: media={activity.mean():.4f}')
"

echo "3. Test inferencia causal..."
timeout 60 python3 -c "
import sys, numpy as np
sys.path.insert(0, 'scikit-neuromsi')
from skneuromsi.neural import Echeveste2020
ssn = Echeveste2020(N_E=50, N_I=50, seed=42)
ssn.load_parameters('scikit-neuromsi/skneuromsi/data/echeveste2020/')
resp = ssn.run(stimulus_contrast=0.32)
df = resp.get_modes()
exc = df['excitatory'].values.reshape(5000, 50)
inh = df['inhibitory'].values.reshape(5000, 50)
state = np.concatenate([exc.mean(0), inh.mean(0)])
causes = ssn.calculate_causes(network_activity=state)
print(f'✅ Causas: {causes[\"num_causes\"]}')
"

echo "🎉 TEST COMPLETO FINALIZADO"
EOF

# Hacer ejecutable
chmod +x test_completo.sh

# Ejecutar
./test_completo.sh
```

### Script de benchmark

```bash
# Crear script de benchmark
cat > benchmark.sh << 'EOF'
#!/bin/bash
echo "⚡ BENCHMARK ECHEVESTE 2020"
echo "==========================="

cd /home/molina/FAMAF/5to-Famaf/TESIS

echo "Benchmarking..."
time python3 ejemplos_rapidos_ipython.py << 'PYTHON'
benchmark_completo()
exit()
PYTHON

echo "✅ Benchmark completo"
EOF

chmod +x benchmark.sh
./benchmark.sh
```

---

## 📊 COMANDOS DE ANÁLISIS

### Analizar resultados

```bash
# Analizar archivos de output
grep "Media:" *.txt | head -10

# Contar líneas de código
find . -name "*.py" -exec wc -l {} + | sort -n

# Buscar funciones específicas
grep -r "def calculate_causes" --include="*.py"

# Buscar errores en logs
grep -i "error\|exception" *.log
```

### Comparar archivos

```bash
# Comparar outputs
diff output_original.txt output_nuestro.txt

# Comparar con formato
diff -u output_original.txt output_nuestro.txt | head -20
```

---

## 🎯 COMANDOS PARA CASOS ESPECÍFICOS

### Problema: "Could not load pre-trained GSM data"

```bash
# Verificar archivos GSM
ls -la scikit-neuromsi/skneuromsi/data/gsm/
cat scikit-neuromsi/skneuromsi/data/gsm/A | head -5
```

### Problema: Error de dimensiones

```bash
# Debug dimensiones
python3 -c "
import numpy as np
A = np.loadtxt('scikit-neuromsi/skneuromsi/data/gsm/A')
print(f'A shape: {A.shape}')
C = np.loadtxt('scikit-neuromsi/skneuromsi/data/gsm/C')
print(f'C shape: {C.shape}')
"
```

### Problema: Timeout en simulaciones

```bash
# Simulación más corta
python3 -c "
import sys
sys.path.insert(0, 'scikit-neuromsi')
from skneuromsi.neural import Echeveste2020
ssn = Echeveste2020(N_E=25, N_I=25, seed=42)  # Red más pequeña
ssn.load_parameters('scikit-neuromsi/skneuromsi/data/echeveste2020/')
resp = ssn.run(stimulus_contrast=0.32, simulation_time=500.0)  # Tiempo más corto
print('✅ Simulación rápida OK')
"
```

---

## 🔄 COMANDOS DE MANTENIMIENTO

### Limpiar archivos temporales

```bash
# Limpiar .pyc
find . -name "*.pyc" -delete

# Limpiar __pycache__
find . -name "__pycache__" -type d -exec rm -rf {} +

# Limpiar outputs temporales
rm -f *.tmp *.log output_*.txt
```

### Backup de resultados importantes

```bash
# Crear backup
tar -czf backup_echeveste_$(date +%Y%m%d).tar.gz \
  *.py *.md *.ipynb scikit-neuromsi/skneuromsi/data/

# Listar backups
ls -la backup_*.tar.gz
```

---

## 🎉 RESUMEN DE COMANDOS MÁS ÚTILES

```bash
# 🚀 INICIO RÁPIDO
cd /home/molina/FAMAF/5to-Famaf/TESIS
ipython
%run ejemplos_rapidos_ipython.py
ejemplo_original()

# 🔬 TEST COMPLETO
python3 test_parametros_distintos.py

# 🧠 INFERENCIA RÁPIDA
python3 -c "%run ejemplos_rapidos_ipython.py; test_causas()"

# ⚡ BENCHMARK
%run ejemplos_rapidos_ipython.py
benchmark_completo()

# 🎯 COMPARACIÓN
comparacion_rapida()

# 🔄 VARIABILIDAD
test_variabilidad(n_runs=10)
```

---

## 💡 TIPS FINALES

1. **Siempre inicia desde el directorio correcto**: `/home/molina/FAMAF/5to-Famaf/TESIS`
2. **Usa timeouts para simulaciones largas**: `timeout 300 python3 script.py`
3. **Carga ejemplos rápidos para test interactivos**: `%run ejemplos_rapidos_ipython.py`
4. **Guarda sesiones importantes**: `%save mi_sesion.py 1-100`
5. **Usa el notebook para análisis visual**: `jupyter notebook tutorial_echeveste_interactivo.ipynb`

🎉 **¡Ya tienes todas las herramientas para dominar el modelo Echeveste 2020!**