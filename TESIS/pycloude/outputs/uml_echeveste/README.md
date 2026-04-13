# Diagramas UML - Echeveste et al. (2020) Implementation

Diagramas UML que documentan la implementación del modelo SSN de Echeveste et al. (2020) en `scikit-neuromsi`.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `01_class_diagram_main.mmd` | Diagrama de clases principal con `Echeveste2020`, `GSM`, `SSNIntegrator` y Data Loaders |
| `02_training_functions.mmd` | Funciones de entrenamiento JAX en `_echeveste_training.py` |
| `03_component_diagram.mmd` | Arquitectura de módulos y dependencias |
| `04_sequence_training.mmd` | Flujo de entrenamiento en dos etapas (Stage 1 + Stage 2) |
| `05_gsm_data_flow.mmd` | Flujo de datos del modelo GSM: generación de estímulos → input h |
| `06_adf_equations.mmd` | Assumed Density Filtering (ADF): ecuaciones 17-20 del paper |
| `07_visualization_module.mmd` | Módulo de visualización de dinámicas |

## Cómo visualizar los diagramas

### Opción 1: Mermaid Live Editor (Online)
1. Ir a https://mermaid.live/
2. Copiar el contenido del archivo `.mmd`
3. El diagrama se renderiza automáticamente
4. Exportar como PNG, SVG o PDF

### Opción 2: VS Code con extensión
1. Instalar extensión "Mermaid Markdown Syntax Highlighting" o "Markdown Preview Mermaid Support"
2. Abrir el archivo `.mmd`
3. Usar preview (Ctrl+Shift+V)

### Opción 3: Generar imágenes con CLI
```bash
# Instalar mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Generar PNG
mmdc -i 01_class_diagram_main.mmd -o 01_class_diagram_main.png -b transparent

# Generar SVG (mejor calidad)
mmdc -i 01_class_diagram_main.mmd -o 01_class_diagram_main.svg -b transparent

# Generar todos los diagramas
for f in *.mmd; do mmdc -i "$f" -o "${f%.mmd}.png" -b transparent; done
```

### Opción 4: Python con pyppeteer
```python
# pip install mermaid-py
from mermaid import Mermaid

with open("01_class_diagram_main.mmd", "r") as f:
    diagram = f.read()

m = Mermaid(diagram)
m.to_png("01_class_diagram_main.png")
```

## Estructura del código documentado

```
scikit-neuromsi/skneuromsi/
├── core/
│   ├── modelabc.py          # SKNMSIMethodABC (clase base)
│   └── ndresult.py          # NDResult (contenedor de resultados)
├── neural/
│   ├── _echeveste2020.py    # Clase principal Echeveste2020 + SSNIntegrator
│   ├── _echeveste_training.py  # Funciones JAX para entrenamiento
│   └── _echeveste_visualization.py  # Funciones de visualización
├── generative/
│   └── _gsm.py              # Modelo GSM (Gaussian Scale Mixture)
└── data/
    ├── echeveste_data_loader.py  # Carga parámetros SSN preentrenados
    ├── gsm_data_loader.py        # Carga datos GSM
    ├── echeveste2020/            # Archivos de parámetros SSN
    └── gsm/                      # Archivos de datos GSM
```

## Referencias

- Echeveste, R., Aitchison, L., Hennequin, G., & Lengyel, M. (2020).
  *Cortical-like dynamics in recurrent circuits optimized for sampling-based probabilistic inference.*
  Nature Neuroscience, 23(9), 1138-1149.

- Código original OCaml: `ssn_inference_optimizer/`
- Código original Python: `ssn_inference_numerical_experiments/`
