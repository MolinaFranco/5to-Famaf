"""
Ejecutar el código GSM de Echeveste y analizar los resultados.
"""
import numpy as np
import sys
import os

# Cambiar al directorio de Echeveste para que encuentre filters.npy
os.chdir('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

# Importar y ejecutar
from GSM import main

print("Ejecutando GSM.main() de Echeveste...")
main()

# Analizar los resultados guardados
results_location = "bumps/no_noise/results"

print("\n" + "="*60)
print("ANÁLISIS DE RESULTADOS")
print("="*60)

# Cargar z_array
z_array = np.loadtxt(results_location + "/z_array")
print(f"\nContraste (z_array): {z_array}")

n_targets = len(z_array)

# Analizar std para cada contraste
print("\n--- Análisis de std_post ---")
for alpha in range(n_targets):
    z = z_array[alpha]

    # std con z conocido
    std_true_z = np.loadtxt(results_location + f"/std_true_z_{alpha}")

    # std con z_MAP
    std_z_MAP = np.loadtxt(results_location + f"/std_z_MAP_{alpha}")

    # std con full inference
    std_full_inf = np.loadtxt(results_location + f"/std_full_inf_{alpha}")

    # mu con z conocido
    mu_true_z = np.loadtxt(results_location + f"/mu_true_z_{alpha}")

    print(f"\nz={z}:")
    print(f"  std_true_z: min={std_true_z.min():.4f}, max={std_true_z.max():.4f}, "
          f"extremos={std_true_z[0]:.4f}, centro={std_true_z[len(std_true_z)//2]:.4f}")
    print(f"  std_z_MAP:  min={std_z_MAP.min():.4f}, max={std_z_MAP.max():.4f}")
    print(f"  std_full:   min={std_full_inf.min():.4f}, max={std_full_inf.max():.4f}")
    print(f"  mu_true_z:  min={mu_true_z.min():.4f}, max={mu_true_z.max():.4f}, "
          f"centro={mu_true_z[len(mu_true_z)//2]:.4f}")

# Graficar comparación
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Seleccionar algunos contrastes para visualizar
contrasts_to_show = [1, 2, 3, 4, 5]  # índices 1-5 (z=0.125 a 2.0)

for i, alpha in enumerate(contrasts_to_show):
    if i >= 3:
        break
    z = z_array[alpha]
    mu = np.loadtxt(results_location + f"/mu_true_z_{alpha}")
    std = np.loadtxt(results_location + f"/std_true_z_{alpha}")

    # Gráfico de mean
    ax = axes[0, i]
    ax.plot(mu, 'g-', linewidth=2)
    ax.set_title(f'Mean posterior (z={z})')
    ax.set_xlabel('Neurona')
    ax.set_ylabel('μ')

    # Gráfico de std
    ax = axes[1, i]
    ax.plot(std, 'g-', linewidth=2)
    ax.set_title(f'Std posterior (z={z})')
    ax.set_xlabel('Neurona')
    ax.set_ylabel('σ')

plt.tight_layout()
plt.savefig('/home/molina/FAMAF/5to-Famaf/TESIS/figures/data/NarrowFig/echeveste_gsm_results.png',
            dpi=150, bbox_inches='tight')
print(f"\nFigura guardada en echeveste_gsm_results.png")

# También mostrar el resumen general
results = np.loadtxt(results_location + "/mu_and_std_vs_contrast_gsm")
print("\n--- Resumen (mu_and_std_vs_contrast_gsm) ---")
print("Columnas: z_true, z_MAP, avg_mu_true_z, max_mu_true_z, avg_std_true_z, "
      "avg_mu_z_MAP, max_mu_z_MAP, avg_std_z_MAP, avg_mu_full, max_mu_full, avg_std_full")
for row in results:
    print(f"z={row[0]:.3f}: avg_std_true_z={row[4]:.4f}, avg_std_full={row[10]:.4f}")
