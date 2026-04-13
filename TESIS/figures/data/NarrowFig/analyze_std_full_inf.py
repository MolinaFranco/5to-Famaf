"""
Analizar detalladamente std_full_inf de Echeveste para ver si tiene forma de U.
"""
import numpy as np
import matplotlib.pyplot as plt

# Directorio de resultados de Echeveste
results_location = "/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM/bumps/no_noise/results"

z_array = np.loadtxt(results_location + "/z_array")
print(f"Contrastes: {z_array}")

n_targets = len(z_array)
D_y = 50  # Número de neuronas

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Títulos para las columnas
axes[0, 0].set_title('Mean posterior', fontsize=14)
axes[0, 1].set_title('Std posterior (full inference)', fontsize=14)

# Colores para diferentes contrastes
colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_targets))

# Subplot 1: Todas las medias superpuestas
ax = axes[0, 0]
for alpha in range(1, n_targets):  # Saltar z=0
    mu = np.loadtxt(results_location + f"/mu_true_z_{alpha}")
    ax.plot(mu, color=colors[alpha], label=f'z={z_array[alpha]}', linewidth=2)
ax.set_xlabel('Neurona/Orientación')
ax.set_ylabel('μ')
ax.legend()
ax.set_xlim(0, D_y-1)

# Subplot 2: Todas las std (full inf) superpuestas
ax = axes[0, 1]
for alpha in range(1, n_targets):  # Saltar z=0
    std = np.loadtxt(results_location + f"/std_full_inf_{alpha}")
    ax.plot(std, color=colors[alpha], label=f'z={z_array[alpha]}', linewidth=2)
ax.set_xlabel('Neurona/Orientación')
ax.set_ylabel('σ')
ax.legend()
ax.set_xlim(0, D_y-1)

# Comparar std_true_z vs std_full_inf para z=1.0
ax = axes[1, 0]
alpha = 4  # z=1.0
std_true = np.loadtxt(results_location + f"/std_true_z_{alpha}")
std_full = np.loadtxt(results_location + f"/std_full_inf_{alpha}")
ax.plot(std_true, 'b-', label='std_true_z (z conocido)', linewidth=2)
ax.plot(std_full, 'g-', label='std_full_inf', linewidth=2)
ax.set_xlabel('Neurona/Orientación')
ax.set_ylabel('σ')
ax.set_title(f'Comparación std para z=1.0')
ax.legend()
ax.set_xlim(0, D_y-1)

# Lo mismo para z=2.0
ax = axes[1, 1]
alpha = 5  # z=2.0
std_true = np.loadtxt(results_location + f"/std_true_z_{alpha}")
std_full = np.loadtxt(results_location + f"/std_full_inf_{alpha}")
ax.plot(std_true, 'b-', label='std_true_z (z conocido)', linewidth=2)
ax.plot(std_full, 'g-', label='std_full_inf', linewidth=2)
ax.set_xlabel('Neurona/Orientación')
ax.set_ylabel('σ')
ax.set_title(f'Comparación std para z=2.0')
ax.legend()
ax.set_xlim(0, D_y-1)

# Ver si la diferencia entre extremos y centro es significativa
ax = axes[2, 0]
for alpha in range(1, n_targets):
    std_full = np.loadtxt(results_location + f"/std_full_inf_{alpha}")
    # Normalizar para ver mejor la forma
    std_norm = (std_full - std_full.min()) / (std_full.max() - std_full.min() + 1e-10)
    ax.plot(std_norm, color=colors[alpha], label=f'z={z_array[alpha]}', linewidth=2)
ax.set_xlabel('Neurona/Orientación')
ax.set_ylabel('σ (normalizado)')
ax.set_title('Std normalizado (full inf) - forma relativa')
ax.legend()
ax.set_xlim(0, D_y-1)

# Mostrar la forma de la media para comparar con std
ax = axes[2, 1]
for alpha in range(1, n_targets):
    mu = np.loadtxt(results_location + f"/mu_true_z_{alpha}")
    std_full = np.loadtxt(results_location + f"/std_full_inf_{alpha}")
    # Invertir la std para ver si coincide con la forma de la media
    std_inv = std_full.max() - std_full
    ax.plot(std_inv / std_inv.max() if std_inv.max() > 0 else std_inv,
            color=colors[alpha], linestyle='--', alpha=0.7)
    ax.plot(mu / mu.max() if mu.max() > 0 else mu,
            color=colors[alpha], linewidth=2)
ax.set_xlabel('Neurona/Orientación')
ax.set_ylabel('Normalizado')
ax.set_title('Mean (sólido) vs -Std invertido (punteado)')
ax.set_xlim(0, D_y-1)

plt.tight_layout()
plt.savefig('/home/molina/FAMAF/5to-Famaf/TESIS/figures/data/NarrowFig/echeveste_std_analysis.png',
            dpi=150, bbox_inches='tight')
print("Figura guardada en echeveste_std_analysis.png")

# Imprimir valores específicos
print("\n=== Valores de std_full_inf en extremos vs centro ===")
for alpha in range(1, n_targets):
    std_full = np.loadtxt(results_location + f"/std_full_inf_{alpha}")
    center = D_y // 2
    print(f"z={z_array[alpha]:.3f}: extremo[0]={std_full[0]:.4f}, "
          f"centro[{center}]={std_full[center]:.4f}, "
          f"extremo[-1]={std_full[-1]:.4f}, "
          f"diferencia (extremo-centro)={(std_full[0] - std_full[center]):.4f}")
