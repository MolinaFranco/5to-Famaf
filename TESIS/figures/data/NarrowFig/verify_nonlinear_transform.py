"""
Verificar que aplicar la transformación no lineal a los momentos del GSM
produce las curvas U-shape en la std como se ve en la figura de Echeveste.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N_exc = 50

def np_th(u):
    """Rectificación (threshold at 0)."""
    return np.maximum(u, 0)

def nl_fun(u, nl_scale, nl_baseline, nl_power):
    """Función no lineal del SSN: rectified power law."""
    return nl_scale * (np_th(u + nl_baseline) ** nl_power)

def get_mean_var_nl(points_1d, mu, std, nl_scale, nl_baseline, nl_power):
    """
    Calcular la media y varianza después de pasar por la no linealidad.

    Para cada neurona i, tenemos y_i ~ N(μ_i, σ_i²).
    Calculamos E[f(y_i)] y Var[f(y_i)] numéricamente.
    """
    new_mean = np.empty(N_exc)
    new_var = np.empty(N_exc)

    for i in range(N_exc):
        mu_i = mu[i]
        std_i = std[i]
        # Puntos reescalados para integrar
        points_resc = std_i * points_1d + mu_i
        # Densidad de probabilidad
        values_dist = norm.pdf(points_resc, loc=mu_i, scale=std_i)
        norm_f = np.sum(values_dist)
        # Valores de la función no lineal
        values_fun = nl_fun(points_resc, nl_scale, nl_baseline, nl_power)
        # Media y varianza
        new_mean[i] = np.sum(values_dist * values_fun) / norm_f
        new_var[i] = np.sum(values_dist * (values_fun - new_mean[i])**2) / norm_f

    return new_mean, np.sqrt(new_var)  # Retornar std, no var


# Parámetros de la no linealidad (del código de Echeveste)
nl_scale = 2.4
nl_power = 0.6
nl_baseline = (3.5 / nl_scale) ** (1.0 / nl_power)
print(f"Parámetros NL: scale={nl_scale}, power={nl_power}, baseline={nl_baseline:.4f}")

# Puntos para integración numérica
points_1d = np.linspace(-4.0, 4.0, num=201)

# Cargar resultados de Echeveste
results_location = ("/home/molina/FAMAF/5to-Famaf/TESIS/"
                    "ssn_inference_numerical_experiments/GSM/bumps/no_noise/results")
z_array = np.loadtxt(results_location + "/z_array")
print(f"Contrastes: {z_array}")

# Procesar cada contraste
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Para comparar con la figura original, usar baseline=3.0 en los targets
baseline = 3.0

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(z_array)))

print("\n=== Resultados después de transformación NL ===")

for alpha in range(1, len(z_array)):  # Saltar z=0
    z = z_array[alpha]

    # Cargar momentos GSM
    mu_gsm = np.loadtxt(results_location + f"/mu_true_z_{alpha}")
    std_gsm = np.loadtxt(results_location + f"/std_true_z_{alpha}")

    # Agregar baseline como hace Echeveste
    mu_gsm_shifted = baseline + mu_gsm

    # Aplicar transformación NL
    mu_nl, std_nl = get_mean_var_nl(points_1d, mu_gsm_shifted, std_gsm,
                                     nl_scale, nl_baseline, nl_power)

    print(f"\nz={z}:")
    print(f"  mu_gsm+baseline: min={mu_gsm_shifted.min():.4f}, "
          f"max={mu_gsm_shifted.max():.4f}, centro={mu_gsm_shifted[N_exc//2]:.4f}")
    print(f"  std_gsm (constante): {std_gsm[0]:.4f}")
    print(f"  mu_nl: min={mu_nl.min():.4f}, max={mu_nl.max():.4f}, "
          f"centro={mu_nl[N_exc//2]:.4f}")
    print(f"  std_nl: min={std_nl.min():.4f}, max={std_nl.max():.4f}, "
          f"extremos={std_nl[0]:.4f}, centro={std_nl[N_exc//2]:.4f}")

    # Gráficos
    # Row 0: Mean
    axes[0, 0].plot(mu_gsm_shifted, color=colors[alpha], label=f'z={z}')
    axes[0, 1].plot(mu_nl, color=colors[alpha], label=f'z={z}')

    # Row 1: Std
    axes[1, 0].plot(std_gsm, color=colors[alpha], label=f'z={z}')
    axes[1, 1].plot(std_nl, color=colors[alpha], label=f'z={z}')

axes[0, 0].set_title('Mean GSM (con baseline)', fontsize=12)
axes[0, 0].set_ylabel('μ')
axes[0, 0].legend()

axes[0, 1].set_title('Mean después de NL', fontsize=12)
axes[0, 1].set_ylabel('E[f(y)]')
axes[0, 1].legend()

axes[1, 0].set_title('Std GSM (constante)', fontsize=12)
axes[1, 0].set_xlabel('Neurona/Orientación')
axes[1, 0].set_ylabel('σ')
axes[1, 0].legend()

axes[1, 1].set_title('Std después de NL (¡U-shape!)', fontsize=12)
axes[1, 1].set_xlabel('Neurona/Orientación')
axes[1, 1].set_ylabel('sqrt(Var[f(y)])')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('/home/molina/FAMAF/5to-Famaf/TESIS/figures/data/NarrowFig/'
            'nonlinear_transform_result.png', dpi=150, bbox_inches='tight')
print("\nFigura guardada en nonlinear_transform_result.png")

# Comparación lado a lado con formato similar a Echeveste
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))

# Colores estilo Echeveste (verdes para posterior)
green_colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(z_array)))

for alpha in range(1, len(z_array)):
    z = z_array[alpha]

    mu_gsm = np.loadtxt(results_location + f"/mu_true_z_{alpha}")
    std_gsm = np.loadtxt(results_location + f"/std_true_z_{alpha}")
    mu_gsm_shifted = baseline + mu_gsm

    mu_nl, std_nl = get_mean_var_nl(points_1d, mu_gsm_shifted, std_gsm,
                                     nl_scale, nl_baseline, nl_power)

    # Eje x en grados (-90 a 90)
    orientations = np.linspace(-90, 90, N_exc)

    # Mean
    axes2[0, 0].plot(orientations, mu_nl, color=green_colors[alpha], linewidth=2)

    # Std
    axes2[1, 0].plot(orientations, std_nl, color=green_colors[alpha], linewidth=2)

axes2[0, 0].set_ylabel('mean $u_E$ [mV]', fontsize=11)
axes2[0, 0].set_title('posterior (NL transform)', fontsize=12)
axes2[0, 0].set_xlim(-90, 90)

axes2[1, 0].set_ylabel('$u_E$ std. [mV]', fontsize=11)
axes2[1, 0].set_xlabel('pref. ori.', fontsize=11)
axes2[1, 0].set_xlim(-90, 90)

# Placeholders para network (columna derecha)
axes2[0, 1].set_title('network', fontsize=12)
axes2[0, 1].text(0.5, 0.5, '(datos de red aquí)', ha='center', va='center',
                 transform=axes2[0, 1].transAxes, fontsize=10, color='gray')
axes2[1, 1].set_xlabel('pref. ori.', fontsize=11)

plt.tight_layout()
plt.savefig('/home/molina/FAMAF/5to-Famaf/TESIS/figures/data/NarrowFig/'
            'posterior_nl_echeveste_style.png', dpi=150, bbox_inches='tight')
print("Figura guardada en posterior_nl_echeveste_style.png")
