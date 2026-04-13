#!/usr/bin/env python3
"""
Encontrar el baseline correcto para que el posterior y network
tengan valores similares en c=0.
"""
import numpy as np
from scipy.stats import norm
from pathlib import Path

N_exc = 50
SIMULATION_DIR = Path('/home/molina/FAMAF/5to-Famaf/TESIS/figures/data/simulations')

# Parámetros NL
nl_scale = 2.4
nl_power = 0.6
nl_baseline = (3.5 / nl_scale) ** (1.0 / nl_power)

def np_th(u):
    return np.maximum(u, 0)

def nl_fun(u):
    return nl_scale * (np_th(u + nl_baseline) ** nl_power)

def get_mean_nl(mu, std, n_points=201):
    """Calcular solo la media después de NL."""
    points_1d = np.linspace(-4.0, 4.0, n_points)
    new_mean = np.empty(len(mu))

    for i in range(len(mu)):
        mu_i = mu[i]
        std_i = std[i]
        points_resc = std_i * points_1d + mu_i
        values_dist = norm.pdf(points_resc, loc=mu_i, scale=std_i)
        norm_f = np.sum(values_dist)
        values_fun = nl_fun(points_resc)
        new_mean[i] = np.sum(values_dist * values_fun) / norm_f

    return new_mean

# Cargar datos de network para c=0
net_data = np.load(SIMULATION_DIR / "contrast_0.000_narrow_wf0.01" / "simulation_data.npz")
net_mean_c0 = net_data['u_E_mean'].mean()  # Promedio sobre todas las neuronas

print(f"Network mean (c=0): {net_mean_c0:.2f} mV")
print(f"Network mean range (c=0): [{net_data['u_E_mean'].min():.2f}, {net_data['u_E_mean'].max():.2f}]")

# GSM posterior para c=0 (prior)
mu_gsm_c0 = np.zeros(N_exc)  # Media del prior es 0
std_gsm_c0 = np.ones(N_exc) * 2.0  # sqrt(C[0,0]) = 2

print(f"\nGSM prior: mu=0, std=2.0")

# Probar diferentes baselines
print("\n--- Buscando baseline correcto ---")
print(f"{'Baseline':<10} {'Post mean (c=0)':<18} {'Diferencia':<12}")
print("-" * 40)

for baseline in np.arange(0.0, 4.0, 0.25):
    mu_shifted = baseline + mu_gsm_c0
    mu_nl = get_mean_nl(mu_shifted, std_gsm_c0)
    post_mean_c0 = mu_nl.mean()
    diff = post_mean_c0 - net_mean_c0
    marker = " <-- CERCANO" if abs(diff) < 0.5 else ""
    print(f"{baseline:<10.2f} {post_mean_c0:<18.2f} {diff:<12.2f}{marker}")

# El baseline que más se acerca
print("\n--- Conclusión ---")
target = net_mean_c0
print(f"Target (network c=0): {target:.2f} mV")

# Búsqueda más fina
best_baseline = None
best_diff = float('inf')
for baseline in np.arange(-1.0, 3.0, 0.05):
    mu_shifted = baseline + mu_gsm_c0
    mu_nl = get_mean_nl(mu_shifted, std_gsm_c0)
    post_mean_c0 = mu_nl.mean()
    diff = abs(post_mean_c0 - target)
    if diff < best_diff:
        best_diff = diff
        best_baseline = baseline

print(f"Mejor baseline: {best_baseline:.2f}")

# Verificar
mu_shifted = best_baseline + mu_gsm_c0
mu_nl = get_mean_nl(mu_shifted, std_gsm_c0)
print(f"Posterior mean (c=0) con baseline={best_baseline:.2f}: {mu_nl.mean():.2f} mV")
