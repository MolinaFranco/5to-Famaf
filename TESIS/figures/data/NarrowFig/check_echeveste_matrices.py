"""
Script para verificar la estructura de las matrices C y ATA de Echeveste.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM')

# Importar funciones de Echeveste
from GSM import get_fourier_base, get_C_from_fourier

# Cargar los filtros de Echeveste
FILTER_FILE = '/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/GSM/filters.npy'
A = np.load(FILTER_FILE)
ATA = np.dot(A.T, A)

D_y = len(A[0])  # Número de neuronas/orientaciones
print(f"Dimensiones: A={A.shape}, D_y={D_y}")

# Construir C como lo hace Echeveste
B = get_fourier_base(D_y)
decay_length = D_y / 50.0
epsilon = 0.01
C = get_C_from_fourier(epsilon, B, decay_length)
C_inv = np.linalg.inv(C)

print(f"\n=== Matriz C ===")
print(f"Diagonal de C: min={np.diag(C).min():.4f}, max={np.diag(C).max():.4f}")
print(f"C[0,0]={C[0,0]:.4f}, C[D_y//2, D_y//2]={C[D_y//2, D_y//2]:.4f}")
print(f"Es C diagonal constante? {np.allclose(np.diag(C), C[0,0])}")
print(f"Primeros 10 valores de diag(C): {np.diag(C)[:10]}")

print(f"\n=== Matriz C_inv ===")
print(f"Diagonal de C_inv: min={np.diag(C_inv).min():.4f}, max={np.diag(C_inv).max():.4f}")
print(f"Es C_inv diagonal constante? {np.allclose(np.diag(C_inv), C_inv[0,0])}")

print(f"\n=== Matriz ATA ===")
print(f"Diagonal de ATA: min={np.diag(ATA).min():.4f}, max={np.diag(ATA).max():.4f}")
print(f"Es ATA diagonal constante? {np.allclose(np.diag(ATA), ATA[0,0])}")
print(f"Primeros 10 valores de diag(ATA): {np.diag(ATA)[:10]}")

# Ahora calcular Sigma_post para z=1.0 y ver cómo varía std
s_x = 10.0
s_x_2 = s_x**2
z = 1.0

M = C_inv + (z*z/s_x_2) * ATA
Sigma_post = np.linalg.inv(M)
std_post = np.sqrt(np.diag(Sigma_post))

print(f"\n=== Sigma_post para z={z} ===")
print(f"Diagonal de Sigma_post: min={np.diag(Sigma_post).min():.4f}, max={np.diag(Sigma_post).max():.4f}")
print(f"std_post: min={std_post.min():.4f}, max={std_post.max():.4f}")
print(f"std_post en extremos (0, -1): {std_post[0]:.4f}, {std_post[-1]:.4f}")
print(f"std_post en centro (D_y//2): {std_post[D_y//2]:.4f}")

# Graficar el perfil de std_post
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Diag(C)
ax = axes[0, 0]
ax.plot(np.diag(C), 'b-')
ax.set_title('Diagonal de C (Echeveste)')
ax.set_xlabel('Neurona/Orientación')
ax.set_ylabel('C[i,i]')

# Diag(ATA)
ax = axes[0, 1]
ax.plot(np.diag(ATA), 'r-')
ax.set_title('Diagonal de ATA (Echeveste)')
ax.set_xlabel('Neurona/Orientación')
ax.set_ylabel('ATA[i,i]')

# std_post para diferentes z
ax = axes[1, 0]
for z in [0.125, 0.25, 0.5, 1.0, 2.0]:
    M = C_inv + (z*z/s_x_2) * ATA
    Sigma_post = np.linalg.inv(M)
    std_post = np.sqrt(np.diag(Sigma_post))
    ax.plot(std_post, label=f'z={z}')
ax.set_title('std_post vs orientación para diferentes z')
ax.set_xlabel('Neurona/Orientación')
ax.set_ylabel('std_post')
ax.legend()

# Matriz C completa
ax = axes[1, 1]
im = ax.imshow(C, cmap='RdBu_r')
ax.set_title('Matriz C completa')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('/home/molina/FAMAF/5to-Famaf/TESIS/figures/data/NarrowFig/echeveste_matrices_check.png',
            dpi=150, bbox_inches='tight')
print(f"\nFigura guardada en echeveste_matrices_check.png")
