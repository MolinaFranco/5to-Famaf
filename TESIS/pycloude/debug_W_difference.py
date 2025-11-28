#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Debug script para entender la diferencia entre W y W_exact."""

import sys
import os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'scikit-neuromsi')
)

import numpy as np
from skneuromsi.neural import Echeveste2020

model = Echeveste2020(N_E=50, N_I=50, seed=42)
model.load_parameters()

# Construir W desde parámetros
W_from_params = model.build_connectivity_matrix()

# Obtener W_exact cargada
W_exact = model._W_exact

print("Comparando W_from_params vs W_exact")
print("=" * 70)

print(f"\nW_from_params shape: {W_from_params.shape}")
print(f"W_exact shape: {W_exact.shape}")

print(f"\nW_from_params stats:")
print(f"  min={W_from_params.min():.6f}")
print(f"  max={W_from_params.max():.6f}")
print(f"  mean={W_from_params.mean():.6f}")
print(f"  std={W_from_params.std():.6f}")

print(f"\nW_exact stats:")
print(f"  min={W_exact.min():.6f}")
print(f"  max={W_exact.max():.6f}")
print(f"  mean={W_exact.mean():.6f}")
print(f"  std={W_exact.std():.6f}")

diff = W_from_params - W_exact
print(f"\nDiferencia (W_from_params - W_exact):")
print(f"  min_diff={diff.min():.6f}")
print(f"  max_diff={diff.max():.6f}")
print(f"  mean_diff={diff.mean():.6f}")
print(f"  std_diff={diff.std():.6f}")

# Verificar si son iguales o muy similares
are_close = np.allclose(W_from_params, W_exact, rtol=1e-10, atol=1e-10)
print(f"\nSon iguales (rtol=1e-10): {are_close}")

# Ver algunos valores específicos
print(f"\nPrimeras 3x3 de cada matriz:")
print(f"W_from_params[0:3, 0:3]:")
print(W_from_params[0:3, 0:3])
print(f"\nW_exact[0:3, 0:3]:")
print(W_exact[0:3, 0:3])
print(f"\nDiff[0:3, 0:3]:")
print(diff[0:3, 0:3])

# Revisar que build_connectivity_matrix esté usando los params correctos
print("\n" + "=" * 70)
print("Verificando qué matriz retorna build_connectivity_matrix():")

# Ver el código interno
if hasattr(model, '_W_full') and model._W_full is not None:
    print("Tiene _W_full guardada")
else:
    print("NO tiene _W_full, debe reconstruir")

if (model._W_EE is not None and
    model._W_EI is not None and
    model._W_IE is not None and
    model._W_II is not None):
    print("Tiene matrices _W_XX guardadas, las usará directamente")
    print(f"  _W_EE shape: {model._W_EE.shape}")
    print(f"  _W_EI shape: {model._W_EI.shape}")
    print(f"  _W_IE shape: {model._W_IE.shape}")
    print(f"  _W_II shape: {model._W_II.shape}")
else:
    print("NO tiene matrices _W_XX, reconstruirá desde parámetros")
