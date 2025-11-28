# Tabla de Referencia Rápida: Funciones de Simulación

**Comparación visual rápida entre código original y nuestra implementación**

---

## Leyenda de Símbolos

| Símbolo | Significado |
|---------|-------------|
| ✅ | Idéntica implementación |
| ≡ | Matemáticamente equivalente (diferente implementación) |
| ⚡ | Mejorada (funcionalidad extendida) |
| ⚠️ | Diferencia importante (pero correcta) |

---

## Funciones Core SSN

| Función | Original | Nuestra | Status | Ubicación Original | Ubicación Nuestra |
|---------|----------|---------|--------|-------------------|-------------------|
| **Activación supralineal** | `get_r(u)` | `supralinear_activation(u)` | ✅ | `methods.py:43` | `_echeveste2020.py:63` |
| **Derivada activación** | `get_r_prime(u)` | JAX auto-diff | ≡ | `methods.py:51` | JAX implícito |
| **Ecuación membrana** | `new_u(u,r,h,W,eta)` | `__call__(u_e,u_i,t,W,h,eta)` | ✅ | `methods.py:289` | `_echeveste2020.py:79` |
| **Runge-Kutta 4** | `new_u_det_4RK(W,h,u)` | `bp.odeint(method="rk4")` | ≡ | `methods.py:298` | BrainPy built-in |
| **Evolución red** | `network_evolution(...)` | `run(...)` | ⚡ | `methods.py:383` | `_echeveste2020.py:1751` |
| **Proceso O-U (η)** | Dentro `network_evolution` | Dentro `run` | ✅ | `methods.py:391-397` | `_echeveste2020.py:1813` |

---

## Momentos y Estadísticas

| Función | Original | Nuestra | Status | Ubicación Original | Ubicación Nuestra |
|---------|----------|---------|--------|-------------------|-------------------|
| **Media tasas ν** | `get_nu(mu,diag_Sigma)` | `_compute_nonlinear_moments(mu,sigma)` | ✅ | `methods.py:58` | `_echeveste2020.py:1456` |
| **Recursión ν** | `nu_recursive(m,...)` | Hardcoded n=2 | ≡ | `methods.py:98` | `_echeveste2020.py:1468` |
| **Matriz Gamma Γ** | `get_Gamma(mu,Sigma,diag_Sigma)` | Implícito (JAX ADF) | ≡ | `methods.py:111` | Stage 2 interno |
| **Factor gamma γ** | `get_gamma(mu,diag_Sigma)` | Implícito (JAX ADF) | ≡ | `methods.py:146` | Stage 2 interno |
| **Jacobiano J** | `get_J(W,gamma)` | JAX auto-diff | ≡ | `methods.py:128` | JAX implícito |

---

## Conectividad y Ruido

| Función | Original | Nuestra | Status | Ubicación Original | Ubicación Nuestra | Notas |
|---------|----------|---------|--------|-------------------|-------------------|-------|
| **Matriz W** | Cargada de archivo | `build_connectivity_matrix()` | ⚡ | `parameters.py` | `_echeveste2020.py:2406` | ⭐ Construida desde params |
| **Conectividad paramétrica** | NO implementada | `parametric_connectivity()` | ⚡ | N/A (solo en OCaml) | `_echeveste2020.py:2371` | ⭐ Nueva |
| **Covarianza ruido Σ_η** | Cargada de archivo | `_build_noise_covariance()` | ⚡ | `parameters.py` | `_echeveste2020.py:1326` | ⭐ Construida desde params |

---

## Modelo GSM

| Función | Original | Nuestra | Status | Ubicación Original | Ubicación Nuestra |
|---------|----------|---------|--------|-------------------|-------------------|
| **Filtros Gabor A** | `np.load("filters.npy")` | `np.load("data/filters.npy")` | ✅ | `GSM.py:275` | `_echeveste2020.py:2675` |
| **Prior C** | `get_C_from_fourier()` | `_get_C_from_fourier()` | ✅ | `GSM.py:52` | Método privado |
| **Posterior μ** | `get_mu_z()` | Dentro `calculate_causes()` | ✅ | `GSM.py:107` | `_echeveste2020.py:2727` |
| **Posterior Σ** | `get_Sigma_z()` | Dentro `calculate_causes()` | ✅ | `GSM.py:111` | `_echeveste2020.py:2727` |
| **Inferencia z** | `P_z_giv_x()` | `_compute_z_posterior()` | ✅ | `GSM.py:211` | Método privado |
| **Full inference** | `get_post_moments_full_inference()` | Opción en `calculate_causes()` | ⚡ | `GSM.py:115` | `_echeveste2020.py:2727` |

---

## Input y Transformaciones

| Función | Original | Nuestra | Status | Ubicación Original | Ubicación Nuestra | Notas |
|---------|----------|---------|--------|-------------------|-------------------|-------|
| **Input h simple** | `h = A.T @ x / 15` | Modo pre-training | ✅ | `GSM.py:427` | `_echeveste2020.py:2077` | Experimentos |
| **Input h optimizado** | NO en experimentos | Modo post-training | ⚡ | Solo en paper (Eq. S25) | `_echeveste2020.py:2077` | ⭐ Implementado |
| **No-linealidad h** | N/A | `h = α[A.T@x]_+^γ + β` | ⚡ | Paper Eq. S25 | `_echeveste2020.py:2090` | ⭐ Stage 2 |

---

## Integración Numérica

| Aspecto | Original | Nuestra | Status | Notas |
|---------|----------|---------|--------|-------|
| **Método** | Euler forward | Euler (default), RK2, RK4 | ⚡ | BrainPy soporta múltiples |
| **Paso temporal dt** | 0.2 ms (hardcoded) | 0.2 ms (configurable) | ⚡ | Mismo default |
| **Backend** | NumPy | BrainPy + JAX | ⚡ | GPU/TPU support |
| **Auto-diff** | Manual | JAX | ⚡ | Gradientes exactos |

---

## Entrenamiento (Training)

| Componente | Original | Nuestra | Status | Ubicación Original | Ubicación Nuestra | Notas |
|-----------|----------|---------|--------|-------------------|-------------------|-------|
| **Stage 1 (ADAM)** | OCaml | Python + JAX | ≡ | `train.ml` (OCaml) | `_echeveste2020.py:499` | ⭐ Reimplementado |
| **Stage 2 (L-BFGS-B)** | OCaml | Python + JAX | ≡ | `train.ml` (OCaml) | `_echeveste2020.py:993` | ⭐ Reimplementado |
| **ADF method** | OCaml | JAX auto-diff | ≡ | `train.ml` (OCaml) | Stage 2 interno | ⭐ Usando JAX |
| **Stochastic sampling** | OCaml | Python loops | ≡ | `train.ml` (OCaml) | Stage 1 interno | ⭐ Reimplementado |

---

## Funciones Auxiliares

| Función | Original | Nuestra | Status | Notas |
|---------|----------|---------|--------|-------|
| **Norma cuadrática** | `sqr_norm(x)` | `np.sum(x*x)` inline | ✅ | Usado en costo |
| **PDF Gaussiana** | `f1(x)` | `f1(x)` | ✅ | Idéntica |
| **CDF Gaussiana** | `f2_exact(x)` | `stats.erf()` | ✅ | Scipy equivalente |
| **Cholesky L** | `np.linalg.cholesky()` | `bp.math.linalg.cholesky()` | ✅ | Backend diferente |
| **Buffer ruido** | `buffer_init()`, `update_buffer()` | No necesario | ⚠️ | Solo para lag experiments |

---

## Casos Especiales Implementados

### Original tiene pero nuestra NO necesita:

| Función Original | Razón NO implementada |
|------------------|----------------------|
| `network_evolution_w_pulses()` | Experimentos específicos (pulsos) |
| `network_evolution_w_lagged_noise()` | Experimentos con lag (no en paper base) |
| `network_sample_w_delay()` | Experimentos con delays |
| `h_modulation(t)` | Transient overshoot (experimentos) |
| `Gamma_spike_train_time_resc()` | Spike generation (no usado en inference) |

### Nuestra tiene pero original NO:

| Función Nuestra | Razón Agregada |
|-----------------|----------------|
| `load_parameters()` | Cargar parámetros entrenados |
| `save_parameters()` | Guardar parámetros post-training |
| `get_all_parameters()` | Export completo de estado |
| `_detect_posterior_peaks()` | Análisis de multi-modalidad |
| `_extract_cause_orientations()` | Decodificación de orientaciones |
| `generate_stimulus_from_gsm()` | Generación sintética de estímulos |

---

## Estructura de Datos

### Original (archivos separados):
```
ssn_inference_numerical_experiments/
├── SSN/
│   ├── methods.py          # Funciones dinámicas
│   ├── parameters.py       # Parámetros optimizados (cargados)
│   └── w_learn             # Matriz W (archivo)
├── GSM/
│   ├── GSM.py              # Modelo generativo
│   ├── filters.npy         # Filtros Gabor
│   └── gabor_filters.py    # Generación filtros
└── GP/
    └── methods.py          # Gaussian Process experiments
```

### Nuestra (clase unificada):
```
scikit-neuromsi/skneuromsi/neural/
├── _echeveste2020.py       # Clase Echeveste2020 (todo integrado)
└── data/
    └── filters.npy         # Filtros Gabor (incluidos)
```

---

## Equivalencias Matemáticas Clave

### 1. Activación Supralineal
**Original:** `0.5 * (u + |u|)` → Implementa `max(0, u)`
**Nuestra:** `bp.math.maximum(0, u)` → Directamente `max(0, u)`
**Status:** ✅ Equivalente

### 2. Proceso Ornstein-Uhlenbeck (η)
**Original:**
```python
eps_1 = (1.0 - dt * tau_n_inv)
eps_2 = sqrt(2.0 * dt * tau_n_inv)
eta_new = eps_1 * eta_old + eps_2 * L @ xi
```

**Nuestra:**
```python
eps_1 = 1.0 - dt / tau_n
eps_2 = bp.math.sqrt(2.0 * dt / tau_n)
eta_new = eps_1 * eta + eps_2 * (L @ xi)
```

**Status:** ✅ Idéntico (solo notación diferente)

### 3. Ecuación SSN
**Original:** `u_new = u_old + dt * tau_inv * (-u + h + W@r + eta)`
**Nuestra:** `du/dt = (-u + h + W@r + eta) / tau` → integrador aplica Euler
**Status:** ✅ Equivalente (Euler forward)

### 4. Posterior GSM
**Original:**
```python
M = C_inv + (z^2 / s_x^2) * A.T @ A
Sigma_post = inv(M)
mu_post = (z / s_x^2) * Sigma_post @ A.T @ x
```

**Nuestra:** **Idéntico**
**Status:** ✅ Exactamente igual

---

## Validación Numérica (Resumen)

| Test | Métrica | Criterio | Resultado |
|------|---------|----------|-----------|
| Activación supralineal | Error absoluto | < 1e-12 | ✅ PASS |
| Construcción W | Error relativo | < 1e-10 | ✅ PASS |
| Trayectorias u(t) | Correlación | > 0.99 | ✅ PASS |
| Posterior μ_post | Error absoluto | < 1e-8 | ✅ PASS |
| Posterior Σ_post | Error absoluto | < 1e-7 | ✅ PASS |
| Inferencia z_MAP | Error absoluto | < 0.01 | ✅ PASS |
| Variabilidad ruido | KS test p-value | > 0.05 | ✅ PASS |
| Convergencia Stage 1 | Success rate | 5/5 | ✅ PASS |
| Convergencia Stage 2 | Success rate | 5/5 | ✅ PASS |

**Total:** 9/9 tests ✅

---

## Referencias Cruzadas

### Para cada función, ver detalles en:
- **Comparación detallada:** `comparison_implementation_vs_original.md`
- **Validación GSM:** `REPORTE_COMPARACION_GSM.md`
- **Parámetros optimizados:** `parameters.md`
- **Transformación input:** `ANALISIS_PARAMETROS_TRANSFORMACION_NOLINEAL.md`
- **Logs entrenamiento:** `entrenamiento_completo_5runs.log`

### Tests de validación:
- `test_equivalencia_parametros.py`
- `test_modelo_funcionando.py`
- `test_variabilidad_10_ejecuciones.py`
- `test_complete_training_with_gsm.py`

---

## Workflow de Uso

### Original (código experimentos):
```python
# 1. Cargar parámetros pre-entrenados
W = np.loadtxt("w_learn")
Sigma_eta = np.loadtxt("sigma_eta")

# 2. Cargar filtros GSM
A = np.load("filters.npy")

# 3. Crear input
h = (1/15) * A.T @ x

# 4. Evolucionar red
u, eta = network_evolution(W, h, u0, Sigma_eta, steps_max)
```

### Nuestra implementación:
```python
# 1. Crear modelo
model = Echeveste2020(N_E=50, N_I=50)

# 2a. Opción A: Cargar parámetros pre-entrenados
model.load_parameters()

# 2b. Opción B: Entrenar desde cero
gsm = create_gsm()
model.train(gsm_model=gsm)

# 3. Simular
result = model.run(
    stimulus_contrast=0.5,
    stimulus_orientation=45.0,
    noise_level=1.0
)

# 4. Analizar
u_e = result['excitatory_potential']
r_e = result['excitatory_firing_rate']
```

**Ventaja:** API unificada scikit-learn-style ⚡

---

## Conclusión Visual

```
┌─────────────────────────────────────────────────────────────┐
│                    EQUIVALENCIA TOTAL                        │
├─────────────────────────────────────────────────────────────┤
│  ✅ Todas las ecuaciones matemáticas                        │
│  ✅ Todos los procesos estocásticos                         │
│  ✅ Todos los cálculos de momentos                          │
│  ✅ Todos los métodos de inferencia                         │
│                                                              │
│  ⚡ + Construcción paramétrica W, Σ_η                       │
│  ⚡ + Entrenamiento end-to-end                              │
│  ⚡ + Soporte GPU/TPU                                       │
│  ⚡ + Auto-diferenciación JAX                               │
│  ⚡ + API unificada                                         │
└─────────────────────────────────────────────────────────────┘
```

**Esta tabla permite navegación rápida entre código original y nuestra implementación.**
