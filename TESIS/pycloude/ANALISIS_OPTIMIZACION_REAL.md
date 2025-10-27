# Análisis: Optimización Real vs Nuestra Implementación

## TL;DR - Respuesta Directa

**¿El código original de Echeveste hace entrenamiento real?**
**SÍ**, hace optimización real completa con:
- Diferenciación automática (autodiff)
- Cálculo de gradientes mediante backpropagation
- Optimizadores reales (L-BFGS-B y ADAM)

**¿Nuestra implementación actual hace lo mismo?**
**NO**, es un esqueleto funcional que:
- ✅ Establece la estructura correcta
- ✅ Inicializa parámetros
- ❌ NO calcula gradientes
- ❌ NO optimiza realmente

---

## 1. ¿Qué Hace el Código Original de Echeveste?

### El Código OCaml Hace OPTIMIZACIÓN REAL

Veamos las piezas clave:

#### A) Diferenciación Automática (Autodiff)

```ocaml
(* objective.ml línea 469 *)
let%diff objective p =
  let open O in
  let noise_reg, noise_width_reg, costs = objectives (module O) p in
  let evolution_cost = Array.fold_left (...) costs in
  evolution_cost + noise_reg + noise_width_reg
```

**¿Qué significa `let%diff`?**
- Es una **extensión de sintaxis de OCaml** para diferenciación automática
- Similar a JAX `@jax.grad` o PyTorch `autograd`
- Genera AUTOMÁTICAMENTE el código para calcular gradientes
- Permite backpropagation a través de toda la función

#### B) Cálculo de Gradientes

```ocaml
(* train.ml línea 176 *)
let _, fdf = grad X.f
let f_df x g =
  let cost, g_ = fdf x in  (* Calcula COSTO Y GRADIENTE simultáneamente *)
  Vec.blit g_ g;
  cost
```

**Esto hace:**
1. `grad X.f` → Genera función que calcula gradiente de `objective`
2. `fdf x` → Retorna tupla `(costo, gradiente)`
3. El gradiente `g_` es el vector ∇f(x) necesario para optimización

#### C) Simulación de la Dinámica

```ocaml
(* objective.ml líneas 377-408 *)
let%diff evolution_costs ~sigma_eta ~w ~h ~target_mu ~target_sigma =
  let rec accumulate ~mu ~sigma ~sigma_star accu t =
    if t = n_time_bins then (...)
    else begin
      (* Calcula momentos no lineales *)
      let nu = V.init n (fun i -> nu_fun (module O) (vget ca i)) in
      let gamma = V.init n (fun i -> gamma_fun (module O) (vget ca i)) in

      (* EVOLUCIONA los momentos μ y Σ paso a paso *)
      let mu' = mu +:|:| (dt *.:| dmu_dt ~w ~mu ~nu ~h) in
      let sigma' = new_sigma ~j_mat ~sigma_star ~j_mat_sigma_star ~sigma in

      (* Calcula COSTO comparando con target *)
      let cost_mean = lambda_mean *. sqr_nrm2 (target_mu - mu) in
      let cost_var = lambda_var *. sqr_nrm2 (target_sigma_diag - sigma_diag) in
      let cost_cov = lambda_cov *. sqr_frob (target_sigma - sigma) in

      (* Recursión para siguiente timestep *)
      accumulate ~mu:mu' ~sigma:sigma' accu' (t+1)
    end
```

**Esto hace:**
1. **Simula** la evolución temporal de momentos μ(t) y Σ(t)
2. **Compara** μ_red(t) con μ_target y Σ_red(t) con Σ_target
3. **Acumula** error en cada timestep
4. Todo con `let%diff` → **GRADIENTES SE CALCULAN AUTOMÁTICAMENTE**

#### D) Optimización Real con L-BFGS-B

```ocaml
(* train.ml líneas 268-277 *)
let rec attempt () =
  try Lbfgs.(F.min
    ~print:(Every 1)
    ~u:X.ub              (* upper bounds *)
    ~factr:1E1
    ~pgtol:0.
    ~corrections:20
    ~stop
    f_df                 (* función que retorna (costo, gradiente) *)
    x                    (* parámetros iniciales *)
  |> ignore)
```

**Esto hace:**
- Usa biblioteca **L-BFGS-B** (optimización quasi-Newton con bounds)
- Itera hasta convergencia
- En cada iteración:
  1. Calcula `f_df(x)` → obtiene costo y gradiente
  2. Usa gradiente para actualizar parámetros
  3. Repite hasta que `||∇f|| < tol`

---

## 2. ¿Por Qué Faltan Esos 3 Puntos en Nuestra Implementación?

### Punto 1: Función de Costo Completa

**¿Qué tenemos?**
```python
# Nuestra implementación actual (placeholder)
optimized_params = self._unpack_parameters(x0)  # Usa parámetros iniciales
```

**¿Qué necesitamos?**
```python
def _compute_cost_and_gradient(self, x, targets):
    """
    Calcula costo total y su gradiente.

    Basado en objective.ml líneas 469-474.
    """
    # 1. Desempaquetar parámetros
    params = self._unpack_parameters(x)

    # 2. Construir matrices W y Σ_η desde parámetros
    W = self._build_W_from_params(params)
    Sigma_eta = self._build_Sigma_eta_from_params(params)

    total_cost = 0.0

    # 3. Para cada target (imagen de entrenamiento)
    for target in targets:
        h = target['h_vec']  # Input al SSN
        mu_target = target['mu_vec']
        sigma_target = target['sigma_mat']

        # 4. Simular evolución de momentos
        mu_final, sigma_final = self._evolve_moments(W, h, Sigma_eta)

        # 5. Calcular costos de moment-matching
        cost_mean = lambda_mean * np.sum((mu_target - mu_final[:N_E])**2)
        cost_var = lambda_var * np.sum((np.diag(sigma_target) - np.diag(sigma_final[:N_E, :N_E]))**2)
        cost_cov = lambda_cov * np.sum((sigma_target - sigma_final[:N_E, :N_E])**2)

        # 6. Acumular
        total_cost += cost_mean + cost_var + cost_cov

    return total_cost
```

**¿Por qué falta esto?**
- **Complejidad**: Requiere simular toda la dinámica del SSN
- **Targets**: Necesitamos datos de entrenamiento (pares (I, μ_posterior, Σ_posterior) del GSM)
- **Momento matching**: Comparar estadísticas de red con estadísticas del GSM posterior

---

### Punto 2: Backpropagation Through Time

**¿Qué es esto?**
Calcular ∂Cost/∂W cuando el costo depende de una **secuencia temporal**.

**El problema:**
```
Parámetros: x = [a_EE, a_EI, ..., d_II]
           ↓
Construir: W(x), Σ_η(x)
           ↓
Simular:   μ(0) → μ(1) → μ(2) → ... → μ(T)
           Σ(0) → Σ(1) → Σ(2) → ... → Σ(T)
           ↓
Costo:     Cost(μ(T), Σ(T); μ_target, Σ_target)
           ↓
Gradiente: ∂Cost/∂x = ?
```

**Solución del código original:**
```ocaml
let%diff evolution_costs ~sigma_eta ~w ~h ~target_mu ~target_sigma =
  let rec accumulate ~mu ~sigma t =
    (* ... evolución paso a paso ... *)
  in
  accumulate ~mu:mu0 ~sigma:sigma0 0
```

El `let%diff` hace **diferenciación automática** de TODO el proceso:
- Guarda operaciones en forward pass
- Calcula gradientes en backward pass
- **Backpropagation automático** a través del tiempo

**¿Qué necesitamos en Python?**

**Opción 1: JAX (más cercano al código original)**
```python
import jax
import jax.numpy as jnp

@jax.jit
def evolution_costs(params, W, Sigma_eta, h, mu_target, sigma_target):
    mu = jnp.zeros(N)
    sigma = 4.0 * jnp.eye(N)

    cost = 0.0

    # Loop temporal
    for t in range(n_time_bins):
        # Momentos no lineales
        nu, gamma = compute_nonlinear_moments(mu, sigma)

        # Evolución
        mu = mu + dt * dmu_dt(W, mu, nu, h)
        sigma = new_sigma(...)

        # Acumular costo
        if t % subsamp == 0:
            cost += jnp.sum((mu_target - mu[:N_E])**2)

    return cost

# JAX calcula el gradiente AUTOMÁTICAMENTE
grad_fn = jax.grad(evolution_costs)
gradient = grad_fn(params, W, Sigma_eta, h, mu_target, sigma_target)
```

**Opción 2: PyTorch**
```python
import torch

def evolution_costs(params, targets):
    # Construir W y Sigma_eta como tensors con requires_grad=True
    W = build_W_differentiable(params)

    mu = torch.zeros(N, requires_grad=True)
    sigma = 4.0 * torch.eye(N, requires_grad=True)

    cost = 0.0

    for t in range(n_time_bins):
        nu, gamma = compute_nonlinear_moments(mu, sigma)
        mu = mu + dt * dmu_dt(W, mu, nu, h)
        sigma = new_sigma(...)

        cost = cost + torch.sum((mu_target - mu[:N_E])**2)

    return cost

# PyTorch calcula gradiente
cost = evolution_costs(params, targets)
cost.backward()  # Calcula gradientes automáticamente
gradient = params.grad
```

**¿Por qué falta esto?**
- **No usamos framework de autodiff**: NumPy no calcula gradientes
- **Necesitamos JAX o PyTorch**: Para diferenciación automática
- **Reescribir código**: Las funciones deben ser compatibles con autodiff

---

### Punto 3: Integración con Optimizadores

**¿Qué tenemos?**
```python
# Placeholder sin optimización real
optimized_params = self._unpack_parameters(x0)
```

**¿Qué necesitamos?**

**Opción A: scipy.optimize.minimize con gradientes**
```python
from scipy.optimize import minimize

def cost_function(x):
    return self._compute_cost_and_gradient(x, targets)

def gradient_function(x):
    _, grad = self._compute_cost_and_gradient(x, targets)
    return grad

result = minimize(
    cost_function,
    x0,
    method='L-BFGS-B',
    jac=gradient_function,  # ← NECESITAMOS ESTO
    bounds=bounds,
    options={'maxiter': 1000}
)

optimized_params = self._unpack_parameters(result.x)
```

**Opción B: JAX + jaxopt**
```python
import jaxopt

optimizer = jaxopt.LBFGS(
    fun=evolution_costs,
    maxiter=1000
)

result = optimizer.run(x0, targets=targets)
optimized_params = self._unpack_parameters(result.params)
```

**Opción C: PyTorch optimizers**
```python
import torch.optim as optim

params = torch.tensor(x0, requires_grad=True)
optimizer = optim.LBFGS([params], max_iter=1000)

def closure():
    optimizer.zero_grad()
    cost = evolution_costs(params, targets)
    cost.backward()
    return cost

optimizer.step(closure)
optimized_params = self._unpack_parameters(params.detach().numpy())
```

**¿Por qué falta esto?**
- **Dependencia**: Necesita el Punto 1 (función de costo) y Punto 2 (gradientes)
- **Sin gradientes**: scipy.minimize necesita la función `jac` (jacobiano/gradiente)
- **Actualmente**: Solo tenemos la función de costo sin gradientes

---

## 3. Resumen Comparativo

| Aspecto | Código Original OCaml | Nuestra Implementación Python |
|---------|----------------------|-------------------------------|
| **Autodiff** | ✅ `let%diff` (automático) | ❌ No implementado |
| **Gradientes** | ✅ `grad X.f` | ❌ No calculados |
| **Simulación dinámica** | ✅ `evolution_costs` con loop temporal | ❌ No implementada |
| **Función de costo** | ✅ Moment-matching completo | ❌ Placeholder |
| **Optimizador** | ✅ L-BFGS-B real | ❌ Sin optimización |
| **Backprop through time** | ✅ Automático vía autodiff | ❌ No implementado |
| **Resultado** | ✅ Parámetros optimizados | ⚠️ Parámetros iniciales |

---

## 4. ¿Qué Tan Complejo Es Implementar Cada Punto?

### Punto 1: Función de Costo ★★☆☆☆ (Moderado)

**Dificultad:** Media
**Tiempo estimado:** 1-2 días

**Por qué es alcanzable:**
- Ya tenemos `_compute_nonlinear_moments()` implementado
- Ya sabemos evolucionar momentos (similar a `run()`)
- Solo falta:
  ```python
  def _evolve_moments_for_training(self, W, h, Sigma_eta, n_steps):
      mu = np.zeros(self._N)
      sigma = 4.0 * np.eye(self._N)
      sigma_star = 4.0 * np.eye(self._N)

      for t in range(n_steps):
          # Momentos no lineales
          nu, gamma = self._compute_nonlinear_moments(mu, sigma)

          # Evolución (ADF - Assumed Density Filtering)
          mu = mu + dt * self._dmu_dt(W, mu, nu, h)
          sigma = self._new_sigma(...)
          sigma_star = self._new_sigma_star(...)

      return mu, sigma
  ```

**Requisito previo:**
- Datos de entrenamiento (targets del GSM)

---

### Punto 2: Backpropagation Through Time ★★★★☆ (Difícil)

**Dificultad:** Alta
**Tiempo estimado:** 1-2 semanas

**Dos caminos posibles:**

#### Camino A: JAX (Recomendado)
```python
# Ventajas:
# - Más similar al código OCaml original
# - Autodiff automático como let%diff
# - JIT compilation (rápido)

# Desventajas:
# - Requiere reescribir código en JAX
# - Cambiar numpy → jax.numpy
# - Aprender paradigma funcional de JAX
```

#### Camino B: PyTorch
```python
# Ventajas:
# - Autodiff sencillo con .backward()
# - Más documentación y ejemplos
# - Integración con BrainPy (también usa PyTorch backend)

# Desventajas:
# - Overhead de tensors
# - Menos eficiente para este tipo de código
```

**Complejidad:**
1. Reescribir `_compute_nonlinear_moments()` en JAX/PyTorch
2. Reescribir evolución de momentos como función diferenciable
3. Asegurar que TODAS las operaciones sean diferenciables
4. Manejar bucles temporales (JAX usa `jax.lax.scan`)

---

### Punto 3: Integración con Optimizadores ★☆☆☆☆ (Fácil)

**Dificultad:** Baja
**Tiempo estimado:** 2-4 horas

**Una vez tengamos Puntos 1 y 2:**
```python
from scipy.optimize import minimize

# Con JAX
import jax

@jax.jit
def cost_fn(x):
    params = self._unpack_parameters(x)
    W = build_W(params)
    Sigma_eta = build_Sigma_eta(params)
    return compute_total_cost(W, Sigma_eta, targets)

grad_fn = jax.grad(cost_fn)

result = minimize(
    cost_fn,
    x0,
    method='L-BFGS-B',
    jac=grad_fn,
    bounds=bounds
)
```

**Esto es trivial** una vez tengamos autodiff funcionando.

---

## 5. Plan de Implementación Sugerido

### Fase 1: Preparación (1 semana)
1. ✅ **DONE**: Estructura de train() (tenemos esto)
2. **TODO**: Generar datos de entrenamiento desde GSM
   - Muestras de imágenes naturales
   - Posteriores μ_target, Σ_target para cada imagen

### Fase 2: Función de Costo sin Gradientes (1 semana)
1. Implementar `_evolve_moments_for_training()` en NumPy puro
2. Implementar `_compute_cost()` comparando con targets
3. **Verificar** que el costo disminuye cuando usamos parámetros pre-entrenados

### Fase 3: Migración a JAX (2 semanas)
1. Instalar JAX: `pip install jax jaxlib`
2. Reescribir funciones clave en JAX:
   - `_compute_nonlinear_moments()` → `jnp` en vez de `np`
   - `_evolve_moments()` → usar `jax.lax.scan` para loop
   - Constructores de matrices W, Σ_η
3. Verificar que `jax.grad()` funciona

### Fase 4: Optimización Real (3 días)
1. Integrar con scipy.optimize.minimize
2. Ejecutar entrenamiento completo
3. Comparar resultados con parámetros de Echeveste

---

## 6. Ejemplo Concreto: Lo Que Falta

Déjame mostrarte **exactamente** lo que hace el código original vs lo que hace el nuestro:

### Código Original (train.ml + objective.ml)

```ocaml
(* 1. Define función objetivo diferenciable *)
let%diff objective p =
  (* Desempaqueta parámetros *)
  let prms = unpack (module O) p in
  let w = w prms.weight_prms in
  let sigma_eta = sigma_eta prms.sigma_eta_prms in

  (* Para cada target, simula y calcula costo *)
  let costs = Array.map (fun targ ->
    let h = h prms targ.h_vec in
    (* SIMULA n_time_bins pasos temporales *)
    let (cost_mean, cost_var, cost_cov), mu, sigma =
      evolution_costs ~w ~h ~sigma_eta
                     ~target_mu:targ.mu_vec
                     ~target_sigma:targ.sigma_mat
    in
    cost_mean + cost_var + cost_cov
  ) targets in

  sum costs

(* 2. Calcula gradiente AUTOMÁTICAMENTE *)
let _, fdf = grad objective

(* 3. Optimiza usando L-BFGS-B *)
let result = Lbfgs.min ~u:bounds f_df x0
```

**Esto retorna:** Parámetros que **MINIMIZAN** el error entre red y targets.

---

### Nuestra Implementación Actual

```python
def _optimize_stage1(self, gsm_model, params):
    # Inicializa parámetros
    initial_params = {
        'a_EE': 0.02,
        'd_EE': 0.8,
        # ...
    }

    # PLACEHOLDER: sin optimización real
    optimized_params = initial_params  # ← Solo copia iniciales

    # Actualiza modelo
    self._a_EE = optimized_params['a_EE']
    # ...

    return {'optimized_params': optimized_params}
```

**Esto retorna:** Los **MISMOS** parámetros iniciales, sin optimizar.

---

## 7. Conclusión

### ¿El código original hace optimización real?
**SÍ, completamente:**
- ✅ Simula dinámica temporal del SSN
- ✅ Calcula costo de moment-matching
- ✅ Calcula gradientes vía autodiff
- ✅ Optimiza con L-BFGS-B hasta convergencia

### ¿Por qué faltan los 3 puntos?
1. **Función de costo**: Necesitamos simular la dinámica y comparar con targets
2. **Backprop**: Necesitamos framework de autodiff (JAX/PyTorch) para calcular gradientes
3. **Optimizadores**: Necesita los puntos 1 y 2 primero, luego es trivial

### ¿Es factible implementarlo?
**SÍ**, pero requiere:
- **Tiempo**: ~1 mes de trabajo
- **Herramientas**: JAX para autodiff
- **Datos**: Targets de entrenamiento del GSM
- **Validación**: Comparar con resultados de Echeveste

### ¿Vale la pena?
**Depende del objetivo:**
- Si solo necesitas **usar** el modelo → usa `load_parameters()` (ya funciona)
- Si quieres **entrenar** desde cero → necesitas implementación completa
- Si es para **tesis/paper** → definitivamente vale la pena

---

## Apéndice: Diferencias Clave OCaml vs Python

| OCaml | Python Equivalente |
|-------|-------------------|
| `let%diff f x = ...` | `@jax.grad` o `torch.autograd` |
| `grad f` | `jax.grad(f)` |
| `V.init n (fun i -> ...)` | `jnp.array([... for i in range(n)])` |
| `let rec accumulate ...` | `for` loop o `jax.lax.scan` |
| `M.sqr_frob matrix` | `jnp.sum(matrix ** 2)` |
| `Lbfgs.min f_df x0` | `scipy.optimize.minimize(..., jac=...)` |
