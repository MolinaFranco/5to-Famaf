# Resumen: ¿Qué Falta Para Optimización Real?

## 📊 Comparación Rápida

```
CÓDIGO ORIGINAL (OCaml)              NUESTRA IMPLEMENTACIÓN (Python)
═══════════════════════════          ═══════════════════════════════

┌─────────────────────┐              ┌─────────────────────┐
│ Parámetros x₀       │              │ Parámetros x₀       │
└──────────┬──────────┘              └──────────┬──────────┘
           │                                    │
           ▼                                    ▼
┌─────────────────────┐              ┌─────────────────────┐
│ Construir W, Σ_η    │              │ Construir W, Σ_η    │
└──────────┬──────────┘              └──────────┬──────────┘
           │                                    │
           ▼                                    │
┌─────────────────────┐                        │
│ Simular dinámica:   │              ❌ FALTA ESTO
│ μ(0)→μ(1)→...→μ(T) │                        │
│ Σ(0)→Σ(1)→...→Σ(T) │                        │
└──────────┬──────────┘                        │
           │                                    │
           ▼                                    │
┌─────────────────────┐                        │
│ Calcular costo:     │              ❌ FALTA ESTO
│ ||μ_red - μ_tgt||²  │                        │
│ ||Σ_red - Σ_tgt||²  │                        │
└──────────┬──────────┘                        │
           │                                    │
           ▼                                    │
┌─────────────────────┐                        │
│ Calcular gradiente: │              ❌ FALTA ESTO
│ ∇Cost = ∂C/∂x       │                        │
│ (autodiff)          │                        │
└──────────┬──────────┘                        │
           │                                    │
           ▼                                    │
┌─────────────────────┐                        │
│ Actualizar params:  │              ❌ FALTA ESTO
│ x ← x - α·∇Cost     │                        │
│ (L-BFGS-B)          │                        │
└──────────┬──────────┘                        │
           │                                    │
           ├─── repetir hasta convergencia     │
           │                                    │
           ▼                                    ▼
┌─────────────────────┐              ┌─────────────────────┐
│ ✅ x_optimizado     │              │ ⚠️  x₀ (sin cambio) │
└─────────────────────┘              └─────────────────────┘
```

---

## 🔍 Explicación Simple: ¿Qué Hace Realmente el Código de Echeveste?

### Imagina que quieres encontrar la mejor receta de pizza:

#### ❌ Lo que hace nuestra implementación:
```
1. Tomas una receta genérica (parámetros iniciales)
2. Dices "esta es la receta óptima"
3. FIN
```

#### ✅ Lo que hace el código original:
```
1. Tomas una receta inicial
2. COCINAS la pizza con esa receta
3. PRUEBAS la pizza
4. CALCULAS qué tan diferente es de la "pizza perfecta"
5. AJUSTAS la receta basándote en el error
6. REPITES pasos 2-5 muchas veces
7. Cuando el error es mínimo → tienes la receta óptima
```

### Traducción a nuestro contexto:

| Analogía Pizza | SSN Training |
|----------------|--------------|
| Receta | Parámetros (a_EE, d_EE, ...) |
| Cocinar pizza | Simular dinámica del SSN |
| Pizza resultante | Actividad μ(T), Σ(T) de la red |
| Pizza perfecta (referencia) | Estadísticas del GSM posterior |
| Probar y medir error | Calcular costo ||μ_red - μ_GSM||² |
| Ajustar receta | Actualizar parámetros con gradiente |

---

## 🎯 Los 3 Puntos Que Faltan (Explicación Simple)

### PUNTO 1: Función de Costo Completa

**¿Qué es?**
Una función que dice "qué tan mal están los parámetros actuales"

**¿Por qué falta?**
Porque para calcularla necesitamos:
1. **Simular** la red con esos parámetros (evolución temporal)
2. **Comparar** el resultado con lo que debería salir (targets del GSM)

**Pseudocódigo:**
```python
def costo(parámetros):
    # 1. Construir red con esos parámetros
    W = construir_conectividad(parámetros)

    # 2. Simular la red por 500ms
    μ_final = simular_500ms(W, estímulo)

    # 3. Comparar con lo que debería salir
    error = ||μ_final - μ_esperado||²

    return error
```

**¿Tenemos esto?** ❌ NO
- Tenemos `run()` que simula, pero no está integrado con el entrenamiento
- No calculamos el error comparando con targets
- No tenemos los targets (datos de entrenamiento del GSM)

---

### PUNTO 2: Backpropagation Through Time

**¿Qué es?**
Calcular "cómo cambiar los parámetros para mejorar el resultado"

**Analogía:**
- Quieres llegar a la cima de una montaña (minimizar costo)
- Estás con los ojos vendados (no ves la montaña)
- El gradiente te dice: "camina 3 pasos al norte, 2 al este"

**¿Por qué se llama "through time"?**
Porque el costo depende de una **secuencia temporal**:
```
x → W → μ(t₀) → μ(t₁) → μ(t₂) → ... → μ(T) → Costo
```

Para calcular ∂Costo/∂x necesitas **backpropagation** desde T hasta 0.

**¿Cómo lo hace el código original?**
```ocaml
let%diff evolution_costs ~w ~h = ...
```
La magia de `let%diff` = diferenciación AUTOMÁTICA

**¿Cómo lo haríamos nosotros?**
```python
import jax

@jax.jit
def evolution_costs(params, h):
    W = build_W(params)
    # ... simular ...
    return cost

# JAX calcula el gradiente automáticamente
gradiente = jax.grad(evolution_costs)(params, h)
```

**¿Tenemos esto?** ❌ NO
- Usamos NumPy (no calcula gradientes)
- Necesitamos JAX o PyTorch
- Necesitamos reescribir código para que sea diferenciable

---

### PUNTO 3: Optimizador Real

**¿Qué es?**
Un algoritmo que **itera** usando el gradiente para mejorar parámetros

**Algoritmo L-BFGS-B (el que usa Echeveste):**
```
Repetir hasta convergencia:
    1. Calcular costo y gradiente en x
    2. Decidir dirección de búsqueda
    3. Hacer line search
    4. Actualizar x
```

**¿Tenemos esto?** ❌ NO
- Scipy tiene L-BFGS-B, pero necesita el gradiente (Punto 2)
- Sin gradiente, no podemos optimizar

---

## 📈 Nivel de Dificultad

```
Punto 1: Función de Costo
████░░░░░░ 40% difícil
- Ya sabemos simular
- Solo falta integrar con targets

Punto 2: Backpropagation
█████████░ 90% difícil  ← EL MÁS COMPLICADO
- Requiere aprender JAX
- Reescribir código
- Entender autodiff

Punto 3: Optimizador
██░░░░░░░░ 20% difícil
- Trivial si tienes Puntos 1 y 2
- Solo llamar a scipy.optimize.minimize
```

---

## 💡 Ejemplo Concreto: Forward Pass

### Lo que hace el código original en CADA ITERACIÓN del optimizador:

```python
# ITERACIÓN 1
params = [a_EE=0.02, d_EE=0.8, ...]
↓
W_EE = 0.02 * exp((cos(...) - 1) / 0.8²)
↓
μ(0) = [0, 0, ..., 0]
μ(1) = μ(0) + dt·(-μ(0) + W·r(0) + h) / τ
μ(2) = μ(1) + dt·(-μ(1) + W·r(1) + h) / τ
...
μ(500) = ...
↓
Costo = ||μ(500) - μ_target||² = 15.3
Gradiente = ∂Costo/∂params = [-0.2, 0.5, ...]
↓
params_nuevo = [a_EE=0.022, d_EE=0.75, ...]  ← Mejora!

# ITERACIÓN 2
params = [a_EE=0.022, d_EE=0.75, ...]
↓
... (repite proceso) ...
↓
Costo = 12.1  ← Bajó!

# ... repite 100-1000 veces ...

# ITERACIÓN FINAL
Costo = 0.001  ← Convergió
params_optimizados = [a_EE=0.0185, d_EE=0.823, ...]
```

### Lo que hace nuestra implementación:

```python
# ÚNICA "ITERACIÓN"
params = [a_EE=0.02, d_EE=0.8, ...]
↓
params_optimizados = [a_EE=0.02, d_EE=0.8, ...]  ← ¡Mismo!
↓
FIN
```

---

## 🚀 ¿Por Qué el Código Original USA Autodiff?

### Sin Autodiff (Calcular Gradiente Manualmente)

Imagina que tu costo depende de 8 parámetros y 500 timesteps:

```
∂Costo/∂a_EE = ∂Costo/∂μ(500) · ∂μ(500)/∂μ(499) · ... · ∂μ(1)/∂W_EE · ∂W_EE/∂a_EE
```

Esto requiere:
1. Calcular 500 derivadas parciales
2. Multiplicarlas usando chain rule
3. Hacer esto para CADA uno de los 8 parámetros
4. **Propenso a errores**
5. **Código súper largo**

**Ejemplo de cómo sería:**
```python
def grad_a_EE_manual(params, targets):
    # Paso 1: Forward pass (guardar todos los estados)
    W = build_W(params)
    mu_history = []
    mu = np.zeros(N)

    for t in range(500):
        mu = mu + dt * dmu_dt(W, mu, ...)
        mu_history.append(mu.copy())

    # Paso 2: Backward pass (calcular derivadas)
    dCost_dmu = 2 * (mu - mu_target)

    dmu_dW = np.zeros((N, N, N))  # Tensor de derivadas
    for t in range(499, -1, -1):
        # Calcular ∂μ(t+1)/∂μ(t)
        dmu_dt_dmu = ...  # Derivada compleja

        # Chain rule
        dCost_dmu = dCost_dmu @ dmu_dt_dmu

        # Acumular
        dmu_dW[t] = ...

    # Paso 3: Derivada de W respecto a a_EE
    dW_da_EE = exp((cos(...) - 1) / d_EE²)

    # Paso 4: Chain rule final
    grad_a_EE = np.sum(dCost_dmu @ dmu_dW @ dW_da_EE)

    return grad_a_EE

# ¡Y esto es solo para UN parámetro!
# Necesitas hacer lo mismo para los otros 7 parámetros
```

**Esto es:**
- ❌ Propenso a bugs
- ❌ Difícil de mantener
- ❌ Lento de implementar

---

### Con Autodiff (Como el Código Original)

```ocaml
(* OCaml con autodiff *)
let%diff cost params =
  let W = build_W params in
  let mu = simulate W in
  sqr_norm (mu - mu_target)

let gradient = grad cost
```

O en Python con JAX:
```python
import jax

def cost(params):
    W = build_W(params)
    mu = simulate(W)
    return jnp.sum((mu - mu_target)**2)

# JAX calcula TODAS las derivadas automáticamente
gradient_fn = jax.grad(cost)
gradient = gradient_fn(params)
```

**Esto es:**
- ✅ Sin bugs (JAX hace el trabajo)
- ✅ Fácil de mantener
- ✅ Rápido de implementar

**Por eso Echeveste usa OCaml con autodiff y nosotros necesitamos JAX.**

---

## 📚 Resumen Final

### ¿El código original hace optimización real?
**SÍ**, hace optimización completa con gradientes.

### ¿Nuestra implementación hace lo mismo?
**NO**, solo establece la estructura sin optimizar realmente.

### ¿Por qué faltan los 3 puntos?

1. **Función de Costo**: Necesitamos simular + comparar con targets
   - **Por qué falta**: No tenemos datos de entrenamiento (targets)

2. **Backpropagation**: Necesitamos calcular gradientes
   - **Por qué falta**: NumPy no hace autodiff, necesitamos JAX

3. **Optimizador**: Necesitamos iterar mejorando parámetros
   - **Por qué falta**: Depende de los puntos 1 y 2

### ¿Es factible implementarlo?
**SÍ**, pero requiere:
- Migrar a JAX (1-2 semanas)
- Generar datos de entrenamiento (3-5 días)
- Integrar optimizador (2-3 días)

**Total: ~1 mes de trabajo**

### ¿Vale la pena?
Depende:
- ✅ Para **usar** el modelo → ya funciona con `load_parameters()`
- ✅ Para **tesis/investigación** → definitivamente sí
- ⚠️  Para **producción** → quizás no (parámetros pre-entrenados son suficientes)
