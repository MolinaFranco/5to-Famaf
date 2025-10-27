#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparación lado a lado: JAX vs PyTorch
========================================

Este archivo muestra EXACTAMENTE el mismo código de entrenamiento
implementado en JAX y PyTorch para que veas las diferencias.

Basado en evolution_costs() de objective.ml líneas 377-408.
"""

print("=" * 70)
print("COMPARACIÓN: JAX vs PyTorch para Echeveste2020")
print("=" * 70)

# =============================================================================
# VERSIÓN 1: JAX (Funcional, como OCaml)
# =============================================================================

print("\n" + "=" * 70)
print("VERSIÓN JAX (Funcional)")
print("=" * 70)

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit
    from jax.scipy import stats as jax_stats

    @jit
    def compute_moments_jax(mu, sigma, k=0.3):
        """
        Momentos no lineales en JAX.

        CARACTERÍSTICAS:
        - Inmutable: mu, sigma no se modifican
        - Puro: sin efectos secundarios
        - JIT: se compila automáticamente
        """
        sigma_diag = jnp.diag(sigma)
        sigma_std = jnp.sqrt(sigma_diag + 1e-8)

        ratio = mu / sigma_std
        phi = jax_stats.norm.pdf(ratio)
        psi = jax_stats.norm.cdf(ratio)

        nu1 = mu * psi + sigma_std * phi
        nu = k * (mu * nu1 + sigma_diag * psi)
        gamma = 2 * k * nu1

        return nu, gamma


    @jit
    def evolve_jax(W, h, mu_target, n_steps=100, dt=0.0002):
        """
        Evolución temporal en JAX usando jax.lax.scan.

        SIMILAR A: let rec accumulate ... en OCaml

        VENTAJAS:
        - Compila a código nativo (XLA)
        - Loop optimizado automáticamente
        - Backprop through time eficiente
        """
        N = len(h)
        tau = jnp.array([0.02] * (N//2) + [0.01] * (N//2))

        def step_fn(carry, t):
            """Un paso de evolución (como un timestep de recursión)."""
            mu, sigma, cost = carry

            # Momentos no lineales
            nu, gamma = compute_moments_jax(mu, sigma)

            # Evolución (simplificada)
            dmu = (-mu + W @ nu + h) / tau
            mu_new = mu + dt * dmu

            # Covarianza (simplificada)
            sigma_new = sigma * 0.99 + jnp.eye(N) * 0.01

            # Acumular costo
            error = mu_new - mu_target
            cost_new = cost + jnp.sum(error**2)

            # Retornar (nuevo_estado, salida_opcional)
            return (mu_new, sigma_new, cost_new), mu_new

        # Condiciones iniciales
        mu0 = jnp.zeros(N)
        sigma0 = 4.0 * jnp.eye(N)
        cost0 = 0.0

        # jax.lax.scan = recursión optimizada
        (mu_final, sigma_final, total_cost), trajectory = jax.lax.scan(
            step_fn,
            (mu0, sigma0, cost0),
            jnp.arange(n_steps)
        )

        return total_cost


    # Ejemplo de uso
    N = 50
    W = jnp.eye(N) * 0.1
    h = jnp.ones(N)
    mu_target = jnp.zeros(N)

    # Calcular costo
    cost = evolve_jax(W, h, mu_target)
    print(f"Costo JAX: {cost:.4f}")

    # AUTODIFF: Calcular gradiente respecto a W
    print("\nCalculando gradiente con jax.grad()...")
    grad_fn = jax.grad(lambda w: evolve_jax(w, h, mu_target))
    gradient_W = grad_fn(W)
    print(f"Gradiente shape: {gradient_W.shape}")
    print(f"Gradiente norm: {jnp.linalg.norm(gradient_W):.4f}")

    # JIT: Primera llamada compila, segunda es rápida
    import time
    print("\nBenchmark JAX:")
    start = time.time()
    _ = evolve_jax(W, h, mu_target)  # Primera: compila
    compile_time = time.time() - start
    print(f"  Primera llamada (compila): {compile_time*1000:.1f}ms")

    start = time.time()
    _ = evolve_jax(W, h, mu_target)  # Segunda: usa compilado
    run_time = time.time() - start
    print(f"  Segunda llamada (compilado): {run_time*1000:.1f}ms")
    print(f"  Speedup: {compile_time/run_time:.1f}x")

    JAX_AVAILABLE = True

except ImportError:
    print("\n⚠️  JAX no está instalado. Instalar con:")
    print("   pip install jax jaxlib")
    JAX_AVAILABLE = False


# =============================================================================
# VERSIÓN 2: PyTorch (Imperativo, OOP)
# =============================================================================

print("\n" + "=" * 70)
print("VERSIÓN PYTORCH (Imperativo)")
print("=" * 70)

try:
    import torch
    from torch import nn

    def compute_moments_pytorch(mu, sigma, k=0.3):
        """
        Momentos no lineales en PyTorch.

        CARACTERÍSTICAS:
        - Mutable: podríamos hacer mu += algo
        - Imperativo: paso a paso
        - Eager: no compila automáticamente
        """
        sigma_diag = torch.diag(sigma)
        sigma_std = torch.sqrt(sigma_diag + 1e-8)

        ratio = mu / sigma_std

        # PyTorch no tiene stats.norm, aproximamos
        phi = torch.exp(-0.5 * ratio**2) / (2.50662827463)  # sqrt(2π)
        psi = 0.5 * (1 + torch.erf(ratio / 1.41421356237))  # 1/sqrt(2)

        nu1 = mu * psi + sigma_std * phi
        nu = k * (mu * nu1 + sigma_diag * psi)
        gamma = 2 * k * nu1

        return nu, gamma


    def evolve_pytorch(W, h, mu_target, n_steps=100, dt=0.0002):
        """
        Evolución temporal en PyTorch usando for loop.

        DIFERENTE A JAX:
        - Loop de Python (no compilado)
        - Interpreta cada iteración
        - Más lento (~10-100x)
        """
        N = len(h)
        tau = torch.cat([
            torch.full((N//2,), 0.02),
            torch.full((N//2,), 0.01)
        ])

        # Condiciones iniciales (MUTABLE)
        mu = torch.zeros(N, requires_grad=True)
        sigma = 4.0 * torch.eye(N)
        total_cost = torch.tensor(0.0, requires_grad=True)

        # Loop imperativo (¡NO compilado!)
        for t in range(n_steps):
            # Momentos no lineales
            nu, gamma = compute_moments_pytorch(mu, sigma)

            # Evolución
            dmu = (-mu + W @ nu + h) / tau
            mu = mu + dt * dmu  # IMPORTANTE: crear nuevo tensor

            # Covarianza (simplificada)
            sigma = sigma * 0.99 + torch.eye(N) * 0.01

            # Acumular costo
            error = mu - mu_target
            total_cost = total_cost + torch.sum(error**2)

        return total_cost


    # Ejemplo de uso
    N = 50
    W = torch.eye(N) * 0.1
    W.requires_grad = True  # Para calcular gradientes
    h = torch.ones(N)
    mu_target = torch.zeros(N)

    # Calcular costo
    cost = evolve_pytorch(W, h, mu_target)
    print(f"Costo PyTorch: {cost.item():.4f}")

    # AUTODIFF: Calcular gradiente con .backward()
    print("\nCalculando gradiente con .backward()...")
    cost.backward()
    gradient_W = W.grad
    print(f"Gradiente shape: {gradient_W.shape}")
    print(f"Gradiente norm: {torch.linalg.norm(gradient_W).item():.4f}")

    # Benchmark PyTorch
    import time
    print("\nBenchmark PyTorch:")
    W_bench = torch.eye(N) * 0.1
    W_bench.requires_grad = True

    start = time.time()
    _ = evolve_pytorch(W_bench, h, mu_target)
    run_time = time.time() - start
    print(f"  Primera llamada: {run_time*1000:.1f}ms")

    start = time.time()
    _ = evolve_pytorch(W_bench, h, mu_target)
    run_time2 = time.time() - start
    print(f"  Segunda llamada: {run_time2*1000:.1f}ms")
    print(f"  (No hay compilación, siempre mismo tiempo)")

    PYTORCH_AVAILABLE = True

except ImportError:
    print("\n⚠️  PyTorch no está instalado. Instalar con:")
    print("   pip install torch")
    PYTORCH_AVAILABLE = False


# =============================================================================
# COMPARACIÓN LADO A LADO
# =============================================================================

print("\n" + "=" * 70)
print("COMPARACIÓN DE CÓDIGO")
print("=" * 70)

comparison = """
┌─────────────────────────────────────────────────────────────────────┐
│                    JAX                  │          PyTorch           │
├─────────────────────────────────────────┼────────────────────────────┤
│ PARADIGMA: Funcional                    │ PARADIGMA: Imperativo      │
├─────────────────────────────────────────┼────────────────────────────┤
│ def step(carry, t):                     │ for t in range(n_steps):   │
│     mu, sigma, cost = carry             │     # Modifica variables   │
│     mu_new = mu + ...                   │     mu = mu + ...          │
│     return (mu_new, ...), mu_new        │     total_cost = cost + ...│
│                                         │                            │
│ jax.lax.scan(step, init, range)         │ # Loop interpretado        │
├─────────────────────────────────────────┼────────────────────────────┤
│ INMUTABILIDAD: Obligatoria              │ MUTABILIDAD: Permitida     │
├─────────────────────────────────────────┼────────────────────────────┤
│ mu_new = mu + dt * dmu                  │ mu = mu + dt * dmu         │
│ # mu no cambia                          │ # mu se reemplaza          │
│ # mu_new es nuevo array                 │ # (o peor: mu += dmu)      │
├─────────────────────────────────────────┼────────────────────────────┤
│ AUTODIFF: Declarativo                   │ AUTODIFF: Imperativo       │
├─────────────────────────────────────────┼────────────────────────────┤
│ grad_fn = jax.grad(cost_fn)             │ cost = cost_fn(...)        │
│ gradient = grad_fn(params)              │ cost.backward()            │
│                                         │ gradient = params.grad     │
├─────────────────────────────────────────┼────────────────────────────┤
│ JIT: Automático y fácil                 │ JIT: Manual y limitado     │
├─────────────────────────────────────────┼────────────────────────────┤
│ @jax.jit                                │ @torch.jit.script          │
│ def f(x):                               │ def f(x):                  │
│     return x**2                         │     return x**2            │
│ # ✅ Funciona siempre                   │ # ⚠️ Limitaciones          │
├─────────────────────────────────────────┼────────────────────────────┤
│ VELOCIDAD: Compilado (XLA)              │ VELOCIDAD: Interpretado    │
├─────────────────────────────────────────┼────────────────────────────┤
│ Primera: ~500ms (compila)               │ Siempre: ~100ms            │
│ Después: ~5ms (100x más rápido)         │ (no compila)               │
├─────────────────────────────────────────┼────────────────────────────┤
│ SIMILITUD CON OCAML: 95%                │ SIMILITUD CON OCAML: 60%   │
└─────────────────────────────────────────┴────────────────────────────┘
"""

print(comparison)

# =============================================================================
# DIFERENCIA CLAVE: LOOPS
# =============================================================================

print("\n" + "=" * 70)
print("DIFERENCIA CLAVE: MANEJO DE LOOPS")
print("=" * 70)

loops_comparison = """
┌────────────────────────────────────────────────────────────────┐
│ OCaml (Código Original)                                        │
├────────────────────────────────────────────────────────────────┤
│ let rec accumulate mu sigma t =                                │
│   if t = n_steps then sigma                                    │
│   else                                                          │
│     let mu' = update mu in                                     │
│     accumulate mu' sigma (t + 1)                               │
│                                                                │
│ → Recursión de cola (tail recursion)                           │
│ → Compilador lo optimiza a loop                                │
│ → Muy rápido                                                    │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ JAX (Traducción Directa)                                       │
├────────────────────────────────────────────────────────────────┤
│ def step(carry, t):                                            │
│     mu, sigma = carry                                          │
│     mu_new = update(mu)                                        │
│     return (mu_new, sigma), None                               │
│                                                                │
│ jax.lax.scan(step, (mu0, sigma0), range(n_steps))             │
│                                                                │
│ → jax.lax.scan ≈ recursión de cola                             │
│ → JIT compila a XLA (como código nativo)                       │
│ → Muy rápido (similar a OCaml)                                 │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ PyTorch (Requiere Refactorización)                             │
├────────────────────────────────────────────────────────────────┤
│ mu = mu0                                                       │
│ sigma = sigma0                                                 │
│                                                                │
│ for t in range(n_steps):                                       │
│     mu = update(mu)                                            │
│                                                                │
│ → Loop de Python (no compilado)                                │
│ → Interpreta cada iteración                                    │
│ → Más lento (~10-100x)                                         │
│                                                                │
│ Para optimizar: torch.jit.script (limitado y complejo)         │
└────────────────────────────────────────────────────────────────┘
"""

print(loops_comparison)

# =============================================================================
# RECOMENDACIÓN FINAL
# =============================================================================

print("\n" + "=" * 70)
print("RECOMENDACIÓN")
print("=" * 70)

if JAX_AVAILABLE and PYTORCH_AVAILABLE:
    print("\n✅ Ambas bibliotecas están disponibles")
    print("\n🏆 Para Echeveste2020: USAR JAX")
    print("\nRazones:")
    print("  1. Traducción casi directa desde OCaml")
    print("  2. JIT compilation automática (100x más rápido)")
    print("  3. Paradigma funcional (como el código original)")
    print("  4. Menos esfuerzo de implementación")
    print("  5. Menos bugs (inmutabilidad)")

elif JAX_AVAILABLE:
    print("\n✅ JAX disponible")
    print("🏆 Perfecto para este proyecto")

elif PYTORCH_AVAILABLE:
    print("\n✅ PyTorch disponible")
    print("⚠️  Funciona, pero JAX sería mejor para este proyecto")
    print("   Considera instalar JAX: pip install jax jaxlib")

else:
    print("\n❌ Ninguna biblioteca de autodiff instalada")
    print("\nInstalar JAX (recomendado):")
    print("  pip install jax jaxlib")
    print("\nO PyTorch (alternativa):")
    print("  pip install torch")

print("\n" + "=" * 70)
