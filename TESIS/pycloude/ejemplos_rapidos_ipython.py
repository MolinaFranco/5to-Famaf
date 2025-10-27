#!/usr/bin/env python3
"""
🚀 EJEMPLOS RÁPIDOS PARA IPYTHON - ECHEVESTE 2020

Script con funciones listas para usar en IPython interactivo.
Simplemente carga este archivo y ejecuta las funciones.

Uso en IPython:
    %run ejemplos_rapidos_ipython.py
    ejemplo_original()
    ejemplo_nuestro()
    comparacion_rapida()
"""

import sys
import numpy as np
import os

# Configurar paths automáticamente
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi')
sys.path.insert(0, '/home/molina/FAMAF/5to-Famaf/TESIS/'
                'ssn_inference_numerical_experiments/GSM')

print("🚀 EJEMPLOS RÁPIDOS CARGADOS")
print("=" * 40)
print("Funciones disponibles:")
print("  • ejemplo_original()        - Ejecutar código original")
print("  • ejemplo_nuestro()         - Ejecutar nuestra implementación")
print("  • comparacion_rapida()      - Comparar ambos")
print("  • test_causas()             - Test de inferencia causal")
print("  • test_variabilidad()       - Test de variabilidad")
print("  • mostrar_ayuda()           - Mostrar esta ayuda")
print("=" * 40)


def ejemplo_original(contraste=0.32, ruido=10.0, mostrar=True):
    """
    🔬 EJEMPLO CÓDIGO ORIGINAL

    Ejecuta el código original de Echeveste et al. (2020).

    Args:
        contraste (float): Contraste del estímulo (default: 0.32)
        ruido (float): Nivel de ruido (default: 10.0)
        mostrar (bool): Mostrar resultados (default: True)

    Returns:
        dict: Resultados del código original
    """
    if mostrar:
        print("🔬 EJECUTANDO CÓDIGO ORIGINAL")
        print(f"   Contraste: {contraste}, Ruido: {ruido}")

    # Guardar directorio actual
    original_dir = os.getcwd()

    try:
        # Cambiar al directorio GSM
        gsm_path = ('/home/molina/FAMAF/5to-Famaf/TESIS/'
                    'ssn_inference_numerical_experiments/GSM')
        os.chdir(gsm_path)

        # Importar GSM
        import GSM

        # Cargar filtros
        A = np.load("filters.npy")

        # Crear estímulo
        D_y = A.shape[1]  # 50 orientaciones
        y = np.zeros(D_y)
        y[0] = 1.0  # Orientación horizontal

        # Generar observación
        x = GSM.get_x(y, contraste, A, ruido)

        # Calcular estadísticas
        resultado = {
            'x': x,
            'media': np.mean(x),
            'std': np.std(x),
            'min': np.min(x),
            'max': np.max(x),
            'forma': x.shape,
            'parametros': {'contraste': contraste, 'ruido': ruido},
            'success': True
        }

        if mostrar:
            print(f"   ✅ ÉXITO:")
            print(f"      Forma: {resultado['forma']}")
            print(f"      Media: {resultado['media']:.4f}")
            print(f"      Std: {resultado['std']:.4f}")
            print(f"      Rango: [{resultado['min']:.4f}, {resultado['max']:.4f}]")

        return resultado

    except Exception as e:
        if mostrar:
            print(f"   ❌ ERROR: {e}")
        return {'success': False, 'error': str(e)}

    finally:
        os.chdir(original_dir)


def ejemplo_nuestro(contraste=0.32, orientacion=0.0, tiempo=1000.0, ruido=0.1, mostrar=True):
    """
    🚀 EJEMPLO NUESTRA IMPLEMENTACIÓN

    Ejecuta nuestra implementación del modelo Echeveste2020.

    Args:
        contraste (float): Contraste del estímulo (default: 0.32)
        orientacion (float): Orientación en grados (default: 0.0)
        tiempo (float): Tiempo de simulación en ms (default: 1000.0)
        ruido (float): Nivel de ruido (default: 0.1)
        mostrar (bool): Mostrar resultados (default: True)

    Returns:
        dict: Resultados de nuestra implementación
    """
    if mostrar:
        print(f"🚀 EJECUTANDO NUESTRA IMPLEMENTACIÓN")
        print(f"   Contraste: {contraste}, Orientación: {orientacion}°")
        print(f"   Tiempo: {tiempo}ms, Ruido: {ruido}")

    try:
        # Importar nuestra implementación
        from skneuromsi.neural import Echeveste2020

        # Crear instancia
        ssn = Echeveste2020(N_E=50, N_I=50, seed=42)

        # Cargar parámetros
        ssn.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')

        # Ejecutar simulación
        response = ssn.run(
            stimulus_contrast=contraste,
            stimulus_orientation=orientacion,
            simulation_time=tiempo,
            noise_level=ruido
        )

        # Extraer actividad
        df = response.get_modes()
        n_times = int(tiempo / 0.2)
        activity_exc = df['excitatory'].values.reshape(n_times, 50)
        activity_inh = df['inhibitory'].values.reshape(n_times, 50)

        # Calcular estadísticas
        resultado = {
            'activity_exc': activity_exc,
            'activity_inh': activity_inh,
            'response': response,
            'ssn': ssn,
            'media': np.mean(activity_exc),
            'std': np.std(activity_exc),
            'min': np.min(activity_exc),
            'max': np.max(activity_exc),
            'forma': activity_exc.shape,
            'media_final': np.mean(activity_exc[-500:]),  # Último medio segundo
            'parametros': {
                'contraste': contraste,
                'orientacion': orientacion,
                'tiempo': tiempo,
                'ruido': ruido
            },
            'success': True
        }

        if mostrar:
            print(f"   ✅ ÉXITO:")
            print(f"      Forma: {resultado['forma']}")
            print(f"      Media: {resultado['media']:.4f}")
            print(f"      Std: {resultado['std']:.4f}")
            print(f"      Rango: [{resultado['min']:.4f}, {resultado['max']:.4f}]")
            print(f"      Media final: {resultado['media_final']:.4f}")

        return resultado

    except Exception as e:
        if mostrar:
            print(f"   ❌ ERROR: {e}")
        return {'success': False, 'error': str(e)}


def comparacion_rapida(contraste=0.32, mostrar=True):
    """
    🎯 COMPARACIÓN RÁPIDA

    Ejecuta ambas implementaciones y las compara.

    Args:
        contraste (float): Contraste para ambas implementaciones
        mostrar (bool): Mostrar resultados

    Returns:
        dict: Resultados de ambas implementaciones
    """
    if mostrar:
        print(f"🎯 COMPARACIÓN RÁPIDA (contraste={contraste})")
        print("=" * 50)

    # Ejecutar original
    orig = ejemplo_original(contraste=contraste, mostrar=False)

    # Ejecutar nuestro
    nues = ejemplo_nuestro(contraste=contraste, mostrar=False)

    if mostrar:
        print(f"📊 RESULTADOS:")
        print(f"   Original:")
        if orig['success']:
            print(f"      Media: {orig['media']:.4f}")
            print(f"      Std: {orig['std']:.4f}")
            print(f"      Forma: {orig['forma']}")
        else:
            print(f"      ❌ Error: {orig.get('error', 'Desconocido')}")

        print(f"   Nuestro:")
        if nues['success']:
            print(f"      Media: {nues['media']:.4f}")
            print(f"      Std: {nues['std']:.4f}")
            print(f"      Forma: {nues['forma']}")
        else:
            print(f"      ❌ Error: {nues.get('error', 'Desconocido')}")

        # Comparación directa
        if orig['success'] and nues['success']:
            ratio_media = orig['media'] / nues['media'] if nues['media'] != 0 else float('inf')
            ratio_std = orig['std'] / nues['std'] if nues['std'] != 0 else float('inf')

            print(f"   Ratios (Original/Nuestro):")
            print(f"      Media: {ratio_media:.2f}x")
            print(f"      Std: {ratio_std:.2f}x")

    return {'original': orig, 'nuestro': nues}


def test_causas(mostrar=True):
    """
    🧠 TEST DE INFERENCIA CAUSAL

    Ejecuta test de inferencia causal usando nuestra implementación.

    Args:
        mostrar (bool): Mostrar resultados

    Returns:
        dict: Resultados de inferencia causal
    """
    if mostrar:
        print(f"🧠 TEST DE INFERENCIA CAUSAL")
        print("=" * 30)

    try:
        # Ejecutar simulación
        resultado = ejemplo_nuestro(mostrar=False)

        if not resultado['success']:
            if mostrar:
                print(f"❌ Error en simulación: {resultado.get('error', 'Desconocido')}")
            return resultado

        # Preparar datos para inferencia
        activity_exc = resultado['activity_exc']
        activity_inh = resultado['activity_inh']
        ssn = resultado['ssn']

        # Calcular estado de red promedio
        mean_exc = np.mean(activity_exc, axis=0)
        mean_inh = np.mean(activity_inh, axis=0)
        network_state = np.concatenate([mean_exc, mean_inh])

        # Inferencia causal
        causes = ssn.calculate_causes(
            network_activity=network_state,
            confidence_threshold=0.95
        )

        # Agregar resultados de causas
        resultado.update({
            'network_state': network_state,
            'causes': causes,
            'n_causas': causes['num_causes']
        })

        if mostrar:
            print(f"   ✅ INFERENCIA COMPLETADA:")
            print(f"      Número de causas: {causes['num_causes']}")
            print(f"      Posiciones: {causes['cause_positions']}")
            print(f"      Contrastes: {causes['cause_contrasts']}")
            print(f"      Confianza: {causes['confidence']:.4f}")

        return resultado

    except Exception as e:
        if mostrar:
            print(f"   ❌ ERROR: {e}")
        return {'success': False, 'error': str(e)}


def test_variabilidad(n_runs=5, mostrar=True):
    """
    🔬 TEST DE VARIABILIDAD

    Ejecuta múltiples simulaciones para analizar variabilidad.

    Args:
        n_runs (int): Número de ejecuciones (default: 5)
        mostrar (bool): Mostrar resultados

    Returns:
        list: Lista de resultados de todas las ejecuciones
    """
    if mostrar:
        print(f"🔬 TEST DE VARIABILIDAD ({n_runs} ejecuciones)")
        print("=" * 40)
        print("Run | Media | Std | Causas")
        print("-" * 30)

    resultados = []
    medias = []
    stds = []
    causas = []

    for i in range(n_runs):
        try:
            # Importar nueva instancia
            from skneuromsi.neural import Echeveste2020

            # Nueva instancia con diferente seed
            ssn = Echeveste2020(N_E=50, N_I=50, seed=42+i)
            ssn.load_parameters('/home/molina/FAMAF/5to-Famaf/TESIS/scikit-neuromsi/skneuromsi/data/echeveste2020/')

            # Simulación
            response = ssn.run(
                stimulus_contrast=0.32,
                stimulus_orientation=0.0,
                simulation_time=1000.0,
                noise_level=0.1
            )

            # Análisis
            df = response.get_modes()
            activity_exc = df['excitatory'].values.reshape(5000, 50)
            activity_inh = df['inhibitory'].values.reshape(5000, 50)

            # Inferencia causal
            mean_e = np.mean(activity_exc, axis=0)
            mean_i = np.mean(activity_inh, axis=0)
            net_state = np.concatenate([mean_e, mean_i])

            causes_result = ssn.calculate_causes(network_activity=net_state)

            # Guardar resultados
            resultado = {
                'run': i+1,
                'media': np.mean(activity_exc),
                'std': np.std(activity_exc),
                'n_causas': causes_result['num_causes'],
                'activity_exc': activity_exc,
                'activity_inh': activity_inh,
                'causes': causes_result,
                'success': True
            }

            resultados.append(resultado)
            medias.append(resultado['media'])
            stds.append(resultado['std'])
            causas.append(resultado['n_causas'])

            if mostrar:
                print(f"{i+1:3d} | {resultado['media']:.3f} | {resultado['std']:.2f} | {resultado['n_causas']:6d}")

        except Exception as e:
            resultado_error = {
                'run': i+1,
                'success': False,
                'error': str(e)
            }
            resultados.append(resultado_error)

            if mostrar:
                print(f"{i+1:3d} | ERROR: {e}")

    # Análisis de variabilidad
    if medias:
        cv = np.std(medias) / np.mean(medias) * 100

        if mostrar:
            print(f"\n📊 ANÁLISIS DE VARIABILIDAD:")
            print(f"   Coeficiente de variación: {cv:.2f}%")
            print(f"   Rango de medias: [{np.min(medias):.3f}, {np.max(medias):.3f}]")
            print(f"   Causas únicas detectadas: {set(causas)}")
            print(f"   Ejecuciones exitosas: {len(medias)}/{n_runs}")

    return resultados


def mostrar_ayuda():
    """📚 MOSTRAR AYUDA COMPLETA"""
    print("📚 AYUDA COMPLETA - EJEMPLOS RÁPIDOS ECHEVESTE 2020")
    print("=" * 60)
    print()

    print("🔧 FUNCIONES PRINCIPALES:")
    print()

    print("1. ejemplo_original(contraste=0.32, ruido=10.0)")
    print("   Ejecuta el código original GSM")
    print("   Retorna: dict con x, estadísticas y parámetros")
    print()

    print("2. ejemplo_nuestro(contraste=0.32, orientacion=0.0, tiempo=1000.0, ruido=0.1)")
    print("   Ejecuta nuestra implementación SSN")
    print("   Retorna: dict con actividad, estadísticas y objetos")
    print()

    print("3. comparacion_rapida(contraste=0.32)")
    print("   Compara ambas implementaciones")
    print("   Retorna: dict con ambos resultados")
    print()

    print("4. test_causas()")
    print("   Test de inferencia causal")
    print("   Retorna: dict con causas detectadas")
    print()

    print("5. test_variabilidad(n_runs=5)")
    print("   Test de variabilidad estocástica")
    print("   Retorna: list de resultados de todas las ejecuciones")
    print()

    print("📊 EJEMPLOS DE USO:")
    print()
    print("# Ejemplo básico original")
    print("orig = ejemplo_original()")
    print()
    print("# Ejemplo básico nuestro")
    print("nues = ejemplo_nuestro()")
    print()
    print("# Comparación directa")
    print("comp = comparacion_rapida(contraste=0.8)")
    print()
    print("# Test de causas")
    print("causas = test_causas()")
    print("print(f'Causas detectadas: {causas[\"n_causas\"]}')")
    print()
    print("# Test de variabilidad")
    print("var = test_variabilidad(n_runs=10)")
    print("medias = [r['media'] for r in var if r['success']]")
    print("print(f'CV: {np.std(medias)/np.mean(medias)*100:.2f}%')")
    print()

    print("🎯 PARÁMETROS TÍPICOS:")
    print("   contraste: 0.05, 0.1, 0.32, 0.8, 1.0")
    print("   orientacion: 0.0, 45.0, 90.0, 135.0, 180.0")
    print("   tiempo: 500.0, 1000.0, 2000.0 (ms)")
    print("   ruido: 0.05, 0.1, 0.2, 0.3")
    print()

    print("💡 TIPS:")
    print("   • Usa mostrar=False para ejecutar sin output")
    print("   • Guarda resultados para análisis posterior")
    print("   • Los objetos ssn y response son reutilizables")
    print("   • Experimenta con diferentes parámetros")


def benchmark_completo():
    """⚡ BENCHMARK COMPLETO"""
    print("⚡ BENCHMARK COMPLETO")
    print("=" * 30)

    import time

    # Benchmark original
    print("🔬 Original...")
    start = time.time()
    orig = ejemplo_original(mostrar=False)
    tiempo_orig = time.time() - start

    # Benchmark nuestro
    print("🚀 Nuestro...")
    start = time.time()
    nues = ejemplo_nuestro(mostrar=False)
    tiempo_nues = time.time() - start

    # Benchmark causas
    print("🧠 Causas...")
    start = time.time()
    causas = test_causas(mostrar=False)
    tiempo_causas = time.time() - start

    # Resultados
    print(f"\n⏱️  TIEMPOS DE EJECUCIÓN:")
    print(f"   Original: {tiempo_orig:.3f}s")
    print(f"   Nuestro: {tiempo_nues:.3f}s")
    print(f"   Causas: {tiempo_causas:.3f}s")
    print(f"   Total: {tiempo_orig + tiempo_nues + tiempo_causas:.3f}s")

    return {
        'original': {'resultado': orig, 'tiempo': tiempo_orig},
        'nuestro': {'resultado': nues, 'tiempo': tiempo_nues},
        'causas': {'resultado': causas, 'tiempo': tiempo_causas}
    }


# Mensaje final
print("\n💡 CONSEJO: Ejecuta mostrar_ayuda() para ver ejemplos de uso detallados")
print("🚀 ¡Listo para usar! Prueba: ejemplo_original() o ejemplo_nuestro()")