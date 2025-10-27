#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo Completo de Uso del Sistema Echeveste 2020.

Este ejemplo demuestra cómo usar el sistema completo de inferencia
probabilística basado en muestreo implementado según Echeveste et al. (2020).

El sistema combina:
1. Un modelo generativo GSM (Gaussian Scale Mixture)
2. Una red neuronal SSN (Stabilized Supralinear Network)
3. Inferencia causal mediante análisis de distribuciones posteriores
   multi-modales

Basado en: Echeveste et al. (2020) "Cortical-like dynamics in recurrent
           circuits optimized for sampling-based probabilistic inference"
"""

import numpy as np
from skneuromsi.generative import create_echeveste_gsm
from skneuromsi.neural import Echeveste2020


def ejemplo_sistema_completo():
    """
    Ejemplo completo mostrando el flujo completo de inferencia.

    1. Generación de estímulos visuales con GSM
    2. Procesamiento con red SSN
    3. Inferencia causal mediante calculate_causes
    """
    print("=" * 70)
    print("EJEMPLO COMPLETO: SISTEMA DE INFERENCIA ECHEVESTE 2020")
    print("=" * 70)

    # PASO 1: CONFIGURACIÓN DEL MODELO GENERATIVO (GSM)
    print("\n🔧 PASO 1: Configurando el modelo generativo GSM...")

    # Crear el modelo GSM con parámetros realistas
    # Fundamento: Echeveste et al. Fig. 1 - Gaussian Scale Mixture model
    gsm = create_echeveste_gsm(
        patch_size=16,        # 16x16 patches de imagen
        n_orientations=50,    # 50 orientaciones diferentes
        random_seed=42        # Para reproducibilidad
    )

    print("   ✅ GSM creado con:")
    print(f"      - Tamaño de patch: {gsm.patch_size}x{gsm.patch_size}")
    print(f"      - Número de orientaciones: {gsm.n_orientations}")
    print(f"      - Dimensión total: {gsm.patch_dim}")

    # PASO 2: CONFIGURACIÓN DE LA RED NEURONAL (SSN)
    print("\n🧠 PASO 2: Configurando la red SSN...")

    # Crear la red SSN con topología de anillo
    # Fundamento: Echeveste et al. Fig. 1c - ring topology neural network
    ssn = Echeveste2020(
        N_E=50,              # 50 neuronas excitatorias
        N_I=50,              # 50 neuronas inhibitorias
        random_seed=42       # Para reproducibilidad
    )

    print("   ✅ SSN creada con:")
    print(f"      - Neuronas excitatorias: {ssn._N_E}")
    print(f"      - Neuronas inhibitorias: {ssn._N_I}")
    print(f"      - Total de neuronas: {ssn._N}")
    print("      - Topología: anillo con conectividad local")

    # PASO 3: GENERACIÓN DE ESTÍMULOS VISUALES
    print("\n📷 PASO 3: Generando estímulos visuales...")

    # Definir diferentes niveles de contraste para probar el sistema
    # Fundamento: Echeveste et al. - el contraste z es la variable latente
    contrastes = [0.1, 0.3, 0.5, 0.8, 1.0]  # Desde bajo a alto contraste
    n_muestras = 3  # Múltiples muestras para cada contraste

    print(f"   🎯 Generando estímulos con contrastes: {contrastes}")

    # Generar estímulos usando el modelo GSM
    respuesta_gsm = gsm.generate_stimuli(
        contrasts=contrastes,
        n_samples=n_muestras
    )

    estimulos = respuesta_gsm['stimuli']      # Patches generados
    entradas_h = respuesta_gsm['h_inputs']    # Entradas para SSN
    contrastes_verdaderos = respuesta_gsm['true_contrasts']

    print("   ✅ Generados:")
    print(f"      - {len(contrastes)} niveles de contraste")
    print(f"      - {n_muestras} muestras por contraste")
    print(f"      - Forma de estímulos: {estimulos.shape}")
    print(f"      - Forma de entradas h: {entradas_h.shape}")

    # PASO 4: SIMULACIÓN DE LA RED SSN
    print("\n⚡ PASO 4: Simulando dinámicas de la red SSN...")

    # Procesar cada estímulo con la red SSN
    actividades_red = []
    dinamicas_temporales = []

    # Tomar una muestra representativa para análisis detallado
    idx_contraste = 2  # Contraste medio (0.5)
    idx_muestra = 0    # Primera muestra

    entrada_prueba = entradas_h[idx_contraste, idx_muestra]

    print(f"   🔍 Analizando estímulo con contraste {contrastes[idx_contraste]}")  # noqa: E501
    print(f"      - Magnitud de entrada h: {np.linalg.norm(entrada_prueba):.3f}")  # noqa: E501

    # Simular la evolución temporal de la red
    # Fundamento: Echeveste et al. Fig. 2 - cortical-like properties
    tiempo_simulacion = 100  # ms
    actividad, dinamica = ssn.simulate(
        h_input=entrada_prueba,
        simulation_time=tiempo_simulacion,
        return_dynamics=True
    )

    actividades_red.append(actividad)
    dinamicas_temporales.append(dinamica)

    # Analizar propiedades de la actividad resultante
    actividad_excitatoria = actividad[:ssn._N_E]
    actividad_inhibitoria = actividad[ssn._N_E:]

    print("   ✅ Simulación completada:")
    print(f"      - Actividad E promedio: {np.mean(actividad_excitatoria):.3f}")  # noqa: E501
    print(f"      - Actividad I promedio: {np.mean(actividad_inhibitoria):.3f}")  # noqa: E501
    print(f"      - Neurona E más activa: {np.argmax(actividad_excitatoria)} ")
    print(f"      - Pico de actividad E: {np.max(actividad_excitatoria):.3f}")

    # PASO 5: INFERENCIA CAUSAL CON calculate_causes
    print("\n🎯 PASO 5: Realizando inferencia causal...")

    # Usar calculate_causes para inferir causas presentes en la escena
    print("   🔍 Ejecutando calculate_causes...")
    print("      Fundamento teórico: Echeveste et al. Fig. 5")
    print("      - Análisis de distribuciones posteriores multi-modales")
    print("      - Detección de picos que corresponden a causas")
    print("      - Extracción de estadísticas causales")

    try:
        # Llamar a calculate_causes con la actividad de red simulada
        resultado_causal = ssn.calculate_causes(
            network_activity=actividad,
            stimulus=entrada_prueba,
            peak_threshold=0.1,      # Umbral para detectar picos
            peak_distance=5,         # Distancia mínima entre picos
            confidence_threshold=0.3  # Confianza mínima para causas válidas
        )

        print("   ✅ Inferencia causal completada:")
        print(f"      - Número de causas detectadas: {resultado_causal['num_causes']}")  # noqa: E501

        if resultado_causal['num_causes'] > 0:
            print(f"      - Contrastes de causas: {resultado_causal['cause_contrasts']}")  # noqa: E501
            causas_conf = resultado_causal['confidence_scores']
            print(f"      - Confianza de causas: {[f'{c:.3f}' for c in causas_conf]}")  # noqa: E501
            orientaciones = resultado_causal.get('cause_orientations', 'N/A')
            print(f"      - Orientaciones de causas: {orientaciones}")

            # Comparar con contraste verdadero
            contraste_verdadero = contrastes_verdaderos[idx_contraste,
                                                        idx_muestra]
            if resultado_causal['num_causes'] > 0:
                contraste_inferido = resultado_causal['cause_contrasts'][0]
            else:
                contraste_inferido = None

            print("   📊 COMPARACIÓN CON VERDAD:")
            print(f"      - Contraste verdadero: {contraste_verdadero:.3f}")
            if contraste_inferido:
                error_rel = abs(contraste_inferido - contraste_verdadero)
                error_rel = error_rel / contraste_verdadero
                print(f"      - Contraste inferido: {contraste_inferido:.3f}")
                print(f"      - Error relativo: {error_rel:.1%}")

                if error_rel < 0.2:
                    print("      ✅ EXCELENTE: Error < 20%")
                elif error_rel < 0.5:
                    print("      ⚠️  BUENO: Error < 50%")
                else:
                    print("      ❌ MEJORABLE: Error > 50%")
        else:
            print("      ⚠️ No se detectaron causas con los umbrales actuales")

    except Exception as e:
        print(f"   ❌ Error en inferencia causal: {e}")
        return

    # PASO 6: ANÁLISIS DE MÚLTIPLES ESTÍMULOS
    print("\n📈 PASO 6: Analizando múltiples estímulos...")

    # Procesar varios estímulos para mostrar robustez del sistema
    resultados_multiple = []

    for i, contraste in enumerate(contrastes[:3]):  # Solo primeros 3
        entrada = entradas_h[i, 0]  # Primera muestra de cada contraste

        # Simular red
        actividad_temp = ssn.simulate(h_input=entrada, simulation_time=50)

        try:
            # Inferir causas
            resultado_temp = ssn.calculate_causes(
                network_activity=actividad_temp,
                stimulus=entrada,
                peak_threshold=0.05,
                confidence_threshold=0.2
            )

            resultados_multiple.append({
                'contraste_real': contraste,
                'contraste_verdadero': contrastes_verdaderos[i, 0],
                'num_causas': resultado_temp['num_causes'],
                'causas_detectadas': resultado_temp['cause_contrasts'],
                'confianza': resultado_temp['confidence_scores']
            })

        except Exception as e:
            print(f"   ⚠️ Error procesando contraste {contraste}: {e}")
            continue

    # Mostrar resumen de resultados
    print("\n   📋 RESUMEN DE RESULTADOS:")
    print("   " + "-" * 50)
    for i, res in enumerate(resultados_multiple):
        print(f"   Estímulo {i+1}:")
        print(f"      Contraste objetivo: {res['contraste_real']:.2f}")
        print(f"      Contraste verdadero: {res['contraste_verdadero']:.3f}")
        print(f"      Causas detectadas: {res['num_causas']}")
        if res['num_causas'] > 0:
            mejor_causa = res['causas_detectadas'][0]
            mejor_conf = res['confianza'][0]
            print(f"      Mejor inferencia: {mejor_causa:.3f} "
                  f"(conf: {mejor_conf:.3f})")

    # PASO 7: INTERPRETACIÓN Y CONCLUSIONES
    print("\n🎓 PASO 7: Interpretación de resultados...")

    print("\n   💡 QUÉ OBSERVAMOS:")
    print("   • La red SSN convierte estímulos visuales en patrones de "
          "actividad")
    print("   • Los patrones reflejan muestras de distribuciones posteriores "
          "P(z|x)")
    print("   • calculate_causes extrae causas analizando multi-modalidad")
    print("   • El sistema infiere contraste y orientación de objetos "
          "visuales")

    print("\n   🔬 FUNDAMENTO CIENTÍFICO:")
    print("   • Implementa muestreo probabilístico Bayesiano con redes "
          "neuronales")
    print("   • Exhibe dinámicas similares a corteza visual (oscilaciones "
          "gamma)")
    print("   • Realiza inferencia causal mediante análisis de picos "
          "posteriores")
    print("   • Combina procesamiento bottom-up y expectativas top-down")

    print("\n   🎯 APLICACIONES POTENCIALES:")
    print("   • Segmentación de objetos en escenas visuales complejas")
    print("   • Inferencia de múltiples causas superpuestas")
    print("   • Modelado de procesos corticales de percepción visual")
    print("   • Sistemas de visión artificial bio-inspirados")

    print("\n" + "=" * 70)
    print("✅ EJEMPLO COMPLETADO EXITOSAMENTE")
    print("=" * 70)


def visualizar_resultados_ejemplo():
    """Función adicional para crear visualizaciones de los resultados."""
    print("\n📊 Función de visualización disponible para análisis detallado")
    print("   (Implementar según necesidades específicas)")


if __name__ == "__main__":
    # Ejecutar el ejemplo completo
    ejemplo_sistema_completo()

    print("\n💬 PENSAMIENTOS Y OBSERVACIONES ESPERADAS:")
    print("-" * 50)
    print("Al ejecutar este ejemplo, deberíamos observar:")
    print()
    print("1. 🏗️  CONSTRUCCIÓN DEL SISTEMA:")
    print("   - GSM y SSN se inicializan correctamente")
    print("   - Las dimensiones de los datos son consistentes")
    print("   - Los parámetros están en rangos biológicamente plausibles")
    print()
    print("2. 🎯 GENERACIÓN DE ESTÍMULOS:")
    print("   - Estímulos más contrastados tienen mayor varianza")
    print("   - Las entradas h capturan información de orientación")
    print("   - Los patrones reflejan la estructura de filtros Gabor")
    print()
    print("3. ⚡ DINÁMICAS DE RED:")
    print("   - La red SSN converge a estados estables")
    print("   - Actividad excitatoria forma picos localizados")
    print("   - Inhibición global controla la actividad total")
    print("   - Puede mostrar oscilaciones transitorias (gamma)")
    print()
    print("4. 🔍 INFERENCIA CAUSAL:")
    print("   - calculate_causes detecta picos en distribuciones "
          "posteriores")
    print("   - Número de causas refleja complejidad del estímulo")
    print("   - Contrastes inferidos aproximan valores verdaderos")
    print("   - Confianza correlaciona con claridad del estímulo")
    print()
    print("5. 📈 COMPORTAMIENTO DEL SISTEMA:")
    print("   - Mejor rendimiento con contrastes medios-altos (0.3-0.8)")
    print("   - Posible sobre-segmentación con umbrales muy bajos")
    print("   - Robustez ante variaciones en parámetros")
    print("   - Comportamiento estocástico por ruido inherente")
    print()
    print("6. 🎓 INTERPRETACIÓN BIOLÓGICA:")
    print("   - Simula procesamiento en corteza visual primaria")
    print("   - Implementa principios de codificación predictiva")
    print("   - Exhibe propiedades como adaptación y contextualización")
    print("   - Proporciona base para entender percepción visual")
