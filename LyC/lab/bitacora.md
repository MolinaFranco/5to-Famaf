# Bitácora de desarrollo – Compilador en Haskell  
**Basado en _Theories of Programming Languages_ de John C. Reynolds**

---

## Etapa inicial

Ya tenía instalado GHC, por lo que comencé directamente con la construcción del proyecto. Mi primer paso fue dividir el sistema en módulos bien definidos, lo cual me permitiría avanzar ordenadamente y escalar el lenguaje progresivamente.

---

## División en archivos

Separé el código en cuatro partes principales, de acuerdo con los tipos de expresiones y estructuras básicas del lenguaje imperativo simple estudiado en el libro de Reynolds:

- **Expresiones booleanas (`BoolExp`)**
- **Expresiones enteras (`IntExp`)**
- **Comandos (`Comm`)**
- **Estado (`State`)**

Esta estructura modular permite trabajar con claridad sobre cada componente por separado, y facilita la extensión del sistema más adelante.

---

## Diseño de gramáticas e intérpretes

Me basé en las gramáticas presentadas en el capítulo 2 del libro para definir cada una de las estructuras:

- **Expresiones enteras**: Incluyen constantes, variables, negaciones y operaciones binarias como suma, resta, multiplicación, división y resto. Las expresiones están representadas como tipos algebraicos y se interpretan mediante una función que evalúa su resultado dado un estado.

- **Expresiones booleanas**: Incorporan constantes lógicas (`true`, `false`), negación, operadores lógicos (conjunción, disyunción, implicación, equivalencia) y comparaciones entre expresiones enteras. Implementé un intérprete que evalúa una expresión booleana en un estado.

- **Comandos**: Incluyen las construcciones básicas de un lenguaje imperativo: `skip`, asignaciones, secuencias, condicionales, definición de variables locales (`newvar`) y bucles `while`. Por el momento decidí no implementar `fail`, entrada/salida ni excepciones para poder establecer primero una semántica sólida.

- **Estado**: Implementé una representación funcional del estado como una función que asigna a cada variable su valor actual. También construí una función auxiliar para actualizar ese estado de forma inmutable.

- **Decision de Diseño**: Modifiqué las firmas y definiciones de todos los intérpretes para que el State se pase como último parámetro. Esta convención es más idiomática en Haskell, facilita la currificación y permite una composición de funciones más natural. Afectó a los módulos IntExp, BoolExp y Comm.

---

## Implementación del intérprete

Para cada estructura sintáctica definí su correspondiente función de interpretación. Estas funciones aplican la semántica del lenguaje sobre una estructura y un estado, devolviendo el valor o estado resultante.

---

## Consideraciones sobre el bucle `while`

Uno de los puntos más complejos fue el manejo del bucle `while`. Al principio intenté representarlo en términos del punto fijo, como sugiere el libro, pero encontré que en Haskell era mucho más natural y claro usar una definición recursiva. Esta solución es más directa, fácil de implementar y mantiene la misma semántica funcional.

Reemplacé la definición recursiva directa del comando while en interpComm por una función auxiliar local evalWhile, de modo que el intérprete de comandos queda más claramente guiado por la sintaxis. Esta reescritura mantiene el estilo funcional, pero facilita la comprensión del flujo de ejecución de bucles y prepara el terreno para futuras extensiones (como seguimiento de pasos, conteo de iteraciones o límites de ejecución).

---

## Bitácora técnica resumida

- Ya tenía GHC instalado.
- Separé los archivos en módulos independientes: expresiones booleanas, expresiones enteras, comandos y estado.
- Para las gramáticas me basé en los capítulos 1 y 2 del libro.
- Implementé los intérpretes de cada módulo.
- Decidí arrancar sin `fail`, ni entrada/salida, para establecer bien las bases y luego hacer crecer el sistema si es necesario.
- Lo más complicado fue comprender y resolver el `while`, que finalmente implementé de forma recursiva, por claridad y funcionalidad.

---

## Próximos pasos

Con la base funcional ya implementada, planeo avanzar en las siguientes direcciones:

- Agregar manejo de errores aritméticos (por ejemplo, división por cero)
- Implementar el comando `fail` y modelar entrada/salida
- Explorar semánticas alternativas como las de continuaciones
- Incorporar verificación de especificaciones como en el capítulo 3 del libro
- Añadir análisis estático o control de tipos en etapas posteriores

---
