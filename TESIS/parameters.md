| Parámetro | Descripción | Valor |
| :--- | :--- | :--- |
| **Modelo Generativo** | | |
| N_x | Número de variables observadas (píxeles) | 256 |
| N_y | Número de variables latentes | 50 |
| Matriz A | Matriz de filtros Gabor orientados | (Extended Data Fig. 1 y repositorio) |
| C | Matriz de covarianza a priori | (Extended Data Fig. 1 y repositorio) |
| sigma_x | Desviación estándar del ruido de píxel | 10.0 |
| K | Parámetro de forma (Shape) de la gamma a priori en contraste | 2.0 |
| theta | Parámetro de escala (Scale) de la gamma a priori en contraste | 2.0 |
| alfa_nl | Escala de la transformación no lineal | 2.4 |
| beta_nl | Línea de base (Baseline) de la transformación no lineal | 1.9 |
| gamma_nl | Potencia (Power) de la transformación no lineal | 0.6 |
| **Red Neuronal** | | |
| N | Número total de neuronas | 100 |
| N_E | Número de células E | 50 |
| N_I | Número de células I | 50 |
| tau_E | Constante de tiempo de membrana de células E | 20 ms |
| tau_I | Constante de tiempo de membrana de células I | 10 ms |
| tau_eta | Constante de tiempo del ruido de proceso | 20 ms |
| W | Matriz de pesos | (optimizado, ver repositorio) |
| n | Exponente en la función de transferencia de célula única | 2 |
| k | Factor de escala en la función de transferencia de célula única | 0.3 |
| Sigma_eta | Covarianza del ruido de proceso | (optimizado, ver repositorio) |
| mu_0 | Media de la distribución inicial | 0 |
| Sigma_0 | Covarianza de la distribución inicial | 41 |
| dt | Paso de tiempo de simulación | 0.2 ms |
| **Entrada (Input)** | | |
| W^ff | Pesos feed-forward | [A A]^T / 15.0 |
| alpha_h | Escala de entrada | 1.96 (optimizado) |
| beta_h | Línea de base de entrada | 0.10 (optimizado) |
| gamma_h | Potencia de entrada | 2.03 (optimizado) |
| **Cálculo del Factor Fano** | | |
| T | Ventana de observación | 100 ms |
| K_ISI | Parámetro de forma (Shape) de la distribución gamma del intervalo inter-espiga | 1.15 |
| **Cálculo de la Conductancia de Entrada** | | |
| C_m | Capacitancia de membrana | 20.0 pF |
| V_rest | Potencial de membrana en reposo (nivel de referencia para u) | -65 mV |
| V^E | Potencial de reversión E | 0 mV |
| V^I | Potencial de reversión I | -80 mV |
| **Optimización** | | |
| E_mean | Peso de costo para el error de la media | 4.0e-5 |
| E_var | Peso de costo para el error de la varianza | 8.0e-5 |
| E_cov | Peso de costo para el error de la covarianza | 8.0e-7 |
| E_slow | Peso de costo para la penalización de lentitud | 4.0e-10 (solo ADF) |
| T_max | Tiempo de presentación del estímulo / Tiempo final de integración | 500 ms |
| T_min | Tiempo inicial de integración | 0 a 450 ms |
| tau_max | Tiempo de integración de autocorrelación | 100 ms |
| N_trials | Número de ensayos para optimización estocástica | 50 |
| dt' | Paso de tiempo de simulación durante la optimización | 0.2 ms |