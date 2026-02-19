## Estrategias de carteras

Este documento resume cómo se construye cada cartera en `generate_portfolios.py`: filtros, ponderaciones y objetivo.

Conceptos básicos:
- **Estimated_Return_1W**: retorno esperado a 1 semana del modelo.
- **Security_Score**: confianza del modelo (0–100; más alto = más confianza).
- **Sentiment_Score**: sentimiento medio reciente de noticias.
- **Sentiment_Accel**: aceleración del sentimiento (mejora/empeora frente a la semana anterior).
- **Combined_Score**: ranking interno = `Estimated_Return_1W * 2 * w_tech + Sentiment_Score * w_sent`.

Proceso común:
- Se filtra el universo según reglas de cada estrategia (mínimo de seguridad, retorno, sentimiento, etc.).
- Se ordena por `Combined_Score` descendente.
- Se prueban carteras con k = 1, 2, …, N mejores valores.
- Para cada k:
  - Se calculan pesos `w_i` según la métrica de la estrategia.
  - Se calcula el retorno esperado de cartera: `R = Σ w_i * Estimated_Return_1W_i`.
  - Se calcula concentración `HHI = Σ w_i²`.
  - Se maximiza `R - div_lambda * HHI` (si `div_lambda > 0`), equilibrando retorno y diversificación.
- Si incluso la mejor cartera tiene retorno medio < 0, se queda en **CASH**.

### Technical Pure

- Objetivo: maximizar retorno del modelo ajustado por confianza.
- Filtros:
  - No tiene mínimo de seguridad (`min_sec = 0`).
- Ponderación:
  - `weight_metric = "tech_conf"`.
  - Base de peso: `Estimated_Return_1W ≥ 0`.
  - Peso final: `w_i ∝ Estimated_Return_1W_i * (Security_Score_i / 100)`.
- Diversificación:
  - `div_lambda ≈ 0.0004`: prefiere varias posiciones si dan retorno similar, pero puede concentrarse si hay un claro ganador.

### Model Raw

- Objetivo: replicar la lógica “raw” del backtest (solo modelo, sin confianza ni sentimiento).
- Filtros:
  - Sin mínimo de seguridad adicional (`min_sec = 0`).
- Ponderación:
  - `weight_metric = "raw_backtest"`.
  - `w_i ∝ max(Estimated_Return_1W_i, 0)`.
  - Es equivalente a usar las predicciones como pesos positivos y normalizarlas.
- Diversificación:
  - `div_lambda ≈ 0.0002`: ligera preferencia por varias posiciones, pero sigue siendo muy agresiva.

### Sentiment Pure

- Objetivo: cartera basada solo en sentimiento de noticias.
- Filtros:
  - Sin mínimo de seguridad.
- Ponderación:
  - `weight_metric = "sentiment"`.
  - `w_i ∝ (Sentiment_Score_i − mínimo del grupo)`, con un pequeño desplazamiento para evitar ceros.
- Diversificación:
  - `div_lambda ≈ 0.0005`: bastante peso a no concentrar toda la posición en un único valor.

### Balanced Aggressive

- Objetivo: combinar señal técnica y sentimiento, con perfil agresivo.
- Filtros:
  - `Security_Score ≥ 20` (descarta los más inciertos).
- Ranking:
  - `w_tech = 0.6`, `w_sent = 0.4` en el `Combined_Score`.
- Ponderación:
  - `weight_metric = "return_exp"`.
  - `w_i ∝ exp(Estimated_Return_1W_i * 50)` (tras truncar negativos a 0).
  - Acentúa mucho al mejor valor cuando hay diferencias claras.
- Diversificación:
  - `div_lambda ≈ 0.0006`: empuja a usar más de un valor si el retorno no cae demasiado.

### Balanced Conservative

- Objetivo: combinación técnica + sentimiento con prioridad a estabilidad.
- Filtros:
  - `Security_Score ≥ 80`.
- Ranking:
  - `w_tech = 0.4`, `w_sent = 0.6` (más peso a sentimiento).
- Ponderación:
  - Usa métrica `"return"` por defecto: `w_i ∝ Estimated_Return_1W_i` ajustado para que todos sean ≥ 0.
- Diversificación:
  - `div_lambda ≈ 0.0008`: fuerte penalización a carteras muy concentradas.

### High Safety Technical

- Objetivo: técnica pura, pero solo con valores muy fiables.
- Filtros:
  - `Security_Score ≥ 85`.
- Ponderación:
  - `weight_metric = "return"` (retorno del modelo).
- Diversificación:
  - `div_lambda ≈ 0.0006`: busca un equilibrio entre concentrar en el mejor y repartir entre varios de alta seguridad.

### High Safety Sentiment

- Objetivo: sentimiento fuerte solo en valores muy seguros.
- Filtros:
  - `Security_Score ≥ 85`.
- Ponderación:
  - `weight_metric = "sentiment"`.
- Diversificación:
  - `div_lambda ≈ 0.0008`: tiende a repartir entre varios nombres con buen sentimiento y alta seguridad.

### Risk Taker (Low Safety)

- Objetivo: buscar oportunidades con baja seguridad (alto riesgo).
- Filtros:
  - `Security_Score ≤ 40` (máximo).
- Ponderación:
  - Misma lógica de combinación técnica/sentimiento que corresponda, sin peso extra a diversificación.
- Diversificación:
  - `div_lambda = 0`: no penaliza la concentración; puede acabar en uno o pocos valores.

### Momentum (High Both)

- Objetivo: valores con buen retorno esperado y buen sentimiento.
- Filtros:
  - `Estimated_Return_1W ≥ 0.005` (0.5% a 1 semana).
  - `Sentiment_Score ≥ 0.2`.
- Ranking:
  - `w_tech = 0.5`, `w_sent = 0.5`.
- Ponderación:
  - Métrica `"return"` por defecto (retorno esperado).
- Diversificación:
  - `div_lambda ≈ 0.0005`: busca varios nombres, no solo uno, siempre que el retorno lo permita.

### Contrarian (Tech>0, Sent<0)

- Objetivo: estrategia contraria al sentimiento.
- Filtros:
  - Solo valores con `Estimated_Return_1W > 0.005` y `Sentiment_Score < 0`.
- Ponderación:
  - Usa `"return"` por defecto (retorno del modelo).
- Diversificación:
  - Sin `div_lambda` explícito: prioriza retorno, con menos énfasis en diversificación que otras carteras defensivas.

### News Hype (Accel)

- Objetivo: capturar subidas cuando el sentimiento está mejorando rápido.
- Filtros:
  - `Sentiment_Score > 0.2`.
  - `Sentiment_Accel > 0.05` (sentimiento reciente subiendo frente a la semana anterior).
- Ponderación:
  - Basada en retorno técnico por defecto; el filtro ya asegura la parte de sentimiento.
- Diversificación:
  - Sin `div_lambda` explícito: se concentra si hay una señal muy fuerte.

### Steady Growth

- Objetivo: crecimiento estable, más defensivo.
- Filtros:
  - `Security_Score ≥ 70`.
  - `Estimated_Return_1W ≥ 0.003`.
- Ponderación:
  - `weight_metric = "return"` (retorno esperado).
- Diversificación:
  - `div_lambda ≈ 0.0007`: favorece carteras con varios valores sólidos frente a una sola apuesta.

### Speculative

- Objetivo: apuestas especulativas con baja seguridad.
- Filtros:
  - `Security_Score ≤ 30`.
- Ponderación:
  - Basada en retorno técnico.
- Diversificación:
  - Sin `div_lambda` explícito: admite una concentración muy fuerte en el mejor valor especulativo.

### Killer

- Objetivo: estrategia muy agresiva para destacar en competiciones (máximo impacto en la mejor oportunidad).
- Filtros:
  - `Security_Score ≥ 20` (evita los muy extremos).
- Ranking:
  - `w_tech = 0.8`, `w_sent = 0.2`.
- Ponderación:
  - `weight_metric = "return_exp"`: `w_i ∝ exp(Estimated_Return_1W_i * 50)`.
  - Esto hace que una pequeña ventaja de retorno se convierta en mucha más inversión.
- Diversificación:
  - Sin `div_lambda`: la cartera puede acabar fuertemente concentrada en 1–2 valores si el modelo ve una oportunidad clara.

