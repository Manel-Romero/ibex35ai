# Sistema de Inversión IA IBEX35

## Descripción del Proyecto
Este proyecto implementa un sistema automatizado para el análisis de *swing trading* en el mercado IBEX35. Integra análisis de sentimiento avanzado mediante modelos Transformer (BERT) con datos técnicos de mercado para generar recomendaciones de inversión con horizonte semanal.

El sistema está diseñado para maximizar retornos mientras gestiona el riesgo a través de estrategias de construcción de carteras diversificadas, que van desde enfoques puramente técnicos hasta selecciones impulsadas por el sentimiento de mercado.

## Arquitectura del Sistema

El flujo de trabajo consta de tres etapas principales:

1.  **Adquisición de Datos y Análisis de Sentimiento (`scraper.py`)**
    *   Extrae noticias financieras de múltiples fuentes para empresas del IBEX35.
    *   Analiza el sentimiento utilizando `pysentimiento` (roberta-base-bne) con una estrategia de fragmentación (*chunking*) para procesar textos largos.
    *   Genera `ibex35_news_sentiment.csv` que contiene puntuaciones de sentimiento calibradas (Probabilidad Positiva - Probabilidad Negativa).

2.  **Modelado Predictivo (`investment_model.py`)**
    *   Fusiona datos de sentimiento con datos históricos de mercado (`ibex35_market_data.csv`).
    *   Calcula indicadores técnicos: Momentum, Volatilidad, Beta, Liquidez y estacionalidad.
    *   Entrena un *Random Forest Regressor* para predecir retornos a una semana.
    *   Produce `investment_report.csv` con retornos estimados y puntuaciones de seguridad.

3.  **Generación de Carteras (`generate_portfolios.py`)**
    *   Construye carteras de inversión diversificadas basadas en diferentes perfiles de riesgo y estrategias.
    *   Las estrategias incluyen: Técnica Pura, Sentimiento Puro, Balanceada Agresiva, Crecimiento Estable, entre otras.
    *   Optimiza la asignación de capital mediante una distribución ponderada basada en el potencial de retorno o puntuaciones de seguridad.
    *   Genera `ibex35_portfolios.csv`.

## Requisitos Previos

*   Python 3.8+
*   Paquetes de Python necesarios (instalar vía `pip install -r requirements.txt` si está disponible, de lo contrario asegurar la instalación de los siguientes):
    *   `pandas`, `numpy`, `scikit-learn`
    *   `pysentimiento`, `torch`
    *   `duckduckgo-search`, `newspaper3k`, `lxml`
    *   `yfinance`

## Guía de Uso

Ejecute los siguientes scripts en orden para realizar el ciclo completo de análisis.

### 1. Recolección de Datos y Análisis de Sentimiento
Ejecute el scraper para recopilar las últimas noticias y calcular puntuaciones de sentimiento. Este proceso puede tomar tiempo dependiendo del volumen de noticias.

```bash
python scraper.py
```

### 2. Entrenamiento del Modelo y Predicción
Entrene el modelo de IA utilizando los últimos datos de sentimiento y mercado para generar predicciones de retorno.

```bash
python investment_model.py
```

### 3. Generación de Carteras
Genere carteras de inversión optimizadas basadas en las predicciones del modelo.

```bash
python generate_portfolios.py
```

## Archivos de Salida

*   **`ibex35_news_sentiment.csv`**: Datos de sentimiento crudos para cada noticia procesada.
*   **`investment_report.csv`**: Informe de predicción detallado por empresa, incluyendo retorno estimado semanal y puntuación de confianza/seguridad.
*   **`ibex35_portfolios.csv`**: Recomendaciones finales de inversión agrupadas por estrategia, incluyendo la asignación de capital específica por empresa.

## Metodología

### Análisis de Sentimiento
*   **Modelo**: Utiliza `pysentimiento/roberta-base-bne`, un modelo pre-entrenado en textos en español.
*   **Fragmentación**: Los artículos se dividen en fragmentos de 1500 caracteres para asegurar una cobertura completa dentro del límite de tokens del modelo.
*   **Puntuación**: La puntuación final es una métrica calibrada derivada de la probabilidad de sentimiento positivo menos la probabilidad de sentimiento negativo.

### Modelo de Inversión
*   **Objetivo**: Retorno a 1 semana, decisión el viernes con datos del jueves.
*   **Características**: Factores técnicos de corto plazo y sentimiento.
*   **Estrategia de Trading**: Rebalanceo semanal sin costes de transacción, ejecución los viernes.

### Asignación de Cartera
*   **Estrategia**: Selección semanal de los mejores activos basada en predicción de retorno a corto plazo.

## Resultados del Backtesting

1978

## Notas Técnicas sobre la Estrategia de Backtesting

Es importante notar que la estrategia utilizada en el *Backtesting Walk-Forward* difiere ligeramente de las carteras generadas para producción (`generate_portfolios.py`):

1.  **Lógica "Pure Alpha"**: El backtest simula una estrategia de **Maximización de Retorno Predicho** sin filtrar por incertidumbre (`Security_Score`). Es equivalente a una versión optimizada de la cartera *Technical Pure* con restricciones sectoriales estrictas.
2.  **Gestión de Riesgo Estructural**:
    *   **Diversificación Forzada**: Máximo 2 activos por sector GICS.
    *   **Universo de Inversión**: Limitado dinámicamente por liquidez (`liquidity_flag`).
3.  **Diferencia con Producción**: Las carteras de producción (ej. *Balanced*, *High Safety*) incorporan capas adicionales de seguridad (filtrado por varianza de predicción) y factores de sentimiento, sacrificando potencial de retorno máximo a cambio de una menor volatilidad y mayor robustez ("Sharpe Ratio" optimizado).

El backtesting valida la **potencia predictiva bruta** del modelo (capacidad de generar Alpha), mientras que las carteras de producción adaptan esa señal para diferentes perfiles de aversión al riesgo.

---
**Aviso Legal**: Este software es solo para fines educativos y de investigación. No constituye asesoramiento financiero.
