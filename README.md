# IBEX35 AI

## Descripción

Sistema en Python para analizar el IBEX35 usando datos de mercado y noticias.

## Instalación rápida

- Python 3.8 o superior  
- Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Flujo básico

1. Noticias y sentimiento:

```bash
python scraper.py
```

2. Datos de mercado:

```bash
python market_data.py
```

3. Modelo de predicción:

```bash
python investment_model.py
```

4. Carteras:

```bash
python generate_portfolios.py
```

5. Backtesting histórico:

```bash
python backtesting.py
```

## Ficheros importantes

- `ibex35_news_sentiment.csv`
- `ibex35_news_sentiment_historic.csv`
- `ibex35_market_data.csv`
- `ibex35_market_data_historic.csv`
- `investment_report.csv`
- `ibex35_portfolios.csv`

## Funcionamiento

- Noticias:
  - Se extraen enlaces con DuckDuckGo y fuentes preferidas.
  - Se analiza sentimiento con `pysentimiento` en español.
  - Se guarda un score diario por empresa en `ibex35_news_sentiment.csv` (campos clave: `date`, `ticker`, `company`, `calibrated_score`).
- Mercado:
  - Se descargan precios diarios con `yfinance` a `ibex35_market_data.csv` (campos clave: `Date`, `Close` o `Adj Close`, `Volume`, `Ticker`, `Company`).
- Factores:
  - Se calculan momentum, volatilidad, beta, distancia a máximos 52 semanas, estacionalidad y liquidez.
- Modelo:
  - Objetivo: retorno a 1 semana centrado por mercado.
  - Algoritmo: `XGBoost`.
  - Se genera `investment_report.csv` con `Estimated_Return_1W` y `Security_Score` (basado en la dispersión de los estimadores).
- Carteras:
  - Se combinan señales técnicas y de sentimiento.
  - Se aplican filtros simples de seguridad/umbral según estrategia.

## Backtesting histórico

- Universo:
  - Usa compañías actuales y antiguas del IBEX (ver `backtesting_companies.py`).
  - Solo afecta al backtesting; producción usa `companies.py`.
- Datos:
  - Por defecto lee `ibex35_market_data_historic.csv` y `ibex35_news_sentiment_historic.csv`.
  - Si no existen, se copian desde los CSV estándar la primera vez.
- Protocolo:
  - Walk-forward semanal con rebalanceo los jueves.
  - Entrenamiento con datos hasta la semana previa.
  - Límite de concentración por sector y filtro de liquidez.
  - Pesos acotados por posición.

## Supuestos y límites

- Sin comisiones ni deslizamientos.
- Se tienen en cuenta datos del viernes, aun comprando bajo el precio de cierre del jueves.
- Ejecución semanal simplificada.
- Dado a que los datos de `yfinance` no son correctos, se han modificado ciertos valores anómalos. Estos representan el 0.92% del total.
- Las noticias pueden ser incompletas o ruidosas.
- Las predicciones no garantizan resultados; `Security_Score` es una proxy de incertidumbre.
