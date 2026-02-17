import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
from factors import compute_factors
from backtesting_companies import IBEX35_SECTORS_HISTORIC as IBEX35_SECTORS

def prepare_merged(market_csv='ibex35_market_data_historic.csv', sentiment_csv='ibex35_news_sentiment_historic.csv', use_sentiment=True):
    if not os.path.exists(market_csv) and os.path.exists('ibex35_market_data.csv'):
        pd.read_csv('ibex35_market_data.csv').to_csv(market_csv, index=False)
    if not os.path.exists(sentiment_csv) and os.path.exists('ibex35_news_sentiment.csv'):
        pd.read_csv('ibex35_news_sentiment.csv').to_csv(sentiment_csv, index=False)
    m = pd.read_csv(market_csv)
    s = pd.read_csv(sentiment_csv)
    m['Date'] = pd.to_datetime(m['Date'])
    s['date'] = pd.to_datetime(s['date'], dayfirst=True, errors='coerce')
    s = s.dropna(subset=['date'])
    s = s[['date','ticker','company','sector','calibrated_score']].rename(columns={'date':'Date'})
    daily = s.groupby(['company','Date'])['calibrated_score'].mean().reset_index().rename(columns={'calibrated_score':'Daily_Sentiment', 'company':'Company'})
    df = pd.merge(m, daily, on=['Company','Date'], how='left')
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df = df[df[price_col].notna() & (df[price_col] > 0)]
    if use_sentiment:
        df['Daily_Sentiment'] = df.groupby('Company')['Daily_Sentiment'].ffill(limit=5).fillna(0.0)
        
    f = compute_factors(df)
    cols = ['momentum_1w','momentum_1m','momentum_3m','momentum_6m','vol_1m','vol_3m','avg_volume_20','liquidity_flag']
    df[cols] = f[cols]
    df['dist_52w_high'] = f['dist_52w_high']
    df['beta_3m'] = f['beta_3m']
    df['month_sin'] = f['month_sin']
    df['month_cos'] = f['month_cos']
    df['turn_of_month'] = f['turn_of_month']
    
    if use_sentiment:
        df['Sentiment_30D'] = df.groupby('Company')['Daily_Sentiment'].rolling(30).mean().reset_index(level=0, drop=True)
        df['Sentiment_7D'] = df.groupby('Company')['Daily_Sentiment'].rolling(7).mean().reset_index(level=0, drop=True)
    df['Return_1D'] = df.groupby('Company')[price_col].pct_change(1)
    df['Return_1M'] = df.groupby('Company')[price_col].pct_change(20)
    df['Forward_Return_1W'] = df.groupby('Company')[price_col].transform(lambda x: x.shift(-5) / x - 1)
    df['Target_1W'] = df['Forward_Return_1W'] - df.groupby('Date')['Forward_Return_1W'].transform('mean')
    return df

def select_universe(df_row, sector_counts, max_per_sector):
    return sector_counts.get(df_row['sector'], 0) < max_per_sector and df_row['liquidity_flag'] == 1

def backtest_walkforward(train_years=5, top_n=6, max_per_sector=2, max_weight=0.2, use_sentiment=True, confidence_threshold=0.005, start_date=None):
    df = prepare_merged(use_sentiment=use_sentiment)
    if start_date is not None:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    base_cols = ['momentum_1w','momentum_1m','momentum_3m','momentum_6m','vol_1m','vol_3m','avg_volume_20','Return_1M','dist_52w_high','beta_3m','month_sin','month_cos','turn_of_month']
    
    if use_sentiment:
        df = df.dropna(subset=base_cols + ['Sentiment_7D','Sentiment_30D'])
        feature_cols = base_cols + ['Sentiment_7D','Sentiment_30D']
    else:
        df = df.dropna(subset=base_cols)
        feature_cols = base_cols
        
    # Identificar Viernes (Weekday 4)
    df['Weekday'] = df['Date'].dt.weekday
    trade_dates = sorted(df[df['Weekday'] == 4]['Date'].unique())
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        start_idx = next((i for i, d in enumerate(trade_dates) if d >= start_dt), len(trade_dates))
    else:
        start_idx = 0
        while start_idx < len(trade_dates) and (trade_dates[start_idx] - trade_dates[0]).days < train_years*365:
            start_idx += 1
    
    print(f"Iniciando Backtest Semanal (Viernes con datos del día previo) - {len(trade_dates)-start_idx} semanas...")
    
    curves = []
    period_returns = []
    benchmark_returns = []
    trade_log = []
    
    # Entrenar cada X semanas para eficiencia (ej. cada 4 semanas = 1 mes)
    retrain_freq = 4
    model = None
    
    for i in range(start_idx, len(trade_dates) - 1):
        curr_date = trade_dates[i]
        next_date = trade_dates[i+1]
        feature_date = df[df['Date'] < curr_date]['Date'].max()
        if pd.isna(feature_date):
            continue
        
        # Retrain logic
        if (i - start_idx) % retrain_freq == 0:
            print(f"Re-entrenando modelo en {curr_date.date()}...")
            # Entrenar con datos DISPONIBLES hasta hoy (excluyendo look-ahead del target)
            # Target_1W en t necesita precio en t+5. 
            # Por tanto, para entrenar en t, solo podemos usar filas donde t_row + 5 <= t
            # Simplificación: Usar datos hasta hace 1 semana para asegurar etiquetas
            train_limit_date = feature_date - timedelta(days=7)
            train_df = df[df['Date'] <= train_limit_date].dropna(subset=['Target_1W'])
            
            if len(train_df) > 100:
                X = train_df[feature_cols]
                y = train_df['Target_1W']
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=12,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y)
        
        if model is None:
            continue
            
        predict_df = df[df['Date'] == feature_date].copy()
        if predict_df.empty:
            continue
            
        # Benchmark Return (Market Average of universe)
        # Calcular retorno del mercado entre curr_date y next_date
        # Necesitamos precios en next_date para el universo disponible en curr_date
        
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        next_prices = df[df['Date'] == next_date].set_index('Company')[price_col]
        curr_prices = df[df['Date'] == curr_date].set_index('Company')[price_col]
        if curr_prices.empty or next_prices.empty:
            period_returns.append(0.0)
            benchmark_returns.append(0.0)
            continue
        
        # Calcular retornos reales para todos los activos
        # Solo para los que existen en ambas fechas
        common_companies = curr_prices.index.intersection(next_prices.index)
        if len(common_companies) == 0:
            period_returns.append(0.0)
            benchmark_returns.append(0.0)
            continue
            
        market_rets = (next_prices[common_companies] / curr_prices[common_companies]) - 1
        market_period_return = market_rets.mean()
        
        # Preparar predicción
        sector_map = predict_df['Company'].map(lambda c: IBEX35_SECTORS.get(predict_df[predict_df['Company']==c]['Ticker'].iloc[0], 'Unknown'))
        predict_df['sector'] = sector_map
        
        Xp = predict_df[feature_cols]
        predict_df['pred'] = model.predict(Xp)
        predict_df = predict_df.sort_values('pred', ascending=False)
        
        # Selección
        sector_counts = {}
        selected = []
        for _, row in predict_df.iterrows():
            if row['Company'] not in common_companies: continue # Skip if delisted next week
            
            sec = row['sector']
            sector_counts.setdefault(sec, 0)
            if select_universe(row, sector_counts, max_per_sector):
                selected.append(row)
                sector_counts[sec] += 1
            if len(selected) >= top_n:
                break
        
        if len(selected) == 0:
            strat_ret = 0.0
        else:
            sel_df = pd.DataFrame(selected)
            avg_pred = sel_df['pred'].mean()
            if avg_pred < confidence_threshold:
                strat_ret = 0.0
            else:
                raw_weights = sel_df['pred'].clip(lower=0.0).to_numpy()
                if raw_weights.sum() == 0:
                    strat_ret = 0.0
                else:
                    weights = raw_weights / raw_weights.sum()
                    if max_weight is not None:
                        weights = np.minimum(weights, max_weight)
                        weights = weights / weights.sum()
                    sel_rets = market_rets[sel_df['Company']].values
                    strat_ret = float(np.sum(weights * sel_rets))
                    
                    for idx, row in sel_df.iterrows():
                        real_ret = market_rets[row['Company']]
                        trade_log.append({
                            'Date': curr_date,
                            'Ticker': row.get('Ticker', 'Unknown'),
                            'Company': row['Company'],
                            'Sector': row['sector'],
                            'Weight': float(weights[sel_df.index.get_loc(idx)]),
                            'Predicted_Return': row['pred'],
                            'Realized_Return': real_ret,
                            'Market_Return': market_period_return,
                            'Alpha': real_ret - market_period_return
                        })
        
        period_returns.append(strat_ret)
        benchmark_returns.append(market_period_return)

    if not period_returns:
        return pd.DataFrame()

    trades_df = pd.DataFrame(trade_log)
    trades_df.to_csv('backtest_trades.csv', index=False)
    print(f"\nGuardado registro detallado de operaciones en 'backtest_trades.csv' ({len(trades_df)} registros)")

    equity_curve = [10000.0]
    benchmark_curve = [10000.0]
    
    for r, b_r in zip(period_returns, benchmark_returns):
        equity_curve.append(equity_curve[-1] * (1 + r))
        benchmark_curve.append(benchmark_curve[-1] * (1 + b_r))
    
    total_return = (equity_curve[-1] / 10000.0) - 1
    benchmark_total_return = (benchmark_curve[-1] / 10000.0) - 1
    
    years = (trade_dates[-1] - trade_dates[start_idx]).days / 365.0
    cagr = (equity_curve[-1] / 10000.0) ** (1/years) - 1 if years > 0 else 0
    benchmark_cagr = (benchmark_curve[-1] / 10000.0) ** (1/years) - 1 if years > 0 else 0
    
    alpha_total = total_return - benchmark_total_return
    alpha_annual = cagr - benchmark_cagr

    print(f"\n--- Resultados Backtest ({years:.1f} años) ---")
    print(f"Capital Final Estrategia: {equity_curve[-1]:.2f}€ (Inicio: 10,000€)")
    print(f"Capital Final Mercado:    {benchmark_curve[-1]:.2f}€")
    print(f"Retorno Total: {total_return*100:.2f}% (Mercado: {benchmark_total_return*100:.2f}%)")
    print(f"CAGR (Anual):  {cagr*100:.2f}% (Mercado: {benchmark_cagr*100:.2f}%)")
    print(f"Alpha Total:   {alpha_total*100:.2f}%")
    print(f"Alpha Anual:   {alpha_annual*100:.2f}%")
    
    return pd.DataFrame({'Date': trade_dates[start_idx:][:len(equity_curve)], 'Equity': equity_curve, 'Benchmark': benchmark_curve})

if __name__ == "__main__":
    backtest_walkforward()
