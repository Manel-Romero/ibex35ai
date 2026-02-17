import pandas as pd
import numpy as np

def compute_factors(market_df):
    df = market_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Company', 'Date'])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0.0)
    df['ret_1d'] = df.groupby('Company')['Close'].pct_change(1)
    market_ret = df.groupby('Date')['ret_1d'].mean().rename('market_ret')
    df = df.merge(market_ret, on='Date', how='left')
    out = []
    for company, g in df.groupby('Company'):
        g = g.copy()
        g['momentum_1w'] = g['Close'].pct_change(5)
        g['momentum_1m'] = g['Close'].pct_change(20)
        g['momentum_3m'] = g['Close'].pct_change(65)
        g['momentum_6m'] = g['Close'].pct_change(130)
        g['vol_1m'] = g['ret_1d'].rolling(20).std()
        g['vol_3m'] = g['ret_1d'].rolling(65).std()
        g['avg_volume_20'] = g['Volume'].rolling(20).mean()
        g['liquidity_flag'] = (g['avg_volume_20'] > g['avg_volume_20'].quantile(0.2)).astype(int)
        high_252 = g['Close'].rolling(252).max()
        g['dist_52w_high'] = g['Close'] / high_252 - 1
        w = 65
        cov = g[['ret_1d','market_ret']].rolling(w).cov().unstack().iloc[:,1]
        var_m = g['market_ret'].rolling(w).var()
        g['beta_3m'] = cov / (var_m + 1e-9)
        m = g['Date'].dt.month
        g['month_sin'] = np.sin(2*np.pi*m/12.0)
        g['month_cos'] = np.cos(2*np.pi*m/12.0)
        g['turn_of_month'] = ((g['Date'].dt.day <= 3) | (g['Date'].dt.day >= 28)).astype(int)
        out.append(g)
    return pd.concat(out)
