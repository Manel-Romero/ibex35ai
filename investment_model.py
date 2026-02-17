import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from factors import compute_factors

def prepare_data():
    market_df = pd.read_csv('ibex35_market_data.csv')
    news_df = pd.read_csv('ibex35_news_sentiment.csv')
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    market_df = market_df.sort_values(['Company', 'Date'])
    
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce', dayfirst=True)
    news_df = news_df.dropna(subset=['date'])
    
    daily_sentiment = news_df.groupby(['company', 'date'])['calibrated_score'].mean().reset_index()
    daily_sentiment.rename(columns={'company': 'Company', 'date': 'Date', 'calibrated_score': 'Daily_Sentiment'}, inplace=True)
    df = pd.merge(market_df, daily_sentiment, on=['Company', 'Date'], how='left')
    df['Daily_Sentiment'] = df.groupby('Company')['Daily_Sentiment'].ffill(limit=5).fillna(0.0)
    f = compute_factors(df)
    df['momentum_1w'] = f['momentum_1w']
    df['momentum_1m'] = f['momentum_1m']
    df['momentum_3m'] = f['momentum_3m']
    df['momentum_6m'] = f['momentum_6m']
    df['vol_1m'] = f['vol_1m']
    df['vol_3m'] = f['vol_3m']
    df['avg_volume_20'] = f['avg_volume_20']
    df['dist_52w_high'] = f['dist_52w_high']
    df['beta_3m'] = f['beta_3m']
    df['month_sin'] = f['month_sin']
    df['month_cos'] = f['month_cos']
    df['turn_of_month'] = f['turn_of_month']
    df['Sentiment_7D'] = df.groupby('Company')['Daily_Sentiment'].rolling(window=7).mean().reset_index(level=0, drop=True)
    df['Sentiment_30D'] = df.groupby('Company')['Daily_Sentiment'].rolling(window=30).mean().reset_index(level=0, drop=True)
    df['Forward_Return_1W'] = df.groupby('Company')['Close'].transform(lambda x: x.shift(-5) / x - 1)
    df['Target_1W'] = df['Forward_Return_1W'] - df.groupby('Date')['Forward_Return_1W'].transform('mean')
    return df

def train_and_predict():
    df = prepare_data()
    feature_cols = [
        'momentum_1w', 'momentum_1m', 'momentum_3m','momentum_6m','vol_1m','vol_3m','avg_volume_20',
        'dist_52w_high','beta_3m','month_sin','month_cos','turn_of_month',
        'Sentiment_7D','Sentiment_30D','Return_1M'
    ]
    df['Return_1M'] = df.groupby('Company')['Close'].pct_change(20)
    df_model = df.dropna(subset=feature_cols).copy()
    train_df = df_model.dropna(subset=['Target_1W'])
    df_model['Weekday'] = df_model['Date'].dt.weekday
    latest_feature_date = df_model[df_model['Weekday'] == 3]['Date'].max()
    if pd.isna(latest_feature_date):
        latest_feature_date = df_model['Date'].max()
    latest_predict_df = df_model[df_model['Date'] == latest_feature_date].copy()
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
    X_pred = latest_predict_df[feature_cols]
    predicted_returns = model.predict(X_pred)
    # Fix warning: Convert to numpy array to avoid feature names mismatch warning when accessing estimators directly
    preds_all = np.array([est.predict(X_pred.values) for est in model.estimators_])
    uncertainty = np.std(preds_all, axis=0)
    results = latest_predict_df[['Company', 'Ticker', 'Close', 'Date']].copy()
    results['Estimated_Return_1W'] = predicted_returns
    results['Uncertainty'] = uncertainty
    max_unc = results['Uncertainty'].quantile(0.95)
    min_unc = results['Uncertainty'].min()
    results['Security_Score'] = 100 * (1 - (results['Uncertainty'] - min_unc) / (max_unc - min_unc + 1e-6))
    results['Security_Score'] = results['Security_Score'].clip(0, 100)
    results = results.sort_values('Estimated_Return_1W', ascending=False)
    results.to_csv('investment_report.csv', index=False)
    return results

if __name__ == "__main__":
    train_and_predict()
