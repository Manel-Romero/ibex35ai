import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from factors import compute_factors

if __name__ == "__main__":
    market_df = pd.read_csv('ibex35_market_data.csv')
    news_df = pd.read_csv('ibex35_news_sentiment.csv')

    market_df['Date'] = pd.to_datetime(market_df['Date'])
    market_df = market_df.sort_values(['Company', 'Date'])

    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce', dayfirst=True)
    news_df = news_df.dropna(subset=['date'])

    daily_sentiment = news_df.groupby(['company', 'date'])['calibrated_score'].mean().reset_index()
    daily_sentiment = daily_sentiment.rename(
        columns={'company': 'Company', 'date': 'Date', 'calibrated_score': 'Daily_Sentiment'}
    )

    data = pd.merge(market_df, daily_sentiment, on=['Company', 'Date'], how='left')
    data['Daily_Sentiment'] = data.groupby('Company')['Daily_Sentiment'].ffill(limit=5).fillna(0.0)

    factors = compute_factors(data)
    data['momentum_1w'] = factors['momentum_1w']
    data['momentum_1m'] = factors['momentum_1m']
    data['momentum_3m'] = factors['momentum_3m']
    data['momentum_6m'] = factors['momentum_6m']
    data['vol_1m'] = factors['vol_1m']
    data['vol_3m'] = factors['vol_3m']
    data['avg_volume_20'] = factors['avg_volume_20']
    data['dist_52w_high'] = factors['dist_52w_high']
    data['beta_3m'] = factors['beta_3m']
    data['month_sin'] = factors['month_sin']
    data['month_cos'] = factors['month_cos']
    data['turn_of_month'] = factors['turn_of_month']

    data['Sentiment_7D'] = (
        data.groupby('Company')['Daily_Sentiment']
        .rolling(window=7)
        .mean()
        .reset_index(level=0, drop=True)
    )
    data['Sentiment_30D'] = (
        data.groupby('Company')['Daily_Sentiment']
        .rolling(window=30)
        .mean()
        .reset_index(level=0, drop=True)
    )

    data['Forward_Return_1W'] = data.groupby('Company')['Close'].transform(
        lambda x: x.shift(-5) / x - 1
    )
    data['Target_1W'] = data['Forward_Return_1W'] - data.groupby('Date')['Forward_Return_1W'].transform('mean')

    feature_cols = [
        'momentum_1w',
        'momentum_1m',
        'momentum_3m',
        'momentum_6m',
        'vol_1m',
        'vol_3m',
        'avg_volume_20',
        'dist_52w_high',
        'beta_3m',
        'month_sin',
        'month_cos',
        'turn_of_month',
        'Sentiment_7D',
        'Sentiment_30D',
        'Return_1M',
    ]

    data['Return_1M'] = data.groupby('Company')['Close'].pct_change(20)
    model_data = data.dropna(subset=feature_cols).copy()
    train_data = model_data.dropna(subset=['Target_1W'])

    model_data['Weekday'] = model_data['Date'].dt.weekday
    latest_feature_date = model_data[model_data['Weekday'] == 3]['Date'].max()
    if pd.isna(latest_feature_date):
        latest_feature_date = model_data['Date'].max()

    predict_data = model_data[model_data['Date'] == latest_feature_date].copy()

    X = train_data[feature_cols]
    y = train_data['Target_1W']

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    X_pred = predict_data[feature_cols]
    predicted_returns = model.predict(X_pred)
    preds_all = np.array([est.predict(X_pred.values) for est in model.estimators_])
    uncertainty = np.std(preds_all, axis=0)

    results = predict_data[['Company', 'Ticker', 'Close', 'Date']].copy()
    results['Estimated_Return_1W'] = predicted_returns
    results['Uncertainty'] = uncertainty

    max_uncertainty = results['Uncertainty'].quantile(0.95)
    min_uncertainty = results['Uncertainty'].min()

    results['Security_Score'] = 100 * (
        1 - (results['Uncertainty'] - min_uncertainty) / (max_uncertainty - min_uncertainty + 1e-6)
    )
    results['Security_Score'] = results['Security_Score'].clip(0, 100)

    results = results.sort_values('Estimated_Return_1W', ascending=False)
    results.to_csv('investment_report.csv', index=False)
