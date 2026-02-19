import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from companies import IBEX35_SECTORS

def parse_date(date_str):
    if pd.isna(date_str): return None
    for fmt in ('%d/%m/%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(str(date_str), fmt)
        except ValueError:
            continue
    return None

def compute_weights(top_picks, strat):
    weight_metric = strat.get("weight_metric", "return")
    if weight_metric == "return":
        base = top_picks['Estimated_Return_1W']
        min_base = base.min()
        base = base - min_base + 1e-6
        weights = base.clip(lower=0.0)
    elif weight_metric == "raw_backtest":
        base = top_picks['Estimated_Return_1W'].clip(lower=0.0)
        weights = base
    elif weight_metric == "tech_conf":
        base = top_picks['Estimated_Return_1W'].clip(lower=0.0)
        conf = top_picks['Security_Score'].clip(lower=0.0) / 100.0
        weights = base * conf
    elif weight_metric == "return_exp":
        base = top_picks['Estimated_Return_1W'].clip(lower=0.0)
        if base.sum() == 0:
            return pd.Series([1] * len(top_picks), index=top_picks.index)
        scaled = np.exp(base * 50.0)
        weights = pd.Series(scaled, index=top_picks.index)
    elif weight_metric == "sentiment":
        base = top_picks['Sentiment_Score']
        min_base = base.min()
        base = base - min_base + 1e-6
        weights = base.clip(lower=0.0)
    elif weight_metric == "security":
        weights = top_picks['Security_Score'].clip(lower=0.0)
    elif weight_metric == "combined":
        base = top_picks['Combined_Score']
        min_base = base.min()
        base = base - min_base + 1e-6
        weights = base.clip(lower=0.0)
    else:
        weights = pd.Series([1] * len(top_picks), index=top_picks.index)
    total = float(weights.sum())
    if total <= 0:
        return pd.Series([1] * len(top_picks), index=top_picks.index) / len(top_picks)
    return weights / total

def generate_portfolios():
    try:
        report_df = pd.read_csv('investment_report.csv')
        sent_df = pd.read_csv('ibex35_news_sentiment.csv')
    except FileNotFoundError:
        print("Error: Faltan archivos CSV necesarios")
        return

    if 'date' not in sent_df.columns:
        if 'published_date' in sent_df.columns:
             sent_df['date'] = sent_df['published_date']
    
    report_df['Date'] = pd.to_datetime(report_df['Date'], errors='coerce')
    reference_date = report_df['Date'].max()
    if pd.isna(reference_date):
        reference_date = datetime.now()
    sent_df['dt'] = pd.to_datetime(sent_df['date'], dayfirst=True, errors='coerce')
    cutoff = reference_date - timedelta(days=120)
    sent_df = sent_df[sent_df['dt'] >= cutoff]
    
    if 'ticker' not in sent_df.columns and 'Ticker' in sent_df.columns:
        sent_df['ticker'] = sent_df['Ticker']
        
    score_col = 'calibrated_score' if 'calibrated_score' in sent_df.columns else 'sentiment_score'
    
    daily_sent = sent_df.groupby(['ticker', 'dt'])[score_col].mean().reset_index()
    daily_sent = daily_sent.sort_values(['ticker', 'dt'])
    daily_sent['Sent_7D'] = daily_sent.groupby('ticker')[score_col].rolling(window=7, min_periods=3).mean().reset_index(level=0, drop=True)
    daily_sent['Sent_7D_prev'] = daily_sent.groupby('ticker')['Sent_7D'].shift(7)
    daily_sent['Sent_Accel'] = daily_sent['Sent_7D'] - daily_sent['Sent_7D_prev']
    latest = daily_sent[daily_sent['dt'] <= reference_date]
    if latest.empty:
        latest = daily_sent
    latest = latest.sort_values(['ticker', 'dt']).groupby('ticker').tail(1)
    latest = latest[['ticker', 'Sent_7D', 'Sent_Accel']]
    latest.rename(columns={'Sent_7D': 'Sentiment_Score', 'Sent_Accel': 'Sentiment_Accel'}, inplace=True)
    
    df = pd.merge(report_df, latest, left_on='Ticker', right_on='ticker', how='left')
    df['Sentiment_Score'] = df['Sentiment_Score'].fillna(0)
    df['Sentiment_Accel'] = df['Sentiment_Accel'].fillna(0)
    
    df['Sector'] = df['Ticker'].map(IBEX35_SECTORS).fillna('Unknown')

    strategies = [
        {"name": "Technical Pure", "w_tech": 1.0, "w_sent": 0.0, "min_sec": 0, "weight_metric": "tech_conf", "div_lambda": 0.0004},
        {"name": "Model Raw", "w_tech": 1.0, "w_sent": 0.0, "min_sec": 0, "weight_metric": "raw_backtest", "div_lambda": 0.0002},
        {"name": "Sentiment Pure", "w_tech": 0.0, "w_sent": 1.0, "min_sec": 0, "div_lambda": 0.0005},
        {"name": "Balanced Aggressive", "w_tech": 0.6, "w_sent": 0.4, "min_sec": 20, "weight_metric": "return_exp", "div_lambda": 0.0006},
        {"name": "Balanced Conservative", "w_tech": 0.4, "w_sent": 0.6, "min_sec": 80, "div_lambda": 0.0008},
        {"name": "High Safety Technical", "w_tech": 1.0, "w_sent": 0.0, "min_sec": 85, "div_lambda": 0.0006},
        {"name": "High Safety Sentiment", "w_tech": 0.0, "w_sent": 1.0, "min_sec": 85, "div_lambda": 0.0008},
        {"name": "Risk Taker (Low Safety)", "w_tech": 0.8, "w_sent": 0.2, "max_sec": 40},
        {"name": "Momentum (High Both)", "w_tech": 0.5, "w_sent": 0.5, "min_return": 0.005, "min_sent": 0.2, "div_lambda": 0.0005},
        {"name": "Contrarian (Tech>0, Sent<0)", "custom": lambda r: r['Estimated_Return_1W'] > 0.005 and r['Sentiment_Score'] < 0},
        {"name": "News Hype (Accel)", "custom": lambda r: r['Sentiment_Score'] > 0.2 and r['Sentiment_Accel'] > 0.05},
        {"name": "Steady Growth", "w_tech": 1.0, "min_sec": 70, "min_return": 0.003, "div_lambda": 0.0007},
        {"name": "Speculative", "w_tech": 1.0, "max_sec": 30},
        {"name": "Killer", "w_tech": 0.8, "w_sent": 0.2, "min_sec": 20, "weight_metric": "return_exp"}
    ]
    
    results = []
    
    BUDGET = 10000
    confidence_threshold = 0.0
    market_avg_return = df['Estimated_Return_1W'].mean()
    
    for strat in strategies:
        temp_df = df.copy()
        
        if "min_sec" in strat:
            temp_df = temp_df[temp_df['Security_Score'] >= strat["min_sec"]]
        if "max_sec" in strat:
            temp_df = temp_df[temp_df['Security_Score'] <= strat["max_sec"]]
        if "min_return" in strat:
            temp_df = temp_df[temp_df['Estimated_Return_1W'] >= strat["min_return"]]
        if "min_sent" in strat:
            temp_df = temp_df[temp_df['Sentiment_Score'] >= strat["min_sent"]]
            
        if "custom" in strat:
            temp_df = temp_df[temp_df.apply(strat["custom"], axis=1)]
        
        if "w_tech" in strat and "w_sent" in strat:
            temp_df['Combined_Score'] = (temp_df['Estimated_Return_1W'] * 2 * strat["w_tech"]) + (temp_df['Sentiment_Score'] * strat["w_sent"])
        elif "w_tech" in strat:
             temp_df['Combined_Score'] = temp_df['Estimated_Return_1W']
        else:
            temp_df['Combined_Score'] = temp_df['Estimated_Return_1W']
            
        temp_df = temp_df.sort_values('Combined_Score', ascending=False)
        
        if temp_df.empty:
            continue
        
        max_n = strat.get("max_n", len(temp_df))
        max_n = min(max_n, len(temp_df))
        
        best_objective = None
        best_top_picks = None
        div_lambda = float(strat.get("div_lambda", 0.0))
        
        for k in range(1, max_n + 1):
            cand = temp_df.head(k).copy()
            weights_k = compute_weights(cand, strat)
            avg_return_k = float((cand['Estimated_Return_1W'] * weights_k).sum())
            if div_lambda > 0.0:
                hhi = float((weights_k ** 2).sum())
                objective = avg_return_k - div_lambda * hhi
            else:
                objective = avg_return_k
            if best_objective is None or objective > best_objective:
                best_objective = objective
                best_top_picks = cand
        
        if best_top_picks is None:
            continue
        
        top_picks = best_top_picks
        
        if top_picks.empty:
            continue
        
        avg_pred = top_picks['Estimated_Return_1W'].mean()
        if avg_pred < confidence_threshold:
            results.append({
                "Portfolio_Name": strat["name"],
                "Tickers_Allocation": f"CASH ({BUDGET}€)",
                "Investment_Euros": BUDGET,
                "Estimated_Return_Pct": 0.0,
                "Estimated_Alpha_Pct": 0.0,
                "Estimated_Gain_Euros": 0.0,
                "Avg_Security_Score": 0.0
            })
            continue
            
        weights = compute_weights(top_picks, strat)
        allocations = weights * BUDGET
        allocations_int = allocations.astype(int)
        budget_int = int(BUDGET)
        diff_int = budget_int - int(allocations_int.sum())
        if len(allocations_int) > 0:
            first_idx = allocations_int.index[0]
            allocations_int.loc[first_idx] += diff_int
        top_picks['Allocation'] = allocations_int
        
        tickers_str = ", ".join([f"{row['Ticker']} ({int(row['Allocation'])}€)" for _, row in top_picks.iterrows()])
        
        avg_return = (top_picks['Estimated_Return_1W'] * weights).sum()
        total_gain = avg_return * BUDGET
        avg_security = (top_picks['Security_Score'] * weights).sum()
        
        estimated_alpha = avg_return - market_avg_return
        
        results.append({
            "Portfolio_Name": strat["name"],
            "Tickers_Allocation": tickers_str,
            "Investment_Euros": BUDGET,
            "Estimated_Return_Pct": round(avg_return * 100, 2),
            "Estimated_Alpha_Pct": round(estimated_alpha * 100, 2),
            "Estimated_Gain_Euros": round(total_gain, 2),
            "Avg_Security_Score": round(avg_security, 2)
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv('ibex35_portfolios.csv', index=False)
    print("Carteras generadas en 'ibex35_portfolios.csv'")
    print(res_df.to_markdown(index=False))

if __name__ == "__main__":
    generate_portfolios()
