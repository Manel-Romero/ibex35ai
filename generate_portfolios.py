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
    
    sent_agg = sent_df.groupby('ticker')[score_col].mean().reset_index()
    sent_agg.rename(columns={score_col: 'Sentiment_Score'}, inplace=True)
    
    df = pd.merge(report_df, sent_agg, left_on='Ticker', right_on='ticker', how='left')
    df['Sentiment_Score'] = df['Sentiment_Score'].fillna(0)
    
    df['Sector'] = df['Ticker'].map(IBEX35_SECTORS).fillna('Unknown')

    strategies = [
        {"name": "Technical Pure", "w_tech": 1.0, "w_sent": 0.0, "min_sec": 0},
        {"name": "Model Raw (Backtest Logic)", "w_tech": 1.0, "w_sent": 0.0, "min_sec": 0},
        {"name": "Sentiment Pure", "w_tech": 0.0, "w_sent": 1.0, "min_sec": 0},
        {"name": "Balanced Aggressive", "w_tech": 0.6, "w_sent": 0.4, "min_sec": 20},
        {"name": "Balanced Conservative", "w_tech": 0.4, "w_sent": 0.6, "min_sec": 80},
        {"name": "High Safety Technical", "w_tech": 1.0, "w_sent": 0.0, "min_sec": 85},
        {"name": "High Safety Sentiment", "w_tech": 0.0, "w_sent": 1.0, "min_sec": 85},
        {"name": "Risk Taker (Low Safety)", "w_tech": 0.8, "w_sent": 0.2, "max_sec": 40},
        {"name": "Momentum (High Both)", "w_tech": 0.5, "w_sent": 0.5, "min_return": 0.005, "min_sent": 0.2},
        {"name": "Contrarian (Tech>0, Sent<0)", "custom": lambda r: r['Estimated_Return_1W'] > 0.005 and r['Sentiment_Score'] < 0},
        {"name": "News Hype (Sent>0.5)", "custom": lambda r: r['Sentiment_Score'] > 0.5},
        {"name": "Steady Growth", "w_tech": 1.0, "min_sec": 70, "min_return": 0.003},
        {"name": "Speculative", "w_tech": 1.0, "max_sec": 30}
    ]
    
    results = []
    
    BUDGET = 10000
    confidence_threshold = 0.005
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
        
        top_n_count = 3
        max_sector = 999
        
        selected_rows = []
        sector_counts = {}
        
        for _, row in temp_df.iterrows():
            sec = row['Sector']
            sector_counts.setdefault(sec, 0)
            
            if sector_counts[sec] < max_sector:
                selected_rows.append(row)
                sector_counts[sec] += 1
                
            if len(selected_rows) >= top_n_count:
                break
                
        top_picks = pd.DataFrame(selected_rows)
        
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
            
        weight_metric = "return"
        
        if weight_metric == "return":
            weights = top_picks['Estimated_Return_1W'].clip(lower=0.01)
        elif weight_metric == "sentiment":
            weights = top_picks['Sentiment_Score'].clip(lower=0.01)
        elif weight_metric == "security":
            weights = top_picks['Security_Score']
        elif weight_metric == "combined":
            weights = top_picks['Combined_Score'].clip(lower=0.01)
        else:
            weights = pd.Series([1]*len(top_picks), index=top_picks.index)
            
        weights = weights / weights.sum()
        top_picks['Allocation'] = weights * BUDGET
        
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
