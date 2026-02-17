import yfinance as yf
import pandas as pd
from companies import IBEX35_COMPANIES
import time
from tqdm import tqdm

def fetch_market_data():
    print("Fetching market data for IBEX35 companies...")
    
    all_data = []
    
    tickers = list(IBEX35_COMPANIES.keys())
    
    for ticker_code, company_name in tqdm(IBEX35_COMPANIES.items()):
        if ticker_code == "MTS":
            yf_ticker = "MTS.MC"
        elif ticker_code == "IBE":
             yf_ticker = "IBE.MC"
        else:
            yf_ticker = f"{ticker_code}.MC"
            
        try:
            df = yf.download(yf_ticker, period="max", interval="1d", progress=False, auto_adjust=True)
            
            if not df.empty:
                df = df.reset_index()
                df['Ticker'] = ticker_code
                df['Company'] = company_name
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if c[0] != 'Price' else c[1] for c in df.columns]
                
                if 'Date' not in df.columns and df.index.name == 'Date':
                    df = df.reset_index()
                
                all_data.append(df)
            else:
                print(f"No data found for {company_name} ({yf_ticker})")
                
        except Exception as e:
            print(f"Error fetching {company_name}: {e}")
            
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        if isinstance(final_df.columns, pd.MultiIndex):
            pass

        output_file = 'ibex35_market_data.csv'
        final_df.to_csv(output_file, index=False)
        print(f"Market data saved to {output_file}. Total rows: {len(final_df)}")
        return final_df
    else:
        print("No data fetched.")
        return None

if __name__ == "__main__":
    fetch_market_data()
