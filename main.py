import yfinance as yf
from groq import Groq
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from supabase import create_client, Client
import os
import urllib.parse
import time
import requests
import pandas as pd

# --- SETUP KEYS FROM GITHUB SECRETS ---
GROQ_KEY = os.environ.get("GROQ_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
FMP_KEY = os.environ.get("FMP_KEY")

client_groq = Groq(api_key=GROQ_KEY)
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- EXPANDED UNIVERSE (FMCG + IT) ---
companies_universe = {
    'ITC.NS': {'name': 'ITC Limited', 'sector': 'FMCG', 'macro_tags': ['Crude Oil (Packaging)', 'Rural Demand', 'Regulatory (Tobacco Tax)', 'INR/USD']},
    'HINDUNILVR.NS': {'name': 'Hindustan Unilever', 'sector': 'FMCG', 'macro_tags': ['Crude Oil (Packaging)', 'Palm Oil (Input)', 'Rural Demand', 'Commodity Inflation']},
    'NESTLEIND.NS': {'name': 'Nestle India', 'sector': 'FMCG', 'macro_tags': ['Milk Prices (Input)', 'Rural Demand', 'Wheat Prices', 'Supply Chain']},
    'BRITANNIA.NS': {'name': 'Britannia Industries', 'sector': 'FMCG', 'macro_tags': ['Wheat Prices (Input)', 'Crude Oil (Packaging)', 'Rural Demand', 'Palm Oil']},
    'DABUR.NS': {'name': 'Dabur India', 'sector': 'FMCG', 'macro_tags': ['Rural Demand', 'Crude Oil (Packaging)', 'Agricultural Output (Herbs)', 'INR/USD']},
    'MARICO.NS': {'name': 'Marico', 'sector': 'FMCG', 'macro_tags': ['Coconut Oil (Input)', 'Rural Demand', 'Crude Oil (Packaging)', 'INR/USD']},
    'GODREJCP.NS': {'name': 'Godrej Consumer Products', 'sector': 'FMCG', 'macro_tags': ['Crude Oil (Packaging)', 'Rural Demand', 'Palm Oil', 'INR/USD']},
    'PGHH.NS': {'name': 'Procter & Gamble Health', 'sector': 'FMCG', 'macro_tags': ['INR/USD', 'Crude Oil (Packaging)', 'API Prices']},
    'COLPAL.NS': {'name': 'Colgate Palmolive', 'sector': 'FMCG', 'macro_tags': ['Crude Oil (Packaging)', 'Rural Demand', 'Commodity Inflation']},
    'TATACONSUM.NS': {'name': 'Tata Consumer Products', 'sector': 'FMCG', 'macro_tags': ['Tea Prices (Input)', 'Rural Demand', 'Wheat Prices', 'INR/USD']},
    'TCS.NS': {'name': 'Tata Consultancy Services', 'sector': 'IT', 'macro_tags': ['INR/USD', 'US H1B Visa Policy', 'AI Disruption Risk', 'US Fed Rates']},
    'INFY.NS': {'name': 'Infosys', 'sector': 'IT', 'macro_tags': ['INR/USD', 'US Tech Spending', 'AI Disruption Risk', 'US Fed Rates']},
    'WIPRO.NS': {'name': 'Wipro', 'sector': 'IT', 'macro_tags': ['INR/USD', 'US Tech Spending', 'AI Disruption Risk', 'US Fed Rates']}
}

def get_val(df, row_names, col):
    """Helper to find a value in a Pandas DataFrame regardless of exact naming."""
    if df is None or df.empty: return 0
    for name in row_names:
        if name in df.index:
            val = df.loc[name, col]
            if pd.notna(val): return val
    return 0

def get_10_year_history(ticker_symbol, stock):
    history_text = ""
    history_data = []
    
    url_inc = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker_symbol}?period=annual&apikey={FMP_KEY}"
    url_bal = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker_symbol}?period=annual&apikey={FMP_KEY}"
    
    try:
        inc_data = requests.get(url_inc).json()
        bal_data = requests.get(url_bal).json()
        
        if isinstance(inc_data, list) and len(inc_data) > 0 and isinstance(bal_data, list) and len(bal_data) > 0:
            max_years = min(len(inc_data), len(bal_data), 10)
            for i in range(max_years - 1, -1, -1):
                try:
                    inc = inc_data[i]
                    bal = bal_data[i]
                    revenue = inc.get('revenue', 0)
                    ebit = inc.get('operatingIncome', 0)
                    net_income = inc.get('netIncome', 0)
                    total_debt = bal.get('totalDebt', 0)
                    total_equity = bal.get('totalStockholdersEquity', 1)
                    current_liab = bal.get('totalCurrentLiabilities', 0)
                    total_assets = bal.get('totalAssets', 0)
                    year = int(inc.get('calendarYear', 2020))
                    
                    roce = round((ebit / (total_assets - current_liab)) * 100, 1) if (total_assets - current_liab) > 0 else 0
                    debt_eq = round(total_debt / total_equity, 2) if total_equity > 0 else 0
                    margin = round((net_income / revenue) * 100, 1) if revenue > 0 else 0
                    
                    history_text += f"\nYear {year}: ROCE={roce}%, Net Margin={margin}%, Debt/Equity={debt_eq}"
                    history_data.append({
                        'year': year, 'ROCE': roce, 'Margin': margin, 'Debt': debt_eq,
                        'Revenue': round(revenue / 10000000, 1),
                        'EBIT': round(ebit / 10000000, 1),
                        'Net_Income': round(net_income / 10000000, 1),
                        'Total_Debt': round(total_debt / 10000000, 1),
                        'Equity': round(total_equity / 10000000, 1)
                    })
                except:
                    continue
            if history_data:
                print("✅ FMP data fetched successfully.")
                return history_text, history_data
    except:
        pass
    
    print("⚠️ FMP failed or blocked. Falling back to Yahoo Finance.")
    try:
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        
        if income_stmt.empty or balance_sheet.empty:
            print("❌ Yahoo Finance returned empty statements for", ticker_symbol)
            return history_text, history_data
            
        for col in income_stmt.columns[:5]:
            try:
                # FIX: Ensure we extract the year as an integer
                if hasattr(col, 'year'):
                    year = col.year
                else:
                    # Fallback if col is a string
                    year = int(str(col)[:4])
                
                revenue = get_val(income_stmt, ['Total Revenue', 'Operating Revenue', 'Revenue'], col)
                ebit = get_val(income_stmt, ['Operating Income', 'EBIT', 'Ebit'], col)
                net_income = get_val(income_stmt, ['Net Income', 'Net Income Common Stockholders'], col)
                
                total_debt = get_val(balance_sheet, ['Total Debt', 'Long Term Debt'], col)
                total_equity = get_val(balance_sheet, ['Stockholders Equity', 'Total Equity Gross Minority Interest', 'Common Stock Equity'], col)
                current_liab = get_val(balance_sheet, ['Current Liabilities'], col)
                total_assets = get_val(balance_sheet, ['Total Assets'], col)
                
                if total_equity == 0: total_equity = 1 # Prevent division by zero
                
                roce = round((ebit / (total_assets - current_liab)) * 100, 1) if (total_assets - current_liab) > 0 else 0
                debt_eq = round(total_debt / total_equity, 2)
                margin = round((net_income / revenue) * 100, 1) if revenue > 0 else 0
                
                history_text += f"\nYear {year}: ROCE={roce}%, Net Margin={margin}%, Debt/Equity={debt_eq}"
                history_data.append({
                    'year': year, 'ROCE': roce, 'Margin': margin, 'Debt': debt_eq,
                    'Revenue': round(revenue / 10000000, 1),
                    'EBIT': round(ebit / 10000000, 1),
                    'Net_Income': round(net_income / 10000000, 1),
                    'Total_Debt': round(total_debt / 10000000, 1),
                    'Equity': round(total_equity / 10000000, 1)
                })
            except Exception as e:
                print(f"Error parsing Yahoo year {col}: {e}")
                continue
    except Exception as e:
        print(f"❌ Yahoo Finance completely failed: {e}")
        
    return history_text, history_data

def get_all_data_and_save(ticker_symbol, name, macro_tags):
    print(f"\n--- Processing {name} ---")
    stock = yf.Ticker(ticker_symbol)
    info = stock.info
    live_data = {
        'price': info.get('currentPrice', 0),
        'pe_ratio': info.get('trailingPE', 0),
        'pb_ratio': info.get('priceToBook', 0),
        'dividend_yield': info.get('dividendYield', 0)
    }
    
    history_text, history_data = get_10_year_history(ticker_symbol, stock)

    prompt1 = f"You are a rational value investor in the style of Warren Buffett. Analyze {name}. Current Price: {live_data['price']}, P/E: {live_data['pe_ratio']}, Div Yield: {live_data['dividend_yield']}. 10-Year Historical Trends: {history_text}. Write a 4-paragraph thesis. Paragraph 1: Moat & Business Quality. Paragraph 2: Financial Health & Capital Allocation. Paragraph 3: Valuation Rationality. Paragraph 4: Value Chain & Industry Structure."
    
    # OPTIMIZATION: Using Llama 3.1 8B to avoid the 429 rate limit errors. It is 5x faster and has massive free limits.
    thesis = client_groq.chat.completions.create(messages=[{"role": "user", "content": prompt1}], model="llama3-8b-8192").choices[0].message.content

    query = urllib.parse.quote(f"{name} India stock")
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en")
    headlines = [entry.title for entry in feed.entries[:10]]
    total_score = sum(sia.polarity_scores(h)['compound'] for h in headlines)
    avg_score = total_score / len(headlines) if headlines else 0
    
    if avg_score > 0.15: sentiment_label = "Positive / Greedy"
    elif avg_score < -0.15: sentiment_label = "Negative / Fearful"
    else: sentiment_label = "Neutral / Uncertain"

    prompt2 = f"You are a rational value investor. Analyze market pulse for {name}. News Sentiment: {avg_score:.2f} ({sentiment_label}). Headlines: {headlines[:5]}. Macro Risks: {macro_tags}. Write a 2-paragraph summary on Market Pulse and Macro Risks."
    pulse = client_groq.chat.completions.create(messages=[{"role": "user", "content": prompt2}], model="llama3-8b-8192").choices[0].message.content

    # Red/Green Flag Engine
    red_flags = []
    green_flags = []
    if history_data and len(history_data) > 0:
        latest = history_data[-1]
        if latest['Debt'] > 1.0: red_flags.append("High Debt/Equity (>1.0)")
        if latest['ROCE'] < 15: red_flags.append("Low ROCE (<15%)")
        if latest['Margin'] < 5: red_flags.append("Low Net Margin (<5%)")
        
        if latest['Debt'] < 0.5: green_flags.append("Low Debt/Equity (<0.5)")
        if latest['ROCE'] > 20: green_flags.append("Excellent ROCE (>20%)")
        if latest['Margin'] > 10: green_flags.append
