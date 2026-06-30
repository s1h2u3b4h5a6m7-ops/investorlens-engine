import yfinance as yf
from groq import Groq
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from supabase import create_client, Client
import os
import urllib.parse
import time
import requests # <-- NEW IMPORT

# --- SETUP KEYS FROM GITHUB SECRETS ---
GROQ_KEY = os.environ.get("GROQ_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
FMP_KEY = os.environ.get("FMP_KEY") # <-- NEW KEY

client_groq = Groq(api_key=GROQ_KEY)
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- EXPANDED UNIVERSE (FMCG + IT) ---
companies_universe = {
    # FMCG Sector
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
    
    # IT Sector
    'TCS.NS': {'name': 'Tata Consultancy Services', 'sector': 'IT', 'macro_tags': ['INR/USD', 'US H1B Visa Policy', 'AI Disruption Risk', 'US Fed Rates']},
    'INFY.NS': {'name': 'Infosys', 'sector': 'IT', 'macro_tags': ['INR/USD', 'US Tech Spending', 'AI Disruption Risk', 'US Fed Rates']},
    'WIPRO.NS': {'name': 'Wipro', 'sector': 'IT', 'macro_tags': ['INR/USD', 'US Tech Spending', 'AI Disruption Risk', 'US Fed Rates']}
}

def get_10_year_history(ticker_symbol):
    """Fetches 10 years of financial history from Financial Modeling Prep."""
    history_text = ""
    history_data = []
    
    # FMP API URLs for Income Statement and Balance Sheet
    url_inc = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker_symbol}?period=annual&apikey={FMP_KEY}"
    url_bal = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker_symbol}?period=annual&apikey={FMP_KEY}"
    
    inc_data = requests.get(url_inc).json()
    bal_data = requests.get(url_bal).json()
    
    # FMP returns most recent year first. We reverse it to get oldest to newest (for 10-year trends)
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
            history_data.append({'year': year, 'ROCE': roce, 'Margin': margin, 'Debt': debt_eq})
        except:
            continue
            
    return history_text, history_data

def get_all_data_and_save(ticker_symbol, name, macro_tags):
    print(f"Processing {name}...")
    
    # 1. Live Data (Still using Yahoo for Live Price)
    stock = yf.Ticker(ticker_symbol)
    info = stock.info
    live_data = {
        'price': info.get('currentPrice', 0),
        'pe_ratio': info.get('trailingPE', 0),
        'pb_ratio': info.get('priceToBook', 0),
        'dividend_yield': info.get('dividendYield', 0)
    }
    
    # 2. 10-Year Historical Data (Using FMP)
    history_text, history_data = get_10_year_history(ticker_symbol)

    # 3. AI Buffett Thesis (Now fed with 10 years of data!)
    prompt1 = f"You are a rational value investor in the style of Warren Buffett. Analyze {name}. Current Price: {live_data['price']}, P/E: {live_data['pe_ratio']}, Div Yield: {live_data['dividend_yield']}. 10-Year Historical Trends: {history_text}. Write a 3-paragraph thesis on Moat, Financial Health, and Valuation."
    thesis = client_groq.chat.completions.create(messages=[{"role": "user", "content": prompt1}], model="llama-3.3-70b-versatile").choices[0].message.content

    # 4. News & Sentiment (Unchanged)
    query = urllib.parse.quote(f"{name} India stock")
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en")
    headlines = [entry.title for entry in feed.entries[:10]]
    total_score = sum(sia.polarity_scores(h)['compound'] for h in headlines)
    avg_score = total_score / len(headlines) if headlines else 0
    
    if avg_score > 0.15: sentiment_label = "Positive / Greedy"
    elif avg_score < -0.15: sentiment_label = "Negative / Fearful"
    else: sentiment_label = "Neutral / Uncertain"

    # 5. AI Market Pulse (Unchanged)
    prompt2 = f"You are a rational value investor. Analyze market pulse for {name}. News Sentiment: {avg_score:.2f} ({sentiment_label}). Headlines: {headlines[:5]}. Macro Risks: {macro_tags}. Write a 2-paragraph summary on Market Pulse and Macro Risks."
    pulse = client_groq.chat.completions.create(messages=[{"role": "user", "content": prompt2}], model="llama-3.3-70b-versatile").choices[0].message.content

    # 6. SAVE TO SUPABASE DATABASE
    data_to_save = {
        'ticker': ticker_symbol, 'name': name, 'live_price': live_data['price'],
        'pe_ratio': live_data['pe_ratio'], 'pb_ratio': live_data['pb_ratio'],
        'dividend_yield': live_data['dividend_yield'], 'history_summary': history_text,
        'historical_data': history_data, 
        'news_sentiment_score': avg_score, 'news_sentiment_label': sentiment_label,
        'macro_tags': macro_tags, 'buffett_thesis': thesis, 'market_pulse': pulse
    }
    
    supabase.table("company_reports").upsert(data_to_save).execute()
    print(f"✅ {name} saved to database successfully!")
    
# --- MAIN EXECUTION ---
if __name__ == "__main__":
    for ticker, data in companies_universe.items():
        try:
            get_all_data_and_save(ticker, data['name'], data['macro_tags'])
            time.sleep(2)
        except Exception as e:
            print(f"Error processing {data['name']}: {e}")
