from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Quant ML Master API", version="5.0")

# Enable CORS so your front-end can communicate with the back-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=10)

# --- ASSET CONFIGURATIONS ---
ASSET_CONFIGS = {
    "equities": {
        "assets": {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK", "Finnifty": "NIFTY_FIN_SERVICE.NS", "Midcap Nifty": "^NSEMDCP50", "Nifty Next 50": "^NSMIDCP", "Sensex": "^BSESN"},
        "div1": "^NSEI", "div2": "^NSEBANK", "div1_name": "Nifty 50", "div2_name": "Bank Nifty", "currency": "₹", "trading_days": 252
    },
    "crypto": {
        "assets": {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Solana (SOL)": "SOL-USD", "Binance Coin (BNB)": "BNB-USD", "Ripple (XRP)": "XRP-USD"},
        "div1": "BTC-USD", "div2": "ETH-USD", "div1_name": "Bitcoin", "div2_name": "Ethereum", "currency": "$", "trading_days": 365
    }
}

# --- QUANT FUNCTIONS ---
def sync_fetch(ticker: str, period: str, interval: str = "1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

async def fetch_data(ticker: str, period: str, interval: str = "1d"):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, sync_fetch, ticker, period, interval)

def get_scalar(series):
    if series.empty: return 0.0
    val = series.iloc[-1]
    return float(val.item()) if hasattr(val, 'item') else float(val)

def calculate_hurst(ts):
    if len(ts) < 20: return 0.5
    lags = range(2, 20)
    reg_val = [np.std(ts.values[lag:] - ts.values[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(reg_val), 1)
    return float(poly[0])

def calc_yang_zhang(df, trading_days):
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    N = len(df)
    if N < 2: return 0.0, 0.0
    log_ho = np.log(df['High'] / df['Open'])
    log_lo = np.log(df['Low'] / df['Open'])
    log_co = np.log(df['Close'] / df['Open'])
    log_oc = np.log(df['Open'] / df['Close'].shift(1))
    
    vol_o = log_oc.dropna().std() ** 2
    vol_c = log_co.dropna().std() ** 2
    rs = (log_ho * (log_ho - log_co)) + (log_lo * (log_lo - log_co))
    vol_rs = rs.dropna().mean()
    
    k = 0.34 / (1.34 + (N + 1) / (N - 1))
    yz_var = vol_o + k * vol_c + (1 - k) * vol_rs
    yz_vol = np.sqrt(yz_var) * np.sqrt(trading_days) * 100
    c2c_vol = np.log(df['Close'] / df['Close'].shift(1)).dropna().std() * np.sqrt(trading_days) * 100
    return float(yz_vol), float(c2c_vol)

# --- API ENDPOINTS ---
@app.get("/api/assets")
def get_assets(asset_class: str = "equities"):
    if asset_class not in ASSET_CONFIGS:
        raise HTTPException(status_code=400, detail="Invalid asset class")
    return {"assets": list(ASSET_CONFIGS[asset_class]["assets"].keys())}

@app.get("/api/synthesis")
async def get_market_synthesis(asset_class: str = "equities", asset_name: str = "Nifty 50"):
    config = ASSET_CONFIGS.get(asset_class)
    if not config or asset_name not in config["assets"]:
        raise HTTPException(status_code=400, detail="Invalid configuration params")
    
    ticker = config["assets"][asset_name]
    trading_days = config["trading_days"]
    
    tasks = [
        fetch_data(ticker, "1y", "1d"),
        fetch_data(ticker, "30d", "15m"),
        fetch_data(config["div1"], "1y", "1d"),
        fetch_data(config["div2"], "1y", "1d")
    ]
    daily_df, intra_df, d1_df, d2_df = await asyncio.gather(*tasks)
    
    if daily_df is None or intra_df is None or d1_df is None or d2_df is None:
        raise HTTPException(status_code=500, detail="Failed to fetch data components")
        
    # Process Synthetic vs Real IV Benchmark
    if asset_class == "equities":
        vix_df = await fetch_data("^INDIAVIX", "1y", "1d")
        current_vix = get_scalar(vix_df['Close'])
    else:
        ret_30 = np.log(daily_df['Close'] / daily_df['Close'].shift(1))
        synth_vix = ret_30.rolling(30).std() * np.sqrt(365) * 100
        current_vix = float(synth_vix.dropna().iloc[-1])
        vix_df = pd.DataFrame(synth_vix.dropna().tail(252), columns=['Close'])

    vix_close = vix_df['Close']
    high_52w, low_52w = float(vix_close.max()), float(vix_close.min())
    ivr = ((current_vix - low_52w) / (high_52w - low_52w) * 100) if (high_52w - low_52w) != 0 else 0.0

    hv_20 = np.log(daily_df['Close'] / daily_df['Close'].shift(1)).rolling(20).std() * np.sqrt(trading_days) * 100
    vrp_val = current_vix - float(hv_20.iloc[-1])

    hurst_val = calculate_hurst(np.log(daily_df['Close']).tail(60))
    hurst_regime = "TRENDING" if hurst_val > 0.55 else ("MEAN REVERTING" if hurst_val < 0.45 else "RANDOM WALK")

    window = 20
    intra_df['Prev_High'] = intra_df['High'].rolling(window).max().shift(1)
    intra_df['Prev_Low'] = intra_df['Low'].rolling(window).min().shift(1)
    is_supply = bool((intra_df['High'].iloc[-1] > intra_df['Prev_High'].iloc[-1]) and (intra_df['Close'].iloc[-1] < intra_df['Prev_High'].iloc[-1]))
    is_demand = bool((intra_df['Low'].iloc[-1] < intra_df['Prev_Low'].iloc[-1]) and (intra_df['Close'].iloc[-1] > intra_df['Prev_Low'].iloc[-1]))
    liq_regime = "SUPPLY SWEEP (Resistance)" if is_supply else ("DEMAND SWEEP (Support)" if is_demand else "PRICE DISCOVERY")

    log_ret1 = np.log(d1_df['Close'] / d1_df['Close'].shift(1)).dropna()
    log_ret2 = np.log(d2_df['Close'] / d2_df['Close'].shift(1)).dropna()
    corr_val = float(log_ret1.rolling(20).corr(log_ret2).iloc[-1])
    div_regime = "HIGH CORRELATION" if corr_val > 0.80 else ("SEVERE DIVERGENCE" if corr_val < 0.50 else "MODERATE DIVERGENCE")

    yz_vol, c2c_vol = calc_yang_zhang(daily_df, trading_days)

    bias = 0
    if hurst_regime == "TRENDING": bias += 1
    if hurst_regime == "MEAN REVERTING": bias -= 1
    if is_demand: bias += 1
    if is_supply: bias -= 1
    if corr_val < 0.50: bias -= 1
    
    macro_state = "Bullish / Trending Focus" if bias > 0 else ("Bearish / Mean Reversion Focus" if bias < 0 else "Neutral / Choppy")
    option_strategy = "Net Short Premium (Credit Spreads/Iron Condors)" if ivr > 50 and vrp_val > 0 else "Net Long Premium (Debit Spreads/Directional)"

    chart_dates = daily_df.index.strftime('%Y-%m-%d').tolist()
    price_series = daily_df['Close'].tolist()

    return {
        "meta": {"asset_name": asset_name, "currency": config["currency"], "spot": get_scalar(daily_df['Close'])},
        "scores": {
            "ivr": round(ivr, 2), "vrp": round(vrp_val, 2), "hurst": round(hurst_val, 3), 
            "hurst_regime": hurst_regime, "liq_regime": liq_regime, "correlation": round(corr_val, 2), 
            "div_regime": div_regime, "macro_state": macro_state, "option_strategy": option_strategy,
            "yz_vol": round(yz_vol, 2), "c2c_vol": round(c2c_vol, 2), "current_vix": round(current_vix, 2)
        },
        "chart_data": {"dates": chart_dates, "prices": price_series}
    }

if __name__ == "__main__":
    import uvicorn
    # This block allows you to run it via `python main.py`
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
