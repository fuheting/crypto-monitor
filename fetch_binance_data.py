import requests
import pandas as pd
import time
from datetime import datetime

BASE_URL = "https://api.binance.com"


def get_usdt_symbols():
    url = f"{BASE_URL}/api/v3/exchangeInfo"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    symbols = []
    for symbol in data["symbols"]:
        if symbol["quoteAsset"] == "USDT" and symbol["status"] == "TRADING":
            if not any(x in symbol["symbol"] for x in ["UP", "DOWN", "BULL", "BEAR"]):
                symbols.append(symbol["symbol"])
    return symbols


def fetch_ohlcv(symbol, interval="1h", start_str=None, end_str=None, limit=1000):
    start_time = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000) if start_str else None
    end_time = int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000) if end_str else None
    
    all_data = []
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        url = f"{BASE_URL}/api/v3/klines"
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            break
        
        all_data.extend(data)
        
        last_timestamp = data[-1][0]
        
        if start_time and last_timestamp >= (end_time or float("inf")):
            break
        
        start_time = last_timestamp + 1
        
        time.sleep(0.2)
    
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    
    df.set_index("timestamp", inplace=True)
    
    return df


if __name__ == "__main__":
    symbols = get_usdt_symbols()
    print(f"Found {len(symbols)} USDT trading pairs")
    print(f"First 10: {symbols[:10]}")
    
    btc_df = fetch_ohlcv("BTCUSDT", interval="1h", start_str="2023-01-01", end_str="2024-01-01")
    print(f"\nBTCUSDT data shape: {btc_df.shape}")
    print(btc_df.head())
    print(btc_df.tail())
