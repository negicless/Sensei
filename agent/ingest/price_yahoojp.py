import pandas as pd
import yfinance as yf

def fetch_price_history(code: str) -> pd.DataFrame:
    ticker = f"{code}.T"
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No price data for {ticker}")
    df = df.rename(columns=str.lower).reset_index().rename(
        columns={"date":"date","open":"o","high":"h","low":"l","close":"c","volume":"v"})
    return df.dropna()
