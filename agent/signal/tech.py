import pandas as pd
import numpy as np


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma20"] = df["c"].rolling(20).mean()
    df["ma50"] = df["c"].rolling(50).mean()
    df["ma200"] = df["c"].rolling(200).mean()
    tr = np.maximum(
        df["h"] - df["l"],
        np.maximum(
            (df["h"] - df["c"].shift()).abs(), (df["l"] - df["c"].shift()).abs()
        ),
    )
    df["atr14"] = tr.rolling(14).mean()
    delta = df["c"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["vol_ma20"] = df["v"].rolling(20).mean()
    return df.dropna()


def find_levels(df: pd.DataFrame, lookback=180):
    window = df.tail(lookback)
    res = window["h"].rolling(5).max().iloc[-1]
    sup = window["l"].rolling(5).min().iloc[-1]
    return {"resistance": float(res), "support": float(sup)}
