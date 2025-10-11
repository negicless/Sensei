# agent/ingest/price_yahoojp.py
import datetime as dt
import pathlib
import pandas as pd

from .price_mujinzou import fetch_prices_mujinzou
from .http import make_session
import io, time, yfinance as yf, requests

CACHE_DIR = pathlib.Path("data/prices")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _save_cache(code: str, df: pd.DataFrame):
    fp = CACHE_DIR / f"{code}.parquet"
    try:
        df.to_parquet(fp, index=False)
    except Exception:
        pass


def _load_cache(code: str) -> pd.DataFrame | None:
    fp = CACHE_DIR / f"{code}.parquet"
    if fp.exists():
        try:
            df = pd.read_parquet(fp)
            if not df.empty:
                return df
        except Exception:
            pass
    return None


def _normalize_yf(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.rename(columns=str.lower)
        .reset_index()
        .rename(
            columns={
                "date": "date",
                "open": "o",
                "high": "h",
                "low": "l",
                "close": "c",
                "volume": "v",
            }
        )
    )
    return df[["date", "o", "h", "l", "c", "v"]].dropna()


def _fetch_stooq(code: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={code}.jp&i=d"
    s = make_session()
    r = s.get(url, timeout=(5, 30))
    r.raise_for_status()
    sdf = pd.read_csv(io.StringIO(r.text))
    if sdf is None or sdf.empty:
        raise RuntimeError("Stooq returned no data")
    sdf = sdf.rename(
        columns={
            "Date": "date",
            "Open": "o",
            "High": "h",
            "Low": "l",
            "Close": "c",
            "Volume": "v",
        }
    )
    sdf["date"] = pd.to_datetime(sdf["date"])
    return sdf[["date", "o", "h", "l", "c", "v"]].dropna()


def _fetch_yahoo(code: str) -> pd.DataFrame:
    ticker = f"{code}.T"
    for i in range(3):
        try:
            df = yf.download(
                ticker,
                period="2y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                timeout=45,
            )
            if df is not None and not df.empty:
                return _normalize_yf(df)
        except Exception:
            time.sleep(1 + i)
    raise RuntimeError("Yahoo failed")


def fetch_price_history(code: str) -> pd.DataFrame:
    code = str(code).zfill(4)
    # 0) try cache
    cached = _load_cache(code)
    # 1) Mujinzou for the last ~2 years (safe range)
    end = dt.date.today()
    start = end - dt.timedelta(days=730)

    try:
        df = fetch_prices_mujinzou(code, start, end)
        if df is not None and not df.empty:
            _save_cache(code, df)
            return df
    except Exception:
        pass

    # 2) Stooq fallback
    try:
        df = _fetch_stooq(code)
        _save_cache(code, df)
        return df
    except Exception:
        pass

    # 3) Yahoo fallback
    try:
        df = _fetch_yahoo(code)
        _save_cache(code, df)
        return df
    except Exception:
        pass

    # 4) last resort: return cached if present
    if cached is not None and not cached.empty:
        return cached

    raise RuntimeError(
        f"Price download failed for {code} from Mujinzou / Stooq / Yahoo"
    )
