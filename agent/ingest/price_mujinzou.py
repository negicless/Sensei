# agent/ingest/price_mujinzou.py
import io, zipfile, datetime as dt, pathlib
import pandas as pd
from .http import make_session  # we added earlier for retries & headers

# CSV schema (per community docs):
# YYYY/MM/DD, <code>, <marketCode>, <code + name>, Open, High, Low, Close, Volume, <marketName>
# Example row shown in docs: 2020/12/1,1301,11,1301 極洋,2824,2824,2780,2787,17900,東証１部

CACHE_DIR = pathlib.Path("data/prices")
DL_DIR = pathlib.Path("data/mujinzou_daily")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DL_DIR.mkdir(parents=True, exist_ok=True)


def _daily_zip_url(day: dt.date) -> str:
    yy = day.year - 2000
    mm = day.month
    dd = day.day
    fname = f"T{yy:02d}{mm:02d}{dd:02d}.zip"

    # try historical (k_data) first, then same-day (d_data) structure
    k_url = f"https://mujinzou.com/k_data/{day.year}/{yy:02d}_{mm:02d}/{fname}"
    d_url = f"https://mujinzou.com/d_data/{day.year}d/{yy:02d}_{mm:02d}d/{fname}"
    return k_url, d_url


def _download_zip(day: dt.date) -> bytes:
    s = make_session()
    k_url, d_url = _daily_zip_url(day)
    for url in (k_url, d_url):
        r = s.get(url, timeout=(5, 30))
        if r.status_code == 200 and r.content:
            return r.content
    raise RuntimeError(f"Mujinzou daily ZIP not found for {day}")


def _parse_daily_csv(zip_bytes: bytes, code: str) -> pd.DataFrame:
    code = str(code).zfill(4)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # assume single CSV inside
        names = zf.namelist()
        if not names:
            return pd.DataFrame()
        with zf.open(names[0]) as f:
            # encoding can vary; try shift_jis then cp932 then utf-8
            b = f.read()
            for enc in ("cp932", "shift_jis", "utf-8", "utf-8-sig"):
                try:
                    text = b.decode(enc)
                    break
                except Exception:
                    continue
            else:
                text = b.decode("cp932", errors="ignore")

    # parse into DataFrame
    df = pd.read_csv(io.StringIO(text), header=None)
    # Filter by code: col1 == code OR col3 startswith(f"{code} ")
    mask = (df.iloc[:, 1].astype(str) == code) | df.iloc[:, 3].astype(
        str
    ).str.startswith(f"{code} ")
    df = df[mask]
    if df.empty:
        return pd.DataFrame()

    df = df[[0, 4, 5, 6, 7, 8]].copy()
    df.columns = ["date", "o", "h", "l", "c", "v"]
    # Normalize types
    df["date"] = pd.to_datetime(df["date"])
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def fetch_prices_mujinzou(code: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Collect daily CSVs between start..end from Mujinzou and build a time series for 'code'.
    """
    code = str(code).zfill(4)
    frames = []
    day = start
    while day <= end:
        try:
            z = _download_zip(day)
            dfd = _parse_daily_csv(z, code)
            if not dfd.empty:
                frames.append(dfd)
        except Exception:
            # skip missing days (holidays or unavailable)
            pass
        day += dt.timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=["date", "o", "h", "l", "c", "v"])
    out = (
        pd.concat(frames, ignore_index=True).sort_values("date").drop_duplicates("date")
    )
    return out
