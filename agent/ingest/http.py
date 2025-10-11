# agent/ingest/https.py
from __future__ import annotations
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HTTP_TIMEOUT = (4, 8)


def _make_retry() -> Retry:
    return Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )


def _make_adapter() -> HTTPAdapter:
    return HTTPAdapter(max_retries=_make_retry(), pool_connections=16, pool_maxsize=32)


def make_session() -> requests.Session:
    s = requests.Session()
    # IMPORTANT: respect system env (proxies, CA bundle) unless explicitly disabled
    s.trust_env = os.getenv("SENSEI_HTTP_TRUST_ENV", "1") != "0"  # default True
    s.headers.update(
        {
            "User-Agent": "SenseiBot/1.0",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
        }
    )
    ad = _make_adapter()
    s.mount("http://", ad)
    s.mount("https://", ad)
    return s


_SESSION = make_session()


def get(url: str, **kw) -> requests.Response:
    kw.setdefault("timeout", HTTP_TIMEOUT)
    return _SESSION.get(url, **kw)
