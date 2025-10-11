# agent/runtime/browser.py
from playwright.sync_api import sync_playwright

_browser = None
_p = None


def get_browser():
    global _browser, _p
    if _browser is not None:
        return _browser
    _p = sync_playwright().start()
    _browser = _p.chromium.launch(headless=True)
    return _browser


def new_page():
    b = get_browser()
    page = b.new_page(user_agent="Mozilla/5.0 (SenseiBot)")
    return page


def shutdown():
    global _browser, _p
    try:
        if _browser:
            _browser.close()
        if _p:
            _p.stop()
    finally:
        _browser = None
        _p = None
