def _r_multiple(entry, stop, targets):
    R = abs(entry - stop) if entry != stop else 1e-9
    out = []
    for t in targets:
        r = (t - entry)/R
        out.append({"target": t, "R": round(r, 2)})
    return out

def compression_breakout(px, funda):
    df = px.copy()
    # crude bandwidth proxy: rolling high-low range vs mean price
    df["bbw"] = (df["h"].rolling(20).max() - df["l"].rolling(20).min()) / (df["c"].rolling(20).mean() + 1e-9)
    recent = df.tail(30)
    cond_narrow = recent["bbw"].mean() < df["bbw"].mean()*0.8
    cond_break = recent["c"].iloc[-1] > recent["h"].rolling(20).max().iloc[-2]
    if cond_narrow and cond_break and df["atr14"].iloc[-1] > 0:
        entry = float(recent["h"].rolling(20).max().iloc[-2])
        stop  = float(recent["l"].rolling(10).min().iloc[-1])
        atr = float(recent["atr14"].iloc[-1])
        t1, t2 = entry + 1.0*atr, entry + 2.0*atr
        return {"name":"Compression Breakout","entry":entry,"stop":stop,"targets":[t1,t2],
                "r_table": _r_multiple(entry, stop, [t1,t2])}
    return None

def trend_pullback(px, funda):
    df = px.copy()
    if df["ma50"].iloc[-1] > df["ma200"].iloc[-1] and df["atr14"].iloc[-1] > 0:
        near_ma = abs(df["c"].iloc[-1]-df["ma20"].iloc[-1]) / (df["ma20"].iloc[-1] + 1e-9) < 0.01
        if near_ma and 40 <= df["rsi14"].iloc[-1] <= 55:
            entry = float(df["c"].iloc[-1] * 1.01)
            stop  = float(df["c"].iloc[-1] - 1.25*df["atr14"].iloc[-1])
            atr = float(df["atr14"].iloc[-1])
            t1, t2 = entry + 1.0*atr, entry + 2.0*atr
            return {"name":"Trend Pullback","entry":entry,"stop":stop,"targets":[t1,t2],
                    "r_table": _r_multiple(entry, stop, [t1,t2])}
    return None

def scan_setups(px, funda):
    ideas = []
    for fn in (compression_breakout, trend_pullback):
        sig = fn(px, funda)
        if sig:
            ideas.append(sig)
    return ideas
