#!/usr/bin/env python3
"""
Binance SPOT analyst: scans the top-100 USDT pairs on 1D / 4H / 1H charts,
scores trend + momentum + volume, finds trade setups, checks liquidity,
and emails BUY / SELL(exit) suggestions with entry, stop, targets and reasons.

Free: uses Binance public market data + your own Gmail. Built for GitHub Actions.

Secrets / environment variables:
    ALERT_EMAIL_USER, ALERT_EMAIL_PASS, ALERT_EMAIL_TO (optional)

Local use:
    python analyst.py --dry                    # analyse and print, no email
    python analyst.py --dry --only BTCUSDT,SOLUSDT
    python analyst.py --test                   # send a test email
    python analyst.py --mode intraday --dry    # 5m/10m/15m/30m analysis (top 30 pairs)
    python analyst.py --mode intraday --loop 300   # VPS: run every 5 minutes, forever

Not financial advice. Signals are rule-based and can be wrong.
"""
import os
import sys
import json
import time
import smtplib
import ssl
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from email.message import EmailMessage
from statistics import pstdev

import requests

# ======================= SETTINGS (edit these) =======================
TOP_N = 100                    # top pairs by 24h USDT volume
MIN_QUOTE_VOLUME = 15_000_000  # ignore pairs with less 24h volume than this (USDT)
ORDER_SIZE_USDT = 1000         # used to estimate slippage for your typical order
TZ_OFFSET_HOURS = 3            # UTC+3 (matches your Binance chart timezone)
TZ_NAME = "UTC+3"
DIGEST_HOUR_LOCAL = 8          # daily digest is sent at/after this local hour
ALERT_GRADES = ("A", "B")      # instant alerts only for these grades
COOLDOWN_HOURS = 12            # don't repeat the same alert sooner than this
MAX_PER_SIDE = 8               # max BUY and max SELL setups listed per email
WORKERS = 5                    # parallel requests (keep low for rate limits)
DEPTH_CHECKS = 30              # order books to check (best setups only)
STATE_FILE = "analyst_state.json"   # intraday mode uses intraday_state.json
INTRADAY_N = 30                # intraday mode scans the top-N most liquid pairs
INTRADAY_DEPTH_CHECKS = 12     # order books to check in intraday mode
INTRADAY_COOLDOWN_HOURS = 2    # repeat-alert cooldown for intraday signals
MIN_TP1_PCT = 0.6              # Target 1 must be at least this % (spot fees ~0.15-0.2% round trip)
MAX_SPREAD_BPS_INTRADAY = 10   # skip intraday setups when the spread is wider than this
# =====================================================================

EMAIL_USER = os.environ.get("ALERT_EMAIL_USER", "")
EMAIL_PASS = os.environ.get("ALERT_EMAIL_PASS", "")
EMAIL_TO = os.environ.get("ALERT_EMAIL_TO") or EMAIL_USER
SMTP_HOST = os.environ.get("ALERT_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("ALERT_SMTP_PORT", "465"))

BASES = ["https://data-api.binance.vision", "https://api.binance.com"]
STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "USD1", "PYUSD", "XUSD", "AEUR", "EUR"}
TF_WEIGHTS = {"1d": 0.35, "4h": 0.35, "1h": 0.30}


# ============================ Networking =============================
def api(path):
    return [b + path for b in BASES]


def get_json(urls, params=None, retries=2):
    last = None
    for url in urls:
        for attempt in range(retries + 1):
            try:
                r = requests.get(url, params=params, timeout=20)
                if r.status_code in (451, 403):
                    last = RuntimeError(f"HTTP {r.status_code} from {url.split('/')[2]} (region blocked)")
                    break
                if r.status_code in (418, 429):
                    last = RuntimeError("rate limited by Binance")
                    time.sleep(2 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last = e
                time.sleep(1)
    raise last


def fetch_candles(symbol, interval, limit=250):
    rows = get_json(api("/api/v3/klines"), {"symbol": symbol, "interval": interval, "limit": limit})
    now_ms = int(time.time() * 1000)
    rows = [k for k in rows if k[6] < now_ms]  # closed candles only
    return {
        "t": [int(k[0]) for k in rows], "h": [float(k[2]) for k in rows],
        "l": [float(k[3]) for k in rows], "c": [float(k[4]) for k in rows],
        "v": [float(k[5]) for k in rows],
    }


# ============================ Indicators =============================
def ema_series(x, n):
    if len(x) < n:
        return []
    k = 2 / (n + 1)
    e = sum(x[:n]) / n
    out = [e]
    for v in x[n:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi_last(c, n=14):
    c = c[-200:]
    if len(c) < n + 1:
        return None
    g = l = 0.0
    for i in range(1, n + 1):
        d = c[i] - c[i - 1]
        g += max(d, 0)
        l += max(-d, 0)
    ag, al = g / n, l / n
    for i in range(n + 1, len(c)):
        d = c[i] - c[i - 1]
        ag = (ag * (n - 1) + max(d, 0)) / n
        al = (al * (n - 1) + max(-d, 0)) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def atr_last(h, l, c, n=14):
    if len(c) < n + 1:
        return None
    trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, len(c))]
    a = sum(trs[:n]) / n
    for tr in trs[n:]:
        a = (a * (n - 1) + tr) / n
    return a


def macd_hist(c):
    e12, e26 = ema_series(c, 12), ema_series(c, 26)
    if len(e26) < 12:
        return None, None
    e12 = e12[-len(e26):]
    m = [a - b for a, b in zip(e12, e26)]
    sig = ema_series(m, 9)
    if len(sig) < 2:
        return None, None
    hist = [a - b for a, b in zip(m[-len(sig):], sig)]
    return hist[-1], hist[-2]


def analyze_tf(cd):
    """Score one timeframe from -100 (very bearish) to +100 (very bullish)."""
    c, h, l, v = cd["c"], cd["h"], cd["l"], cd["v"]
    if len(c) < 60:
        return None
    close = c[-1]
    e20, e50 = ema_series(c, 20)[-1], ema_series(c, 50)[-1]
    e200s = ema_series(c, 200)
    e200 = e200s[-1] if e200s else None
    r, a = rsi_last(c), atr_last(h, l, c)
    mh, mhp = macd_hist(c)
    if r is None or a is None:
        return None
    s20 = sum(c[-20:]) / 20
    sd = pstdev(c[-20:])
    bb = (close - (s20 - 2 * sd)) / (4 * sd) if sd > 0 else 0.5
    base = v[-21:-1]
    vr = v[-1] / (sum(base) / len(base)) if base and sum(base) > 0 else 1.0

    s = 15 if close > e50 else -15
    s += 10 if e20 > e50 else -10
    if e200 is not None:
        s += 15 if close > e200 else -15
    if mh is not None:
        s += (10 if mh > 0 else -10) + (10 if mh > mhp else -10)
    s += max(-20, min(20, r - 50))
    if vr >= 1.3:
        s += 10 if close > c[-2] else -10
    if e200 is None:
        s *= 90 / 75
    s = max(-100, min(100, s))

    return {
        "close": close, "e20": e20, "e50": e50, "e200": e200, "rsi": r, "atr": a,
        "mh": mh, "mhp": mhp, "bb": bb, "vr": vr, "score": s,
        "label": "bull" if s >= 25 else "bear" if s <= -25 else "neutral",
        "hi20": max(h[-21:-1]), "lo20": min(l[-21:-1]),
        "hi40": max(h[-40:]), "lo40": min(l[-40:]),
        "last_t": cd["t"][-1],
    }


# ============================ Setup logic ============================
def pick_setup(a1d, a4, a1, overall):
    """Return (side, name, base_confidence, reasons, counter_trend) or None."""
    s1d, s4, s1 = a1d["score"], a4["score"], a1["score"]
    ext = a4["rsi"] >= 75 or a1["rsi"] >= 78
    cands = []

    # ---------- BUY side ----------
    if s1d >= 25 and s4 >= 25 and a1["rsi"] <= 45 and a1["close"] <= a1["e50"] * 1.01:
        cands.append(("BUY", "Trend pullback", 40 + 0.2 * (s1d + s4), [
            "Daily and 4H trends are up, price has dipped on 1H "
            f"(RSI {a1['rsi']:.0f}, near the 1H EMA50): a classic buy-the-dip spot.",
        ], False))
    if a4["close"] > a4["hi20"] and a4["vr"] >= 1.5 and s4 >= 20 and a4["rsi"] < 80:
        cands.append(("BUY", "Breakout", 40 + 0.2 * (s4 + s1d) + min(10, 5 * (a4["vr"] - 1.5)), [
            f"4H candle closed above its 20-candle high on {a4['vr']:.1f}x average volume.",
        ], False))
    if a4["rsi"] <= 30 and a4["bb"] <= 0.05 and s1d > -40:
        cands.append(("BUY", "Oversold bounce (counter-trend)", 42, [
            f"4H RSI {a4['rsi']:.0f} and price at the lower Bollinger band: a bounce is possible, "
            "but this goes against the trend, so use a small size.",
        ], True))
    if overall >= 30 and not ext:
        cands.append(("BUY", "Trend following", 25 + 0.8 * overall, [
            f"Trend and momentum agree across timeframes (overall score {overall:+.0f}).",
        ], False))

    # ---------- SELL / EXIT side ----------
    if a4["close"] < a4["lo20"] and a4["vr"] >= 1.5 and s4 <= -20:
        cands.append(("SELL", "Breakdown", 40 + 0.2 * (-s4 - s1d) + min(10, 5 * (a4["vr"] - 1.5)), [
            f"4H candle closed below its 20-candle low on {a4['vr']:.1f}x average volume.",
        ], False))
    if a4["rsi"] >= 78 and a4["bb"] >= 1.0 and a4["mh"] is not None and a4["mh"] < a4["mhp"]:
        cands.append(("SELL", "Overbought exhaustion (take profit)", 45 + min(15, (a4["rsi"] - 78) * 2), [
            f"4H RSI {a4['rsi']:.0f}, price above the upper Bollinger band and momentum is fading: "
            "good place to take profit.",
        ], False))
    if overall <= -30:
        cands.append(("SELL", "Trend turned bearish (exit / avoid)", 25 + 0.8 * abs(overall), [
            f"Trend and momentum are down across timeframes (overall score {overall:+.0f}).",
        ], False))

    return max(cands, key=lambda c: c[2]) if cands else None


def horizon(side, s1d, s4, s1):
    sign = 1 if side == "BUY" else -1
    if sign * s1d >= 25 and sign * s4 >= 25:
        return "Swing trade: days to weeks"
    if sign * s4 >= 25:
        return "Short swing: 1 to 5 days"
    return "Intraday / short-term: hours to a day or two"


def grade_of(conf):
    return "A" if conf >= 70 else "B" if conf >= 55 else "C" if conf >= 40 else None


def build_setup(sym, info, an, btc, spread_bps):
    a1d, a4, a1 = an["1d"], an["4h"], an["1h"]
    overall = sum(TF_WEIGHTS[k] * an[k]["score"] for k in TF_WEIGHTS)
    pick = pick_setup(a1d, a4, a1, overall)
    if not pick:
        return None
    side, name, conf, reasons, counter = pick
    price = a1["close"]
    s1d, s4, s1 = a1d["score"], a4["score"], a1["score"]
    reasons = list(reasons)

    # --- trade plan ---
    risk = 1.5 * a4["atr"]
    if side == "BUY":
        if risk >= price or risk / price * 100 > 25:
            return None
        plan = {
            "entry_lo": price - 0.5 * a1["atr"], "entry_hi": price,
            "stop": price - risk, "tp1": price + 1.5 * risk, "tp2": price + 3 * risk,
            "risk_pct": risk / price * 100, "res": a4["hi40"],
        }
        if plan["res"] > price and plan["res"] < plan["tp1"]:
            conf -= 8
            reasons.append(f"Caution: 4H resistance near {fmt_p(plan['res'])} sits before target 1.")
        if plan["risk_pct"] > 12:
            conf -= 10
            reasons.append("Caution: very volatile, so the stop is wide. Reduce your position size.")
    else:
        plan = {
            "s1": price - 1.5 * a4["atr"], "s2": price - 3 * a4["atr"],
            "swing": a4["lo40"] if a4["lo40"] < price else None,
            "invalid": a4["e50"] if a4["e50"] > price else price + 1.5 * a4["atr"],
        }

    # --- adjustments ---
    sign = 1 if side == "BUY" else -1
    if side == "BUY" and a1d["rsi"] >= 75:
        conf -= 5
        reasons.append(f"Caution: daily RSI is high ({a1d['rsi']:.0f}). Consider a smaller size or waiting for a dip.")
    if all(sign * s >= 15 for s in (s1d, s4, s1)):
        conf += 8
        reasons.append("1D, 4H and 1H all point the same way.")
    if btc and sym != "BTCUSDT":
        if side == "BUY":
            if btc["bull"]:
                conf += 5
                reasons.append("BTC trend is supportive.")
            elif btc["bear"]:
                conf -= 10
                reasons.append("Caution: BTC trend is bearish, which is a headwind for buys.")
        else:
            if btc["bear"]:
                conf += 5
                reasons.append("BTC trend is also bearish.")
            elif btc["bull"]:
                conf -= 5
                reasons.append("BTC is bullish, so this may only be a pullback.")
    if info["qv"] < 30_000_000:
        conf -= 5
    if spread_bps is not None:
        conf -= 10 if spread_bps > 10 else 4 if spread_bps > 5 else 0
    if counter:
        conf = min(conf, 54)

    return {
        "symbol": sym, "side": side, "setup": name, "conf": conf, "grade": grade_of(conf),
        "price": price, "overall": overall, "s1d": s1d, "s4": s4, "s1": s1,
        "tf_text": [tf_text("1D", a1d), tf_text("4H", a4), tf_text("1H", a1)],
        "plan": plan, "horizon": horizon(side, s1d, s4, s1), "reasons": reasons,
        "liq": {"qv": info["qv"], "spread": spread_bps, "chg": info["chg"]}, "new": False,
    }


def tf_text(name, a):
    lab = {"bull": "Bull", "bear": "Bear", "neutral": "Neutral"}[a["label"]]
    return f"{name} {lab} (RSI {a['rsi']:.0f})"


# ============================ Liquidity ==============================
def depth_stats(symbol):
    d = get_json(api("/api/v3/depth"), {"symbol": symbol, "limit": 100})
    bids = [(float(p), float(q)) for p, q in d["bids"]]
    asks = [(float(p), float(q)) for p, q in d["asks"]]
    if not bids or not asks:
        return None

    def slip(levels):
        remaining, cost, qty = ORDER_SIZE_USDT, 0.0, 0.0
        for p, q in levels:
            take = min(remaining, p * q)
            cost += take
            qty += take / p
            remaining -= take
            if remaining <= 1e-9:
                break
        if remaining > 1e-9 or qty == 0:
            return None
        return abs(cost / qty - levels[0][0]) / levels[0][0] * 100

    return {
        "bid_depth": sum(p * q for p, q in bids), "ask_depth": sum(p * q for p, q in asks),
        "slip_buy": slip(asks), "slip_sell": slip(bids),
    }


def apply_depth(s, ds):
    s["liq"]["depth"] = ds
    if not ds:
        return
    slip = ds["slip_buy"] if s["side"] == "BUY" else ds["slip_sell"]
    if slip is None or slip > 0.15:
        s["conf"] -= 10
        s["liq"]["thin"] = True
        s["reasons"].append(f"Caution: thin order book, so a ${ORDER_SIZE_USDT:,} order may slip noticeably.")
    elif slip > 0.05:
        s["conf"] -= 3
    s["grade"] = grade_of(s["conf"])


# ============================== Scan =================================
def get_spreads(notes):
    spreads = {}
    try:
        for t in get_json(api("/api/v3/ticker/bookTicker")):
            bid, ask = float(t["bidPrice"]), float(t["askPrice"])
            if bid > 0 and ask > 0:
                spreads[t["symbol"]] = (ask - bid) / ((ask + bid) / 2) * 1e4
    except Exception as e:
        notes.append(f"Spread data unavailable: {e}")
    return spreads


def load_universe(only=None):
    data = get_json(api("/api/v3/ticker/24hr"))
    rows = []
    for t in data:
        sym = t["symbol"]
        if not sym.endswith("USDT") or sym[:-4] in STABLES:
            continue
        qv = float(t["quoteVolume"])
        if qv < MIN_QUOTE_VOLUME and not (only and sym in only):
            continue
        rows.append((qv, sym, float(t.get("priceChangePercent") or 0)))
    rows.sort(reverse=True)
    if only:
        rows = [r for r in rows if r[1] in only]
    else:
        rows = rows[:TOP_N]
    return {sym: {"qv": qv, "chg": chg} for qv, sym, chg in rows}


def analyze_symbol(sym):
    out = {}
    try:
        for name in ("1d", "4h", "1h"):
            a = analyze_tf(fetch_candles(sym, name))
            if a is None:
                return sym, None, "not enough history"
            out[name] = a
        return sym, out, None
    except Exception as e:
        return sym, None, str(e)


def scan(only=None):
    notes = []
    universe = load_universe(only)
    spreads = get_spreads(notes)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(analyze_symbol, list(universe)))
    analyses = {s: a for s, a, _ in results if a}
    errors = [e for _, a, e in results if e and not a]
    print(f"Analysed {len(analyses)}/{len(universe)} pairs, {len(errors)} errors")
    if not analyses:
        raise RuntimeError(errors[0] if errors else "no data")

    btc = None
    if "BTCUSDT" in analyses:
        b = analyses["BTCUSDT"]
        btc = {
            "bull": b["4h"]["score"] >= 25 and b["1d"]["score"] >= 0,
            "bear": b["4h"]["score"] <= -25 and b["1d"]["score"] <= 0,
            "text": f"BTC: 1D {b['1d']['label']} (RSI {b['1d']['rsi']:.0f}), "
                    f"4H {b['4h']['label']} (RSI {b['4h']['rsi']:.0f}), 1H {b['1h']['label']}",
        }

    setups = []
    for sym, an in analyses.items():
        s = build_setup(sym, universe[sym], an, btc, spreads.get(sym))
        if s and s["grade"]:
            setups.append(s)

    # order-book check for the strongest candidates only (saves API weight)
    setups.sort(key=lambda s: -s["conf"])
    to_check = setups[:DEPTH_CHECKS]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        def _d(s):
            try:
                return depth_stats(s["symbol"])
            except Exception:
                return None
        for s, ds in zip(to_check, ex.map(_d, to_check)):
            apply_depth(s, ds)
    setups = [s for s in setups if s["grade"]]
    setups.sort(key=lambda s: -s["conf"])

    extended = [sym for sym, an in analyses.items()
                if an["4h"]["rsi"] >= 75 and an["4h"]["score"] > 0]
    overview = market_overview(analyses, universe, setups, btc)
    return setups, extended, overview, notes


def market_overview(analyses, universe, setups, btc):
    n = len(analyses)
    pct = lambda tf: sum(1 for a in analyses.values() if a[tf]["score"] >= 25) / n * 100
    pct_bear4 = sum(1 for a in analyses.values() if a["4h"]["score"] <= -25) / n * 100
    return {
        "n": n, "p1d": pct("1d"), "p4h": pct("4h"), "p1h": pct("1h"), "bear4": pct_bear4,
        "chg": sum(universe[s]["chg"] for s in analyses) / n,
        "btc": btc, "buys": sum(1 for s in setups if s["side"] == "BUY"),
        "sells": sum(1 for s in setups if s["side"] == "SELL"),
    }



# ================= INTRADAY MODE: 5m / 10m / 15m / 30m ================
INTRA_W = {"30m": 0.30, "15m": 0.30, "10m": 0.20, "5m": 0.20}


def resample_10m(cd):
    """Binance has no 10m chart, so build it from pairs of closed 5m candles."""
    buckets, order = {}, []
    for t, h, l, c, v in zip(cd["t"], cd["h"], cd["l"], cd["c"], cd["v"]):
        b = t - t % 600000
        x = buckets.get(b)
        if x is None:
            buckets[b] = {"h": h, "l": l, "c": c, "v": v, "n": 1}
            order.append(b)
        else:
            x["h"], x["l"], x["c"] = max(x["h"], h), min(x["l"], l), c
            x["v"] += v
            x["n"] += 1
    keep = [b for b in order if buckets[b]["n"] == 2]  # drop incomplete buckets
    return {"t": keep, "h": [buckets[b]["h"] for b in keep], "l": [buckets[b]["l"] for b in keep],
            "c": [buckets[b]["c"] for b in keep], "v": [buckets[b]["v"] for b in keep]}


def analyze_intraday(sym):
    try:
        c5 = fetch_candles(sym, "5m", 500)
        tfs = {
            "5m": analyze_tf(c5), "10m": analyze_tf(resample_10m(c5)),
            "15m": analyze_tf(fetch_candles(sym, "15m")), "30m": analyze_tf(fetch_candles(sym, "30m")),
            "1h": analyze_tf(fetch_candles(sym, "1h")), "4h": analyze_tf(fetch_candles(sym, "4h")),
        }
        if any(v is None for v in tfs.values()):
            return sym, None, "not enough history"
        return sym, tfs, None
    except Exception as e:
        return sym, None, str(e)


def pick_intraday(tfs, overall):
    a5, a15, a30, a1 = tfs["5m"], tfs["15m"], tfs["30m"], tfs["1h"]
    cands = []
    # ---------- BUY ----------
    if (a1["score"] >= 10 and a30["score"] >= 25 and a15["rsi"] <= 45
            and a15["close"] <= a15["e50"] * 1.003 and a5["mh"] is not None and a5["mh"] > a5["mhp"]):
        cands.append(("BUY", "Intraday pullback", 40 + 0.2 * (a30["score"] + a1["score"]), [
            f"30m and 1H are up, 15m dipped (RSI {a15['rsi']:.0f}, at the 15m EMA50) and 5m "
            "momentum is turning back up: a dip-buy with a tight stop.",
        ]))
    if (a15["close"] > a15["hi20"] and a15["vr"] >= 1.8 and a30["score"] >= 15
            and a1["score"] >= 0 and a15["rsi"] < 78):
        cands.append(("BUY", "Intraday breakout",
                      40 + 0.2 * (a15["score"] + a30["score"]) + min(10, 4 * (a15["vr"] - 1.8)), [
            f"15m candle closed above its 20-candle high on {a15['vr']:.1f}x average volume.",
        ]))
    if overall >= 35 and a1["score"] >= 10 and a15["rsi"] < 72 and a5["rsi"] < 75:
        cands.append(("BUY", "Intraday momentum", 25 + 0.8 * overall, [
            f"5m, 10m, 15m and 30m trends agree upward (intraday score {overall:+.0f}).",
        ]))
    # ---------- SELL / EXIT ----------
    if (a15["close"] < a15["lo20"] and a15["vr"] >= 1.8 and a30["score"] <= -15 and a1["score"] <= 0):
        cands.append(("SELL", "Intraday breakdown",
                      40 + 0.2 * (-a15["score"] - a30["score"]) + min(10, 4 * (a15["vr"] - 1.8)), [
            f"15m candle closed below its 20-candle low on {a15['vr']:.1f}x average volume.",
        ]))
    if a15["rsi"] >= 78 and a15["bb"] >= 1.0 and a15["mh"] is not None and a15["mh"] < a15["mhp"]:
        cands.append(("SELL", "Intraday overbought (take profit)", 45 + min(15, (a15["rsi"] - 78) * 2), [
            f"15m RSI {a15['rsi']:.0f}, price above the upper band and momentum fading: take profit.",
        ]))
    if overall <= -35 and a1["score"] <= -10:
        cands.append(("SELL", "Intraday momentum down (exit / avoid)", 25 + 0.8 * abs(overall), [
            f"5m, 10m, 15m and 30m trends agree downward (intraday score {overall:+.0f}).",
        ]))
    return max(cands, key=lambda c: c[2]) if cands else None


def intraday_horizon(side, tfs):
    sign = 1 if side == "BUY" else -1
    s = lambda k: sign * tfs[k]["score"]
    if s("30m") >= 25 and s("15m") >= 25:
        return "Intraday: 1-4 hours. Trade on the 15m chart, confirm on 30m."
    if s("15m") >= 25:
        return "Short intraday: 30-90 minutes. Trade on the 15m chart."
    return "Scalp: 5-45 minutes. Trade on the 5m/10m chart and take profit fast."


def build_intraday_setup(sym, info, tfs, btc, spread_bps):
    if spread_bps is not None and spread_bps > MAX_SPREAD_BPS_INTRADAY:
        return None
    overall = sum(INTRA_W[k] * tfs[k]["score"] for k in INTRA_W)
    pick = pick_intraday(tfs, overall)
    if not pick:
        return None
    side, name, conf, reasons = pick
    a5, a15, a1, a4 = tfs["5m"], tfs["15m"], tfs["1h"], tfs["4h"]
    price = a5["close"]
    reasons = list(reasons)
    risk = 1.5 * a15["atr"]
    if risk >= price or risk / price * 100 > 6:
        return None

    if side == "BUY":
        tp1_pct = 1.5 * risk / price * 100
        if tp1_pct < MIN_TP1_PCT:
            return None  # target too small after trading fees
        plan = {"entry_lo": price - 0.5 * a5["atr"], "entry_hi": price, "stop": price - risk,
                "tp1": price + 1.5 * risk, "tp2": price + 3 * risk, "risk_pct": risk / price * 100,
                "res": a15["hi40"]}
        if plan["res"] > price and plan["res"] < plan["tp1"]:
            conf -= 8
            reasons.append(f"Caution: 15m resistance near {fmt_p(plan['res'])} sits before target 1.")
        extra = (f"Time stop: if Target 1 is not hit in about 3 hours, exit. Fees ~0.15-0.2% round trip "
                 f"(Target 1 is {tp1_pct:.1f}%).")
    else:
        plan = {"s1": price - 1.5 * a15["atr"], "s2": price - 3 * a15["atr"],
                "swing": a15["lo40"] if a15["lo40"] < price else None, "tf": "15m",
                "invalid": a15["e50"] if a15["e50"] > price else price + 1.5 * a15["atr"]}
        extra = "Intraday signals can fade within minutes. Act quickly or skip."

    sign = 1 if side == "BUY" else -1
    if all(sign * tfs[k]["score"] >= 15 for k in INTRA_W):
        conf += 8
        reasons.append("5m, 10m, 15m and 30m all point the same way.")
    if btc and sym != "BTCUSDT":
        if side == "BUY":
            if btc["bull"]:
                conf += 5
                reasons.append("BTC short-term trend is supportive.")
            elif btc["bear"]:
                conf -= 10
                reasons.append("Caution: BTC short-term trend is bearish.")
        else:
            if btc["bear"]:
                conf += 5
                reasons.append("BTC short-term trend is also bearish.")
            elif btc["bull"]:
                conf -= 5
                reasons.append("BTC is bullish, so this may only be a dip.")
    reasons.append(f"Higher timeframes: {tf_text('1H', a1)}, {tf_text('4H', a4)}.")
    if info["qv"] < 30_000_000:
        conf -= 5
    if spread_bps is not None:
        conf -= 10 if spread_bps > 5 else 4 if spread_bps > 3 else 0

    return {
        "symbol": sym, "side": side, "setup": name, "conf": conf, "grade": grade_of(conf),
        "price": price, "overall": overall, "intraday": True,
        "tf_text": [tf_text(k, tfs[k]) for k in ("5m", "10m", "15m", "30m")],
        "plan": plan, "horizon": intraday_horizon(side, tfs), "reasons": reasons, "extra": extra,
        "liq": {"qv": info["qv"], "spread": spread_bps, "chg": info["chg"]}, "new": False,
    }


def scan_intraday(only=None):
    notes = []
    universe = load_universe(only)
    symbols = list(universe) if only else list(universe)[:INTRADAY_N]
    if "BTCUSDT" in universe and "BTCUSDT" not in symbols:
        symbols.append("BTCUSDT")
    spreads = get_spreads(notes)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(analyze_intraday, symbols))
    analyses = {s: a for s, a, _ in results if a}
    errors = [e for _, a, e in results if e and not a]
    print(f"Analysed {len(analyses)}/{len(symbols)} pairs, {len(errors)} errors")
    if not analyses:
        raise RuntimeError(errors[0] if errors else "no data")

    btc = None
    if "BTCUSDT" in analyses:
        b = analyses["BTCUSDT"]
        btc = {"bull": b["1h"]["score"] >= 10 and b["4h"]["score"] >= 0,
               "bear": b["1h"]["score"] <= -10 and b["4h"]["score"] <= 0,
               "text": "BTC: " + ", ".join(f"{k} {b[k]['label']}" for k in ("5m", "15m", "30m", "1h"))}

    setups = []
    for sym, tfs in analyses.items():
        s = build_intraday_setup(sym, universe[sym], tfs, btc, spreads.get(sym))
        if s and s["grade"]:
            setups.append(s)
    setups.sort(key=lambda s: -s["conf"])

    def _d(s):
        try:
            return depth_stats(s["symbol"])
        except Exception:
            return None
    to_check = setups[:INTRADAY_DEPTH_CHECKS]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for s, ds in zip(to_check, ex.map(_d, to_check)):
            apply_depth(s, ds)
    setups = sorted([s for s in setups if s["grade"]], key=lambda s: -s["conf"])
    return setups, intraday_overview(analyses, btc), notes


def intraday_overview(analyses, btc):
    n = len(analyses)
    p = lambda k: sum(1 for a in analyses.values() if a[k]["score"] >= 25) / n * 100
    L = [f"Pairs scanned: top {n} by volume",
         f"Trending up: {p('5m'):.0f}% on 5m | {p('10m'):.0f}% on 10m | {p('15m'):.0f}% on 15m | "
         f"{p('30m'):.0f}% on 30m"]
    if btc:
        L.append(btc["text"])
    if p("15m") >= 60 and p("30m") >= 60:
        L.append("Read: short-term momentum is strong. Favour buying dips on the 15m chart.")
    elif p("15m") <= 35 and p("30m") <= 35:
        L.append("Read: short-term weakness. Avoid new buys and protect profits.")
    else:
        L.append("Read: choppy. Intraday signals are less reliable, so trade smaller.")
    return L


def build_intraday_email(shown, overview, notes, local_now):
    buys = [s for s in shown if s["side"] == "BUY"][:MAX_PER_SIDE]
    sells = [s for s in shown if s["side"] == "SELL"][:MAX_PER_SIDE]
    parts = [f"Spot INTRADAY alert (5m / 10m / 15m / 30m) - {local_now:%Y-%m-%d %H:%M} {TZ_NAME}\n"]
    parts += [f"NOTE: {n}\n" for n in notes]
    if buys:
        parts.append("=== INTRADAY BUY ===\n\n" + "\n\n".join(fmt_setup(s) for s in buys) + "\n")
    if sells:
        parts.append("=== INTRADAY SELL / EXIT ===\n\n" + "\n\n".join(fmt_setup(s) for s in sells) + "\n")
    parts.append("-" * 36 + "\nINTRADAY OVERVIEW\n" + "\n".join(overview))
    parts.append("\nShort-term rules: fees (~0.15-0.2% round trip) eat small profits, so prefer A/B grades. "
                 "Use limit orders inside the entry zone, always set the stop, and risk no more than 1% "
                 "of capital per trade. Signals use closed candles and can be wrong. Not financial advice.")
    top = shown[0]
    subject = f"Intraday: {len(shown)} new - {top['symbol']} {top['side']} ({top['grade']})"
    return subject, "\n".join(parts)


# =============================== Email ===============================
def fmt_p(x):
    if x >= 1000:
        return f"{x:,.1f}"
    if x < 0.0001:
        return f"{x:.8f}"
    return f"{x:.5g}"


def pct_from(price, level):
    return (level - price) / price * 100


def fmt_setup(s):
    p, pl, lq = s["price"], s["plan"], s["liq"]
    lines = [f"{s['side']}  {s['symbol']}   Grade {s['grade']} ({s['conf']:.0f}/100)"
             + ("  [NEW]" if s["new"] else ""),
             f"  Setup: {s['setup']}",
             f"  Price: {fmt_p(p)}   24h: {lq['chg']:+.1f}%",
             "  Charts: " + " | ".join(s["tf_text"])]
    if s["side"] == "BUY":
        lines += [
            f"  Entry zone: {fmt_p(pl['entry_lo'])} - {fmt_p(pl['entry_hi'])}",
            f"  Stop: {fmt_p(pl['stop'])} ({pct_from(p, pl['stop']):+.1f}%)",
            f"  Target 1: {fmt_p(pl['tp1'])} ({pct_from(p, pl['tp1']):+.1f}%)  [1.5R]",
            f"  Target 2: {fmt_p(pl['tp2'])} ({pct_from(p, pl['tp2']):+.1f}%)  [3R]",
        ]
    else:
        lines += [
            "  Action: exit or trim if you hold it. Do not buy.",
            f"  Downside levels: {fmt_p(pl['s1'])} ({pct_from(p, pl['s1']):+.1f}%), "
            f"{fmt_p(pl['s2'])} ({pct_from(p, pl['s2']):+.1f}%)"
            + (f", {pl.get('tf', '4H')} swing low {fmt_p(pl['swing'])} ({pct_from(p, pl['swing']):+.1f}%)" if pl["swing"] else ""),
            f"  Idea invalid if {pl.get('tf', '4H')} closes above: {fmt_p(pl['invalid'])}",
        ]
    lines.append(f"  Horizon: {s['horizon']}")
    if s.get("extra"):
        lines.append(f"  {s['extra']}")
    liq = f"  Liquidity: 24h vol ${lq['qv'] / 1e6:,.0f}M"
    if lq["spread"] is not None:
        liq += f" | spread {lq['spread']:.1f} bps"
    d = lq.get("depth")
    if d:
        slip = d["slip_buy"] if s["side"] == "BUY" else d["slip_sell"]
        liq += (f" | book depth ${min(d['bid_depth'], d['ask_depth']) / 1e3:,.0f}K"
                f" | est. slippage on ${ORDER_SIZE_USDT:,}: "
                + (f"{slip:.2f}%" if slip is not None else "too thin"))
    lines.append(liq)
    lines.append("  Why:")
    lines += [f"   - {r}" for r in s["reasons"]]
    return "\n".join(lines)


def overview_text(o, extended):
    L = ["MARKET OVERVIEW",
         f"Pairs analysed: {o['n']} | average 24h change {o['chg']:+.1f}%",
         f"Trending up: {o['p1d']:.0f}% on 1D | {o['p4h']:.0f}% on 4H | {o['p1h']:.0f}% on 1H "
         f"(trending down on 4H: {o['bear4']:.0f}%)"]
    if o["btc"]:
        L.append(o["btc"]["text"])
    btc = o["btc"]
    if o["p4h"] >= 60 and (not btc or not btc["bear"]):
        bias = "Risk-on: buyers are in control. Favour buying pullbacks in strong trends."
    elif o["p4h"] <= 40 or (btc and btc["bear"]):
        bias = "Risk-off: weak breadth. Protect capital, trade smaller, and avoid chasing."
    else:
        bias = "Mixed: be selective. Stick to A/B grade setups with good liquidity."
    L.append(f"Bias: {bias}")
    if o["p1d"] >= 55 and o["p4h"] >= 55:
        L.append("Timeframe read: 1D and 4H agree, so swing trades have the better odds right now.")
    elif o["p1h"] >= 60 and o["p4h"] < 45:
        L.append("Timeframe read: only the 1H is strong, so treat moves as short-term and take profits quickly.")
    if extended:
        names = ", ".join(s.replace("USDT", "") for s in extended[:10])
        L.append(f"Extended (4H RSI 75+), so avoid chasing: {names}")
    L.append("")
    L.append("Risk rules: risk no more than 1% of your capital per trade (position size = "
             "amount you can lose / distance to stop). Take partial profit at Target 1 and "
             "move your stop to entry. Signals use closed candles and can be wrong. "
             "This is not financial advice.")
    return "\n".join(L)


def build_email(shown, overview, extended, notes, digest, local_now):
    buys = [s for s in shown if s["side"] == "BUY"][:MAX_PER_SIDE]
    sells = [s for s in shown if s["side"] == "SELL"][:MAX_PER_SIDE]
    head = (f"Spot analyst {'daily digest' if digest else 'alert'} - "
            f"{local_now:%Y-%m-%d %H:%M} {TZ_NAME}\n")
    parts = [head] + [f"NOTE: {n}\n" for n in notes]
    if buys:
        parts.append("=== BUY SETUPS ===\n\n" + "\n\n".join(fmt_setup(s) for s in buys) + "\n")
    if sells:
        parts.append("=== SELL / EXIT ===\n\n" + "\n\n".join(fmt_setup(s) for s in sells) + "\n")
    if not buys and not sells:
        parts.append("No graded setups right now. The market is not offering clear trades.\n")
    parts.append("-" * 36)
    parts.append(overview_text(overview, extended))
    if digest:
        subject = f"Spot digest: {len(buys)} buy / {len(sells)} sell setups"
    else:
        top = shown[0]
        subject = (f"Spot alert: {len(shown)} new - {top['symbol']} {top['side']} "
                   f"({top['grade']})")
    return subject, "\n".join(parts)


# ========================= State & sending ===========================
def load_state():
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
    except Exception:
        st = {}
    st.setdefault("alerts", {})
    st.setdefault("last_digest", "")
    cutoff = time.time() - 3 * 24 * 3600
    st["alerts"] = {k: v for k, v in st["alerts"].items() if v > cutoff}
    return st


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=1)


def send_email(subject, body):
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, EMAIL_USER, EMAIL_TO
    msg.set_content(body)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context()) as s:
        s.login(EMAIL_USER, EMAIL_PASS)
        s.send_message(msg)


def run_intraday(dry, only, state, local_now):
    today = local_now.strftime("%Y-%m-%d")
    try:
        setups, overview, notes = scan_intraday(only)
    except Exception as e:
        print(f"Scan failed: {e}")
        if not dry and state.get("last_error_day") != today:
            send_email("Spot intraday: data problem", f"The scan failed: {e}\n"
                       "If this says 'region blocked', Binance is blocking this server's location.")
            state["last_error_day"] = today
            save_state(state)
        return
    now = time.time()
    for s in setups:
        key = f"I:{s['side']}:{s['symbol']}"
        s["new"] = s["grade"] in ALERT_GRADES and now - state["alerts"].get(key, 0) > INTRADAY_COOLDOWN_HOURS * 3600
    new = [s for s in setups if s["new"]]
    print(f"{len(setups)} graded intraday setups, {len(new)} new")
    if not new:
        print("Nothing to send.")
        if not dry:
            save_state(state)
        return
    subject, body = build_intraday_email(new, overview, notes, local_now)
    print(subject + "\n\n" + body)
    if dry:
        return
    send_email(subject, body)
    for s in new:
        state["alerts"][f"I:{s['side']}:{s['symbol']}"] = now
    save_state(state)
    print("Email sent.")


def main():
    global STATE_FILE
    dry = "--dry" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].upper().split(","))
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "swing"
    if mode == "intraday":
        STATE_FILE = "intraday_state.json"
    if "--test" in sys.argv:
        send_email("Spot analyst test", "Your analyst can send email.")
        print("Test email sent.")
        return
    if not dry and not (EMAIL_USER and EMAIL_PASS):
        sys.exit("Set ALERT_EMAIL_USER and ALERT_EMAIL_PASS first.")

    state = load_state()
    local_now = datetime.now(timezone(timedelta(hours=TZ_OFFSET_HOURS)))
    if mode == "intraday":
        return run_intraday(dry, only, state, local_now)
    today = local_now.strftime("%Y-%m-%d")

    try:
        setups, extended, overview, notes = scan(only)
    except Exception as e:
        print(f"Scan failed: {e}")
        if not dry and state.get("last_error_day") != today:
            send_email("Spot analyst: data problem", f"The scan failed: {e}\n"
                       "If this says 'region blocked', Binance is blocking this server's location.")
            state["last_error_day"] = today
            save_state(state)
        return

    now = time.time()
    for s in setups:
        key = f"{s['side']}:{s['symbol']}"
        last = state["alerts"].get(key, 0)
        s["new"] = s["grade"] in ALERT_GRADES and now - last > COOLDOWN_HOURS * 3600

    digest_due = local_now.hour >= DIGEST_HOUR_LOCAL and state["last_digest"] != today
    new = [s for s in setups if s["new"]]
    print(f"{len(setups)} graded setups, {len(new)} new, digest due: {digest_due}")

    if not digest_due and not new:
        print("Nothing to send.")
        if not dry:
            save_state(state)
        return

    shown = setups if digest_due else new
    subject, body = build_email(shown, overview, extended, notes, digest_due, local_now)
    print(subject + "\n\n" + body)
    if dry:
        return
    send_email(subject, body)
    for s in new:
        state["alerts"][f"{s['side']}:{s['symbol']}"] = now
    if digest_due:
        state["last_digest"] = today
    save_state(state)
    print("Email sent.")


if __name__ == "__main__":
    if "--loop" in sys.argv:  # for a VPS: python analyst.py --mode intraday --loop 300
        secs = int(sys.argv[sys.argv.index("--loop") + 1])
        while True:
            try:
                main()
            except SystemExit:
                raise
            except Exception as e:
                print(f"Run failed: {e}")
            time.sleep(secs)
    else:
        main()
