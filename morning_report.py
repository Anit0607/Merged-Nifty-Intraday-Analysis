import yfinance as yf
import numpy as np
import pandas as pd
import requests
import math
import json
import re
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import pytz

nltk.download("vader_lexicon", quiet=True)

TELE_TOKEN = "8623520423:AAH4mSilxgtMHXXyRwlwlFfB-FjB42sDrwQ"
TELE_CHAT  = 1083142887

HIGH_IMPACT = [
    "rbi", "repo rate", "rate decision", "rate cut", "rate hike",
    "federal reserve", "fed meeting", "fomc",
    "election result", "budget", "ceasefire", "war declaration",
    "default", "sovereign", "emergency", "circuit breaker"
]
MEDIUM_IMPACT = [
    "gdp", "inflation", "cpi", "wpi", "iip", "pmi",
    "quarterly result", "earnings", "crude shock", "oil price",
    "ipo listing", "fii selling", "fii buying", "npa", "credit policy"
]


def tele(text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELE_TOKEN}/sendMessage",
            json={"chat_id": TELE_CHAT, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        print(f"Telegram error: {e}")


def fetch_news_sentiment():
    """Fetch India market headlines from Google News RSS and score with VADER."""
    queries = [
        "nifty+50+india+stock+market+today",
        "india+NSE+BSE+market+outlook",
        "RBI+Fed+FII+india+economy"
    ]
    headlines = []
    sia = SentimentIntensityAnalyzer()

    for q in queries:
        try:
            url  = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:6]:
                title = re.sub(r"\s*-\s*[^-]+$", "", entry.title).strip()
                headlines.append(title)
        except Exception:
            pass

    if not headlines:
        return 0.0, [], "Neutral", "Low", 0.0

    # Deduplicate
    seen, unique = set(), []
    for h in headlines:
        key = h[:50].lower()
        if key not in seen:
            seen.add(key)
            unique.append(h)
    headlines = unique[:12]

    scores    = [sia.polarity_scores(h)["compound"] for h in headlines]
    avg_score = sum(scores) / len(scores)

    if   avg_score >  0.15: bias = "Bullish"
    elif avg_score < -0.15: bias = "Bearish"
    else:                   bias = "Neutral"

    all_text = " ".join(headlines).lower()
    if any(kw in all_text for kw in HIGH_IMPACT):
        impact_level, impact_adj = "High",   0.10
    elif any(kw in all_text for kw in MEDIUM_IMPACT):
        impact_level, impact_adj = "Medium", 0.05
    else:
        impact_level, impact_adj = "Low",    0.0

    return avg_score, headlines, bias, impact_level, impact_adj


def fetch_fii_dii():
    """Try NSE API for FII/DII flows. Returns (fii_net_cr, dii_net_cr) or (None, None)."""
    try:
        session = requests.Session()
        bhdrs = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/market-data/fii-dii-activity",
        }
        session.get("https://www.nseindia.com", headers=bhdrs, timeout=8)
        resp = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers=bhdrs, timeout=8
        )
        data    = resp.json()
        fii_net = dii_net = None
        for item in data:
            cat = str(item.get("category", "")).upper()
            val = str(item.get("netVal", "0")).replace(",", "")
            try:
                net = float(val)
            except ValueError:
                continue
            if "FII" in cat or "FPI" in cat:
                fii_net = net
            elif "DII" in cat:
                dii_net = net
        return fii_net, dii_net
    except Exception:
        return None, None


def fii_label(val):
    if val is None: return "Unavailable"
    if   val >  2000: return f"Strong Buying (+{val:,.0f} Cr)"
    elif val >   500: return f"Buying (+{val:,.0f} Cr)"
    elif val <  -2000: return f"Strong Selling ({val:,.0f} Cr)"
    elif val <  -500: return f"Selling ({val:,.0f} Cr)"
    else:              return f"Neutral ({val:+,.0f} Cr)"


def main():
    ist       = pytz.timezone("Asia/Calcutta")
    today     = datetime.now(ist)
    today_str = today.strftime("%d %b %Y")

    # 1. Nifty OHLC
    df = yf.Ticker("^NSEI").history(period="10y")[["Open","High","Low","Close"]].dropna()
    last_date = df.index[-1].date()
    bdays     = np.busday_count(last_date, today.date())
    if bdays > 1:
        tele(f"NSE closed today ({today_str}) — no analysis")
        return

    prev      = df.iloc[-1]
    pH, pL, pC = float(prev.High), float(prev.Low), float(prev.Close)
    prev2_close = float(df.iloc[-2].Close)

    # 2. VIX
    try:    vix = float(yf.Ticker("^INDIAVIX").history(period="5d")["Close"].iloc[-1])
    except: vix = 19.5

    # 3. Global markets
    def chg(t):
        try:
            d = yf.Ticker(t).history(period="5d")["Close"]
            return (d.iloc[-1] - d.iloc[-2]) / d.iloc[-2] * 100
        except: return 0.0

    sp = chg("^GSPC"); nq = chg("^IXIC")
    nk = chg("^N225"); hs = chg("^HSI")
    try:    brent  = float(yf.Ticker("BZ=F").history(period="2d")["Close"].iloc[-1])
    except: brent  = 0.0
    try:    usdinr = float(yf.Ticker("USDINR=X").history(period="2d")["Close"].iloc[-1])
    except: usdinr = 0.0

    # 4. News & Sentiment
    print("Fetching news...")
    news_score, headlines, news_bias, impact_level, impact_adj = fetch_news_sentiment()

    # 5. FII / DII
    print("Fetching FII/DII...")
    fii_net, dii_net = fetch_fii_dii()

    # 6. Markov
    ret  = df["Close"].pct_change().dropna()
    rm   = ret.rolling(20).mean().dropna()
    p33, p67 = float(np.percentile(rm, 33)), float(np.percentile(rm, 67))
    lbl  = rm.apply(lambda x: "Bear" if x <= p33 else ("Bull" if x >= p67 else "Sideways"))
    states = ["Bear", "Sideways", "Bull"]
    T      = pd.DataFrame(0, index=states, columns=states)
    for i in range(len(lbl) - 1):
        T.loc[lbl.iloc[i], lbl.iloc[i+1]] += 1
    Tp        = T.div(T.sum(axis=1), axis=0)
    cr        = lbl.iloc[-1]
    pers      = float(Tp.loc[cr, cr])
    esc       = 1.0 - pers
    days_left = round(1 / (1 - pers), 1) if pers < 1 else 999
    drift     = float(ret.reindex(lbl.index)[lbl == cr].mean() * pC)

    # 7. Params
    try:
        resp_p = requests.get(
            "https://raw.githubusercontent.com/Anit0607/Merged-Nifty-Intraday-Analysis/main/params.json",
            timeout=6
        )
        p = resp_p.json()
    except Exception:
        p = {"call_buffer_pts": 15, "put_buffer_pts": 50,
             "markov_weight": 0.60, "intraday_weight": 0.40,
             "vix_range_mult": 1.5, "vix_assumed": 19.5}

    vix_mult = float(p.get("vix_range_mult", 1.5))
    mw       = float(p.get("markov_weight",   0.60))
    iw       = float(p.get("intraday_weight", 0.40))

    # 8. CPR + pivot levels
    pivot = (pH + pL + pC) / 3
    BC    = (pH + pL) / 2
    TC    = 2 * pivot - BC
    cpu   = max(TC, BC)
    cpl   = min(TC, BC)
    cpw   = cpu - cpl
    R1    = 2 * pivot - pL
    R2    = pivot + (pH - pL)
    S1    = 2 * pivot - pH
    S2    = pivot - (pH - pL)

    vh = (vix / 100) * pC * (1 / math.sqrt(252)) * vix_mult / 2
    ub = round(pC + vh * 2, 0)
    lb = round(pC - vh * 2, 0)

    # 9. Probability model
    m_down = float(Tp.loc[cr, "Bear"])
    m_side = float(Tp.loc[cr, "Sideways"])
    m_up   = float(Tp.loc[cr, "Bull"])

    # Intraday component from gap + VIX
    gap_pct = (pC - prev2_close) / prev2_close * 100
    if   gap_pct >  1.0:  i_up, i_side, i_down = 0.45, 0.33, 0.22
    elif gap_pct >  0.3:  i_up, i_side, i_down = 0.40, 0.35, 0.25
    elif gap_pct < -1.0:  i_up, i_side, i_down = 0.22, 0.33, 0.45
    elif gap_pct < -0.3:  i_up, i_side, i_down = 0.25, 0.35, 0.40
    else:                 i_up, i_side, i_down = 0.35, 0.38, 0.27

    if   vix < 14: i_side += 0.05; i_up -= 0.03; i_down -= 0.02
    elif vix > 22: i_side -= 0.05; i_up += 0.02; i_down += 0.03

    # Merge
    pd_ = mw * m_down + iw * i_down
    ps_ = mw * m_side + iw * i_side
    pu  = mw * m_up   + iw * i_up

    # News overlay
    if news_bias == "Bullish" and impact_adj > 0:
        pu  += impact_adj; pd_ -= impact_adj * 0.6; ps_ -= impact_adj * 0.4
    elif news_bias == "Bearish" and impact_adj > 0:
        pd_ += impact_adj; pu  -= impact_adj * 0.6; ps_ -= impact_adj * 0.4

    # FII flow adjustment
    if fii_net is not None:
        if   fii_net >  2000: pu += 0.03; pd_ -= 0.03
        elif fii_net >   500: pu += 0.02; pd_ -= 0.02
        elif fii_net < -2000: pd_ += 0.03; pu -= 0.03
        elif fii_net <  -500: pd_ += 0.02; pu -= 0.02

    # Normalise to 100%
    tot = pd_ + ps_ + pu
    pd_ /= tot; ps_ /= tot; pu /= tot

    # 10. Strikes
    ca = max(cpu, pC + vh)
    cs = math.ceil((ca + p["call_buffer_pts"]) / 50) * 50
    if cs <= ca: cs += 50

    pa = min(S2, pC - vh)
    ps_strike = math.floor((pa - p["put_buffer_pts"]) / 50) * 50
    if ps_strike >= pa: ps_strike -= 50

    if cr == "Bear" and pers > 0.83: ps_strike -= 50
    if cr == "Bull" and pers > 0.85: cs        += 50
    if esc > 0.20:                   cs += 50; ps_strike -= 50

    # 11. Labels
    re_icon   = {"Bear": "🔴", "Sideways": "🟡", "Bull": "🟢"}.get(cr, "")
    dt_label  = {"Bear": "Sell-on-rise", "Sideways": "Range-selling", "Bull": "Buy-on-dip"}.get(cr, "")
    vix_label = ("Very Low" if vix < 12 else "Normal" if vix < 16 else
                 "Elevated" if vix < 20 else "High" if vix < 25 else "Extreme")
    news_icon = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(news_bias, "⚪")
    imp_icon  = {"High": "⚡", "Medium": "🔔", "Low": "📎"}.get(impact_level, "")
    sg = "+" if sp >= 0 else ""
    ng = "+" if nk >= 0 else ""

    # 12. Message 1 — Regime + Levels + Strikes
    m1 = (
        f"<b>NIFTY PRE-MARKET — {today_str}</b>\n\n"
        f"<b>REGIME: {cr}</b> {re_icon}\n"
        f"Persistence: <b>{pers:.1%}</b>  Escape: {esc:.1%}  ~{days_left}d left\n"
        f"Expected drift: <b>{drift:+.0f} pts/day</b>\n\n"
        f"<b>KEY LEVELS</b>\n"
        f"<code>"
        f"Prev Close : {pC:.0f}\n"
        f"CPR        : {cpl:.0f} - {cpu:.0f}  (W:{cpw:.0f})\n"
        f"Pivot      : {pivot:.0f}\n"
        f"R1 / R2    : {R1:.0f}  /  {R2:.0f}\n"
        f"S1 / S2    : {S1:.0f}  /  {S2:.0f}\n"
        f"VIX        : {vix:.1f}  [{vix_label}]\n"
        f"Day Range  : {lb:.0f} - {ub:.0f}  ({(ub-lb):.0f} pts)"
        f"</code>\n\n"
        f"<b>OPTION SELLER STRIKES</b>\n"
        f"SELL CALL : <b>{cs} CE</b>  ({cs-pC:.0f} pts OTM)\n"
        f"SELL PUT  : <b>{ps_strike} PE</b>  ({pC-ps_strike:.0f} pts OTM)\n"
        f"Entry: 9:30-10:00 AM | Stop: 15-min close beyond strike"
    )

    # 13. Message 2 — Probability + Global
    m2 = (
        f"<b>PROBABILITY  (base {pC:.0f})</b>\n"
        f"Upside:   <b>{pu:.0%}</b>\n"
        f"Sideways: <b>{ps_:.0%}</b>\n"
        f"Downside: <b>{pd_:.0%}</b>\n"
        f"<i>Markov {mw:.0%} + Intraday {iw:.0%}"
        + (f" + {news_bias} news {impact_adj:.0%}" if impact_adj > 0 else "")
        + "</i>\n\n"
        f"<b>GLOBAL CUES</b>\n"
        f"S&amp;P500: {sg}{sp:.1f}%  Nasdaq: {nq:+.1f}%\n"
        f"Nikkei: {ng}{nk:.1f}%  HSI: {hs:+.1f}%\n"
        f"Brent: ${brent:.1f}  USD/INR: {usdinr:.2f}\n\n"
        f"<b>TRADE GUIDANCE — {dt_label}</b>\n"
        f"Sellers : {cs} CE / {ps_strike} PE strangle, entry 9:30-10 AM\n"
        f"Buyers  : {'Puts on bounce' if cr=='Bear' else 'Calls on dip' if cr=='Bull' else 'Wait for breakout'}\n"
        f"Futures : {'Short on rise' if cr=='Bear' else 'Long on dip' if cr=='Bull' else 'Fade range extremes'}"
    )

    # 14. Message 3 — News & Sentiment + FII/DII
    hdl_lines = "\n".join(
        f"  {i+1}. {h[:88]}" for i, h in enumerate(headlines[:6])
    ) if headlines else "  (Headlines unavailable)"

    m3 = (
        f"<b>NEWS &amp; SENTIMENT</b>\n"
        f"{news_icon} Overall bias: <b>{news_bias}</b>  (score {news_score:+.2f})\n"
        f"{imp_icon} Impact level: <b>{impact_level}</b>"
        + (f"  → prob adj {impact_adj:+.0%}" if impact_adj > 0 else " → no prob adjustment")
        + f"\n\n<b>Top Headlines</b>\n{hdl_lines}\n\n"
        f"<b>INSTITUTIONAL FLOWS (prev day)</b>\n"
        f"FII: {fii_label(fii_net)}\n"
        f"DII: {fii_label(dii_net)}\n\n"
        f"<i>Markov + CPR + News + FII | {today_str} | GitHub Actions</i>"
    )

    tele(m1)
    tele(m2)
    tele(m3)
    print(f"Sent | Regime:{cr} | News:{news_bias}({impact_level}) | FII:{fii_net} | pu:{pu:.0%} ps:{ps_:.0%} pd:{pd_:.0%}")


if __name__ == "__main__":
    main()