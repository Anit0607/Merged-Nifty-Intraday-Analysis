import yfinance as yf
import numpy as np
import pandas as pd
import requests, math, json, re
import feedparser, nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import pytz

nltk.download("vader_lexicon", quiet=True)

TELE_TOKEN = "8623520423:AAH4mSilxgtMHXXyRwlwlFfB-FjB42sDrwQ"
TELE_CHAT  = 1083142887

HIGH_KW   = ["rbi","repo rate","rate decision","rate cut","rate hike",
             "federal reserve","fed meeting","fomc","election result",
             "budget","ceasefire","war","default","emergency","circuit breaker"]
MEDIUM_KW = ["gdp","inflation","cpi","wpi","pmi","quarterly result",
             "earnings","crude","oil price","fii selling","fii buying","npa"]

def tele(text):
    try:
        requests.post(f"https://api.telegram.org/bot{TELE_TOKEN}/sendMessage",
                      json={"chat_id":TELE_CHAT,"text":text,"parse_mode":"HTML"},
                      timeout=15)
    except Exception as e:
        print(f"Telegram err: {e}")

def load_params():
    try:
        with open("params.json") as f: return json.load(f)
    except:
        return {"call_buffer_pts":15,"put_buffer_pts":50,"vix_range_mult":1.5,
                "markov_weight":0.60,"intraday_weight":0.40}

def fetch_news():
    sia = SentimentIntensityAnalyzer()
    queries = ["Nifty 50 stock market India today",
               "Indian stock market NSE outlook",
               "Sensex Nifty prediction"]
    headlines, scores, impact = [], [], "Low"
    for q in queries:
        try:
            url = f"https://news.google.com/rss/search?q={q.replace(' ','+')}&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            for e in feed.entries[:3]:
                t = re.sub(r"<[^>]+>","",e.get("title",""))
                headlines.append(t)
                sc = sia.polarity_scores(t)["compound"]
                scores.append(sc)
                tl = t.lower()
                if any(k in tl for k in HIGH_KW):   impact = "High"
                if any(k in tl for k in MEDIUM_KW) and impact!="High": impact = "Medium"
        except: pass
    avg = sum(scores)/len(scores) if scores else 0.0
    sent = "Bullish" if avg>0.05 else ("Bearish" if avg<-0.05 else "Neutral")
    return avg, sent, impact, headlines[:6]

def fetch_fii():
    try:
        h = {"User-Agent":"Mozilla/5.0","Accept":"application/json",
             "Referer":"https://www.nseindia.com/"}
        r = requests.get("https://www.nseindia.com/api/fiidiiTradeReact",
                         headers=h, timeout=10)
        d = r.json()
        fii = float(d[0].get("netVal",0)) if d else 0.0
        dii = float(d[1].get("netVal",0)) if len(d)>1 else 0.0
        return fii, dii
    except: return 0.0, 0.0

def pct_chg(t):
    try:
        d = yf.Ticker(t).history(period="5d")["Close"].dropna()
        return float((d.iloc[-1]-d.iloc[-2])/d.iloc[-2]*100)
    except: return 0.0

def main():
    ist       = pytz.timezone("Asia/Calcutta")
    today     = datetime.now(ist)
    today_disp= today.strftime("%d %b %Y")

    # ── 1. NIFTY DATA (dual-fetch to bypass yfinance staleness) ───────────
    df_long  = yf.Ticker("^NSEI").history(period="10y")[["Open","High","Low","Close"]].dropna()
    df_fresh = yf.Ticker("^NSEI").history(period="1d")[["Open","High","Low","Close"]].dropna()
    df = pd.concat([df_long, df_fresh])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    last_date = df.index[-1].date()
    bdays = int(np.busday_count(last_date, today.date()))
    if bdays > 3:
        tele(f"NSE closed today ({today_disp}) — no analysis"); return

    prev = df.iloc[-1]
    pH, pL, pC = float(prev.High), float(prev.Low), float(prev.Close)

    # ── 2. VIX ─────────────────────────────────────────────────────────────
    try:
        vix = float(yf.Ticker("^INDIAVIX").history(period="5d")["Close"].dropna().iloc[-1])
    except: vix = 19.5

    # Try to get today Nifty opening price (available after 9:15 AM IST)
    today_open = None
    try:
        import pytz as _ptz
        _intra = yf.Ticker("^NSEI").history(period="1d", interval="1m")[["Open"]].dropna()
        if len(_intra) > 0:
            _bar0 = _intra.index[0]
            if hasattr(_bar0, "tzinfo") and _bar0.tzinfo is not None:
                _bar0 = _bar0.astimezone(_ptz.timezone("Asia/Calcutta"))
            if _bar0.date() == today.date():
                today_open = round(float(_intra["Open"].iloc[0]))
    except Exception:
        pass

    if   vix < 12: vix_lbl = "Very Low"
    elif vix < 16: vix_lbl = "Normal"
    elif vix < 20: vix_lbl = "Elevated"
    elif vix < 25: vix_lbl = "High"
    else:          vix_lbl = "Extreme"

    # ── 3. GLOBAL CUES ──────────────────────────────────────────────────────
    sp=pct_chg("^GSPC"); nq=pct_chg("^IXIC")
    nk=pct_chg("^N225"); hs=pct_chg("^HSI")
    try: brent  = float(yf.Ticker("BZ=F").history(period="2d")["Close"].iloc[-1])
    except: brent = 0.0
    try: usdinr = float(yf.Ticker("USDINR=X").history(period="2d")["Close"].iloc[-1])
    except: usdinr = 0.0

    # ── 4. NEWS & FII ───────────────────────────────────────────────────────
    print("Fetching news...")
    n_score, n_sent, n_impact, headlines = fetch_news()
    print("Fetching FII/DII...")
    fii_net, dii_net = fetch_fii()

    # ── 5. MARKOV ───────────────────────────────────────────────────────────
    ret  = df["Close"].pct_change().dropna()
    rm   = ret.rolling(20).mean().dropna()
    p33, p67 = float(np.percentile(rm,33)), float(np.percentile(rm,67))
    lbl  = rm.apply(lambda x:"Bear" if x<=p33 else("Bull" if x>=p67 else "Sideways"))

    states = ["Bear","Sideways","Bull"]
    T = pd.DataFrame(0,index=states,columns=states)
    for i in range(len(lbl)-1): T.loc[lbl.iloc[i],lbl.iloc[i+1]] += 1
    Tp = T.div(T.sum(axis=1),axis=0)

    cr    = lbl.iloc[-1]
    pers  = float(Tp.loc[cr,cr])
    esc   = 1.0 - pers
    drift = float(ret.reindex(lbl.index)[lbl==cr].mean() * pC)  # INDEX FIX

    m_down = float(Tp.loc[cr,"Bear"])
    m_side = float(Tp.loc[cr,"Sideways"])
    m_up   = float(Tp.loc[cr,"Bull"])

    # ── 6. CPR + LEVELS ─────────────────────────────────────────────────────
    p = load_params()
    vix_mult = p.get("vix_range_mult", 1.5)
    mw = p.get("markov_weight", 0.60)
    iw = 1.0 - mw

    pivot = (pH+pL+pC)/3
    BC    = (pH+pL)/2
    TC    = 2*pivot - BC
    cpu   = max(TC,BC); cpl = min(TC,BC); cpw = cpu-cpl
    R1=2*pivot-pL; R2=pivot+(pH-pL)
    S1=2*pivot-pH; S2=pivot-(pH-pL)

    vix_half = (vix/100)*pC*(1/math.sqrt(252))*vix_mult/2
    exp_high = round(pC + vix_half*2)
    exp_low  = round(pC - vix_half*2)
    exp_rng  = exp_high - exp_low

    # ── 7. FIVE-FACTOR INTRADAY SCORING ─────────────────────────────────────
    # Each factor scored 0-1 (0=bearish,0.5=neutral,1=bullish)

    # F1: Price Action & Gap Context (35%)
    global_avg = (sp*0.4 + nk*0.3 + hs*0.2 + nq*0.1)
    f1 = max(0.0, min(1.0, 0.5 + global_avg/8))

    # F2: India VIX (20%) — lower VIX = more bullish
    f2 = max(0.0, min(1.0, 0.5 + (18-vix)/20))

    # F3: Derivatives/CPR (20%) — narrow CPR → trending; skew by regime
    regime_skew = {"Bull":0.65,"Sideways":0.50,"Bear":0.35}.get(cr,0.5)
    cpr_skew    = 0.65 if cpw<30 else (0.35 if cpw>60 else 0.5)
    f3 = (regime_skew*0.6 + cpr_skew*0.4)

    # F4: Global & Macro (15%)
    macro_score = 0.5 + global_avg/10
    if brent > 100: macro_score -= 0.10
    if usdinr > 86: macro_score -= 0.05
    f4 = max(0.0, min(1.0, macro_score))

    # F5: FII/DII (10%)
    flow = fii_net + dii_net*0.5
    f5 = max(0.0, min(1.0, 0.5 + flow/5000))

    intraday_bull = f1*0.35 + f2*0.20 + f3*0.20 + f4*0.15 + f5*0.10
    # Decompose into up/side/down
    i_up   = max(0.0, (intraday_bull - 0.5)*1.8)
    i_down = max(0.0, (0.5 - intraday_bull)*1.8)
    i_side = max(0.1, 1.0 - i_up - i_down)
    tot_i  = i_up + i_down + i_side
    i_up/=tot_i; i_down/=tot_i; i_side/=tot_i

    # ── 8. NEWS OVERLAY ──────────────────────────────────────────────────────
    news_adj = {"High":0.10,"Medium":0.05,"Low":0.0}.get(n_impact,0.0)
    if n_sent == "Bearish": news_adj = -news_adj

    # ── 9. FINAL MERGED PROBABILITIES ────────────────────────────────────────
    pu = mw*m_up   + iw*i_up   + (news_adj if news_adj>0 else 0)
    pd_= mw*m_down + iw*i_down + (abs(news_adj) if news_adj<0 else 0)
    ps_= mw*m_side + iw*i_side

    tot = pu+pd_+ps_
    prob_up   = round(pu/tot*100)
    prob_down = round(pd_/tot*100)
    prob_side = 100-prob_up-prob_down

    # ── 10. STRIKES ──────────────────────────────────────────────────────────
    cb = p.get("call_buffer_pts",15); pb2 = p.get("put_buffer_pts",50)
    ca = max(cpu, pC+vix_half)
    cs = math.ceil((ca+cb)/50)*50
    if cs<=ca: cs+=50
    pa = min(S2, pC-vix_half)
    ps = math.floor((pa-pb2)/50)*50
    if ps>=pa: ps-=50
    if cr=="Bear" and pers>0.83: ps-=50
    if cr=="Bull" and pers>0.85: cs+=50
    if esc>0.20: cs+=50; ps-=50
    call_otm = cs-pC; put_otm = pC-ps

    # ── 11. CONFIDENCE SCORES ────────────────────────────────────────────────
    # Range: VIX stability + regime persistence
    conf_range  = round(min(88, 38 + pers*30 + max(0,(20-vix))*1.2))

    # Direction: leading probability
    lead_prob = max(prob_up, prob_down, prob_side)
    conf_dir  = lead_prob

    # Seller: how far strikes sit outside the VIX-implied range
    call_margin_pct = (cs-exp_high)/pC*100
    put_margin_pct  = (exp_low-ps)/pC*100
    conf_seller = round(min(88, 50 + min(call_margin_pct,put_margin_pct)*6))
    conf_seller = max(30, conf_seller)

    # Buyer: only high when strong directional signal exists
    conf_buyer = round(min(82, max(prob_up,prob_down)*1.1)) if max(prob_up,prob_down)>52 else max(prob_up,prob_down)

    # Futures: directional × regime persistence
    conf_futures = round(min(82, max(prob_up,prob_down)*pers*1.1))

    # Overall: weighted composite
    conf_overall = round(conf_range*0.15 + conf_dir*0.25 + conf_seller*0.20 +
                         conf_buyer*0.20 + conf_futures*0.20)

    # ── 12. LABELS & GUIDANCE ────────────────────────────────────────────────
    ri = {"Bear":"🔴","Sideways":"🟡","Bull":"🟢"}.get(cr,"")
    fii_lbl = "Buying" if fii_net>500 else("Selling" if fii_net<-500 else "Neutral")

    if prob_up>=prob_down and prob_up>=prob_side:
        dir_lbl="CLOSE > OPEN"; dir_icon="⬆️"
    elif prob_down>prob_up and prob_down>=prob_side:
        dir_lbl="CLOSE < OPEN"; dir_icon="⬇️"
    else:
        dir_lbl="CLOSE ≈ OPEN"; dir_icon="↔️"

    # Directional option seller skew
    if   cr=="Bear" and pers>0.83:
        dir_skew = f"Lean SHORT: focus {ps} PE\nCall strike is safety hedge only"
    elif cr=="Bull" and pers>0.85:
        dir_skew = f"Lean LONG: focus {cs} CE\nPut strike is safety hedge only"
    elif esc>0.20:
        dir_skew = "Regime unstable — keep both legs, no lean"
    else:
        dir_skew = "Symmetric — no strong directional lean"

    # Option buyer
    if prob_up>55:
        buyer = f"BUY CALL — {cs-100} CE (ATM-1)\nEntry: Confirm open above CPR {cpu:.0f}\nTarget: {round(pC+vix_half*1.5)} | SL: below {round(pC-vix_half*0.5)}"
    elif prob_down>55:
        buyer = f"BUY PUT — {ps+100} PE (ATM-1)\nEntry: Confirm open below CPR {cpl:.0f}\nTarget: {round(pC-vix_half*1.5)} | SL: above {round(pC+vix_half*0.5)}"
    else:
        buyer = f"WAIT — no clear edge\nAct only on confirmed break above {exp_high} (calls)\nor below {exp_low} (puts)"

    # Futures
    if prob_up>55:
        fut = f"LONG | Entry dip: {round(pC-vix_half*0.4)}-{round(pC)}\nStop: {round(pC-vix_half*0.8)} | Tgt: {round(pC+vix_half*1.5)}\nWindow: 9:30-11 AM"
    elif prob_down>55:
        fut = f"SHORT | Entry rise: {round(pC)}-{round(pC+vix_half*0.4)}\nStop: {round(pC+vix_half*0.8)} | Tgt: {round(pC-vix_half*1.5)}\nWindow: 9:30-11 AM"
    else:
        fut = f"FADE EXTREMES | Buy near {exp_low+50} / Sell near {exp_high-50}\nNo breakout trades until range is confirmed\nWindow: 10 AM-1 PM"

    # ── 13. BUILD MESSAGES ───────────────────────────────────────────────────
    msg1 = (
        f"<b>NIFTY PRE-MARKET {today_disp}</b>\n"
        f"Ref Price (Prev Close): <b>{pC:.0f}</b>\n"
        f"Ref Price (Prev Close): <b>{pC:.0f}</b> | Nifty Open: <b>" +
        (str(today_open) + " [LIVE]" if today_open else "awaiting 9:15 AM") +
        "</b>\n"
        f"Regime: <b>{cr}</b> {ri} | Persist: {pers:.0%} | VIX: {vix:.1f} [{vix_lbl}]\n"
        f"Drift: {drift:+.0f} pts/day | Escape prob: {esc:.0%}\n"
        f"Global: S&amp;P {sp:+.1f}% Nsdq {nq:+.1f}% Nikkei {nk:+.1f}% HSI {hs:+.1f}%\n"
        f"Brent: ${brent:.0f} | USD/INR: {usdinr:.1f} | FII: {fii_lbl}\n"
        f"News: {n_sent} ({n_impact} impact, score {n_score:+.2f})\n\n"

        f"<b>1. EXPECTED DAY RANGE</b>\n"
        f"High: <b>{exp_high}</b> | Low: <b>{exp_low}</b> | Width: {exp_rng} pts\n"
        f"CPR: {cpl:.0f}–{cpu:.0f} (W:{cpw:.0f}) | Pivot: {pivot:.0f}\n"
        f"R1/R2: {R1:.0f} / {R2:.0f} | S1/S2: {S1:.0f} / {S2:.0f}\n"
        f"Confidence: <b>{conf_range}%</b>\n\n"

        f"<b>2. CLOSE vs OPEN</b>\n"
        f"{dir_icon} <b>{dir_lbl}</b>\n"
        f"Up: {prob_up}% | Side: {prob_side}% | Down: {prob_down}%\n"
        f"(Markov {mw:.0%} + Intraday {iw:.0%} + News overlay)\n"
        f"Confidence: <b>{conf_dir}%</b>"
    )

    msg2 = (
        f"<b>3. OPTION SELLER</b>\n"
        f"Non-Directional Strangle:\n"
        f"  SELL CALL <b>{cs} CE</b> ({call_otm:.0f} pts OTM)\n"
        f"  SELL PUT  <b>{ps} PE</b>  ({put_otm:.0f} pts OTM)\n"
        f"  Entry: 9:30-10:00 AM\n"
        f"  Stop: 15-min close beyond either strike\n"
        f"Directional lean: {dir_skew}\n"
        f"Confidence: <b>{conf_seller}%</b>\n\n"

        f"<b>4. OPTION BUYER</b>\n"
        f"{buyer}\n"
        f"Confidence: <b>{conf_buyer}%</b>\n\n"

        f"<b>5. FUTURES TRADER</b>\n"
        f"{fut}\n"
        f"Confidence: <b>{conf_futures}%</b>\n\n"

        f"<b>OVERALL CONFIDENCE: {conf_overall}%</b>\n"
        f"Markov+CPR+VIX+News+FII | {today_disp} | GitHub Actions"
    )


    # Save session state so backtest.py uses exact same reference data
    session = {
        "date": today_disp,
        "date_iso": today.strftime("%Y-%m-%d"),
        "prev_high": pH, "prev_low": pL, "prev_close": pC,
        "cpr_upper": round(cpu,1), "cpr_lower": round(cpl,1), "cpw": round(cpw,1),
        "pivot": round(pivot,1),
        "r1": round(R1,1), "r2": round(R2,1),
        "s1": round(S1,1), "s2": round(S2,1),
        "call_strike": cs, "put_strike": ps,
        "regime": cr, "persistence": round(pers,4), "escape": round(esc,4),
        "prob_up": prob_up, "prob_side": prob_side, "prob_down": prob_down,
        "vix": vix, "exp_high": exp_high, "exp_low": exp_low,
        "conf_overall": conf_overall
    }
    with open("session_state.json","w") as f:
        json.dump(session, f, indent=2)
    print("session_state.json saved")
    tele(msg1); tele(msg2)
    print(f"Sent | {cr} | Up:{prob_up}% Side:{prob_side}% Down:{prob_down}% | Overall:{conf_overall}%")

if __name__ == "__main__":
    main()
