import yfinance as yf, numpy as np, pandas as pd
import requests, math, json
from datetime import datetime
import pytz

TELE_TOKEN = "8623520423:AAH4mSilxgtMHXXyRwlwlFfB-FjB42sDrwQ"
TELE_CHAT  = 1083142887

def tele(text):
    requests.post(f"https://api.telegram.org/bot{TELE_TOKEN}/sendMessage",
                  json={"chat_id":TELE_CHAT,"text":text,"parse_mode":"HTML"})

def main():
    ist = pytz.timezone("Asia/Calcutta")
    today = datetime.now(ist)
    today_str = today.strftime("%d %b %Y")

    df = yf.Ticker("^NSEI").history(period="10y")[["Open","High","Low","Close"]].dropna()
    last_date = df.index[-1].date()
    bdays = np.busday_count(last_date, today.date())
    if bdays > 1:
        tele(f"NSE closed today ({today_str}) - no analysis"); return

    prev = df.iloc[-1]
    pH, pL, pC = prev.High, prev.Low, prev.Close

    try: vix = yf.Ticker("^INDIAVIX").history(period="5d")["Close"].iloc[-1]
    except: vix = 19.5

    def chg(t):
        try:
            d = yf.Ticker(t).history(period="5d")["Close"]
            return (d.iloc[-1]-d.iloc[-2])/d.iloc[-2]*100
        except: return 0.0

    sp=chg("^GSPC"); nq=chg("^IXIC"); nk=chg("^N225"); hs=chg("^HSI")
    try: brent = yf.Ticker("BZ=F").history(period="2d")["Close"].iloc[-1]
    except: brent = 0
    try: usdinr = yf.Ticker("USDINR=X").history(period="2d")["Close"].iloc[-1]
    except: usdinr = 0

    # Markov
    ret = df["Close"].pct_change().dropna()
    rm  = ret.rolling(20).mean().dropna()
    p33,p67 = np.percentile(rm,33), np.percentile(rm,67)
    lbl = rm.apply(lambda x: "Bear" if x<=p33 else ("Bull" if x>=p67 else "Sideways"))
    states = ["Bear","Sideways","Bull"]
    T = pd.DataFrame(0,index=states,columns=states)
    for i in range(len(lbl)-1): T.loc[lbl.iloc[i],lbl.iloc[i+1]] += 1
    Tp = T.div(T.sum(axis=1),axis=0)
    cr = lbl.iloc[-1]
    pers = Tp.loc[cr,cr]; esc = 1-pers
    days_left = round(1/(1-pers),1) if pers<1 else 999
    drift = ret[lbl==cr].mean() * pC

    try:
        with open("params.json") as f: p = json.load(f)
    except:
        p = {"call_buffer_pts":15,"put_buffer_pts":50,"markov_weight":0.60,"intraday_weight":0.40,"vix_assumed":19.5}

    pivot=(pH+pL+pC)/3; BC=(pH+pL)/2; TC=2*pivot-BC
    cpu=max(TC,BC); cpl=min(TC,BC); cpw=cpu-cpl
    R1=2*pivot-pL; R2=pivot+(pH-pL); S1=2*pivot-pH; S2=pivot-(pH-pL)
    vh=(vix/100)*pC*(1/math.sqrt(252))*1.5/2
    ub=pC+vh; lb=pC-vh

    ca=max(cpu,pC+vh); cs=math.ceil((ca+p["call_buffer_pts"])/50)*50
    if cs<=ca: cs+=50
    pa=min(S2,pC-vh); ps=math.floor((pa-p["put_buffer_pts"])/50)*50
    if ps>=pa: ps-=50
    if cr=="Bear" and pers>0.83: ps-=50
    if cr=="Bull" and pers>0.85: cs+=50
    if esc>0.20: cs+=50; ps-=50

    mw=p["markov_weight"]; iw=p["intraday_weight"]
    md=Tp.loc[cr,"Bear"]; ms=Tp.loc[cr,"Sideways"]; mu=Tp.loc[cr,"Bull"]
    pd_=mw*md+iw*0.35; ps_=mw*ms+iw*0.40; pu=mw*mu+iw*0.25
    tot=pd_+ps_+pu; pd_/=tot; ps_/=tot; pu/=tot

    re={"Bear":"🔴","Sideways":"🟡","Bull":"🟢"}.get(cr,"")
    dt={"Bear":"Sell-on-rise","Sideways":"Range-selling","Bull":"Buy-on-dip"}.get(cr,"")
    sg=f"+" if sp>=0 else ""; ng=f"+" if nk>=0 else ""

    m1=f"""<b>📈 NIFTY PRE-MARKET — {today_str}</b>

🧠 <b>REGIME: {cr}</b> {re}
Persistence: <b>{pers:.1%}</b> | Escape: {esc:.1%}
Drift: <b>{drift:+.0f} pts/day</b> | ~{days_left} days left

📊 <b>KEY LEVELS</b>
<code>Prev C : {pC:.0f}
CPR    : {cpl:.0f} - {cpu:.0f}  W:{cpw:.0f}
Pivot  : {pivot:.0f}
R1/R2  : {R1:.0f} / {R2:.0f}
S1/S2  : {S1:.0f} / {S2:.0f}
VIX    : {vix:.1f}
Range  : {lb:.0f} - {ub:.0f}</code>

🎯 <b>OPTION SELLER STRIKES</b>
⬆️ SELL CALL: <b>{cs} CE</b>
⬇️ SELL PUT:  <b>{ps} PE</b>
⏰ Entry: 9:30-10:00 AM
⛔ Stop call: 15min close above {cs}
⛔ Stop put:  15min close below {ps}"""

    m2=f"""🎲 <b>PROBABILITY</b>
⬆️ Upside:   <b>{pu:.0%}</b>
↔️ Sideways: <b>{ps_:.0%}</b>
⬇️ Downside: <b>{pd_:.0%}</b>

🌍 <b>GLOBAL CUES (prev close)</b>
US: S&amp;P <b>{sg}{sp:.1f}%</b> | Nsdq {nq:+.1f}%
Asia: Nikkei {ng}{nk:.1f}% | HSI {hs:+.1f}%
Brent: ${brent:.0f} | USD/INR: {usdinr:.1f}

🧑‍💼 <b>TRADE GUIDANCE</b>
<b>Day type: {dt}</b>
Sellers: {cs} CE / {ps} PE strangle. Entry 9:30-10 AM.
Buyers: Puts below {cpl:.0f}; Calls above {R1:.0f}.
Futures: {"Short near "+str(int(R1))+" stop "+str(int(R2)) if cr=="Bear" else "Long near "+str(int(S1))+" stop "+str(int(S2))}

<i>🤖 Markov+CPR | {today_str} | GitHub Actions</i>"""

    tele(m1); tele(m2)
    print(f"Morning report sent: {today_str}")

if __name__=="__main__": main()