import yfinance as yf
import numpy as np
import pandas as pd
import requests
import math
import json
import csv
import os
from datetime import datetime
import pytz

TELE_TOKEN = "8623520423:AAH4mSilxgtMHXXyRwlwlFfB-FjB42sDrwQ"
TELE_CHAT  = 1083142887

RESULTS_FILE   = "results.csv"
LEARNINGS_FILE = "learnings.md"
PARAMS_FILE    = "params.json"

RESULTS_HEADER = [
    "date","regime","actual_next_regime","regime_persisted",
    "prev_close","open","high","low","close","gap_pct","open_vs_cpr",
    "cpr_lower","cpr_upper","pivot","r1","r2","s1","s2",
    "call_strike","put_strike",
    "call_survived","put_survived","strangle_profitable",
    "call_margin","put_margin",
    "predicted_direction","actual_direction","direction_correct",
    "expected_drift_pts","actual_drift_pts",
    "actual_range","vix_implied_range","actual_range_ratio",
    "call_otm_mult_used","put_otm_mult_used","markov_weight_used","vix_range_mult_used"
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


def load_params():
    defaults = {
        "version": 2,
        "call_otm_multiplier": 1.12,
        "put_otm_multiplier":  1.18,
        "call_buffer_pts":     15,
        "put_buffer_pts":      50,
        "vix_range_mult":      1.5,
        "markov_weight":       0.60,
        "intraday_weight":     0.40,
        "vix_assumed":         19.5,
        "consecutive_losses":  0,
        "emergency_widen":     False,
        "last_updated":        None,
        "call_survival_rate_10d":  None,
        "put_survival_rate_10d":   None,
        "strangle_win_rate_10d":   None,
        "regime_accuracy_10d":     None,
        "notes":               "Initial defaults"
    }
    try:
        with open(PARAMS_FILE) as f:
            loaded = json.load(f)
        for k, v in defaults.items():
            loaded.setdefault(k, v)
        return loaded
    except Exception:
        return defaults


def save_params(p):
    with open(PARAMS_FILE, "w") as f:
        json.dump(p, f, indent=2)


def classify_direction(close, open_price, threshold_pct=0.4):
    chg = (close - open_price) / open_price * 100
    if   chg >  threshold_pct: return "Up"
    elif chg < -threshold_pct: return "Down"
    else:                      return "Sideways"


def main():
    ist        = pytz.timezone("Asia/Calcutta")
    today      = datetime.now(ist)
    today_date = today.date()
    today_str  = today.strftime("%Y-%m-%d")
    today_disp = today.strftime("%d %b %Y")

    # 1. Fetch Nifty data
    df_long  = yf.Ticker("^NSEI").history(period="10y")[["Open","High","Low","Close"]].dropna()
    df_fresh = yf.Ticker("^NSEI").history(period="1d")[["Open","High","Low","Close"]].dropna()
    df = pd.concat([df_long, df_fresh])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    last_date = df.index[-1].date()

    if last_date != today_date:
        tele(
            f"<b>BACKTEST SKIPPED — {today_disp}</b>\n"
            f"yfinance last date: {last_date}\n"
            f"Expected: {today_date}\n"
            "Market may be closed or data not yet updated."
        )
        print(f"No data for {today_date}. Last available: {last_date}")
        return

    today_row = df.iloc[-1]
    prev_row  = df.iloc[-2]

    pH, pL, pC = float(prev_row.High),  float(prev_row.Low),  float(prev_row.Close)
    tO, tH, tL, tC = (float(today_row.Open), float(today_row.High),
                      float(today_row.Low),  float(today_row.Close))

    # 2. VIX
    try:
        vdf = yf.Ticker("^INDIAVIX").history(period="10d")["Close"].dropna()
        vix = float(vdf.iloc[-2] if len(vdf) >= 2 else vdf.iloc[-1])
    except Exception:
        vix = 19.5

    # 3. Load params
    p = load_params()
    vix_mult = p.get("vix_range_mult", 1.5)

    # 4. CPR + pivot levels
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

    if   tO > cpu + 50:  open_vs_cpr = "strong_above_cpr"
    elif tO > cpu + 10:  open_vs_cpr = "above_cpr"
    elif tO >= cpl - 10: open_vs_cpr = "inside_cpr"
    elif tO < cpl - 50:  open_vs_cpr = "strong_below_cpr"
    else:                open_vs_cpr = "below_cpr"

    gap_pct = (tO - pC) / pC * 100
    gap_str = f"{gap_pct:+.2f}%"

    # 5. Markov
    ret  = df["Close"].pct_change().dropna()
    rm   = ret.rolling(20).mean().dropna()
    p33, p67 = float(np.percentile(rm, 33)), float(np.percentile(rm, 67))
    lbl  = rm.apply(lambda x: "Bear" if x <= p33 else ("Bull" if x >= p67 else "Sideways"))

    lbl_prev = lbl.iloc[:-1]
    states   = ["Bear", "Sideways", "Bull"]
    T        = pd.DataFrame(0, index=states, columns=states)
    for i in range(len(lbl_prev) - 1):
        T.loc[lbl_prev.iloc[i], lbl_prev.iloc[i+1]] += 1
    Tp = T.div(T.sum(axis=1), axis=0)

    cr   = lbl_prev.iloc[-1]
    pers = float(Tp.loc[cr, cr])
    esc  = 1.0 - pers
    drift = float(ret.reindex(lbl_prev.index)[lbl_prev == cr].mean() * pC)

    actual_cr        = lbl.iloc[-1]
    regime_persisted = (actual_cr == cr)

    # 6. Strike computation
    vix_half = (vix / 100) * pC * (1 / math.sqrt(252)) * vix_mult / 2

    ca = max(cpu, pC + vix_half)
    cs = math.ceil((ca + p["call_buffer_pts"]) / 50) * 50
    if cs <= ca: cs += 50

    pa = min(S2, pC - vix_half)
    ps_strike = math.floor((pa - p["put_buffer_pts"]) / 50) * 50
    if ps_strike >= pa: ps_strike -= 50

    if cr == "Bear" and pers > 0.83: ps_strike -= 50
    if cr == "Bull" and pers > 0.85: cs        += 50
    if esc > 0.20:                   cs += 50; ps_strike -= 50

    # 7. Score
    call_survived       = bool(tH < cs)
    put_survived        = bool(tL > ps_strike)
    strangle_profitable = call_survived and put_survived
    call_margin         = cs - tH
    put_margin          = tL - ps_strike

    predicted_direction = ("Down" if cr == "Bear" else
                           "Up"   if cr == "Bull"  else "Sideways")
    actual_direction    = classify_direction(tC, tO)
    direction_correct   = (predicted_direction == actual_direction)

    actual_range      = round(tH - tL, 1)
    vix_implied_range = round((vix / 100) * pC * (1 / math.sqrt(252)) * vix_mult, 1)
    range_ratio       = round(actual_range / vix_implied_range, 3) if vix_implied_range > 0 else 1.0

    # 8. Append to results.csv
    row = {
        "date":              today_str,
        "regime":            cr,
        "actual_next_regime":actual_cr,
        "regime_persisted":  str(regime_persisted),
        "prev_close":        round(pC,  2),
        "open":              round(tO,  2),
        "high":              round(tH,  2),
        "low":               round(tL,  2),
        "close":             round(tC,  2),
        "gap_pct":           gap_str,
        "open_vs_cpr":       open_vs_cpr,
        "cpr_lower":         round(cpl,   1),
        "cpr_upper":         round(cpu,   1),
        "pivot":             round(pivot, 1),
        "r1":                round(R1,    1),
        "r2":                round(R2,    1),
        "s1":                round(S1,    1),
        "s2":                round(S2,    1),
        "call_strike":       int(cs),
        "put_strike":        int(ps_strike),
        "call_survived":     str(call_survived),
        "put_survived":      str(put_survived),
        "strangle_profitable": str(strangle_profitable),
        "call_margin":       f"{call_margin:+.1f}",
        "put_margin":        f"{put_margin:+.1f}",
        "predicted_direction": predicted_direction,
        "actual_direction":  actual_direction,
        "direction_correct": str(direction_correct),
        "expected_drift_pts": f"{drift:+.1f}",
        "actual_drift_pts":  f"{(tC - tO):+.1f}",
        "actual_range":      actual_range,
        "vix_implied_range": vix_implied_range,
        "actual_range_ratio":range_ratio,
        "call_otm_mult_used": p["call_otm_multiplier"],
        "put_otm_mult_used":  p["put_otm_multiplier"],
        "markov_weight_used": p["markov_weight"],
        "vix_range_mult_used": vix_mult
    }

    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # 9. Rolling stats
    results_df = pd.read_csv(RESULTS_FILE)
    n          = len(results_df)
    window     = results_df.tail(10)
    nw         = len(window)

    call_rate  = (window["call_survived"]       == "True").sum() / nw
    put_rate   = (window["put_survived"]        == "True").sum() / nw
    win_rate   = (window["strangle_profitable"] == "True").sum() / nw
    regime_acc = (window["direction_correct"]   == "True").sum() / nw

    # 10. Learnings.md
    pnl_label = ("PROFIT"  if strangle_profitable else
                 "PARTIAL" if (call_survived or put_survived) else "LOSS")
    ci = "SAFE"     if call_survived else "BREACHED"
    pi = "SAFE"     if put_survived  else "BREACHED"

    entry = (
        f"\n## {today_str} | {cr} -> {actual_cr} | {pnl_label}\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| OHLC | O:{tO:.0f} H:{tH:.0f} L:{tL:.0f} C:{tC:.0f} |\n"
        f"| Gap | {gap_str} ({open_vs_cpr}) |\n"
        f"| CPR | {cpl:.0f} - {cpu:.0f} (W={cpw:.0f} pts) |\n"
        f"| Call {cs} CE | {ci} ({call_margin:+.0f} pts) |\n"
        f"| Put {ps_strike} PE | {pi} ({put_margin:+.0f} pts) |\n"
        f"| Range | actual {actual_range:.0f} vs VIX-implied {vix_implied_range:.0f} pts ({range_ratio:.2f}x) |\n"
        f"| Direction | {predicted_direction} -> {actual_direction} "
            f"({'CORRECT' if direction_correct else 'WRONG'}) |\n"
        f"| Regime | {cr} -> {actual_cr} "
            f"({'PERSISTED' if regime_persisted else 'TRANSITIONED'}) |\n"
        f"| 10d Win Rate | {win_rate:.0%} "
            f"(Call: {call_rate:.0%} | Put: {put_rate:.0%}) |\n\n"
    )

    if os.path.isfile(LEARNINGS_FILE):
        with open(LEARNINGS_FILE, "r", encoding="utf-8") as f:
            existing = f.read()
        lines     = existing.split("\n")
        insert_at = next(
            (i for i, ln in enumerate(lines) if ln.startswith("## ")),
            len(lines)
        )
        new_content = "\n".join(lines[:insert_at]) + entry + "\n".join(lines[insert_at:])
        with open(LEARNINGS_FILE, "w", encoding="utf-8") as f:
            f.write(new_content)
    else:
        with open(LEARNINGS_FILE, "w", encoding="utf-8") as f:
            f.write("# Nifty 50 Strategy Learnings\n\nAuto-generated.\n" + entry)

    # 11. AUTO-HEALING
    params_changed = False
    change_notes   = []

    # Consecutive loss tracker (no session minimum)
    if strangle_profitable:
        p["consecutive_losses"] = 0
        if p.get("emergency_widen", False):
            p["emergency_widen"] = False
            change_notes.append("Emergency widen cleared — profit restored")
            params_changed = True
    else:
        p["consecutive_losses"] = p.get("consecutive_losses", 0) + 1
        if p["consecutive_losses"] >= 3 and not p.get("emergency_widen", False):
            p["call_buffer_pts"] = min(p["call_buffer_pts"] + 50, 300)
            p["put_buffer_pts"]  = min(p["put_buffer_pts"]  + 50, 300)
            p["emergency_widen"] = True
            change_notes.append(
                f"EMERGENCY WIDEN ({p['consecutive_losses']} consecutive losses): "
                f"call_buffer->{p['call_buffer_pts']}, put_buffer->{p['put_buffer_pts']}")
            params_changed = True

    if n >= 5:
        p["call_survival_rate_10d"] = round(call_rate,  3)
        p["put_survival_rate_10d"]  = round(put_rate,   3)
        p["strangle_win_rate_10d"]  = round(win_rate,   3)
        p["regime_accuracy_10d"]    = round(regime_acc, 3)
        p["last_updated"]           = today_str

        # Rule 1: Call buffer
        if call_rate < 0.70 and not p.get("emergency_widen", False):
            old = p["call_buffer_pts"]
            p["call_buffer_pts"] = min(old + 25, 200)
            if p["call_buffer_pts"] != old:
                change_notes.append(f"call_buffer {old}->{p['call_buffer_pts']} (call {call_rate:.0%}<70%)")
                params_changed = True
        elif call_rate > 0.90 and p["call_buffer_pts"] > 15:
            old = p["call_buffer_pts"]
            p["call_buffer_pts"] = max(old - 10, 15)
            if p["call_buffer_pts"] != old:
                change_notes.append(f"call_buffer {old}->{p['call_buffer_pts']} (call {call_rate:.0%}>90% tighten)")
                params_changed = True

        # Rule 2: Put buffer
        if put_rate < 0.70 and not p.get("emergency_widen", False):
            old = p["put_buffer_pts"]
            p["put_buffer_pts"] = min(old + 25, 200)
            if p["put_buffer_pts"] != old:
                change_notes.append(f"put_buffer {old}->{p['put_buffer_pts']} (put {put_rate:.0%}<70%)")
                params_changed = True
        elif put_rate > 0.90 and p["put_buffer_pts"] > 25:
            old = p["put_buffer_pts"]
            p["put_buffer_pts"] = max(old - 10, 25)
            if p["put_buffer_pts"] != old:
                change_notes.append(f"put_buffer {old}->{p['put_buffer_pts']} (put {put_rate:.0%}>90% tighten)")
                params_changed = True

        # Rule 3: Markov weight self-correction
        if regime_acc < 0.45:
            old_mw = p["markov_weight"]
            p["markov_weight"]   = round(max(0.35, p["markov_weight"] - 0.05), 2)
            p["intraday_weight"] = round(1.0 - p["markov_weight"], 2)
            if p["markov_weight"] != old_mw:
                change_notes.append(
                    f"markov_weight {old_mw}->{p['markov_weight']} (direction {regime_acc:.0%}<45%, trust intraday more)")
                params_changed = True
        elif regime_acc > 0.65:
            old_mw = p["markov_weight"]
            p["markov_weight"]   = round(min(0.75, p["markov_weight"] + 0.05), 2)
            p["intraday_weight"] = round(1.0 - p["markov_weight"], 2)
            if p["markov_weight"] != old_mw:
                change_notes.append(
                    f"markov_weight {old_mw}->{p['markov_weight']} (direction {regime_acc:.0%}>65%, trust Markov more)")
                params_changed = True

        # Rule 4: VIX range multiplier
        if "actual_range_ratio" in results_df.columns:
            recent_ratios = pd.to_numeric(window["actual_range_ratio"], errors="coerce").dropna()
            if len(recent_ratios) >= 5:
                med = float(recent_ratios.median())
                old_mult = p.get("vix_range_mult", 1.5)
                if med > 1.3:
                    p["vix_range_mult"] = round(min(2.5, old_mult + 0.1), 2)
                    if p["vix_range_mult"] != old_mult:
                        change_notes.append(
                            f"vix_range_mult {old_mult}->{p['vix_range_mult']} (actual {med:.2f}x VIX-implied, widen strikes)")
                        params_changed = True
                elif med < 0.7:
                    p["vix_range_mult"] = round(max(1.0, old_mult - 0.1), 2)
                    if p["vix_range_mult"] != old_mult:
                        change_notes.append(
                            f"vix_range_mult {old_mult}->{p['vix_range_mult']} (actual only {med:.2f}x VIX-implied, tighten)")
                        params_changed = True

    if change_notes:
        p["notes"] = " | ".join(change_notes)
    save_params(p)

    # 12. Telegram
    ri  = {"Bear": "🔴", "Sideways": "🟡", "Bull": "🟢"}.get(cr, "")
    ok  = ("✅✅" if strangle_profitable else
           "⚠️"  if (call_survived or put_survived) else "❌❌")
    ci2 = "✅" if call_survived else "❌"
    pi2 = "✅" if put_survived  else "❌"
    label = ("BOTH LEGS SAFE" if strangle_profitable
             else "ONE LEG BREACHED" if (call_survived or put_survived)
             else "BOTH LEGS BREACHED")

    msg = (
        f"<b>NIFTY BACKTEST — {today_disp}</b>\n"
        f"{ok} <b>{label}</b>\n\n"
        f"Regime: <b>{cr}</b> {ri} to {actual_cr} "
            f"({'Persisted' if regime_persisted else 'Transitioned'})\n"
        f"OHLC: O:{tO:.0f} H:{tH:.0f} L:{tL:.0f} C:{tC:.0f}\n\n"
        f"<b>STRIKES</b>\n"
        f"{ci2} CALL {cs} CE  margin: {call_margin:+.0f} pts\n"
        f"{pi2} PUT  {ps_strike} PE  margin: {put_margin:+.0f} pts\n"
        f"Range: actual {actual_range:.0f} pts vs VIX-implied {vix_implied_range:.0f} pts ({range_ratio:.2f}x)\n"
        f"Direction: {predicted_direction} to {actual_direction} "
            f"({'CORRECT' if direction_correct else 'WRONG'})\n\n"
        f"<b>10-DAY STATS</b> (n={nw})\n"
        f"Win: <b>{win_rate:.0%}</b>  Call: {call_rate:.0%}  Put: {put_rate:.0%}  Dir: {regime_acc:.0%}"
    )

    if params_changed:
        msg += "\n\n<b>AUTO-HEAL TRIGGERED</b>\n" + "\n".join(f"  {c}" for c in change_notes)
    else:
        msg += "\n\nParams stable — no adjustment"

    msg += f"\n\n<i>GitHub Actions | {today_disp}</i>"
    tele(msg)

    print(f"Done: {today_str} | {pnl_label} | Call:{cs} {'OK' if call_survived else 'HIT'} | Put:{ps_strike} {'OK' if put_survived else 'HIT'}")
    if params_changed:
        print("Auto-heal:", change_notes)


if __name__ == "__main__":
    main()