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
SESSION_FILE   = "session_state.json"

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
            "https://api.telegram.org/bot" + TELE_TOKEN + "/sendMessage",
            json={"chat_id": TELE_CHAT, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        print("Telegram error: " + str(e))


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

    # 1. Fetch Nifty data (dual-fetch to avoid yfinance staleness)
    df_long  = yf.Ticker("^NSEI").history(period="10y")[["Open","High","Low","Close"]].dropna()
    df_fresh = yf.Ticker("^NSEI").history(period="1d")[["Open","High","Low","Close"]].dropna()
    df = pd.concat([df_long, df_fresh])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    last_date = df.index[-1].date()

    if last_date != today_date:
        tele(
            "<b>BACKTEST SKIPPED - " + today_disp + "</b>\n"
            "yfinance last date: " + str(last_date) + "\n"
            "Expected: " + str(today_date) + "\n"
            "Market may be closed or data not yet updated."
        )
        print("No data for " + str(today_date) + ". Last available: " + str(last_date))
        return

    today_row = df.iloc[-1]
    tO = float(today_row.Open)
    tH = float(today_row.High)
    tL = float(today_row.Low)
    tC = float(today_row.Close)

    # 2. Load session_state.json (written by morning_report.py at 9:10 AM)
    # Gives correct prev-day OHLC/CPR/strikes even when yfinance period=10y is stale
    session_ok = False
    ss = {}
    try:
        with open(SESSION_FILE) as f:
            ss = json.load(f)
        if ss.get("date_iso") == today_str:
            session_ok = True
            print("session_state.json OK: prev_close=" + str(ss["prev_close"]) +
                  ", call=" + str(ss["call_strike"]) + ", put=" + str(ss["put_strike"]))
        else:
            print("session_state.json date mismatch: " + str(ss.get("date_iso")) +
                  " != " + today_str + ", fallback to yfinance")
    except Exception as e:
        print("session_state.json not available (" + str(e) + "), fallback to yfinance")

    # 3. Prev-day OHLC: prefer session_state (accurate) over df.iloc[-2] (may be stale)
    if session_ok:
        pH = float(ss["prev_high"])
        pL = float(ss["prev_low"])
        pC = float(ss["prev_close"])
    else:
        prev_row = df.iloc[-2]
        pH = float(prev_row.High)
        pL = float(prev_row.Low)
        pC = float(prev_row.Close)

    # 4. VIX
    try:
        vdf = yf.Ticker("^INDIAVIX").history(period="10d")["Close"].dropna()
        vix = float(vdf.iloc[-2] if len(vdf) >= 2 else vdf.iloc[-1])
    except Exception:
        vix = float(ss.get("vix", 19.5)) if session_ok else 19.5

    # 5. Load params
    p = load_params()
    vix_mult = p.get("vix_range_mult", 1.5)

    # 6. CPR + pivot levels: prefer session_state over recomputing from stale prev_row
    if session_ok:
        cpu   = float(ss["cpr_upper"])
        cpl   = float(ss["cpr_lower"])
        cpw   = float(ss["cpw"])
        pivot = float(ss["pivot"])
        R1    = float(ss["r1"])
        R2    = float(ss["r2"])
        S1    = float(ss["s1"])
        S2    = float(ss["s2"])
    else:
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
    gap_sign = "+" if gap_pct >= 0 else ""
    gap_str  = gap_sign + "{:.2f}%".format(gap_pct)

    # 7. Markov (always computed fresh - needed for actual_cr)
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

    # Regime: prefer session_state (matched to morning analysis)
    if session_ok:
        cr   = ss["regime"]
        pers = float(ss["persistence"])
        esc  = float(ss["escape"])
    else:
        cr   = lbl_prev.iloc[-1]
        pers = float(Tp.loc[cr, cr])
        esc  = 1.0 - pers

    actual_cr        = lbl.iloc[-1]
    regime_persisted = (actual_cr == cr)
    drift = float(ret.reindex(lbl_prev.index)[lbl_prev == cr].mean() * pC)

    # 8. Strike computation: prefer session_state (same strikes morning used)
    if session_ok:
        cs        = int(ss["call_strike"])
        ps_strike = int(ss["put_strike"])
    else:
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

    # 9. Score
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

    # helpers for string formatting
    def fmt(v): return ("+" if v >= 0 else "") + "{:.1f}".format(v)
    def fmti(v): return ("+" if v >= 0 else "") + str(round(v))

    # 10. Append to results.csv
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
        "call_margin":       fmt(call_margin),
        "put_margin":        fmt(put_margin),
        "predicted_direction": predicted_direction,
        "actual_direction":  actual_direction,
        "direction_correct": str(direction_correct),
        "expected_drift_pts": fmt(drift),
        "actual_drift_pts":  fmt(tC - tO),
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

    # 11. Rolling stats — with CSV header resilience check
    # If existing file has wrong column count, reset it (keeps current row)
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as _f:
            _first = _f.readline().strip()
        if len(_first.split(",")) != len(RESULTS_HEADER):
            print("results.csv header mismatch (" + str(len(_first.split(","))) +
                  " cols vs " + str(len(RESULTS_HEADER)) + " expected) - resetting")
            with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as _f:
                _w = csv.DictWriter(_f, fieldnames=RESULTS_HEADER)
                _w.writeheader()
                _w.writerow(row)
    except Exception as _e:
        print("CSV check error: " + str(_e))

    results_df = pd.read_csv(RESULTS_FILE)
    n          = len(results_df)
    window     = results_df.tail(10)
    nw         = len(window)

    call_rate  = (window["call_survived"]       == "True").sum() / nw
    put_rate   = (window["put_survived"]        == "True").sum() / nw
    win_rate   = (window["strangle_profitable"] == "True").sum() / nw
    regime_acc = (window["direction_correct"]   == "True").sum() / nw

    # 12. Learnings.md
    pnl_label = ("PROFIT"  if strangle_profitable else
                 "PARTIAL" if (call_survived or put_survived) else "LOSS")
    ci = "SAFE"    if call_survived else "BREACHED"
    pi = "SAFE"    if put_survived  else "BREACHED"
    ss_src  = "session_state" if session_ok else "yfinance-fallback"
    dc_word = "CORRECT" if direction_correct else "WRONG"
    rp_word = "PERSISTED" if regime_persisted else "TRANSITIONED"

    entry = (
        "\n## " + today_str + " | " + cr + " -> " + str(actual_cr) +
        " | " + pnl_label + " [" + ss_src + "]\n"
        "| Metric | Value |\n|--------|-------|\n"
        "| OHLC | O:" + str(round(tO)) + " H:" + str(round(tH)) +
        " L:" + str(round(tL)) + " C:" + str(round(tC)) + " |\n"
        "| Gap | " + gap_str + " (" + open_vs_cpr + ") |\n"
        "| CPR | " + str(round(cpl)) + " - " + str(round(cpu)) +
        " (W=" + str(round(cpw)) + " pts) |\n"
        "| Call " + str(cs) + " CE | " + ci + " (" + fmti(call_margin) + " pts) |\n"
        "| Put " + str(ps_strike) + " PE | " + pi + " (" + fmti(put_margin) + " pts) |\n"
        "| Range | actual " + str(round(actual_range)) + " vs VIX-implied " +
        str(round(vix_implied_range)) + " pts (" + str(range_ratio) + "x) |\n"
        "| Direction | " + predicted_direction + " -> " + actual_direction +
        " (" + dc_word + ") |\n"
        "| Regime | " + cr + " -> " + str(actual_cr) + " (" + rp_word + ") |\n"
        "| 10d Win Rate | " + str(round(win_rate * 100)) + "% (Call: " +
        str(round(call_rate * 100)) + "% | Put: " + str(round(put_rate * 100)) + "%) |\n\n"
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

    # 13. AUTO-HEALING
    params_changed = False
    change_notes   = []

    if strangle_profitable:
        p["consecutive_losses"] = 0
        if p.get("emergency_widen", False):
            p["emergency_widen"] = False
            change_notes.append("Emergency widen cleared - profit restored")
            params_changed = True
    else:
        p["consecutive_losses"] = p.get("consecutive_losses", 0) + 1
        if p["consecutive_losses"] >= 3 and not p.get("emergency_widen", False):
            p["call_buffer_pts"] = min(p["call_buffer_pts"] + 50, 300)
            p["put_buffer_pts"]  = min(p["put_buffer_pts"]  + 50, 300)
            p["emergency_widen"] = True
            change_notes.append(
                "EMERGENCY WIDEN (" + str(p["consecutive_losses"]) + " consecutive losses): "
                "call_buffer->" + str(p["call_buffer_pts"]) + ", put_buffer->" + str(p["put_buffer_pts"]))
            params_changed = True

    if n >= 5:
        p["call_survival_rate_10d"] = round(call_rate,  3)
        p["put_survival_rate_10d"]  = round(put_rate,   3)
        p["strangle_win_rate_10d"]  = round(win_rate,   3)
        p["regime_accuracy_10d"]    = round(regime_acc, 3)
        p["last_updated"]           = today_str

        if call_rate < 0.70 and not p.get("emergency_widen", False):
            old = p["call_buffer_pts"]
            p["call_buffer_pts"] = min(old + 25, 200)
            if p["call_buffer_pts"] != old:
                change_notes.append("call_buffer " + str(old) + "->" + str(p["call_buffer_pts"]) +
                    " (call " + str(round(call_rate*100)) + "%<70%)")
                params_changed = True
        elif call_rate > 0.90 and p["call_buffer_pts"] > 15:
            old = p["call_buffer_pts"]
            p["call_buffer_pts"] = max(old - 10, 15)
            if p["call_buffer_pts"] != old:
                change_notes.append("call_buffer " + str(old) + "->" + str(p["call_buffer_pts"]) +
                    " (call " + str(round(call_rate*100)) + "%>90% tighten)")
                params_changed = True

        if put_rate < 0.70 and not p.get("emergency_widen", False):
            old = p["put_buffer_pts"]
            p["put_buffer_pts"] = min(old + 25, 200)
            if p["put_buffer_pts"] != old:
                change_notes.append("put_buffer " + str(old) + "->" + str(p["put_buffer_pts"]) +
                    " (put " + str(round(put_rate*100)) + "%<70%)")
                params_changed = True
        elif put_rate > 0.90 and p["put_buffer_pts"] > 25:
            old = p["put_buffer_pts"]
            p["put_buffer_pts"] = max(old - 10, 25)
            if p["put_buffer_pts"] != old:
                change_notes.append("put_buffer " + str(old) + "->" + str(p["put_buffer_pts"]) +
                    " (put " + str(round(put_rate*100)) + "%>90% tighten)")
                params_changed = True

        if regime_acc < 0.45:
            old_mw = p["markov_weight"]
            p["markov_weight"]   = round(max(0.35, p["markov_weight"] - 0.05), 2)
            p["intraday_weight"] = round(1.0 - p["markov_weight"], 2)
            if p["markov_weight"] != old_mw:
                change_notes.append("markov_weight " + str(old_mw) + "->" + str(p["markov_weight"]) +
                    " (direction " + str(round(regime_acc*100)) + "%<45%, trust intraday more)")
                params_changed = True
        elif regime_acc > 0.65:
            old_mw = p["markov_weight"]
            p["markov_weight"]   = round(min(0.75, p["markov_weight"] + 0.05), 2)
            p["intraday_weight"] = round(1.0 - p["markov_weight"], 2)
            if p["markov_weight"] != old_mw:
                change_notes.append("markov_weight " + str(old_mw) + "->" + str(p["markov_weight"]) +
                    " (direction " + str(round(regime_acc*100)) + "%>65%, trust Markov more)")
                params_changed = True

        if "actual_range_ratio" in results_df.columns:
            recent_ratios = pd.to_numeric(window["actual_range_ratio"], errors="coerce").dropna()
            if len(recent_ratios) >= 5:
                med = float(recent_ratios.median())
                old_mult = p.get("vix_range_mult", 1.5)
                if med > 1.3:
                    p["vix_range_mult"] = round(min(2.5, old_mult + 0.1), 2)
                    if p["vix_range_mult"] != old_mult:
                        change_notes.append("vix_range_mult " + str(old_mult) + "->" +
                            str(p["vix_range_mult"]) + " (actual " + str(round(med,2)) + "x VIX-implied, widen)")
                        params_changed = True
                elif med < 0.7:
                    p["vix_range_mult"] = round(max(1.0, old_mult - 0.1), 2)
                    if p["vix_range_mult"] != old_mult:
                        change_notes.append("vix_range_mult " + str(old_mult) + "->" +
                            str(p["vix_range_mult"]) + " (actual only " + str(round(med,2)) + "x VIX-implied, tighten)")
                        params_changed = True

    if change_notes:
        p["notes"] = " | ".join(change_notes)
    save_params(p)

    # 14. Telegram
    ri_map = {"Bear": chr(0x1F534), "Sideways": chr(0x1F7E1), "Bull": chr(0x1F7E2)}
    ri  = ri_map.get(cr, "")
    ck  = chr(0x2705)
    cx  = chr(0x274C)
    warn = chr(0x26A0) + chr(0xFE0F)
    ok  = (ck + ck) if strangle_profitable else (warn if (call_survived or put_survived) else (cx + cx))
    ci2 = ck if call_survived else cx
    pi2 = ck if put_survived  else cx
    label = ("BOTH LEGS SAFE" if strangle_profitable
             else "ONE LEG BREACHED" if (call_survived or put_survived)
             else "BOTH LEGS BREACHED")
    src_note  = " [session_state]" if session_ok else " [yfinance-fallback]"
    dc_word2  = "CORRECT" if direction_correct else "WRONG"
    rp_word2  = "Persisted" if regime_persisted else "Transitioned"

    msg = (
        "<b>NIFTY BACKTEST - " + today_disp + "</b>\n"
        + ok + " <b>" + label + "</b>" + src_note + "\n\n"
        + "Regime: <b>" + cr + "</b> " + ri + " to " + str(actual_cr) + " (" + rp_word2 + ")\n"
        + "OHLC: O:" + str(round(tO)) + " H:" + str(round(tH)) +
        " L:" + str(round(tL)) + " C:" + str(round(tC)) + "\n\n"
        + "<b>STRIKES</b>\n"
        + ci2 + " CALL " + str(cs) + " CE  margin: " + fmti(call_margin) + " pts\n"
        + pi2 + " PUT  " + str(ps_strike) + " PE  margin: " + fmti(put_margin) + " pts\n"
        + "Range: actual " + str(round(actual_range)) + " pts vs VIX-implied " +
        str(round(vix_implied_range)) + " pts (" + str(range_ratio) + "x)\n"
        + "Direction: " + predicted_direction + " to " + actual_direction + " (" + dc_word2 + ")\n\n"
        + "<b>10-DAY STATS</b> (n=" + str(nw) + ")\n"
        + "Win: <b>" + str(round(win_rate*100)) + "%</b>  Call: " +
        str(round(call_rate*100)) + "%  Put: " + str(round(put_rate*100)) + "%  Dir: " +
        str(round(regime_acc*100)) + "%"
    )

    if params_changed:
        msg += "\n\n<b>AUTO-HEAL TRIGGERED</b>\n" + "\n".join("  " + c for c in change_notes)
    else:
        msg += "\n\nParams stable - no adjustment"

    msg += "\n\n<i>GitHub Actions | " + today_disp + "</i>"
    tele(msg)

    ok_c = "OK" if call_survived else "HIT"
    ok_p = "OK" if put_survived  else "HIT"
    print("Done: " + today_str + " | " + pnl_label +
          " | Call:" + str(cs) + " " + ok_c +
          " | Put:" + str(ps_strike) + " " + ok_p)
    if params_changed:
        print("Auto-heal:", change_notes)


if __name__ == "__main__":
    main()
