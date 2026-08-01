# -*- coding: utf-8 -*-
"""
PHASE + PERIODICITY Transparency Script (NO prediction model)

It produces:
1) periodicity_report.csv    -> dominant cycle candidates (returns + volatility proxy) per symbol
2) regimes_report.csv        -> Markov-switching (2 regimes) summary per symbol (vol levels, transition probs, current regime)
3) changepoints_report.csv   -> simple abrupt-change flags based on rolling volatility shifts

Requirements:
pip install numpy pandas scipy statsmodels

Run:
python phase_periodicity.py
"""

import json
import numpy as np
import pandas as pd

from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


# -------------------------
# USER INPUT
# -------------------------
JSONL_PATH = "test.jsonl"
symbols = [
    'COMP_0','COMP_1','COMP_2','COMP_3','COMP_4','COMP_5','COMP_6','COMP_7','COMP_8','COMP_9',
    'EXT_0','EXT_1','EXT_2'
]

# Periodicity settings
MAX_ACF_LAG = 2000          # how far to look for repeated patterns (ticks)
TOP_N_FREQ = 5              # how many top frequencies to report per series
MIN_PERIOD_TICKS = 5        # ignore too-short cycles
MAX_PERIOD_TICKS = 5000     # ignore too-long cycles

# Phase detection settings (Markov switching)
K_REGIMES = 2               # stable vs volatile
MIN_N_RET = 1500            # minimum returns to run MS model

# Change-point proxy settings (rolling volatility)
ROLL_WIN = 200              # rolling window for volatility proxy
Z_THRESH = 4.0              # how strong a jump in rolling-vol zscore counts as abrupt
MIN_CP_GAP = 50             # minimum spacing between detected points

# outputs
OUT_PERIOD = "periodicity_report.csv"
OUT_REGIMES = "regimes_report.csv"
OUT_CPS = "changepoints_report.csv"


# -------------------------
# Data extraction (ref_price only; EXT has no book but is fine)
# -------------------------
def extract_ref_price_series(jsonl_path: str, symbol: str) -> pd.Series:
    vals = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for t, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if symbol in obj:
                price, buyQ, sellQ = obj[symbol]
                vals.append((t, float(price)))

    if not vals:
        return pd.Series(dtype=float)

    s = pd.Series({t: p for t, p in vals}).sort_index()
    return s


def log_returns(price: pd.Series) -> pd.Series:
    if price.empty:
        return pd.Series(dtype=float)
    return np.log(price).diff().dropna()


# -------------------------
# 1) Periodicity tests
#   - periodogram peaks on returns
#   - periodogram peaks on volatility proxy |r|
#   - ACF peak lags on returns and |r|
# -------------------------
def top_periodogram_periods(x: np.ndarray, top_n=5):
    # x should be (approximately) stationary; use returns or abs returns
    x = x[np.isfinite(x)]
    if len(x) < 500:
        return []

    x = x - np.mean(x)
    freqs, pxx = periodogram(x, scaling="spectrum")  # normalized frequency [0..0.5] in "cycles per sample"
    # remove zero freq
    freqs, pxx = freqs[1:], pxx[1:]
    if len(freqs) == 0:
        return []

    # convert to period in ticks: period = 1/frequency
    # avoid division by 0 (already removed)
    periods = 1.0 / freqs

    # filter reasonable periods
    mask = (periods >= MIN_PERIOD_TICKS) & (periods <= MAX_PERIOD_TICKS)
    freqs2, pxx2, periods2 = freqs[mask], pxx[mask], periods[mask]
    if len(freqs2) == 0:
        return []

    # top peaks by power
    idx = np.argsort(pxx2)[::-1][:top_n]
    out = []
    for i in idx:
        out.append({
            "period_ticks": float(periods2[i]),
            "frequency": float(freqs2[i]),
            "power": float(pxx2[i]),
        })
    return out


def acf_peak_lags(x: np.ndarray, max_lag=2000, top_n=5):
    x = x[np.isfinite(x)]
    if len(x) < 500:
        return []

    x = x - np.mean(x)
    a = acf(x, nlags=max_lag, fft=True)
    # ignore lag 0
    a0 = a[1:]
    lags = np.arange(1, len(a))

    # pick local maxima (very simple local peak)
    peaks = []
    for i in range(1, len(a0) - 1):
        if a0[i] > a0[i-1] and a0[i] > a0[i+1]:
            peaks.append((lags[i], a0[i]))

    if not peaks:
        return []

    # sort by absolute acf strength
    peaks.sort(key=lambda t: abs(t[1]), reverse=True)

    # return top_n distinct-ish peaks (avoid near-duplicates)
    out = []
    used = []
    for lag, val in peaks:
        if any(abs(lag - u) < 5 for u in used):  # de-dup within 5 ticks
            continue
        used.append(lag)
        out.append({"acf_peak_lag": int(lag), "acf_value": float(val)})
        if len(out) >= top_n:
            break
    return out


def run_periodicity_for_symbol(r: pd.Series, sym: str):
    if r.empty:
        return []

    x_r = r.values
    x_v = np.abs(x_r)  # volatility proxy

    rows = []

    # periodogram peaks on returns
    for j, d in enumerate(top_periodogram_periods(x_r, top_n=TOP_N_FREQ), start=1):
        rows.append({
            "symbol": sym,
            "series": "returns",
            "method": "periodogram",
            "rank": j,
            **d
        })

    # periodogram peaks on abs returns
    for j, d in enumerate(top_periodogram_periods(x_v, top_n=TOP_N_FREQ), start=1):
        rows.append({
            "symbol": sym,
            "series": "abs_returns",
            "method": "periodogram",
            "rank": j,
            **d
        })

    # ACF peak lags on returns
    for j, d in enumerate(acf_peak_lags(x_r, max_lag=MAX_ACF_LAG, top_n=TOP_N_FREQ), start=1):
        rows.append({
            "symbol": sym,
            "series": "returns",
            "method": "acf_peaks",
            "rank": j,
            "period_ticks": float(d["acf_peak_lag"]),
            "frequency": float(1.0 / d["acf_peak_lag"]),
            "power": float(abs(d["acf_value"])),
        })

    # ACF peak lags on abs returns
    for j, d in enumerate(acf_peak_lags(x_v, max_lag=MAX_ACF_LAG, top_n=TOP_N_FREQ), start=1):
        rows.append({
            "symbol": sym,
            "series": "abs_returns",
            "method": "acf_peaks",
            "rank": j,
            "period_ticks": float(d["acf_peak_lag"]),
            "frequency": float(1.0 / d["acf_peak_lag"]),
            "power": float(abs(d["acf_value"])),
        })

    return rows


# -------------------------
# 2) Market Phases: Markov Switching (2 regimes)
#   Model: r_t = c_{s_t} + e_t, var switches by regime
# -------------------------
def markov_regime_summary(r: pd.Series, sym: str):
    r = r.dropna()
    n = len(r)
    if n < MIN_N_RET:
        return {"symbol": sym, "n_ret": n, "note": f"too_few_samples(<{MIN_N_RET})"}

    # standardize for numerical stability
    x = (r - r.mean()) / (r.std(ddof=1) + 1e-12)

    try:
        # switching variance is the key (stable vs volatile)
        mod = MarkovRegression(
            x, k_regimes=K_REGIMES, trend="c", switching_variance=True
        )
        res = mod.fit(disp=False)

        # smoothed regime probabilities (n x k)
        probs = res.smoothed_marginal_probabilities

        # Determine which regime is "high vol" by comparing estimated variances
        # params ordering: depends on model; easiest: use regime-specific sigma2 from res.params
        # We'll estimate realized std per regime using probabilities as weights:
        p0 = probs[0].values
        p1 = probs[1].values

        std0 = float(np.sqrt(np.average(x.values**2, weights=p0)))
        std1 = float(np.sqrt(np.average(x.values**2, weights=p1)))

        high_regime = 0 if std0 > std1 else 1
        low_regime  = 1 - high_regime

        # current regime prob
        last_p_high = float(probs[high_regime].iloc[-1])
        last_regime = int(high_regime if last_p_high >= 0.5 else low_regime)

        # transition matrix (A)
        # statsmodels stores it in res.regime_transition
        A = res.regime_transition  # shape (k,k)
        p00, p01 = float(A[0,0]), float(A[0,1])
        p10, p11 = float(A[1,0]), float(A[1,1])

        return {
            "symbol": sym,
            "n_ret": n,
            "note": "",
            "regime0_weighted_std": std0,
            "regime1_weighted_std": std1,
            "high_vol_regime": int(high_regime),
            "current_high_vol_prob": last_p_high,
            "current_regime": last_regime,
            "trans_p00": p00, "trans_p01": p01,
            "trans_p10": p10, "trans_p11": p11,
            "loglike": float(res.llf),
            "aic": float(res.aic),
            "bic": float(res.bic),
        }

    except Exception as e:
        return {"symbol": sym, "n_ret": n, "note": f"markov_failed: {type(e).__name__}"}


# -------------------------
# 3) Abrupt shifts: simple change-point proxy on rolling volatility
#   (not a full CP algorithm, but good for "abrupt phase shifts")
# -------------------------
def changepoints_from_rolling_vol(r: pd.Series, sym: str):
    r = r.dropna()
    if len(r) < ROLL_WIN * 3:
        return []

    vol = r.abs().rolling(ROLL_WIN).mean().dropna()
    z = (vol - vol.mean()) / (vol.std(ddof=1) + 1e-12)

    # detect spikes in z (abrupt increases/decreases)
    idx = z.index.values
    spikes = idx[np.where(np.abs(z.values) >= Z_THRESH)[0]]

    # enforce min spacing
    out = []
    last = None
    for t in spikes:
        if last is None or (t - last) >= MIN_CP_GAP:
            out.append(int(t))
            last = t

    rows = []
    for t in out:
        rows.append({
            "symbol": sym,
            "changepoint_t": t,
            "rolling_vol_z": float(z.loc[t]),
            "rolling_vol": float(vol.loc[t]),
        })
    return rows


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # Load all series once
    prices = {s: extract_ref_price_series(JSONL_PATH, s) for s in symbols}
    rets   = {s: log_returns(prices[s]) for s in symbols}

    # --- Periodicity ---
    per_rows = []
    for s in symbols:
        per_rows.extend(run_periodicity_for_symbol(rets[s], s))

    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(OUT_PERIOD, index=False)
    print("Saved:", OUT_PERIOD)

    # --- Regimes ---
    reg_rows = []
    for s in symbols:
        reg_rows.append(markov_regime_summary(rets[s], s))

    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(OUT_REGIMES, index=False)
    print("Saved:", OUT_REGIMES)

    # --- Change-points (proxy) ---
    cp_rows = []
    for s in symbols:
        cp_rows.extend(changepoints_from_rolling_vol(rets[s], s))

    cp_df = pd.DataFrame(cp_rows)
    cp_df.to_csv(OUT_CPS, index=False)
    print("Saved:", OUT_CPS)

    # Quick console summary (optional)
    if not reg_df.empty:
        print("\nTop symbols with strongest high-vol confidence (last prob):")
        tmp = reg_df[reg_df["note"].eq("")].copy()
        if not tmp.empty:
            print(tmp.sort_values("current_high_vol_prob", ascending=False)[
                ["symbol","current_high_vol_prob","high_vol_regime","regime0_weighted_std","regime1_weighted_std"]
            ].head(10))
