# -*- coding: utf-8 -*-
# app.py — محلل أنماط Crash المتقدم

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2, ks_2samp
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import find_peaks, welch
from scipy.optimize import curve_fit
from collections import Counter, defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🔬 Crash Pattern Analyzer Pro",
    page_icon="🔬",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════
#                    دوال مساعدة
# ══════════════════════════════════════════════════════════════
def to_python(obj):
    if isinstance(obj, np.bool_):   return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(i) for i in obj]
    return obj


def categorize(v):
    if v < 1.5:  return 'VL'
    if v < 2.0:  return 'L'
    if v < 3.0:  return 'M'
    if v < 5.0:  return 'H'
    if v < 10.0: return 'VH'
    return 'EX'

CAT_LABELS = {
    'VL':'<1.5x','L':'1.5-2x','M':'2-3x',
    'H':'3-5x','VH':'5-10x','EX':'>10x'
}
CAT_COLORS = {
    'VL':'#e74c3c','L':'#e67e22','M':'#f1c40f',
    'H':'#2ecc71','VH':'#3498db','EX':'#9b59b6'
}
CAT_ORDER = ['VL','L','M','H','VH','EX']


# ══════════════════════════════════════════════════════════════
#         1. محلل الأنماط العميق
# ══════════════════════════════════════════════════════════════
class DeepPatternAnalyzer:
    """
    يحلل البنية العميقة لتسلسل Crash:
    - توزيع Power Law والانحراف عنه
    - الارتباط الذاتي متعدد المستويات
    - تحليل طيفي متقدم
    - نمذجة GARCH للتقلب
    - تحليل Hurst Exponent
    - كشف نقاط التحول
    """

    def __init__(self, data: list):
        self.raw      = np.array(data, dtype=float)
        self.log_raw  = np.log(np.maximum(self.raw, 1.0))
        self.binary   = (self.raw >= 2.0).astype(int)
        self.cats     = [categorize(v) for v in data]
        self.n        = len(data)
        self.returns  = np.diff(self.log_raw)

    # ── 1.1 تحليل توزيع Power Law ───────────────────────────
    def analyze_distribution(self) -> dict:
        """
        يحلل كيف تتوزع القيم ويكتشف الانحراف
        عن التوزيع النظري لـ Crash
        """
        arr = self.raw
        n   = self.n

        # المعاملات النظرية
        house_edge = 0.99
        thresholds = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0]

        empirical  = []
        theoretical = []
        for t in thresholds:
            emp  = float(np.mean(arr >= t))
            theo = float(min(house_edge / t, 1.0))
            empirical.append(emp)
            theoretical.append(theo)
            
        # اختبار Kolmogorov-Smirnov على log(data)
        log_data = self.log_raw
        mu_hat   = float(np.mean(log_data))
        std_hat  = float(np.std(log_data))
        
        # قيم نظرية من التوزيع المتوقع
        theoretical_samples = np.random.exponential(
            scale=mu_hat, size=n
        )
        ks_stat, ks_p = ks_2samp(log_data, theoretical_samples)

        # اكتشاف القيم الشاذة (Anomalies)
        z_scores = np.abs(stats.zscore(log_data))
        anomalies = np.where(z_scores > 3)[0]

        # مقارنة عبر نوافذ زمنية
        window_size = min(50, n // 4)
        window_stats = []
        for i in range(0, n - window_size, window_size // 2):
            seg = arr[i:i+window_size]
            window_stats.append({
                'start'   : int(i),
                'end'     : int(i + window_size),
                'pct_high': round(float(np.mean(seg >= 2.0))*100, 1),
                'mean'    : round(float(np.mean(seg)), 2),
                'max'     : round(float(np.max(seg)), 2),
                'median'  : round(float(np.median(seg)), 2)
            })

        return {
            'empirical_probs'    : [round(e, 4) for e in empirical],
            'theoretical_probs'  : [round(t, 4) for t in theoretical],
            'thresholds'         : thresholds,
            'deviations'         : [
                round(float(e - t), 4)
                for e, t in zip(empirical, theoretical)
            ],
            'ks_statistic'       : round(float(ks_stat), 4),
            'ks_pvalue'          : round(float(ks_p), 4),
            'anomaly_positions'  : anomalies.tolist(),
            'n_anomalies'        : int(len(anomalies)),
            'log_mean'           : round(mu_hat, 4),
            'log_std'            : round(std_hat, 4),
            'window_stats'       : window_stats,
            'skewness'           : round(float(stats.skew(log_data)), 4),
            'kurtosis'           : round(float(stats.kurtosis(log_data)), 4)
        }

    # ── 1.2 Hurst Exponent ──────────────────────────────────
    def hurst_exponent(self) -> dict:
        """
        H > 0.5 = تتجه (trending) — الأنماط مستمرة
        H = 0.5 = عشوائي تام
        H < 0.5 = معكوس (mean-reverting)
        """
        ts   = self.log_raw
        lags = range(2, min(20, self.n // 4))
        tau  = []
        lag_vals = []

        for lag in lags:
            if len(ts) > lag:
                diffs = ts[lag:] - ts[:-lag]
                tau.append(float(np.std(diffs)))
                lag_vals.append(int(lag))

        if len(tau) < 3:
            return {'hurst': 0.5, 'interpretation': 'بيانات غير كافية'}

        try:
            log_lags = np.log(lag_vals)
            log_tau  = np.log(np.array(tau) + 1e-9)
            slope, intercept, r, p, se = stats.linregress(
                log_lags, log_tau
            )
            H = float(slope)
        except Exception:
            H = 0.5

        H = max(0.0, min(1.0, H))

        if H > 0.6:
            interp = f"📈 H={H:.3f} — تسلسل متجه (Trending): الأنماط تستمر"
        elif H < 0.4:
            interp = f"🔄 H={H:.3f} — انعكاس للمتوسط: بعد الارتفاع ينخفض"
        else:
            interp = f"🎲 H={H:.3f} — شبه عشوائي: صعب التنبؤ"

        return {
            'hurst'         : round(H, 4),
            'interpretation': interp,
            'lags'          : lag_vals,
            'tau_values'    : [round(t, 4) for t in tau],
            'r_squared'     : round(float(r**2), 4)
        }

    # ── 1.3 تحليل التقلب GARCH-like ────────────────────────
    def volatility_analysis(self) -> dict:
        """
        يحلل كيف يتغير التقلب عبر الوقت
        فترات هدوء تسبق قفزات كبيرة؟
        """
        returns  = self.returns
        n        = len(returns)
        window   = min(10, n // 5)

        # تقلب متحرك
        vol_series = []
        for i in range(window, n):
            seg = returns[i-window:i]
            vol_series.append(float(np.std(seg)))

        vol_arr = np.array(vol_series)

        # اكتشاف فترات التقلب المنخفض
        low_vol_threshold  = float(np.percentile(vol_arr, 25))
        high_vol_threshold = float(np.percentile(vol_arr, 75))

        low_vol_periods  = np.where(vol_arr < low_vol_threshold)[0]
        high_vol_periods = np.where(vol_arr > high_vol_threshold)[0]

        # هل يسبق التقلب المنخفض قفزات كبيرة؟
        next_high_after_low_vol = 0
        count_low_vol = 0
        for idx in low_vol_periods:
            actual_idx = idx + window
            if actual_idx + 5 < self.n:
                count_low_vol += 1
                next_5 = self.raw[actual_idx:actual_idx+5]
                if np.any(next_5 >= 3.0):
                    next_high_after_low_vol += 1

        prob_high_after_low_vol = (
            float(next_high_after_low_vol / count_low_vol)
            if count_low_vol > 0 else 0.5
        )

        # نمذجة ARCH: هل تتجمع التقلبات؟
        sq_returns  = returns**2
        arch_corr   = float(np.corrcoef(
            sq_returns[1:], sq_returns[:-1]
        )[0,1]) if len(sq_returns) > 1 else 0.0

        return {
            'vol_series'              : [round(v, 4) for v in vol_series],
            'mean_vol'                : round(float(np.mean(vol_arr)), 4),
            'current_vol'             : round(float(vol_arr[-1]), 4),
            'low_vol_threshold'       : round(low_vol_threshold, 4),
            'high_vol_threshold'      : round(high_vol_threshold, 4),
            'prob_high_after_low_vol' : round(prob_high_after_low_vol, 4),
            'arch_correlation'        : round(arch_corr, 4),
            'volatility_clustering'   : bool(arch_corr > 0.15),
            'current_regime'          : (
                'هادئ 🟢' if vol_arr[-1] < low_vol_threshold
                else 'متقلب 🔴' if vol_arr[-1] > high_vol_threshold
                else 'طبيعي 🟡'
            )
        }

    # ── 1.4 تحليل طيفي متقدم ────────────────────────────────
    def advanced_spectral(self) -> dict:
        """
        تحليل Welch PSD لاكتشاف الدوريات الحقيقية
        """
        log_data = self.log_raw
        n        = self.n

        # Welch Power Spectral Density
        try:
            nperseg = min(64, n // 4)
            freqs, psd = welch(
                log_data - log_data.mean(),
                nperseg=nperseg
            )
        except Exception:
            freqs = np.array([0.0])
            psd   = np.array([0.0])

        # اكتشاف القمم في PSD
        if len(psd) > 3:
            peaks, props = find_peaks(
                psd,
                height=psd.mean() * 1.5,
                distance=2
            )
        else:
            peaks = np.array([])

        detected_cycles = []
        for pk in peaks:
            if float(freqs[pk]) > 0:
                period = float(1.0 / freqs[pk])
                if period <= n / 2:
                    detected_cycles.append({
                        'period_rounds': round(period, 1),
                        'power'        : round(float(psd[pk]), 6),
                        'relative_power': round(
                            float(psd[pk]) / (psd.max() + 1e-9), 4
                        )
                    })

        detected_cycles.sort(
            key=lambda x: x['power'], reverse=True
        )

        # FFT الكلاسيكي
        fft_vals = np.abs(fft(log_data - log_data.mean()))
        half     = n // 2
        fft_h    = fft_vals[:half]
        freq_h   = fftfreq(n)[:half]
        dom_ratio= float(fft_h.max() / (fft_h.mean() + 1e-9))

        # اكتشاف الأنماط الدورية في الثنائي
        bin_fft  = np.abs(fft(self.binary - self.binary.mean()))
        bin_fft_h= bin_fft[:half]
        bin_peaks, _ = find_peaks(
            bin_fft_h,
            height=bin_fft_h.mean() * 2
        )
        binary_cycles = []
        for pk in bin_peaks[:5]:
            if float(freq_h[pk]) > 0:
                period = float(1.0 / freq_h[pk])
                binary_cycles.append({
                    'period_rounds': round(period, 1),
                    'power'        : round(float(bin_fft_h[pk]), 4)
                })

        return {
            'welch_cycles'  : detected_cycles[:8],
            'binary_cycles' : binary_cycles,
            'dominance_ratio': round(dom_ratio, 2),
            'has_pattern'   : bool(dom_ratio > 8),
            'freqs'         : freqs.tolist(),
            'psd'           : psd.tolist(),
            'n_peaks'       : int(len(peaks))
        }

    # ── 1.5 كشف نقاط التحول ────────────────────────────────
    def detect_changepoints(self) -> dict:
        """
        يكتشف متى تتغير خصائص التسلسل
        (فترات ساخنة vs فترات باردة)
        """
        arr      = self.raw
        n        = self.n
        window   = min(20, n // 5)
        step     = max(1, window // 4)

        means    = []
        stds     = []
        pct_high = []
        positions= []

        for i in range(0, n - window, step):
            seg = arr[i:i+window]
            means.append(float(np.mean(seg)))
            stds.append(float(np.std(seg)))
            pct_high.append(float(np.mean(seg >= 2.0)))
            positions.append(int(i + window // 2))

        # اكتشاف التغيرات المفاجئة
        means_arr = np.array(means)
        pct_arr   = np.array(pct_high)

        changepoints = []
        for i in range(1, len(means_arr) - 1):
            delta = abs(pct_arr[i] - pct_arr[i-1])
            if delta > 0.20:
                direction = (
                    '🔥 تحول ساخن' if pct_arr[i] > pct_arr[i-1]
                    else '❄️ تحول بارد'
                )
                changepoints.append({
                    'position'   : int(positions[i]),
                    'before_pct' : round(float(pct_arr[i-1])*100, 1),
                    'after_pct'  : round(float(pct_arr[i])*100, 1),
                    'delta'      : round(float(delta)*100, 1),
                    'direction'  : direction
                })

        # الحالة الراهنة
        recent_pct  = float(np.mean(arr[-20:] >= 2.0))
        hist_pct    = float(np.mean(arr >= 2.0))
        current_state = (
            '🔥 ساخن' if recent_pct > hist_pct * 1.15
            else '❄️ بارد' if recent_pct < hist_pct * 0.85
            else '⚖️ طبيعي'
        )

        return {
            'positions'    : positions,
            'means'        : [round(m, 2) for m in means],
            'pct_high'     : [round(p*100, 1) for p in pct_high],
            'changepoints' : changepoints,
            'current_state': current_state,
            'recent_pct'   : round(recent_pct*100, 1),
            'hist_pct'     : round(hist_pct*100, 1)
        }

    # ── 1.6 تحليل الفجوات بين القفزات ──────────────────────
    def gap_analysis(self) -> dict:
        """
        الأهم: بعد كم جولة تأتي القفزة؟
        يحسب توزيع الفجوات ويتنبأ بالقادمة
        """
        arr = self.raw
        results = {}

        for threshold in [2.0, 3.0, 5.0, 10.0, 20.0]:
            positions = np.where(arr >= threshold)[0]

            if len(positions) < 2:
                continue

            gaps = np.diff(positions).tolist()
            gaps = [int(g) for g in gaps]
            last_pos = int(positions[-1])
            current_gap = int(len(arr) - 1 - last_pos)

            # توزيع الفجوات
            gap_arr = np.array(gaps)
            mean_g  = float(np.mean(gap_arr))
            std_g   = float(np.std(gap_arr))
            median_g= float(np.median(gap_arr))
            p25     = float(np.percentile(gap_arr, 25))
            p75     = float(np.percentile(gap_arr, 75))

            # احتمال الظهور بعد k جولة من الآن
            # بافتراض توزيع هندسي للفجوات
            prob_appearances = {}
            prob_rate = min(float(len(positions) / len(arr)), 0.99)
            for k in range(1, 16):
                prob = float(
                    prob_rate * (1 - prob_rate)**(k-1)
                )
                prob_appearances[f'بعد {k} جولة'] = round(prob, 4)

            # نسبة الاستحقاق
            due_ratio = float(current_gap / (mean_g + 1e-9))

            # هل نحن في منطقة الخطر؟
            if current_gap >= p75:
                zone = '🔴 منطقة الخطر — متأخر!'
            elif current_gap >= median_g:
                zone = '🟡 منطقة التوقع — قريب'
            else:
                zone = '🟢 منطقة الأمان — مبكر'

            # توقع الجولات المتبقية
            expected_remaining = max(0, mean_g - current_gap)

            results[f'>= {threshold}x'] = {
                'count'              : int(len(positions)),
                'gaps'               : gaps[-15:],
                'mean_gap'           : round(mean_g, 1),
                'std_gap'            : round(std_g, 1),
                'median_gap'         : round(median_g, 1),
                'p25_gap'            : round(p25, 1),
                'p75_gap'            : round(p75, 1),
                'current_gap'        : current_gap,
                'due_ratio'          : round(due_ratio, 2),
                'zone'               : zone,
                'expected_remaining' : round(expected_remaining, 1),
                'prob_in_next_5'     : round(float(
                    1 - (1-prob_rate)**5
                ), 4),
                'prob_appearances'   : prob_appearances
            }

        return results

    # ── 1.7 نمذجة سلسلة ماركوف عالية الدقة ────────────────
    def advanced_markov(self) -> dict:
        """
        سلاسل ماركوف من درجة 1، 2، 3، 4
        مع حساب احتمالات الثبات (Stationary Distribution)
        """
        cats = self.cats

        # درجة 1
        trans1 = defaultdict(Counter)
        for i in range(len(cats)-1):
            trans1[cats[i]][cats[i+1]] += 1

        matrix1 = {}
        for c in CAT_ORDER:
            total = sum(trans1[c].values())
            if total > 0:
                matrix1[c] = {
                    cc: round(trans1[c].get(cc,0)/total, 4)
                    for cc in CAT_ORDER
                }
            else:
                matrix1[c] = {cc: 0.0 for cc in CAT_ORDER}

        # التوزيع الثابت (Stationary)
        try:
            M = np.array([[matrix1[r][c] for c in CAT_ORDER]
                          for r in CAT_ORDER])
            eigvals, eigvecs = np.linalg.eig(M.T)
            idx = np.argmin(np.abs(eigvals - 1.0))
            pi  = np.real(eigvecs[:, idx])
            pi  = np.abs(pi) / np.abs(pi).sum()
            stationary = {
                CAT_ORDER[i]: round(float(pi[i]), 4)
                for i in range(len(CAT_ORDER))
            }
        except Exception:
            stationary = {c: round(1/6, 4) for c in CAT_ORDER}

        # درجة 2
        trans2 = defaultdict(Counter)
        for i in range(len(cats)-2):
            key = (cats[i], cats[i+1])
            trans2[key][cats[i+2]] += 1

        # درجة 3
        trans3 = defaultdict(Counter)
        for i in range(len(cats)-3):
            key = (cats[i], cats[i+1], cats[i+2])
            trans3[key][cats[i+3]] += 1

        # درجة 4
        trans4 = defaultdict(Counter)
        for i in range(len(cats)-4):
            key = (cats[i],cats[i+1],cats[i+2],cats[i+3])
            trans4[key][cats[i+4]] += 1

        def top_patterns(trans, min_occ=3, min_prob=0.55):
            patterns = []
            for key, counts in trans.items():
                total = sum(counts.values())
                if total < min_occ:
                    continue
                for next_cat, cnt in counts.items():
                    prob = cnt / total
                    if prob >= min_prob:
                        if isinstance(key, tuple):
                            pat_str = ' → '.join(
                                CAT_LABELS[k] for k in key
                            )
                        else:
                            pat_str = CAT_LABELS[key]
                        patterns.append({
                            'pattern'    : pat_str,
                            'next'       : CAT_LABELS[next_cat],
                            'next_cat'   : next_cat,
                            'probability': round(float(prob), 4),
                            'occurrences': int(total)
                        })
            return sorted(
                patterns,
                key=lambda x: (x['probability'], x['occurrences']),
                reverse=True
            )[:10]

        return {
            'matrix1'    : matrix1,
            'stationary' : stationary,
            'order1'     : top_patterns(trans1, 5, 0.50),
            'order2'     : top_patterns(trans2, 3, 0.55),
            'order3'     : top_patterns(trans3, 3, 0.60),
            'order4'     : top_patterns(trans4, 2, 0.65)
        }

    def run_all(self) -> dict:
        return {
            'distribution'  : self.analyze_distribution(),
            'hurst'         : self.hurst_exponent(),
            'volatility'    : self.volatility_analysis(),
            'spectral'      : self.advanced_spectral(),
            'changepoints'  : self.detect_changepoints(),
            'gaps'          : self.gap_analysis(),
            'markov'        : self.advanced_markov()
        }


# ══════════════════════════════════════════════════════════════
#         2. محرك التنبؤ المتقدم
# ══════════════════════════════════════════════════════════════
class AdvancedPredictor:
    """
    يدمج 8 طرق تنبؤ بأوزان ديناميكية
    """

    def __init__(self, data: list, analysis: dict):
        self.raw      = np.array(data, dtype=float)
        self.cats     = [categorize(v) for v in data]
        self.bin      = [1 if v >= 2.0 else 0 for v in data]
        self.analysis = analysis
        self.n        = len(data)

    def _current_state(self) -> dict:
        """الحالة الراهنة"""
        low_streak  = 0
        high_streak = 0
        for v in reversed(self.bin):
            if v == 0 and high_streak == 0:
                low_streak += 1
            elif v == 1 and low_streak == 0:
                high_streak += 1
            else:
                break

        recent10 = self.raw[-10:]
        recent20 = self.raw[-20:]
        hist_avg = float(np.mean(self.raw))
        cur_vol  = self.analysis['volatility']['current_vol']
        low_vol  = self.analysis['volatility']['low_vol_threshold']

        return {
            'last1'      : self.cats[-1],
            'last2'      : self.cats[-2] if self.n >= 2 else 'M',
            'last3'      : self.cats[-3] if self.n >= 3 else 'M',
            'last4'      : self.cats[-4] if self.n >= 4 else 'M',
            'last_value' : float(self.raw[-1]),
            'low_streak' : int(low_streak),
            'high_streak': int(high_streak),
            'recent_avg10': round(float(np.mean(recent10)), 2),
            'recent_avg20': round(float(np.mean(recent20)), 2),
            'hist_avg'   : round(hist_avg, 2),
            'current_vol': float(cur_vol),
            'is_low_vol' : bool(cur_vol < low_vol),
            'position_in_cycle': int(self.n % 20)
        }

    def predict_markov4(self, state: dict) -> dict:
        """توقع Markov درجة 4"""
        markov   = self.analysis['markov']
        patterns = markov.get('order4', [])
        key_str  = (
            f"{CAT_LABELS[state['last4']]} → "
            f"{CAT_LABELS[state['last3']]} → "
            f"{CAT_LABELS[state['last2']]} → "
            f"{CAT_LABELS[state['last1']]}"
        )
        for p in patterns:
            if p['pattern'] == key_str:
                return {
                    'category'  : p['next_cat'],
                    'confidence': float(p['probability']),
                    'found'     : True,
                    'method'    : 'Markov-4'
                }
        # fallback درجة 3
        patterns3 = markov.get('order3', [])
        key3 = (
            f"{CAT_LABELS[state['last3']]} → "
            f"{CAT_LABELS[state['last2']]} → "
            f"{CAT_LABELS[state['last1']]}"
        )
        for p in patterns3:
            if p['pattern'] == key3:
                return {
                    'category'  : p['next_cat'],
                    'confidence': float(p['probability']) * 0.9,
                    'found'     : True,
                    'method'    : 'Markov-3 (fallback)'
                }
        return {
            'category'  : 'M',
            'confidence': 0.35,
            'found'     : False,
            'method'    : 'Markov (no match)'
        }

    def predict_markov1(self, state: dict) -> dict:
        """توقع Markov درجة 1"""
        matrix = self.analysis['markov']['matrix1']
        probs  = matrix.get(state['last1'], {})
        if not probs:
            return {'category':'M','confidence':0.33,'method':'Markov-1'}
        best = max(probs, key=probs.get)
        return {
            'category'  : best,
            'confidence': float(probs.get(best, 0.33)),
            'all_probs' : probs,
            'method'    : 'Markov-1'
        }

    def predict_gap_analysis(self, state: dict) -> dict:
        """توقع بناءً على تحليل الفجوات"""
        gaps = self.analysis['gaps']

        best_signal = None
        best_conf   = 0.0

        for threshold_key, info in gaps.items():
            due   = float(info['due_ratio'])
            cur_g = int(info['current_gap'])
            mean_g= float(info['mean_gap'])

            if due >= 1.2:
                threshold = float(
                    threshold_key.replace('>=','').replace('x','').strip()
                )
                conf = min(float(due) * 0.4, 0.80)
                if threshold <= 3.0 and conf > best_conf:
                    best_conf   = conf
                    best_signal = {
                        'category'  : 'H',
                        'confidence': round(conf, 3),
                        'threshold' : threshold_key,
                        'due_ratio' : round(due, 2),
                        'zone'      : info['zone'],
                        'method'    : 'Gap Due Analysis'
                    }

        if best_signal:
            return best_signal

        return {
            'category'  : 'M',
            'confidence': 0.40,
            'method'    : 'Gap Analysis (neutral)'
        }

    def predict_hurst(self, state: dict) -> dict:
        """توقع بناءً على Hurst"""
        H = float(self.analysis['hurst']['hurst'])
        last_cat = state['last1']

        if H > 0.6:
            # متجه: نفس الاتجاه
            high_cats = ['M','H','VH','EX']
            cat = last_cat if last_cat in ['H','VH','EX'] else 'M'
            conf = float(H)
        elif H < 0.4:
            # معكوس
            reversal = {
                'VL':'M','L':'M','M':'L','H':'L','VH':'L','EX':'L'
            }
            cat  = reversal.get(last_cat, 'M')
            conf = float(1 - H)
        else:
            cat  = 'M'
            conf = 0.40

        return {
            'category'  : cat,
            'confidence': round(min(conf, 0.75), 3),
            'hurst'     : round(H, 3),
            'method'    : 'Hurst Exponent'
        }

    def predict_volatility(self, state: dict) -> dict:
        """توقع بناءً على نظام التقلب"""
        vol_info = self.analysis['volatility']
        regime   = vol_info['current_regime']
        prob_h   = float(vol_info['prob_high_after_low_vol'])

        if state['is_low_vol'] and prob_h > 0.55:
            return {
                'category'  : 'H',
                'confidence': round(prob_h, 3),
                'regime'    : regime,
                'method'    : 'Volatility Regime'
            }
        elif 'متقلب' in regime:
            return {
                'category'  : 'VL',
                'confidence': 0.55,
                'regime'    : regime,
                'method'    : 'Volatility Regime'
            }
        return {
            'category'  : 'M',
            'confidence': 0.42,
            'regime'    : regime,
            'method'    : 'Volatility Regime'
        }

    def predict_cycle_position(self) -> dict:
        """توقع بناءً على موقع الدورة"""
        cycles = self.analysis['spectral']['welch_cycles']
        if not cycles:
            return {'category':'M','confidence':0.38,'method':'Cycle'}

        best   = cycles[0]
        period = float(best['period_rounds'])
        phase  = float((self.n % max(int(period), 1)) / max(period, 1))

        if 0.1 <= phase <= 0.35:
            cat, conf = 'H', 0.58
        elif 0.35 < phase <= 0.65:
            cat, conf = 'VH', 0.52
        elif 0.65 < phase <= 0.85:
            cat, conf = 'M', 0.50
        else:
            cat, conf = 'VL', 0.55

        return {
            'category'  : cat,
            'confidence': conf,
            'period'    : round(period, 1),
            'phase'     : round(phase, 3),
            'method'    : 'Spectral Cycle'
        }

    def predict_mean_reversion(self, state: dict) -> dict:
        """انتقال للمتوسط"""
        r10  = float(state['recent_avg10'])
        r20  = float(state['recent_avg20'])
        hist = float(state['hist_avg'])
        ratio= r10 / (hist + 1e-9)
        trend= r10 - r20

        if ratio < 0.70:
            cat, conf = 'H', min(float(1.1 - ratio), 0.78)
        elif ratio < 0.85:
            cat, conf = 'M', 0.58
        elif ratio > 1.40:
            cat, conf = 'VL', min(float(ratio - 0.6), 0.72)
        elif ratio > 1.20:
            cat, conf = 'L', 0.55
        else:
            cat, conf = 'M', 0.42

        # تعديل بناءً على الاتجاه
        if trend < 0 and cat in ['H','VH']:
            conf = min(conf + 0.05, 0.85)
        elif trend > 0 and cat in ['VL','L']:
            conf = min(conf + 0.05, 0.85)

        return {
            'category'  : cat,
            'confidence': round(min(conf, 0.82), 3),
            'ratio'     : round(float(ratio), 3),
            'trend'     : round(float(trend), 3),
            'method'    : 'Mean Reversion'
        }

    def predict_streak_bayesian(self, state: dict) -> dict:
        """Bayesian بناءً على التسلسل المنخفض"""
        low_s = int(state['low_streak'])
        bin_  = self.bin
        n     = len(bin_)

        # حساب احتمال مشروط
        transitions = 0
        occurrences = 0
        k = min(low_s, 6)

        for i in range(n - k - 1):
            if all(bin_[i:i+k] == np.zeros(k, dtype=int)):
                occurrences += 1
                if bin_[i+k] == 1:
                    transitions += 1

        if occurrences >= 3:
            # Bayesian مع prior
            alpha  = float(transitions) + 2.0
            beta_  = float(occurrences - transitions) + 1.0
            prob_h = float(alpha / (alpha + beta_))
        else:
            prob_h = float(np.mean(bin_))

        if low_s >= 5:
            prob_h = min(prob_h + 0.10, 0.88)

        cat = 'M' if prob_h >= 0.55 else ('L' if prob_h >= 0.45 else 'VL')
        if prob_h >= 0.65:
            cat = 'H'

        return {
            'category'  : cat,
            'confidence': round(prob_h if prob_h >= 0.5 else 1-prob_h, 3),
            'prob_high' : round(prob_h, 4),
            'low_streak': low_s,
            'occurrences': int(occurrences),
            'method'    : 'Bayesian Streak'
        }

    def ensemble_predict(self) -> dict:
        """التنبؤ الجماعي النهائي"""
        state = self._current_state()

        # تشغيل كل الطرق
        methods = {
            'markov4'       : (self.predict_markov4(state),      0.22),
            'markov1'       : (self.predict_markov1(state),      0.15),
            'gap'           : (self.predict_gap_analysis(state), 0.20),
            'hurst'         : (self.predict_hurst(state),        0.10),
            'volatility'    : (self.predict_volatility(state),   0.12),
            'cycle'         : (self.predict_cycle_position(),    0.08),
            'mean_reversion': (self.predict_mean_reversion(state),0.08),
            'streak_bayes'  : (self.predict_streak_bayesian(state),0.05)
        }

        # تجميع الأصوات
        votes = defaultdict(float)
        for name, (pred, weight) in methods.items():
            cat  = pred['category']
            conf = float(pred.get('confidence', 0.5))
            votes[cat] += weight * conf

        total  = sum(votes.values()) + 1e-9
        winner = max(votes, key=votes.get)
        final_conf = float(votes[winner]) / total

        # احتمال >= 2x
        high_cats = ['M','H','VH','EX']
        prob_high = sum(
            float(votes[c]) for c in high_cats
        ) / total

        # تعديل بالتسلسل المنخفض
        ls = int(state['low_streak'])
        if ls >= 4:
            boost     = min(0.08 * (ls - 3), 0.18)
            prob_high = min(prob_high + boost, 0.90)

        # تعديل بـ Hurst
        H = float(self.analysis['hurst']['hurst'])
        if H < 0.4 and state['last1'] in ['H','VH','EX']:
            prob_high = max(prob_high - 0.08, 0.10)

        # الفئة النهائية
        if prob_high >= 0.72:   final_cat = 'H'
        elif prob_high >= 0.58: final_cat = 'M'
        elif prob_high <= 0.28: final_cat = 'VL'
        elif prob_high <= 0.42: final_cat = 'L'
        else:                   final_cat = winner

        # توقع عدد الجولات للقفزة القادمة
        gaps_info    = self.analysis['gaps']
        rounds_to_jump = {}
        for thr, info in gaps_info.items():
            rem = float(info['expected_remaining'])
            rounds_to_jump[thr] = {
                'expected_in'  : max(0.0, round(rem, 1)),
                'zone'         : info['zone'],
                'current_gap'  : info['current_gap'],
                'mean_gap'     : info['mean_gap']
            }

        # النطاق القيمي
        ranges = {
            'VL':(1.0,1.5),'L':(1.5,2.0),'M':(2.0,3.0),
            'H':(3.0,5.0),'VH':(5.0,10.0),'EX':(10.0,30.0)
        }
        lo, hi = ranges[final_cat]

        return {
            'final_cat'       : final_cat,
            'final_label'     : CAT_LABELS[final_cat],
            'value_range'     : f"{lo:.1f}x — {hi:.1f}x",
            'est_value'       : round((lo+hi)/2, 2),
            'prob_high'       : round(float(prob_high), 4),
            'prob_high_pct'   : round(float(prob_high)*100, 1),
            'final_conf'      : round(final_conf, 4),
            'confidence_level': (
                '🟢 عالية'   if final_conf >= 0.65 else
                '🟡 متوسطة' if final_conf >= 0.50 else
                '🔴 منخفضة'
            ),
            'low_streak'      : ls,
            'current_state'   : state,
            'method_votes'    : {
                k: round(float(v)/total, 4)
                for k, v in sorted(
                    votes.items(),
                    key=lambda x: x[1], reverse=True
                )
            },
            'methods_detail'  : {
                k: v[0] for k, v in methods.items()
            },
            'rounds_to_jump'  : rounds_to_jump,
            'hurst'           : H,
            'volatility_regime': self.analysis['volatility']['current_regime']
        }


# ══════════════════════════════════════════════════════════════
#                    بيانات نموذجية
# ══════════════════════════════════════════════════════════════
SAMPLE_DATA = [
    8.72,6.75,1.86,2.18,1.25,2.28,1.24,1.20,1.54,24.46,
    4.16,1.49,1.09,1.47,1.54,1.53,2.10,32.04,11.0,1.17,
    1.70,2.61,1.26,22.23,1.77,1.93,3.35,7.01,1.83,9.39,
    3.31,2.04,1.30,6.65,1.16,3.39,1.95,10.85,1.65,1.22,
    1.60,4.67,1.85,2.72,1.00,3.02,1.35,1.30,1.37,17.54,
    1.18,1.00,14.40,1.11,6.15,2.39,2.22,1.42,1.23,2.42,
    1.07,1.24,2.55,7.26,1.69,5.10,2.59,5.51,2.31,2.12,
    1.97,1.50,3.01,2.29,1.36,4.95,5.09,8.50,1.77,5.52,
    3.93,1.50,2.28,2.49,18.25,1.68,1.42,2.12,4.17,1.04,
    2.35,1.00,1.01,5.46,1.13,2.84,3.39,2.79,1.59,1.53,
    4.34,2.96,1.06,1.72,2.16,2.20,3.61,2.34,4.49,1.72,
    1.78,9.27,8.49,2.86,1.66,4.63,9.25,1.35,1.00,1.64,
    1.86,2.81,2.44,1.74,1.10,1.29,1.45,8.92,1.24,6.39,
    1.16,1.19,2.40,4.64,3.17,24.21,1.17,1.42,2.13,1.12,
    3.78,1.12,1.52,22.81,1.31,1.90,1.38,1.47,2.86,1.79,
    1.49,1.38,1.84,1.06,3.30,5.97,1.00,2.92,1.64,5.32,
    3.26,1.78,2.24,3.16,1.60,1.08,1.55,1.07,1.02,1.23,
    1.08,5.22,3.32,24.86,3.37,5.16,1.69,2.31,1.07,1.10,
]


# ══════════════════════════════════════════════════════════════
#                    واجهة المستخدم
# ══════════════════════════════════════════════════════════════
st.title("🔬 محلل أنماط Crash المتقدم")
st.caption(
    "Hurst Exponent • GARCH Volatility • Markov-4 • "
    "Welch PSD • Gap Analysis • Bayesian Streak • "
    "Ensemble Prediction"
)

# ── إدخال البيانات ──────────────────────────────────────────
st.header("📥 البيانات")
method = st.radio(
    "الإدخال:",
    ["📝 يدوي","📂 CSV","🎲 نموذجية"],
    horizontal=True
)
raw_data = None

if method == "📝 يدوي":
    txt = st.text_area(
        "قيم Crash (50+ للحصول على أفضل تحليل):",
        height=120,
        placeholder="1.23  4.56  2.10  8.92 ..."
    )
    if txt.strip():
        try:
            raw_data = [
                float(x) for x in
                txt.replace('\n',' ').split() if x.strip()
            ]
            st.success(f"✅ {len(raw_data)} قيمة")
        except Exception:
            st.error("❌ أرقام فقط")

elif method == "📂 CSV":
    up = st.file_uploader("CSV — عمود crash_point", type=['csv'])
    if up:
        try:
            df_u = pd.read_csv(up)
            if 'crash_point' in df_u.columns:
                raw_data = [float(x) for x in
                            df_u['crash_point'].dropna()]
                st.success(f"✅ {len(raw_data)} قيمة")
            else:
                st.error(f"الأعمدة: {list(df_u.columns)}")
        except Exception as e:
            st.error(str(e))
else:
    raw_data = SAMPLE_DATA
    st.info(f"🎲 {len(raw_data)} قيمة")

if raw_data:
    arr = np.array(raw_data, dtype=float)
    n   = int(len(arr))

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("العدد",   str(n))
    c2.metric("متوسط",   f"{arr.mean():.2f}x")
    c3.metric("وسيط",    f"{np.median(arr):.2f}x")
    c4.metric("أقصى",    f"{arr.max():.2f}x")
    c5.metric(">=2x",    f"{np.mean(arr>=2)*100:.1f}%")
    c6.metric("آخر قيمة",f"{arr[-1]:.2f}x")

    if n < 50:
        st.warning(f"⚠️ يُفضَّل 50+ قيمة (لديك {n})")
    else:
        st.markdown("---")
        if st.button(
            "🚀 تحليل عميق + تنبؤ متقدم",
            type="primary",
            use_container_width=True
        ):
            prog = st.progress(0)
            stat = st.empty()

            stat.info("⏳ التحليل العميق...")
            analyzer = DeepPatternAnalyzer(raw_data)
            analysis = analyzer.run_all()
            prog.progress(60)

            stat.info("⏳ حساب التنبؤ الجماعي...")
            predictor = AdvancedPredictor(raw_data, analysis)
            pred      = predictor.ensemble_predict()
            prog.progress(100)

            stat.empty(); prog.empty()
            st.balloons()

            # ════════════════════════════════════════════
            #            بطاقة التنبؤ الرئيسية
            # ════════════════════════════════════════════
            st.markdown("---")
            st.header("🎯 التنبؤ بالجولة القادمة")

            cat   = pred['final_cat']
            color = CAT_COLORS[cat]
            ls    = pred['low_streak']

            col_main, col_side = st.columns([1, 2])

            with col_main:
                st.markdown(
                    f"""
                    <div style="
                        background:{color}18;
                        border:3px solid {color};
                        border-radius:20px;
                        padding:30px;
                        text-align:center;
                    ">
                    <div style="font-size:3em;">{
                        '🔴' if cat=='VL' else
                        '🟠' if cat=='L'  else
                        '🟡' if cat=='M'  else
                        '🟢' if cat=='H'  else
                        '🔵' if cat=='VH' else '🟣'
                    }</div>
                    <h2 style="color:{color};margin:8px 0;">
                        {pred['final_label']}
                    </h2>
                    <h3 style="margin:4px 0;">
                        ≈ {pred['est_value']}x
                    </h3>
                    <p style="color:#888;margin:4px 0;">
                        نطاق: {pred['value_range']}
                    </p>
                    <hr style="border-color:{color}44;">
                    <b>احتمال >= 2x:</b>
                    {pred['prob_high_pct']}%<br>
                    <b>الثقة:</b> {pred['confidence_level']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if ls >= 3:
                    st.warning(
                        f"⚡ تسلسل منخفض = {ls} جولات\n\n"
                        f"احتمال ارتفاع متزايد!"
                    )

            with col_side:
                # أصوات الطرق
                votes_data = {
                    CAT_LABELS.get(k, k): round(v*100, 1)
                    for k, v in pred['method_votes'].items()
                }
                fig_v = px.bar(
                    x=list(votes_data.keys()),
                    y=list(votes_data.values()),
                    title="توزيع أصوات طرق التنبؤ (%)",
                    color=list(votes_data.values()),
                    color_continuous_scale='RdYlGn',
                    labels={'x':'الفئة','y':'الوزن %'}
                )
                fig_v.update_layout(
                    height=280, margin=dict(t=35,b=5)
                )
                st.plotly_chart(fig_v, use_container_width=True)

                # تفاصيل الطرق
                m_rows = []
                for mname, minfo in pred['methods_detail'].items():
                    m_rows.append({
                        'الطريقة'  : mname,
                        'التوقع'   : CAT_LABELS.get(
                            minfo['category'], minfo['category']
                        ),
                        'الثقة %'  : f"{minfo.get('confidence',0)*100:.1f}%"
                    })
                st.dataframe(
                    pd.DataFrame(m_rows),
                    use_container_width=True,
                    hide_index=True
                )

            # ── توقع القفزات القادمة ─────────────────────
            st.markdown("---")
            st.subheader("⏱️ متى تأتي القفزة القادمة؟")

            jump_rows = []
            for thr, info in pred['rounds_to_jump'].items():
                exp_in = float(info['expected_in'])
                jump_rows.append({
                    'العتبة'           : thr,
                    'الفجوة الحالية'   : f"{info['current_gap']} جولة",
                    'متوسط الفجوة'     : f"{info['mean_gap']:.1f}",
                    'متوقعة بعد'       : (
                        f"≈ {exp_in:.0f} جولة"
                        if exp_in > 0 else "متأخرة!"
                    ),
                    'المنطقة'          : info['zone']
                })
            st.dataframe(
                pd.DataFrame(jump_rows),
                use_container_width=True,
                hide_index=True
            )

            # ════════════════════════════════════════════
            #              التحليلات التفصيلية
            # ════════════════════════════════════════════
            st.markdown("---")
            st.header("📊 التحليلات العلمية")

            (t_gap, t_hurst, t_vol, t_spec,
             t_chg, t_mk, t_dist, t_hist) = st.tabs([
                "⏱️ تحليل الفجوات",
                "📐 Hurst",
                "📉 التقلب",
                "📡 الطيف",
                "🔄 نقاط التحول",
                "🔗 Markov",
                "📊 التوزيع",
                "📜 التاريخ"
            ])

            # ── تحليل الفجوات ────────────────────────────
            with t_gap:
                st.subheader("⏱️ تحليل الفجوات بين القفزات")
                gaps = analysis['gaps']

                for thr_key, info in gaps.items():
                    with st.expander(
                        f"🎯 {thr_key} — الفجوة الحالية: "
                        f"{info['current_gap']} | {info['zone']}"
                    ):
                        ca,cb,cc,cd,ce = st.columns(5)
                        ca.metric("عدد الظهورات", info['count'])
                        cb.metric("متوسط الفجوة",
                                  f"{info['mean_gap']:.1f}")
                        cc.metric("وسيط الفجوة",
                                  f"{info['median_gap']:.1f}")
                        cd.metric("الفجوة الحالية",
                                  f"{info['current_gap']}")
                        ce.metric("نسبة الاستحقاق",
                                  f"{info['due_ratio']}x")

                        if info['gaps']:
                            fig_g = go.Figure()
                            fig_g.add_trace(go.Bar(
                                y=info['gaps'],
                                marker_color=[
                                    '#e74c3c' if g > info['mean_gap']
                                    else '#2ecc71'
                                    for g in info['gaps']
                                ],
                                name='الفجوة'
                            ))
                            fig_g.add_hline(
                                y=info['mean_gap'],
                                line_dash="dash",
                                line_color="blue",
                                annotation_text="المتوسط"
                            )
                            fig_g.add_hline(
                                y=info['median_gap'],
                                line_dash="dot",
                                line_color="orange",
                                annotation_text="الوسيط"
                            )
                            fig_g.update_layout(
                                title=f"تاريخ الفجوات — {thr_key}",
                                height=300
                            )
                            st.plotly_chart(
                                fig_g, use_container_width=True
                            )

                        # احتمالية الظهور
                        probs = info['prob_appearances']
                        p_keys = list(probs.keys())[:10]
                        p_vals = [probs[k] for k in p_keys]
                        fig_p = px.bar(
                            x=p_keys, y=p_vals,
                            title=f"احتمال ظهور {thr_key} خلال الجولات القادمة",
                            labels={'x':'','y':'الاحتمال'},
                            color=p_vals,
                            color_continuous_scale='Blues'
                        )
                        fig_p.update_layout(height=250)
                        st.plotly_chart(
                            fig_p, use_container_width=True
                        )

            # ── Hurst ────────────────────────────────────
            with t_hurst:
                hurst = analysis['hurst']
                H     = float(hurst['hurst'])

                st.subheader("📐 Hurst Exponent — طبيعة التسلسل")
                ca, cb = st.columns(2)
                ca.metric(
                    "Hurst H",
                    f"{H:.4f}",
                    delta=hurst['interpretation']
                )
                cb.metric("R²", f"{hurst['r_squared']:.4f}")

                # مقياس بصري
                fig_h = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=H,
                    title={'text':"Hurst Exponent"},
                    gauge={
                        'axis': {'range':[0,1]},
                        'bar' : {'color':'darkblue'},
                        'steps': [
                            {'range':[0,0.4],
                             'color':'#e74c3c'},
                            {'range':[0.4,0.6],
                             'color':'#f1c40f'},
                            {'range':[0.6,1],
                             'color':'#2ecc71'}
                        ],
                        'threshold': {
                            'line':{'color':'black','width':3},
                            'thickness':0.75,
                            'value':H
                        }
                    }
                ))
                fig_h.update_layout(height=320)
                st.plotly_chart(fig_h, use_container_width=True)

                st.info(hurst['interpretation'])
                st.markdown("""
| H | المعنى |
|---|--------|
| H > 0.6 | تسلسل متجه — الأنماط تستمر |
| H ≈ 0.5 | عشوائي تام |
| H < 0.4 | انعكاس للمتوسط |
                """)

                if len(hurst['lags']) > 0:
                    fig_hr = px.scatter(
                        x=np.log(hurst['lags']),
                        y=np.log(
                            np.array(hurst['tau_values']) + 1e-9
                        ),
                        trendline='ols',
                        title="تقدير Hurst (log-log regression)",
                        labels={'x':'log(lag)','y':'log(τ)'}
                    )
                    st.plotly_chart(fig_hr, use_container_width=True)

            # ── التقلب ───────────────────────────────────
            with t_vol:
                vol = analysis['volatility']
                st.subheader("📉 تحليل التقلب — GARCH-like")

                ca,cb,cc = st.columns(3)
                ca.metric(
                    "نظام التقلب الحالي",
                    vol['current_regime']
                )
                cb.metric(
                    "P(ارتفاع | تقلب منخفض)",
                    f"{vol['prob_high_after_low_vol']*100:.1f}%"
                )
                cc.metric(
                    "تجمع التقلبات (ARCH)",
                    "نعم" if vol['volatility_clustering'] else "لا",
                    delta=f"corr={vol['arch_correlation']:.3f}"
                )

                if vol['vol_series']:
                    fig_vol = go.Figure()
                    vs = vol['vol_series']
                    fig_vol.add_trace(go.Scatter(
                        y=vs, mode='lines',
                        name='التقلب',
                        line=dict(color='purple', width=2)
                    ))
                    fig_vol.add_hline(
                        y=vol['low_vol_threshold'],
                        line_dash="dash", line_color="green",
                        annotation_text="حد التقلب المنخفض"
                    )
                    fig_vol.add_hline(
                        y=vol['high_vol_threshold'],
                        line_dash="dash", line_color="red",
                        annotation_text="حد التقلب العالي"
                    )
                    fig_vol.update_layout(
                        title="التقلب عبر الزمن",
                        yaxis_title="التقلب",
                        height=380
                    )
                    st.plotly_chart(
                        fig_vol, use_container_width=True
                    )
                    st.caption(
                        "المناطق الخضراء (تقلب منخفض) تسبق غالباً القفزات الكبيرة"
                    )

            # ── الطيف ────────────────────────────────────
            with t_spec:
                spec = analysis['spectral']
                st.subheader("📡 التحليل الطيفي — الدورات الزمنية")

                ca, cb = st.columns(2)
                ca.metric("نسبة الهيمنة",
                          f"{spec['dominance_ratio']}x")
                cb.metric(
                    "يوجد نمط قوي؟",
                    "نعم 🔴" if spec['has_pattern'] else "لا ✅"
                )

                if spec['welch_cycles']:
                    df_wc = pd.DataFrame(spec['welch_cycles'])
                    fig_wc = px.bar(
                        df_wc,
                        x='period_rounds',
                        y='relative_power',
                        title="دورات Welch PSD (الأقوى)",
                        color='relative_power',
                        color_continuous_scale='Reds',
                        labels={
                            'period_rounds':'دورة (جولات)',
                            'relative_power':'القوة النسبية'
                        },
                        text='period_rounds'
                    )
                    fig_wc.update_traces(
                        texttemplate='%{text:.0f}j',
                        textposition='outside'
                    )
                    st.plotly_chart(
                        fig_wc, use_container_width=True
                    )

                if len(spec['freqs']) > 1:
                    fig_psd = go.Figure()
                    fig_psd.add_trace(go.Scatter(
                        x=spec['freqs'],
                        y=spec['psd'],
                        mode='lines',
                        fill='tozeroy',
                        name='PSD',
                        line=dict(color='steelblue')
                    ))
                    fig_psd.update_layout(
                        title="طيف القدرة (Welch PSD)",
                        xaxis_title="التردد",
                        yaxis_title="القدرة",
                        height=350
                    )
                    st.plotly_chart(
                        fig_psd, use_container_width=True
                    )

                if spec['binary_cycles']:
                    st.subheader("دورات ثنائية (High/Low)")
                    df_bc = pd.DataFrame(spec['binary_cycles'])
                    fig_bc = px.bar(
                        df_bc, x='period_rounds', y='power',
                        title="دورات في التسلسل الثنائي",
                        color='power',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(
                        fig_bc, use_container_width=True
                    )

            # ── نقاط التحول ──────────────────────────────
            with t_chg:
                chg = analysis['changepoints']
                st.subheader("🔄 نقاط التحول والأنظمة")

                ca, cb, cc = st.columns(3)
                ca.metric("الحالة الراهنة",
                          chg['current_state'])
                cb.metric("% مرتفع حديثاً",
                          f"{chg['recent_pct']}%")
                cc.metric("% مرتفع تاريخياً",
                          f"{chg['hist_pct']}%")

                if chg['positions']:
                    fig_chg = make_subplots(rows=2, cols=1)
                    fig_chg.add_trace(
                        go.Scatter(
                            x=chg['positions'],
                            y=chg['pct_high'],
                            mode='lines+markers',
                            name='% مرتفع',
                            line=dict(color='steelblue')
                        ), row=1, col=1
                    )
                    fig_chg.add_hline(
                        y=chg['hist_pct'],
                        line_dash="dash",
                        line_color="red",
                        row=1, col=1,
                        annotation_text="المتوسط التاريخي"
                    )
                    fig_chg.add_trace(
                        go.Scatter(
                            x=chg['positions'],
                            y=chg['means'],
                            mode='lines',
                            name='متوسط القيمة',
                            line=dict(color='green')
                        ), row=2, col=1
                    )
                    for cp in chg['changepoints']:
                        for row in [1, 2]:
                            fig_chg.add_vline(
                                x=cp['position'],
                                line_dash="dot",
                                line_color="red",
                                row=row, col=1
                            )
                    fig_chg.update_layout(
                        height=500,
                        title="تطور النظام عبر الزمن"
                    )
                    st.plotly_chart(
                        fig_chg, use_container_width=True
                    )

                if chg['changepoints']:
                    st.subheader("نقاط التحول المكتشفة")
                    df_cp = pd.DataFrame(chg['changepoints'])
                    st.dataframe(
                        df_cp, use_container_width=True,
                        hide_index=True
                    )

            # ── Markov ───────────────────────────────────
            with t_mk:
                mk = analysis['markov']
                st.subheader("🔗 سلاسل ماركوف المتقدمة")

                # مصفوفة الانتقال
                matrix = mk['matrix1']
                df_mat = pd.DataFrame(matrix).T
                df_mat = df_mat.rename(
                    index=CAT_LABELS, columns=CAT_LABELS
                )
                fig_heat = px.imshow(
                    df_mat.values,
                    x=list(df_mat.columns),
                    y=list(df_mat.index),
                    color_continuous_scale='RdYlGn',
                    title="مصفوفة الانتقال (Markov-1)",
                    zmin=0, zmax=1,
                    text_auto='.2f'
                )
                fig_heat.update_layout(height=400)
                st.plotly_chart(fig_heat, use_container_width=True)

                # التوزيع الثابت
                st.subheader("التوزيع الثابت (Stationary Distribution)")
                stat_df = pd.DataFrame({
                    'الفئة'     : [CAT_LABELS[c]
                                   for c in CAT_ORDER],
                    'الاحتمال'  : [
                        mk['stationary'].get(c, 0)
                        for c in CAT_ORDER
                    ]
                })
                fig_stat = px.pie(
                    stat_df, values='الاحتمال', names='الفئة',
                    color='الفئة',
                    title="التوزيع الثابت لـ Markov"
                )
                st.plotly_chart(fig_stat, use_container_width=True)

                # الأنماط
                for order, key, min_prob in [
                    ('2','order2',0.55),
                    ('3','order3',0.60),
                    ('4','order4',0.65)
                ]:
                    st.subheader(f"أنماط Markov-{order}")
                    pats = mk.get(key, [])
                    if pats:
                        df_p = pd.DataFrame(pats)
                        df_p['probability'] = df_p[
                            'probability'
                        ].apply(lambda x: f"{x*100:.1f}%")
                        st.dataframe(
                            df_p.drop(
                                columns=['next_cat'],
                                errors='ignore'
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info(
                            f"لا أنماط بثقة >= {int(min_prob*100)}%"
                        )

            # ── التوزيع ──────────────────────────────────
            with t_dist:
                dist = analysis['distribution']
                st.subheader("📊 تحليل التوزيع")

                ca,cb,cc = st.columns(3)
                ca.metric("الانحراف (Skewness)",
                          f"{dist['skewness']:.3f}")
                cb.metric("التفرطح (Kurtosis)",
                          f"{dist['kurtosis']:.3f}")
                cc.metric("شوائب (Anomalies)",
                          str(dist['n_anomalies']))

                # مقارنة الاحتمالات
                df_probs = pd.DataFrame({
                    'العتبة': [f">={t}x" for t in dist['thresholds']],
                    'الفعلي': [
                        round(e*100,1)
                        for e in dist['empirical_probs']
                    ],
                    'النظري': [
                        round(t*100,1)
                        for t in dist['theoretical_probs']
                    ],
                    'الانحراف %': [
                        round(d*100,2)
                        for d in dist['deviations']
                    ]
                })

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name='فعلي',
                    x=df_probs['العتبة'],
                    y=df_probs['الفعلي'],
                    marker_color='steelblue'
                ))
                fig_comp.add_trace(go.Bar(
                    name='نظري',
                    x=df_probs['العتبة'],
                    y=df_probs['النظري'],
                    marker_color='tomato',
                    opacity=0.7
                ))
                fig_comp.update_layout(
                    barmode='group',
                    title="مقارنة الاحتمالات الفعلية بالنظرية",
                    height=380
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                st.dataframe(
                    df_probs, use_container_width=True,
                    hide_index=True
                )
                st.info(
                    f"اختبار KS: stat={dist['ks_statistic']} | "
                    f"p={dist['ks_pvalue']}"
                )

            # ── التاريخ ──────────────────────────────────
            with t_hist:
                st.subheader("📜 آخر 100 جولة")
                last_n  = min(100, len(raw_data))
                last100 = raw_data[-last_n:]
                cats100 = [categorize(v) for v in last100]
                colors100 = [CAT_COLORS[c] for c in cats100]

                fig_h100 = go.Figure()
                fig_h100.add_trace(go.Bar(
                    x=list(range(len(last100))),
                    y=last100,
                    marker_color=colors100,
                    name='القيمة',
                    hovertemplate=(
                        "جولة %{x}<br>"
                        "القيمة: %{y:.2f}x<extra></extra>"
                    )
                ))
                for thr, color in [
                    (2.0,'blue'),(5.0,'green'),(10.0,'purple')
                ]:
                    fig_h100.add_hline(
                        y=thr, line_dash="dash",
                        line_color=color,
                        annotation_text=f"{thr}x"
                    )
                fig_h100.update_layout(
                    title=f"آخر {last_n} جولة",
                    xaxis_title="الجولة",
                    yaxis_title="المضاعف",
                    height=420
                )
                st.plotly_chart(
                    fig_h100, use_container_width=True
                )

                # تسلسل الفئات
                st.subheader("تسلسل الفئات — آخر 30 جولة")
                last30 = raw_data[-30:]
                cats30 = [categorize(v) for v in last30]
                fig_seq = go.Figure()
                for i, (val, cat) in enumerate(
                    zip(last30, cats30)
                ):
                    fig_seq.add_trace(go.Scatter(
                        x=[i], y=[val],
                        mode='markers+text',
                        marker=dict(
                            color=CAT_COLORS[cat],
                            size=20,
                            symbol='circle'
                        ),
                        text=[f"{val:.1f}"],
                        textposition='top center',
                        showlegend=False
                    ))
                fig_seq.update_layout(
                    title="تسلسل القيم — اللون يدل على الفئة",
                    height=300
                )
                st.plotly_chart(
                    fig_seq, use_container_width=True
                )

            # ── الاستنتاج ────────────────────────────────
            st.markdown("---")
            st.header("📝 الاستنتاج العلمي")

            H   = float(analysis['hurst']['hurst'])
            vol = analysis['volatility']
            chg = analysis['changepoints']

            findings = []
            if H > 0.6:
                findings.append(
                    f"📈 H={H:.3f}: تسلسل متجه — الأنماط مستمرة"
                )
            elif H < 0.4:
                findings.append(
                    f"🔄 H={H:.3f}: انعكاس للمتوسط — بعد الارتفاع ينخفض"
                )
            if vol['volatility_clustering']:
                findings.append(
                    "📊 تجمع التقلبات: فترات هادئة تسبق القفزات"
                )
            if analysis['spectral']['has_pattern']:
                findings.append("📡 دورة زمنية قوية مكتشفة")

            strong_mk = [
                p for p in analysis['markov']['order3']
                if p['probability'] >= 0.65
            ]
            if strong_mk:
                findings.append(
                    f"🔗 {len(strong_mk)} نمط Markov-3 بثقة >= 65%"
                )

            urgent_gaps = [
                f"{k}: {v['zone']}"
                for k, v in analysis['gaps'].items()
                if v['due_ratio'] >= 1.5
            ]
            if urgent_gaps:
                findings.append(
                    f"⏱️ قفزات متأخرة: {', '.join(urgent_gaps)}"
                )

            for f in findings:
                st.success(f"✅ {f}")

            if not findings:
                st.info("لا أنماط قوية — التسلسل يقترب من العشوائية")

            # تحميل التقرير
            st.markdown("---")
            report = to_python({
                'total_samples': n,
                'prediction'   : {
                    'category' : pred['final_cat'],
                    'label'    : pred['final_label'],
                    'est_value': pred['est_value'],
                    'prob_high': pred['prob_high'],
                    'confidence':pred['final_conf'],
                    'rounds_to_jump': pred['rounds_to_jump']
                },
                'hurst'        : analysis['hurst'],
                'volatility'   : {
                    k: v for k, v in analysis['volatility'].items()
                    if k != 'vol_series'
                },
                'changepoints' : analysis['changepoints']['changepoints'],
                'top_patterns' : {
                    'order2': analysis['markov']['order2'][:5],
                    'order3': analysis['markov']['order3'][:5],
                    'order4': analysis['markov']['order4'][:5]
                },
                'gaps_summary' : {
                    k: {
                        kk: vv for kk, vv in v.items()
                        if kk not in ['gaps','prob_appearances']
                    }
                    for k, v in analysis['gaps'].items()
                },
                'key_findings' : findings
            })

            st.download_button(
                "📥 تحميل التقرير الكامل (JSON)",
                data=json.dumps(
                    report, ensure_ascii=False, indent=2
                ),
                file_name="crash_deep_analysis.json",
                mime="application/json"
            )

st.markdown("---")
st.caption(
    "🎓 مشروع تخرج | Hurst • GARCH • Markov-4 • "
    "Welch PSD • Gap Analysis • Bayesian • Ensemble"
)
