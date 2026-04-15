# -*- coding: utf-8 -*-
# app.py — محلل أنماط Crash الذكي

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, welch
from collections import Counter, defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🔬 Crash Pattern Intelligence",
    page_icon="🔬",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════
#                    دوال مساعدة
# ══════════════════════════════════════════════════════════════
def to_python(obj):
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(i) for i in obj]
    return obj


def safe_linregress(x, y):
    """انحدار خطي آمن بدون statsmodels"""
    try:
        slope, intercept, r, p, se = stats.linregress(x, y)
        return float(slope), float(intercept), float(r**2)
    except Exception:
        return 0.0, 0.0, 0.0


# ══════════════════════════════════════════════════════════════
#         1. محرك اكتشاف الأنماط الذكي
# ══════════════════════════════════════════════════════════════
class SmartPatternEngine:
    """
    يكتشف الأنماط الحقيقية التي تسبق القفزات:
    - بعد كم رقم صغير تأتي القفزة؟
    - ما الأرقام التي تسبق القفزات مباشرة؟
    - هل هناك قواعد "إذا ظهر X ثم Y بعد N جولة"؟
    - ما التسلسلات المتكررة قبل القفزات؟
    """

    def __init__(self, data: list):
        self.raw    = np.array(data, dtype=float)
        self.n      = len(data)
        self.data   = list(data)

    # ─────────────────────────────────────────────────────────
    # 1.1 قانون الفجوات — بعد كم رقم صغير تأتي القفزة؟
    # ─────────────────────────────────────────────────────────
    def discover_gap_laws(self) -> dict:
        """
        يكتشف قوانين من نوع:
        "بعد X رقم < 2 تأتي قفزة >= Y بنسبة Z%"
        """
        arr     = self.raw
        results = {}

        for jump_thr in [2.0, 3.0, 5.0, 10.0]:
            jump_positions = np.where(arr >= jump_thr)[0]
            if len(jump_positions) < 3:
                continue

            # احسب الفجوة (عدد الأرقام الصغيرة) قبل كل قفزة
            gaps_before = []
            for pos in jump_positions:
                if pos == 0:
                    continue
                count = 0
                i     = int(pos) - 1
                while i >= 0 and arr[i] < jump_thr:
                    count += 1
                    i     -= 1
                gaps_before.append(int(count))

            if not gaps_before:
                continue

            gaps_arr = np.array(gaps_before)

            # إحصاءات الفجوات
            mean_g   = float(np.mean(gaps_arr))
            median_g = float(np.median(gaps_arr))
            std_g    = float(np.std(gaps_arr))
            p25      = float(np.percentile(gaps_arr, 25))
            p75      = float(np.percentile(gaps_arr, 75))

            # توزيع الفجوات
            gap_dist = Counter(gaps_before)
            most_common_gap = gap_dist.most_common(1)[0]

            # قانون النطاق الأكثر احتمالاً
            # P(قفزة | الفجوة الحالية = k)
            gap_probs = {}
            for k in range(0, int(p75) + 5):
                count_k     = sum(1 for g in gaps_before if g == k)
                total_occ_k = sum(
                    1 for i in range(len(arr))
                    if self._count_consecutive_low(arr, i, jump_thr) == k
                )
                if total_occ_k >= 2:
                    gap_probs[k] = round(
                        min(float(count_k / total_occ_k), 1.0), 4
                    )

            results[f'>={jump_thr}x'] = {
                'n_jumps'        : int(len(jump_positions)),
                'mean_gap'       : round(mean_g, 1),
                'median_gap'     : round(median_g, 1),
                'std_gap'        : round(std_g, 1),
                'p25_gap'        : round(p25, 1),
                'p75_gap'        : round(p75, 1),
                'most_common_gap': {
                    'gap'  : int(most_common_gap[0]),
                    'count': int(most_common_gap[1])
                },
                'gap_distribution': {
                    str(k): int(v)
                    for k, v in sorted(gap_dist.items())
                },
                'conditional_probs': gap_probs,
                'gaps_before_jumps': gaps_before[:30]
            }

        return results

    def _count_consecutive_low(self, arr, pos, thr):
        """عد الأرقام الصغيرة المتتالية حتى pos"""
        count = 0
        i     = int(pos) - 1
        while i >= 0 and arr[i] < thr:
            count += 1
            i     -= 1
        return count

    # ─────────────────────────────────────────────────────────
    # 1.2 اكتشاف المُحفِّزات — ما الذي يسبق القفزات؟
    # ─────────────────────────────────────────────────────────
    def discover_triggers(self) -> dict:
        """
        يكتشف قوانين من نوع:
        "إذا ظهر رقم >= X فبعد Y جولة تأتي قفزة >= Z بنسبة W%"
        مثل: "إذا ظهر 5x فبعد 7 جولات تأتي قفزة >= 3x بنسبة 65%"
        """
        arr     = self.raw
        n       = self.n
        results = []

        trigger_thrs = [1.5, 2.0, 3.0, 5.0, 10.0]
        target_thrs  = [2.0, 3.0, 5.0, 10.0]
        delays       = list(range(1, 16))

        for trig_thr in trigger_thrs:
            for tgt_thr in target_thrs:
                if trig_thr >= tgt_thr:
                    continue
                for delay in delays:
                    # إيجاد مواضع المحفز
                    trigger_pos = np.where(arr >= trig_thr)[0]
                    hits        = 0
                    total       = 0

                    for pos in trigger_pos:
                        target_pos = int(pos) + delay
                        if target_pos < n:
                            total += 1
                            if arr[target_pos] >= tgt_thr:
                                hits += 1

                    if total >= 5:
                        prob = float(hits / total)
                        # الاحتمال الأساسي
                        base = float(np.mean(arr >= tgt_thr))
                        lift = prob - base

                        if prob >= 0.55 and lift > 0.05:
                            results.append({
                                'trigger'     : f">={trig_thr}x",
                                'delay'       : int(delay),
                                'target'      : f">={tgt_thr}x",
                                'probability' : round(prob, 4),
                                'base_rate'   : round(base, 4),
                                'lift'        : round(lift, 4),
                                'hits'        : int(hits),
                                'total'       : int(total),
                                'rule'        : (
                                    f"بعد {trig_thr}x ⟹ "
                                    f"بعد {delay} جولة ⟹ "
                                    f"{tgt_thr}x "
                                    f"({prob*100:.1f}%)"
                                )
                            })

        # ترتيب بالأهمية
        results.sort(
            key=lambda x: (x['lift'], x['probability']),
            reverse=True
        )
        return {'rules': results[:20]}

    # ─────────────────────────────────────────────────────────
    # 1.3 اكتشاف التسلسلات قبل القفزات
    # ─────────────────────────────────────────────────────────
    def discover_pre_jump_sequences(self) -> dict:
        """
        يكتشف التسلسلات المتكررة قبل القفزات
        مثل: "L L L H" أو "M L L" تسبق قفزة >= 5x
        """
        arr        = self.raw
        n          = self.n
        window     = 4  # نافذة التسلسل
        results    = {}

        def encode(v):
            if v < 1.5:  return 'VL'
            if v < 2.0:  return 'L'
            if v < 3.0:  return 'M'
            if v < 5.0:  return 'H'
            if v < 10.0: return 'VH'
            return 'EX'

        labels = {
            'VL':'<1.5','L':'1.5-2','M':'2-3',
            'H':'3-5','VH':'5-10','EX':'>10'
        }

        for jump_thr in [3.0, 5.0, 10.0]:
            jump_pos = np.where(arr >= jump_thr)[0]
            if len(jump_pos) < 3:
                continue

            # جمع التسلسلات قبل كل قفزة
            seq_counter    = Counter()
            all_seq_counter= Counter()

            for pos in jump_pos:
                start = int(pos) - window
                if start < 0:
                    continue
                seq = tuple(
                    encode(arr[i])
                    for i in range(start, int(pos))
                )
                seq_counter[seq] += 1

            # جمع كل التسلسلات في البيانات
            for i in range(n - window):
                seq = tuple(
                    encode(arr[i + j])
                    for j in range(window)
                )
                all_seq_counter[seq] += 1

            # احتمال كل تسلسل أن يسبق قفزة
            patterns = []
            for seq, count_before_jump in seq_counter.most_common(15):
                total_occ = all_seq_counter.get(seq, 0)
                if total_occ >= 3:
                    prob = float(count_before_jump / total_occ)
                    base = float(len(jump_pos) / max(n - window, 1))
                    lift = prob - base
                    if prob >= 0.40:
                        seq_label = ' → '.join(
                            labels.get(s, s) for s in seq
                        )
                        patterns.append({
                            'sequence'     : seq_label,
                            'probability'  : round(prob, 4),
                            'base_rate'    : round(base, 4),
                            'lift'         : round(lift, 4),
                            'seen_before_jump': int(count_before_jump),
                            'total_seen'   : int(total_occ)
                        })

            patterns.sort(
                key=lambda x: x['probability'], reverse=True
            )
            results[f'>={jump_thr}x'] = patterns[:10]

        return results

    # ─────────────────────────────────────────────────────────
    # 1.4 قانون التراكم — بعد كم قفزة تأتي القفزة الكبيرة؟
    # ─────────────────────────────────────────────────────────
    def discover_accumulation_law(self) -> dict:
        """
        يكتشف:
        - كم جولة بين قفزتين متتاليتين في المتوسط؟
        - هل توجد دورة منتظمة؟
        - بعد قفزة كبيرة كم جولة حتى التالية؟
        """
        arr     = self.raw
        results = {}

        for thr in [2.0, 3.0, 5.0, 10.0, 20.0]:
            positions = np.where(arr >= thr)[0]
            if len(positions) < 3:
                continue

            gaps = np.diff(positions).tolist()
            gaps = [int(g) for g in gaps]

            # إحصاءات
            g_arr  = np.array(gaps)
            mean_g = float(np.mean(g_arr))
            std_g  = float(np.std(g_arr))
            med_g  = float(np.median(g_arr))

            # هل الفجوات منتظمة؟
            cv = std_g / (mean_g + 1e-9)  # معامل التباين

            # آخر فجوة حالية
            last_pos     = int(positions[-1])
            current_gap  = int(len(arr) - 1 - last_pos)
            due_ratio    = float(current_gap / (mean_g + 1e-9))

            # توقع متى تأتي التالية
            expected_next = max(0.0, mean_g - current_gap)

            # هل الفجوة الحالية كبيرة جداً؟
            z_score = float(
                (current_gap - mean_g) / (std_g + 1e-9)
            )

            # تحليل الأنماط في الفجوات
            # هل الفجوات الكبيرة تتبعها فجوات صغيرة؟
            long_short = []
            for i in range(len(gaps) - 1):
                long_short.append((gaps[i], gaps[i+1]))

            corr_gaps = 0.0
            if len(gaps) > 2:
                corr_gaps = float(np.corrcoef(
                    gaps[:-1], gaps[1:]
                )[0, 1])

            results[f'>={thr}x'] = {
                'count'          : int(len(positions)),
                'mean_gap'       : round(mean_g, 1),
                'std_gap'        : round(std_g, 1),
                'median_gap'     : round(med_g, 1),
                'cv'             : round(cv, 3),
                'is_regular'     : bool(cv < 0.5),
                'current_gap'    : int(current_gap),
                'due_ratio'      : round(due_ratio, 2),
                'z_score'        : round(z_score, 2),
                'expected_next_in': round(expected_next, 1),
                'gap_autocorr'   : round(corr_gaps, 3),
                'last_gaps'      : gaps[-10:],
                'zone'           : (
                    '🔴 متأخر جداً!' if z_score > 1.5
                    else '🟠 متأخر'   if z_score > 0.5
                    else '🟡 في الوقت'if z_score > -0.5
                    else '🟢 مبكر'
                )
            }

        return results

    # ─────────────────────────────────────────────────────────
    # 1.5 قانون ما بعد القفزة — ماذا يأتي بعدها؟
    # ─────────────────────────────────────────────────────────
    def discover_post_jump_law(self) -> dict:
        """
        بعد قفزة >= X ماذا يأتي في الجولات 1,2,3,...,10؟
        """
        arr     = self.raw
        n       = self.n
        results = {}

        for jump_thr in [2.0, 3.0, 5.0, 10.0]:
            jump_pos = np.where(arr >= jump_thr)[0]
            if len(jump_pos) < 3:
                continue

            # ماذا يأتي بعد k جولة؟
            post_stats = {}
            for k in range(1, 11):
                values_after = []
                for pos in jump_pos:
                    next_pos = int(pos) + k
                    if next_pos < n:
                        values_after.append(float(arr[next_pos]))

                if values_after:
                    va = np.array(values_after)
                    post_stats[k] = {
                        'mean'      : round(float(np.mean(va)), 2),
                        'median'    : round(float(np.median(va)), 2),
                        'pct_high'  : round(float(np.mean(va>=2.0))*100, 1),
                        'pct_low'   : round(float(np.mean(va<2.0))*100, 1),
                        'pct_very_low': round(float(np.mean(va<1.5))*100,1)
                    }

            # عدد الأرقام الصغيرة المتتالية بعد القفزة
            cooling_counts = []
            for pos in jump_pos:
                count = 0
                i     = int(pos) + 1
                while i < n and arr[i] < jump_thr:
                    count += 1
                    i     += 1
                cooling_counts.append(int(count))

            cc_arr = np.array(cooling_counts)
            results[f'>={jump_thr}x'] = {
                'post_round_stats'  : post_stats,
                'cooling_mean'      : round(float(np.mean(cc_arr)), 1),
                'cooling_median'    : round(float(np.median(cc_arr)), 1),
                'cooling_std'       : round(float(np.std(cc_arr)), 1),
                'cooling_counts'    : cooling_counts[:20]
            }

        return results

    # ─────────────────────────────────────────────────────────
    # 1.6 اكتشاف قواعد "إذا-ثم" المعقدة
    # ─────────────────────────────────────────────────────────
    def discover_if_then_rules(self) -> dict:
        """
        قواعد مركبة من نوع:
        "إذا ظهر X ثم Y ثم Z → بعد N جولة تأتي قفزة"
        """
        arr    = self.raw
        n      = self.n
        rules  = []

        def bucket(v):
            if v < 1.5:  return 'tiny'
            if v < 2.0:  return 'small'
            if v < 3.0:  return 'med'
            if v < 5.0:  return 'big'
            return 'huge'

        label = {
            'tiny':'<1.5','small':'1.5-2',
            'med':'2-3','big':'3-5','huge':'>5'
        }

        # أنماط ثلاثية → قفزة بعد k
        for k in range(1, 8):
            pattern_counts = defaultdict(int)
            pattern_hits   = defaultdict(int)

            for i in range(n - 3 - k):
                pat = (
                    bucket(arr[i]),
                    bucket(arr[i+1]),
                    bucket(arr[i+2])
                )
                pattern_counts[pat] += 1
                if arr[i + 3 + k - 1] >= 3.0:
                    pattern_hits[pat] += 1

            base = float(np.mean(arr >= 3.0))

            for pat, total in pattern_counts.items():
                if total < 4:
                    continue
                hits = pattern_hits.get(pat, 0)
                prob = float(hits / total)
                lift = prob - base

                if prob >= 0.55 and lift > 0.08:
                    pat_str = (
                        f"{label[pat[0]]} → "
                        f"{label[pat[1]]} → "
                        f"{label[pat[2]]}"
                    )
                    rules.append({
                        'pattern'    : pat_str,
                        'delay'      : int(k),
                        'probability': round(prob, 4),
                        'lift'       : round(lift, 4),
                        'hits'       : int(hits),
                        'total'      : int(total),
                        'rule_text'  : (
                            f"إذا رأيت [{pat_str}] "
                            f"→ توقع قفزة >= 3x "
                            f"بعد {k} جولة "
                            f"({prob*100:.1f}%)"
                        )
                    })

        rules.sort(
            key=lambda x: (x['lift'], x['probability']),
            reverse=True
        )
        return {'if_then_rules': rules[:15]}

    # ─────────────────────────────────────────────────────────
    # 1.7 تحليل Hurst وطبيعة التسلسل
    # ─────────────────────────────────────────────────────────
    def hurst_analysis(self) -> dict:
        ts   = np.log(np.maximum(self.raw, 1.0))
        lags = list(range(2, min(20, self.n // 4)))
        tau  = []

        for lag in lags:
            d = ts[lag:] - ts[:-lag]
            tau.append(float(np.std(d)))

        if len(tau) < 3:
            return {'H': 0.5, 'interpretation': 'بيانات غير كافية'}

        slope, intercept, r2 = safe_linregress(
            np.log(lags), np.log(np.array(tau) + 1e-9)
        )
        H = max(0.0, min(1.0, float(slope)))

        return {
            'H'             : round(H, 4),
            'r_squared'     : round(r2, 4),
            'lags'          : lags,
            'tau'           : [round(t, 4) for t in tau],
            'log_lags'      : np.log(lags).tolist(),
            'log_tau'       : np.log(
                np.array(tau) + 1e-9
            ).tolist(),
            'trend_y'       : (
                np.array(np.log(lags)) * slope + intercept
            ).tolist(),
            'interpretation': (
                f"📈 H={H:.3f} متجه — الأنماط تستمر"
                if H > 0.6
                else f"🔄 H={H:.3f} معكوس — يرتد للمتوسط"
                if H < 0.4
                else f"🎲 H={H:.3f} شبه عشوائي"
            )
        }

    # ─────────────────────────────────────────────────────────
    # 1.8 التحليل الطيفي
    # ─────────────────────────────────────────────────────────
    def spectral_analysis(self) -> dict:
        log_data = np.log(np.maximum(self.raw, 1.0))
        n        = self.n

        try:
            nperseg = min(64, n // 4)
            freqs, psd = welch(
                log_data - log_data.mean(), nperseg=nperseg
            )
            psd   = psd.tolist()
            freqs = freqs.tolist()
        except Exception:
            freqs = []
            psd   = []

        # FFT
        fft_vals = np.abs(fft(log_data - log_data.mean()))
        half     = n // 2
        fft_h    = fft_vals[:half]
        freq_h   = fftfreq(n)[:half]

        top_idx  = np.argsort(fft_h)[-8:][::-1]
        cycles   = []
        for idx in top_idx:
            f = float(freq_h[idx])
            if f > 0:
                period = float(1.0 / f)
                if 2 <= period <= n // 2:
                    cycles.append({
                        'period' : round(period, 1),
                        'power'  : round(float(fft_h[idx]), 4),
                        'rel_pow': round(
                            float(fft_h[idx]) /
                            (fft_h.max() + 1e-9), 4
                        )
                    })

        dom = float(fft_h.max() / (fft_h.mean() + 1e-9))
        return {
            'cycles'     : cycles[:6],
            'dominance'  : round(dom, 2),
            'has_cycle'  : bool(dom > 8),
            'freqs'      : freqs,
            'psd'        : psd
        }

    def run_all(self) -> dict:
        return {
            'gap_laws'      : self.discover_gap_laws(),
            'triggers'      : self.discover_triggers(),
            'pre_jump_seqs' : self.discover_pre_jump_sequences(),
            'accumulation'  : self.discover_accumulation_law(),
            'post_jump'     : self.discover_post_jump_law(),
            'if_then'       : self.discover_if_then_rules(),
            'hurst'         : self.hurst_analysis(),
            'spectral'      : self.spectral_analysis()
        }


# ══════════════════════════════════════════════════════════════
#         2. محرك التنبؤ الذكي
# ══════════════════════════════════════════════════════════════
class SmartPredictor:

    def __init__(self, data: list, patterns: dict):
        self.raw      = np.array(data, dtype=float)
        self.data     = list(data)
        self.n        = len(data)
        self.patterns = patterns

    def _current_context(self) -> dict:
        arr = self.raw

        # تسلسل منخفض حالي
        low_streak = 0
        for v in reversed(arr):
            if v < 2.0: low_streak += 1
            else:        break

        # تسلسل مرتفع حالي
        high_streak = 0
        for v in reversed(arr):
            if v >= 2.0: high_streak += 1
            else:         break

        # آخر ظهور لكل عتبة
        def last_seen(thr):
            for i in range(len(arr)-1, -1, -1):
                if arr[i] >= thr:
                    return int(len(arr) - 1 - i)
            return int(len(arr))

        def encode(v):
            if v < 1.5:  return 'tiny'
            if v < 2.0:  return 'small'
            if v < 3.0:  return 'med'
            if v < 5.0:  return 'big'
            return 'huge'

        return {
            'low_streak'   : int(low_streak),
            'high_streak'  : int(high_streak),
            'last_value'   : float(arr[-1]),
            'last_3'       : [float(v) for v in arr[-3:]],
            'last_5'       : [float(v) for v in arr[-5:]],
            'pattern_3'    : tuple(encode(v) for v in arr[-3:]),
            'pattern_4'    : tuple(encode(v) for v in arr[-4:]),
            'recent_avg10' : float(np.mean(arr[-10:])),
            'hist_avg'     : float(np.mean(arr)),
            'last_seen_2x' : last_seen(2.0),
            'last_seen_3x' : last_seen(3.0),
            'last_seen_5x' : last_seen(5.0),
            'last_seen_10x': last_seen(10.0)
        }

    def _apply_gap_law(self, ctx: dict) -> dict:
        """تطبيق قانون الفجوات"""
        gap_laws = self.patterns['gap_laws']
        signals  = []

        for thr_key, info in gap_laws.items():
            thr   = float(thr_key.replace('>=','').replace('x',''))
            ls    = int(ctx['low_streak'])
            cprob = info['conditional_probs'].get(ls, None)

            if cprob is not None and cprob >= 0.45:
                mean_g = info['mean_gap']
                ls_normalized = ls / (mean_g + 1e-9)
                signals.append({
                    'threshold'   : thr_key,
                    'current_gap' : ls,
                    'mean_gap'    : mean_g,
                    'prob'        : cprob,
                    'normalized'  : round(ls_normalized, 2)
                })

        # أقوى إشارة
        if signals:
            best = max(signals, key=lambda x: x['prob'])
            return {
                'signal'     : True,
                'probability': best['prob'],
                'details'    : best,
                'all_signals': signals
            }
        return {'signal': False, 'probability': 0.40}

    def _apply_trigger_rules(self, ctx: dict) -> dict:
        """تطبيق قواعد المحفزات"""
        rules    = self.patterns['triggers']['rules']
        arr      = self.raw
        n        = self.n
        matching = []

        for rule in rules:
            trig_thr = float(
                rule['trigger'].replace('>=','').replace('x','')
            )
            delay    = int(rule['delay'])
            # هل يوجد محفز قبل delay جولة؟
            check_pos = n - delay
            if check_pos >= 0 and arr[check_pos] >= trig_thr:
                matching.append(rule)

        if matching:
            best = max(matching, key=lambda x: x['probability'])
            return {
                'signal'      : True,
                'probability' : float(best['probability']),
                'lift'        : float(best['lift']),
                'rule'        : best['rule'],
                'n_matching'  : int(len(matching))
            }
        return {'signal': False, 'probability': 0.38}

    def _apply_if_then(self, ctx: dict) -> dict:
        """تطبيق قواعد إذا-ثم"""
        rules   = self.patterns['if_then']['if_then_rules']
        arr     = self.raw

        def bucket(v):
            if v < 1.5:  return 'tiny'
            if v < 2.0:  return 'small'
            if v < 3.0:  return 'med'
            if v < 5.0:  return 'big'
            return 'huge'

        label = {
            'tiny':'<1.5','small':'1.5-2',
            'med':'2-3','big':'3-5','huge':'>5'
        }

        matching = []
        if len(arr) >= 3:
            cur_pattern = (
                f"{label[bucket(arr[-3])]} → "
                f"{label[bucket(arr[-2])]} → "
                f"{label[bucket(arr[-1])]}"
            )
            for rule in rules:
                if rule['pattern'] == cur_pattern and rule['delay'] == 1:
                    matching.append(rule)

        if matching:
            best = max(matching, key=lambda x: x['probability'])
            return {
                'signal'     : True,
                'probability': float(best['probability']),
                'rule_text'  : best['rule_text'],
                'lift'       : float(best['lift'])
            }
        return {'signal': False, 'probability': 0.42}

    def _apply_accumulation(self, ctx: dict) -> dict:
        """تطبيق قانون التراكم"""
        acc     = self.patterns['accumulation']
        signals = []

        for thr_key, info in acc.items():
            thr = float(
                thr_key.replace('>=','').replace('x','')
            )
            if thr > 5.0:
                continue

            z      = float(info['z_score'])
            due    = float(info['due_ratio'])
            last_s = int(ctx.get(
                f"last_seen_{int(thr)}x",
                info['current_gap']
            ))

            if z > 1.0 or due >= 1.3:
                prob = min(0.40 + z * 0.15 + due * 0.10, 0.82)
                signals.append({
                    'threshold': thr_key,
                    'z_score'  : z,
                    'due'      : due,
                    'zone'     : info['zone'],
                    'prob'     : round(prob, 3)
                })

        if signals:
            best = max(signals, key=lambda x: x['prob'])
            return {
                'signal'     : True,
                'probability': best['prob'],
                'details'    : best
            }
        return {'signal': False, 'probability': 0.40}

    def _apply_hurst(self, ctx: dict) -> dict:
        """تطبيق Hurst"""
        H      = float(self.patterns['hurst']['H'])
        last_v = float(ctx['last_value'])

        if H < 0.4:
            # معكوس: بعد منخفض → ارتفاع
            if last_v < 1.5:
                return {
                    'probability': 0.62,
                    'logic'      : f"H={H:.3f} معكوس + آخر قيمة منخفضة"
                }
        elif H > 0.6:
            # متجه
            recent = float(np.mean(self.raw[-5:]))
            if recent > float(np.mean(self.raw)):
                return {
                    'probability': 0.60,
                    'logic'      : f"H={H:.3f} متجه + اتجاه صاعد"
                }
        return {'probability': 0.45}

    def full_predict(self) -> dict:
        """التنبؤ الكامل"""
        ctx = self._current_context()

        # تشغيل كل المحركات
        gap_sig  = self._apply_gap_law(ctx)
        trig_sig = self._apply_trigger_rules(ctx)
        ifthen   = self._apply_if_then(ctx)
        acc_sig  = self._apply_accumulation(ctx)
        hurst_sig= self._apply_hurst(ctx)

        # أوزان ديناميكية
        weights = {
            'gap'    : 0.30,
            'trigger': 0.25,
            'ifthen' : 0.20,
            'acc'    : 0.15,
            'hurst'  : 0.10
        }

        probs = {
            'gap'    : float(gap_sig.get('probability', 0.40)),
            'trigger': float(trig_sig.get('probability', 0.38)),
            'ifthen' : float(ifthen.get('probability', 0.42)),
            'acc'    : float(acc_sig.get('probability', 0.40)),
            'hurst'  : float(hurst_sig.get('probability', 0.45))
        }

        # تعزيز إذا وُجدت إشارات قوية
        signals_active = sum([
            bool(gap_sig.get('signal', False)),
            bool(trig_sig.get('signal', False)),
            bool(ifthen.get('signal', False)),
            bool(acc_sig.get('signal', False))
        ])

        # الاحتمال المرجح
        prob_high = sum(
            weights[k] * probs[k] for k in weights
        )

        # تعديل بعدد الإشارات النشطة
        if signals_active >= 3:
            prob_high = min(prob_high + 0.10, 0.88)
        elif signals_active >= 2:
            prob_high = min(prob_high + 0.05, 0.82)

        # تعديل بالتسلسل المنخفض
        ls = int(ctx['low_streak'])
        if ls >= 6:
            prob_high = min(prob_high + 0.12, 0.88)
        elif ls >= 4:
            prob_high = min(prob_high + 0.06, 0.82)

        # تحليل ما بعد القفزة
        post = self.patterns['post_jump']
        post_signal = None
        if '>= 5.0x' in post or '>=5.0x' in post:
            key = '>= 5.0x' if '>= 5.0x' in post else '>=5.0x'
            pj  = post[key]
            ls5 = int(ctx['last_seen_5x'])
            cool= float(pj.get('cooling_mean', 5.0))
            if ls5 < cool:
                prob_high = max(prob_high - 0.08, 0.15)
                post_signal = (
                    f"تبريد بعد قفزة 5x: {ls5}/{cool:.0f} جولة"
                )

        prob_high = float(np.clip(prob_high, 0.05, 0.95))

        # توقع الجولات
        acc = self.patterns['accumulation']
        jump_forecast = {}
        for thr_key, info in acc.items():
            exp = float(info['expected_next_in'])
            jump_forecast[thr_key] = {
                'expected_in': max(0.0, round(exp, 1)),
                'zone'       : info['zone'],
                'z_score'    : info['z_score'],
                'current_gap': info['current_gap']
            }

        # الحكم النهائي
        if prob_high >= 0.70:
            verdict = '🟢 ارتفاع قوي محتمل'
            rec     = 'الظروف مواتية للقفزة'
        elif prob_high >= 0.58:
            verdict = '🟡 ارتفاع محتمل'
            rec     = 'إشارات إيجابية متعددة'
        elif prob_high <= 0.35:
            verdict = '🔴 منخفض مرجح'
            rec     = 'لا تزال في فترة تبريد'
        else:
            verdict = '⚪ غير محدد'
            rec     = 'انتظر إشارات أوضح'

        return {
            'prob_high'      : round(prob_high, 4),
            'prob_high_pct'  : round(prob_high * 100, 1),
            'signals_active' : int(signals_active),
            'verdict'        : verdict,
            'recommendation' : rec,
            'low_streak'     : int(ls),
            'context'        : ctx,
            'post_signal'    : post_signal,
            'jump_forecast'  : jump_forecast,
            'method_probs'   : {
                k: round(v, 4) for k, v in probs.items()
            },
            'signals': {
                'gap'    : gap_sig,
                'trigger': trig_sig,
                'ifthen' : ifthen,
                'acc'    : acc_sig,
                'hurst'  : hurst_sig
            }
        }


# ══════════════════════════════════════════════════════════════
#                    البيانات النموذجية
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
    5.22,3.32,24.86,3.37,5.16,1.69,2.31,1.07,1.10,1.01,
    1.36,1.38,1.54,5.34,2.68,5.78,3.63,1.89,8.41,4.06,
    1.44,1.50,3.17,1.02,1.80,1.90,1.86,1.85,1.73,3.86,
]


# ══════════════════════════════════════════════════════════════
#                    واجهة المستخدم
# ══════════════════════════════════════════════════════════════
st.title("🔬 محلل أنماط Crash الذكي")
st.caption(
    "يكتشف: قوانين الفجوات • محفزات القفزات • "
    "قواعد إذا-ثم • تسلسلات ما قبل القفزة • "
    "Hurst • التنبؤ الجماعي"
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
        "قيم Crash (50+ للأفضل):",
        height=130,
        placeholder="1.23  4.56  2.10  8.92  22.3 ..."
    )
    if txt.strip():
        try:
            raw_data = [
                float(x) for x in
                txt.replace('\n',' ').split()
                if x.strip()
            ]
            st.success(f"✅ {len(raw_data)} قيمة")
        except Exception:
            st.error("❌ أرقام فقط")

elif method == "📂 CSV":
    up = st.file_uploader("CSV — عمود crash_point",type=['csv'])
    if up:
        try:
            df_u = pd.read_csv(up)
            if 'crash_point' in df_u.columns:
                raw_data = [
                    float(x) for x in
                    df_u['crash_point'].dropna()
                ]
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
    c1.metric("العدد",    str(n))
    c2.metric("متوسط",    f"{arr.mean():.2f}x")
    c3.metric("وسيط",     f"{np.median(arr):.2f}x")
    c4.metric("أقصى",     f"{arr.max():.2f}x")
    c5.metric(">=2x",     f"{np.mean(arr>=2)*100:.1f}%")
    c6.metric("آخر قيمة", f"{arr[-1]:.2f}x")

    if n < 50:
        st.warning(f"⚠️ يُفضَّل 50+ (لديك {n})")
    else:
        st.markdown("---")
        if st.button(
            "🚀 تحليل الأنماط والتنبؤ",
            type="primary",
            use_container_width=True
        ):
            prog = st.progress(0)
            stat = st.empty()

            stat.info("⏳ اكتشاف قوانين الفجوات...")
            engine   = SmartPatternEngine(raw_data)
            prog.progress(15)

            stat.info("⏳ اكتشاف المحفزات...")
            patterns = engine.run_all()
            prog.progress(70)

            stat.info("⏳ حساب التنبؤ...")
            predictor = SmartPredictor(raw_data, patterns)
            pred      = predictor.full_predict()
            prog.progress(100)

            stat.empty()
            prog.empty()
            st.balloons()

            # ════════════════════════════════════════════
            #           بطاقة التنبؤ
            # ════════════════════════════════════════════
            st.markdown("---")
            st.header("🎯 التنبؤ بالجولة القادمة")

            prob = pred['prob_high']
            color = (
                '#2ecc71' if prob >= 0.65
                else '#f39c12' if prob >= 0.50
                else '#e74c3c'
            )

            col_a, col_b = st.columns([1,2])

            with col_a:
                st.markdown(
                    f"""
                    <div style="
                        background:{color}15;
                        border:3px solid {color};
                        border-radius:18px;
                        padding:28px;
                        text-align:center;
                    ">
                    <h1 style="color:{color};margin:0;">
                        {pred['prob_high_pct']}%
                    </h1>
                    <h3 style="margin:6px 0;">
                        احتمال >= 2x
                    </h3>
                    <hr style="border-color:{color}55;">
                    <b>{pred['verdict']}</b><br>
                    <small>{pred['recommendation']}</small>
                    <hr style="border-color:{color}33;">
                    <small>إشارات نشطة: 
                    {pred['signals_active']}/4</small><br>
                    <small>تسلسل منخفض: 
                    {pred['low_streak']} جولة</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if pred['post_signal']:
                    st.warning(f"⚠️ {pred['post_signal']}")

            with col_b:
                # مخطط الاحتمالات
                method_labels = {
                    'gap'    :'قانون الفجوات',
                    'trigger':'المحفزات',
                    'ifthen' :'قواعد إذا-ثم',
                    'acc'    :'قانون التراكم',
                    'hurst'  :'Hurst'
                }
                mp = pred['method_probs']
                fig_mp = go.Figure(go.Bar(
                    x=[method_labels[k] for k in mp],
                    y=[v*100 for v in mp.values()],
                    marker_color=[
                        '#2ecc71' if v >= 0.60
                        else '#f39c12' if v >= 0.50
                        else '#e74c3c'
                        for v in mp.values()
                    ],
                    text=[f"{v*100:.1f}%" for v in mp.values()],
                    textposition='outside'
                ))
                fig_mp.add_hline(
                    y=50, line_dash="dash",
                    line_color="gray",
                    annotation_text="50% خط الحياد"
                )
                fig_mp.update_layout(
                    title="احتمال كل طريقة (%)",
                    yaxis_range=[0,100],
                    height=300,
                    margin=dict(t=40,b=5)
                )
                st.plotly_chart(fig_mp, use_container_width=True)

                # الإشارات النشطة
                sigs = pred['signals']
                sig_rows = []
                for k, label in method_labels.items():
                    s = sigs.get(k, {})
                    sig_rows.append({
                        'الطريقة': label,
                        'إشارة'  : (
                            '✅ نشطة'
                            if s.get('signal', False)
                            else '⬜ غائبة'
                        ),
                        'الاحتمال': f"{s.get('probability',0)*100:.1f}%"
                    })
                st.dataframe(
                    pd.DataFrame(sig_rows),
                    use_container_width=True,
                    hide_index=True
                )

            # ── توقع القفزات ─────────────────────────────
            st.markdown("---")
            st.subheader("⏱️ متى تأتي القفزات القادمة؟")

            jf = pred['jump_forecast']
            jf_rows = []
            for thr_key, info in jf.items():
                exp  = float(info['expected_in'])
                z    = float(info['z_score'])
                jf_rows.append({
                    'العتبة'       : thr_key,
                    'الفجوة الحالية': f"{info['current_gap']} جولة",
                    'متوقعة بعد'   : (
                        f"≈ {exp:.0f} جولة"
                        if exp > 0 else "⚠️ متأخرة!"
                    ),
                    'المنطقة'      : info['zone'],
                    'Z-Score'      : f"{z:+.2f}"
                })
            if jf_rows:
                st.dataframe(
                    pd.DataFrame(jf_rows),
                    use_container_width=True,
                    hide_index=True
                )

            # ════════════════════════════════════════════
            #           الأنماط المكتشفة
            # ════════════════════════════════════════════
            st.markdown("---")
            st.header("🔬 الأنماط المكتشفة")

            (t1, t2, t3, t4,
             t5, t6, t7) = st.tabs([
                "📏 قانون الفجوات",
                "⚡ المحفزات",
                "🔤 التسلسلات",
                "🔮 قواعد إذا-ثم",
                "📊 ما بعد القفزة",
                "📐 Hurst",
                "📡 الدورات"
            ])

            # ── قانون الفجوات ────────────────────────────
            with t1:
                st.subheader(
                    "📏 القانون: بعد كم رقم صغير تأتي القفزة؟"
                )
                gl = patterns['gap_laws']

                for thr_key, info in gl.items():
                    st.markdown(f"### 🎯 {thr_key}")

                    ca,cb,cc,cd,ce = st.columns(5)
                    ca.metric("عدد القفزات", info['n_jumps'])
                    cb.metric("متوسط الفجوة", f"{info['mean_gap']:.1f}")
                    cc.metric("وسيط الفجوة", f"{info['median_gap']:.1f}")
                    cd.metric("الأكثر شيوعاً",
                              f"{info['most_common_gap']['gap']} جولة")
                    ce.metric("تكراره",
                              f"{info['most_common_gap']['count']}x")

                    # مخطط توزيع الفجوات
                    gd  = info['gap_distribution']
                    keys= [int(k) for k in gd.keys()]
                    vals= list(gd.values())
                    if keys:
                        fig_gd = px.bar(
                            x=keys, y=vals,
                            title=(
                                f"توزيع الفجوات قبل {thr_key} — "
                                f"معظمها بعد "
                                f"{info['most_common_gap']['gap']} رقم"
                            ),
                            labels={
                                'x':'عدد الأرقام الصغيرة',
                                'y':'عدد المرات'
                            },
                            color=vals,
                            color_continuous_scale='Blues'
                        )
                        fig_gd.add_vline(
                            x=info['mean_gap'],
                            line_dash="dash",
                            line_color="red",
                            annotation_text="المتوسط"
                        )
                        fig_gd.add_vline(
                            x=info['median_gap'],
                            line_dash="dot",
                            line_color="orange",
                            annotation_text="الوسيط"
                        )
                        st.plotly_chart(
                            fig_gd, use_container_width=True
                        )

                    # احتمالات مشروطة
                    cp = info['conditional_probs']
                    if cp:
                        cp_keys = sorted(cp.keys())[:15]
                        cp_vals = [cp[k] for k in cp_keys]
                        fig_cp = go.Figure(go.Bar(
                            x=cp_keys,
                            y=[v*100 for v in cp_vals],
                            marker_color=[
                                '#2ecc71' if v >= 0.60
                                else '#f39c12' if v >= 0.45
                                else '#e74c3c'
                                for v in cp_vals
                            ],
                            text=[f"{v*100:.1f}%" for v in cp_vals],
                            textposition='outside'
                        ))
                        fig_cp.add_hline(
                            y=float(np.mean(arr>=float(
                                thr_key.replace('>=','').replace('x','')
                            ))*100),
                            line_dash="dash",
                            line_color="blue",
                            annotation_text="الاحتمال الأساسي"
                        )
                        fig_cp.update_layout(
                            title=(
                                f"P({thr_key} | فجوة=k) — "
                                "هل الفجوة الحالية مميزة؟"
                            ),
                            xaxis_title="طول الفجوة k",
                            yaxis_title="الاحتمال %",
                            yaxis_range=[0, 110],
                            height=350
                        )
                        st.plotly_chart(
                            fig_cp, use_container_width=True
                        )

                        # تسليط الضوء على الفجوة الحالية
                        ls = int(pred['low_streak'])
                        if ls in cp:
                            cur_prob = cp[ls]
                            base_rate = float(
                                np.mean(arr >= float(
                                    thr_key.replace('>=','').replace('x','')
                                ))
                            )
                            if cur_prob > base_rate * 1.1:
                                st.success(
                                    f"🎯 **الفجوة الحالية = {ls}** | "
                                    f"احتمال {thr_key}: "
                                    f"**{cur_prob*100:.1f}%** "
                                    f"(الأساسي: {base_rate*100:.1f}%)"
                                )

                    st.markdown("---")

            # ── المحفزات ─────────────────────────────────
            with t2:
                st.subheader(
                    "⚡ قانون المحفزات: بعد X جولات من ظهور Y"
                )
                rules = patterns['triggers']['rules']
                if rules:
                    df_rules = pd.DataFrame(rules[:15])

                    # مخطط المحفزات
                    fig_rules = px.scatter(
                        df_rules,
                        x='delay',
                        y='probability',
                        color='lift',
                        size='total',
                        hover_data=['rule'],
                        title=(
                            "قواعد المحفزات — "
                            "المحور X: التأخير، Y: الاحتمال"
                        ),
                        color_continuous_scale='RdYlGn',
                        labels={
                            'delay'      :'التأخير (جولات)',
                            'probability':'الاحتمال',
                            'lift'       :'الارتفاع عن الأساس'
                        }
                    )
                    fig_rules.add_hline(
                        y=0.5, line_dash="dash",
                        line_color="gray"
                    )
                    st.plotly_chart(
                        fig_rules, use_container_width=True
                    )

                    # جدول القواعد
                    df_show = df_rules[[
                        'rule','probability','lift',
                        'hits','total'
                    ]].copy()
                    df_show['probability'] = df_show[
                        'probability'
                    ].apply(lambda x: f"{x*100:.1f}%")
                    df_show['lift'] = df_show[
                        'lift'
                    ].apply(lambda x: f"+{x*100:.1f}%")
                    df_show.columns = [
                        'القاعدة','الاحتمال',
                        'الارتفاع','النجاحات','الإجمالي'
                    ]
                    st.dataframe(
                        df_show, use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("لم تُكتشف قواعد محفزات قوية")

            # ── التسلسلات ────────────────────────────────
            with t3:
                st.subheader(
                    "🔤 التسلسلات التي تسبق القفزات"
                )
                seqs = patterns['pre_jump_seqs']

                for thr_key, pats in seqs.items():
                    if not pats:
                        continue
                    st.markdown(f"### 🎯 {thr_key}")
                    df_seq = pd.DataFrame(pats[:8])
                    df_seq['probability'] = df_seq[
                        'probability'
                    ].apply(lambda x: f"{x*100:.1f}%")
                    df_seq['lift'] = df_seq[
                        'lift'
                    ].apply(lambda x: f"{x*100:+.1f}%")
                    df_seq.columns = [
                        'التسلسل','الاحتمال','الأساسي',
                        'الارتفاع','قبل قفزة','إجمالي'
                    ]
                    st.dataframe(
                        df_seq, use_container_width=True,
                        hide_index=True
                    )

                    if pats:
                        fig_seq = px.bar(
                            x=[p['sequence'] for p in pats[:8]],
                            y=[p['probability']*100 for p in pats[:8]],
                            title=f"احتمال التسلسل يسبق {thr_key}",
                            labels={
                                'x':'التسلسل',
                                'y':'الاحتمال %'
                            },
                            color=[
                                p['probability'] for p in pats[:8]
                            ],
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(
                            fig_seq, use_container_width=True
                        )

            # ── قواعد إذا-ثم ─────────────────────────────
            with t4:
                st.subheader("🔮 قواعد إذا-ثم المكتشفة")
                it_rules = patterns['if_then']['if_then_rules']
                if it_rules:
                    for rule in it_rules[:10]:
                        conf_color = (
                            "success" if rule['probability'] >= 0.65
                            else "warning" if rule['probability'] >= 0.55
                            else "info"
                        )
                        getattr(st, conf_color)(
                            f"📌 **{rule['rule_text']}**\n\n"
                            f"الثقة: {rule['probability']*100:.1f}% | "
                            f"الارتفاع: +{rule['lift']*100:.1f}% | "
                            f"العينات: {rule['total']}"
                        )

                    # مخطط
                    df_it = pd.DataFrame(it_rules[:10])
                    fig_it = px.scatter(
                        df_it,
                        x='delay',
                        y='probability',
                        size='total',
                        color='lift',
                        hover_data=['pattern'],
                        title="قواعد إذا-ثم",
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(
                        fig_it, use_container_width=True
                    )
                else:
                    st.info("لم تُكتشف قواعد إذا-ثم قوية")

            # ── ما بعد القفزة ─────────────────────────────
            with t5:
                st.subheader("📊 ماذا يأتي بعد القفزة؟")
                post = patterns['post_jump']

                for thr_key, info in post.items():
                    st.markdown(f"### بعد {thr_key}")

                    ca, cb, cc = st.columns(3)
                    ca.metric(
                        "متوسط فترة التبريد",
                        f"{info['cooling_mean']:.1f} جولة"
                    )
                    cb.metric(
                        "وسيط التبريد",
                        f"{info['cooling_median']:.1f}"
                    )
                    cc.metric(
                        "انحراف التبريد",
                        f"{info['cooling_std']:.1f}"
                    )

                    # مخطط ما بعد القفزة
                    ps = info['post_round_stats']
                    if ps:
                        ks = sorted(ps.keys())
                        pct_high = [
                            ps[k]['pct_high'] for k in ks
                        ]
                        pct_low  = [
                            ps[k]['pct_low'] for k in ks
                        ]
                        means    = [
                            ps[k]['mean'] for k in ks
                        ]

                        fig_post = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=[
                                '% >= 2x بعد k جولة',
                                'متوسط القيمة بعد k جولة'
                            ]
                        )
                        fig_post.add_trace(
                            go.Bar(
                                x=ks, y=pct_high,
                                name='% مرتفع',
                                marker_color='#2ecc71'
                            ), row=1, col=1
                        )
                        fig_post.add_hline(
                            y=float(np.mean(arr>=2.0)*100),
                            line_dash="dash",
                            line_color="red",
                            row=1, col=1
                        )
                        fig_post.add_trace(
                            go.Scatter(
                                x=ks, y=means,
                                mode='lines+markers',
                                name='المتوسط',
                                line=dict(color='steelblue')
                            ), row=1, col=2
                        )
                        fig_post.update_layout(
                            height=350,
                            title=f"سلوك السوق بعد {thr_key}"
                        )
                        st.plotly_chart(
                            fig_post, use_container_width=True
                        )
                    st.markdown("---")

            # ── Hurst ─────────────────────────────────────
            with t6:
                hurst = patterns['hurst']
                H     = float(hurst['H'])

                st.subheader("📐 Hurst Exponent")
                ca, cb = st.columns(2)
                ca.metric(
                    "H",
                    f"{H:.4f}",
                    delta=hurst['interpretation']
                )
                cb.metric("R²", f"{hurst['r_squared']:.4f}")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=H,
                    title={'text':"Hurst H"},
                    gauge={
                        'axis': {'range':[0,1]},
                        'bar' : {'color':'navy'},
                        'steps': [
                            {'range':[0.0,0.4],
                             'color':'#e74c3c'},
                            {'range':[0.4,0.6],
                             'color':'#f1c40f'},
                            {'range':[0.6,1.0],
                             'color':'#2ecc71'}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=320)
                st.plotly_chart(
                    fig_gauge, use_container_width=True
                )

                # رسم الانحدار
                if len(hurst['lags']) > 1:
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(
                        x=hurst['log_lags'],
                        y=hurst['log_tau'],
                        mode='markers',
                        name='البيانات',
                        marker=dict(
                            color='steelblue', size=10
                        )
                    ))
                    fig_h.add_trace(go.Scatter(
                        x=hurst['log_lags'],
                        y=hurst['trend_y'],
                        mode='lines',
                        name=f'انحدار H={H:.3f}',
                        line=dict(
                            color='red', width=2,
                            dash='dash'
                        )
                    ))
                    fig_h.update_layout(
                        title=(
                            f"log-log regression | "
                            f"R²={hurst['r_squared']:.3f}"
                        ),
                        xaxis_title='log(lag)',
                        yaxis_title='log(τ)',
                        height=380
                    )
                    st.plotly_chart(
                        fig_h, use_container_width=True
                    )

            # ── الدورات ──────────────────────────────────
            with t7:
                spec = patterns['spectral']
                st.subheader("📡 الدورات الزمنية")

                ca, cb = st.columns(2)
                ca.metric(
                    "نسبة الهيمنة",
                    f"{spec['dominance']}x"
                )
                cb.metric(
                    "دورة قوية؟",
                    "نعم 🔴" if spec['has_cycle'] else "لا ✅"
                )

                if spec['cycles']:
                    df_cy = pd.DataFrame(spec['cycles'])
                    fig_cy = px.bar(
                        df_cy,
                        x='period', y='rel_pow',
                        title="الدورات الزمنية (FFT)",
                        color='rel_pow',
                        color_continuous_scale='Reds',
                        labels={
                            'period' :'الدورة (جولات)',
                            'rel_pow':'القوة النسبية'
                        },
                        text='period'
                    )
                    fig_cy.update_traces(
                        texttemplate='%{text:.0f}j',
                        textposition='outside'
                    )
                    st.plotly_chart(
                        fig_cy, use_container_width=True
                    )

                if spec['freqs'] and spec['psd']:
                    fig_psd = go.Figure()
                    fig_psd.add_trace(go.Scatter(
                        x=spec['freqs'],
                        y=spec['psd'],
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color='steelblue')
                    ))
                    fig_psd.update_layout(
                        title="طيف القدرة (PSD)",
                        xaxis_title="التردد",
                        yaxis_title="القدرة",
                        height=350
                    )
                    st.plotly_chart(
                        fig_psd, use_container_width=True
                    )

            # ── آخر 50 جولة ──────────────────────────────
            st.markdown("---")
            st.subheader("📜 آخر 50 جولة")

            def get_color(v):
                if v >= 10: return '#9b59b6'
                if v >= 5:  return '#3498db'
                if v >= 3:  return '#2ecc71'
                if v >= 2:  return '#f1c40f'
                if v >= 1.5:return '#e67e22'
                return '#e74c3c'

            last50  = raw_data[-50:]
            colors50= [get_color(v) for v in last50]

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=list(range(len(last50))),
                y=last50,
                marker_color=colors50,
                hovertemplate=(
                    "جولة %{x}<br>"
                    "القيمة: %{y:.2f}x<extra></extra>"
                )
            ))
            for thr, col, lbl in [
                (2.0,'blue','2x'),
                (5.0,'green','5x'),
                (10.0,'purple','10x')
            ]:
                fig_hist.add_hline(
                    y=thr, line_dash="dash",
                    line_color=col,
                    annotation_text=lbl
                )
            fig_hist.update_layout(
                title="آخر 50 جولة",
                xaxis_title="الجولة",
                yaxis_title="المضاعف",
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # ── الاستنتاج ────────────────────────────────
            st.markdown("---")
            st.header("📋 الاستنتاج والقواعد المكتشفة")

            # أبرز القواعد
            gl = patterns['gap_laws']
            it = patterns['if_then']['if_then_rules']
            tr = patterns['triggers']['rules']

            st.subheader("🏆 أبرز الاكتشافات")

            findings = []

            for thr_key, info in gl.items():
                mcg = info['most_common_gap']
                findings.append(
                    f"📏 **{thr_key}**: تأتي غالباً بعد "
                    f"**{mcg['gap']} رقم صغير** "
                    f"(تكرر {mcg['count']}x) | "
                    f"متوسط الفجوة: {info['mean_gap']:.1f}"
                )

            for rule in it[:3]:
                findings.append(
                    f"🔮 **قاعدة إذا-ثم**: {rule['rule_text']}"
                )

            for rule in tr[:3]:
                findings.append(
                    f"⚡ **محفز**: {rule['rule']}"
                )

            for f in findings:
                st.success(f)

            # تحميل التقرير
            st.markdown("---")
            report = to_python({
                'total_samples'  : n,
                'prediction'     : {
                    'prob_high'     : pred['prob_high'],
                    'verdict'       : pred['verdict'],
                    'signals_active': pred['signals_active'],
                    'low_streak'    : pred['low_streak'],
                    'jump_forecast' : pred['jump_forecast']
                },
                'gap_laws_summary': {
                    k: {
                        'mean_gap'       : v['mean_gap'],
                        'median_gap'     : v['median_gap'],
                        'most_common_gap': v['most_common_gap']
                    }
                    for k, v in patterns['gap_laws'].items()
                },
                'top_trigger_rules': tr[:5],
                'top_if_then_rules': it[:5],
                'hurst'            : {
                    'H'             : patterns['hurst']['H'],
                    'interpretation': patterns['hurst']['interpretation']
                },
                'key_findings'     : findings
            })

            st.download_button(
                "📥 تحميل تقرير الأنماط (JSON)",
                data=json.dumps(
                    report, ensure_ascii=False, indent=2
                ),
                file_name="crash_pattern_report.json",
                mime="application/json"
            )

st.markdown("---")
st.caption(
    "🎓 محلل أنماط Crash الذكي | "
    "قوانين الفجوات • المحفزات • إذا-ثم • "
    "التسلسلات • Hurst • التنبؤ الجماعي"
)
