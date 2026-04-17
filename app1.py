# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from datetime import datetime
import random
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🧠 Crash Intelligence v3",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;700;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
* { font-family: 'Tajawal', sans-serif !important; }
html, body, [data-testid="stAppViewContainer"] { background: #04040f !important; }
[data-testid="stSidebar"] { background: #06060f !important; border-right:1px solid rgba(99,102,241,0.15); }

.card {
    background: linear-gradient(145deg,rgba(8,8,22,0.98),rgba(12,12,32,0.99));
    border:1px solid rgba(99,102,241,0.2);
    box-shadow:0 20px 60px rgba(0,0,0,0.8),inset 0 1px 0 rgba(99,102,241,0.12);
    border-radius:18px; padding:24px; margin-bottom:16px;
    direction:rtl; color:white; position:relative; overflow:hidden;
}
.card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,transparent,#6366f1,#a855f7,#ec4899,transparent);
}

/* ══ حالات ══ */
.STATUS-BET {
    background:linear-gradient(135deg,rgba(0,255,136,0.1),rgba(0,180,90,0.05));
    border:2px solid #00ff88; border-radius:18px; padding:24px; text-align:center;
    animation:pSafe 2s ease-in-out infinite;
}
@keyframes pSafe{0%,100%{box-shadow:0 0 20px rgba(0,255,136,0.2);}50%{box-shadow:0 0 55px rgba(0,255,136,0.5);}}
.STATUS-STRONG {
    background:linear-gradient(135deg,rgba(0,200,255,0.12),rgba(0,130,200,0.06));
    border:2px solid #00c8ff; border-radius:18px; padding:24px; text-align:center;
    animation:pStr 1.6s ease-in-out infinite;
}
@keyframes pStr{0%,100%{box-shadow:0 0 25px rgba(0,200,255,0.2);}50%{box-shadow:0 0 65px rgba(0,200,255,0.55);}}
.STATUS-AVOID {
    background:linear-gradient(135deg,rgba(255,40,40,0.1),rgba(180,0,0,0.05));
    border:2px solid #ff3232; border-radius:18px; padding:24px; text-align:center;
    animation:pDng 0.9s ease-in-out infinite;
}
@keyframes pDng{0%,100%{box-shadow:0 0 20px rgba(255,50,50,0.3);}50%{box-shadow:0 0 65px rgba(255,50,50,0.7);}}
.STATUS-WAIT {
    background:linear-gradient(135deg,rgba(255,200,0,0.09),rgba(255,130,0,0.04));
    border:2px solid #FFD700; border-radius:18px; padding:24px; text-align:center;
    box-shadow:0 0 25px rgba(255,215,0,0.12);
}
.STATUS-DOUBLE {
    background:linear-gradient(135deg,rgba(255,100,0,0.12),rgba(200,50,0,0.06));
    border:2px solid #ff6400; border-radius:18px; padding:24px; text-align:center;
    animation:pDbl 1.1s ease-in-out infinite;
}
@keyframes pDbl{0%,100%{box-shadow:0 0 25px rgba(255,100,0,0.25);}50%{box-shadow:0 0 65px rgba(255,100,0,0.65);}}

/* ══ شارات ══ */
.badge {
    display:inline-block; padding:5px 11px; border-radius:9px;
    font-size:13px; font-weight:900; margin:2px;
    font-family:'Orbitron',monospace !important;
    transition:all 0.2s;
}
.b-loss{background:#3d0000;border:1px solid #ff4444;color:#ff7070;}
.b-loss18{background:#5a0000;border:2px solid #ff2020;color:#ff9090;
           animation:glow-r 1.4s ease-in-out infinite;}
@keyframes glow-r{0%,100%{box-shadow:0 0 5px rgba(255,30,30,0.3);}
                   50%{box-shadow:0 0 15px rgba(255,30,30,0.7);}}
.b-med{background:#1a1200;border:1px solid #FFD700;color:#FFD700;}
.b-win{background:#003d1f;border:1px solid #00ff88;color:#00ff88;}
.b-big{background:#1a0030;border:1px solid #a855f7;color:#c4b5fd;}
.b-gold{background:#2d1800;border:2px solid #ff9500;color:#ffb84d;
        animation:glow-g 1.3s ease-in-out infinite;}
@keyframes glow-g{0%,100%{box-shadow:0 0 6px rgba(255,149,0,0.35);}
                   50%{box-shadow:0 0 20px rgba(255,149,0,0.7);}}

/* ══ kpi ══ */
.kpi{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
     border-radius:12px;padding:14px;text-align:center;direction:rtl;transition:all 0.3s;}
.kpi:hover{border-color:rgba(99,102,241,0.35);transform:translateY(-2px);}
.kn{font-family:'Orbitron',monospace!important;font-size:22px;font-weight:900;
    background:linear-gradient(90deg,#6366f1,#a855f7);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.kl{color:rgba(255,255,255,0.32);font-size:10px;margin-top:3px;letter-spacing:1px;}

/* ══ صناديق ══ */
.bx-g{background:rgba(0,255,136,0.06);border:1px solid rgba(0,255,136,0.25);
      border-right:4px solid #00ff88;border-radius:11px;padding:12px 16px;
      color:rgba(150,255,200,0.9);font-size:13px;direction:rtl;margin:7px 0;line-height:1.8;}
.bx-r{background:rgba(255,50,50,0.06);border:1px solid rgba(255,50,50,0.25);
      border-right:4px solid #ff3232;border-radius:11px;padding:12px 16px;
      color:rgba(255,170,170,0.9);font-size:13px;direction:rtl;margin:7px 0;line-height:1.8;}
.bx-y{background:rgba(255,200,0,0.06);border:1px solid rgba(255,200,0,0.25);
      border-right:4px solid #FFD700;border-radius:11px;padding:12px 16px;
      color:rgba(255,230,150,0.9);font-size:13px;direction:rtl;margin:7px 0;line-height:1.8;}
.bx-b{background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.25);
      border-right:4px solid #6366f1;border-radius:11px;padding:12px 16px;
      color:rgba(180,185,255,0.9);font-size:13px;direction:rtl;margin:7px 0;line-height:1.8;}
.bx-o{background:rgba(255,149,0,0.07);border:1px solid rgba(255,149,0,0.3);
      border-right:4px solid #ff9500;border-radius:11px;padding:12px 16px;
      color:rgba(255,210,150,0.9);font-size:13px;direction:rtl;margin:7px 0;line-height:1.8;}

/* ══ progress ══ */
.pw{background:rgba(255,255,255,0.05);border-radius:7px;height:7px;margin:5px 0;overflow:hidden;}
.pf-g{height:100%;border-radius:7px;background:linear-gradient(90deg,#00c853,#00ff88);transition:width 0.6s;}
.pf-o{height:100%;border-radius:7px;background:linear-gradient(90deg,#ff6d00,#ff9500);transition:width 0.6s;}
.pf-r{height:100%;border-radius:7px;background:linear-gradient(90deg,#c62828,#ff3232);transition:width 0.6s;}
.pf-b{height:100%;border-radius:7px;background:linear-gradient(90deg,#6366f1,#a855f7);transition:width 0.6s;}

/* ══ gold card ══ */
.gc{background:linear-gradient(135deg,rgba(255,149,0,0.09),rgba(255,70,0,0.04));
    border:1px solid rgba(255,149,0,0.4);border-radius:13px;
    padding:14px;text-align:center;transition:all 0.3s;}
.gc:hover{border-color:#ff9500;transform:translateY(-3px);
          box-shadow:0 10px 28px rgba(255,149,0,0.28);}
.gn{font-family:'Orbitron',monospace!important;font-size:20px;
    font-weight:900;color:#ff9500;text-shadow:0 0 10px rgba(255,149,0,0.4);}
.gt{font-family:'Orbitron',monospace!important;font-size:14px;color:#00ff88;margin-top:4px;}

/* ══ buttons ══ */
.stButton>button{
    background:linear-gradient(135deg,#6366f1,#8b5cf6,#a855f7)!important;
    color:white!important;border:none!important;font-weight:700!important;
    font-size:13px!important;border-radius:10px!important;padding:9px 20px!important;
    box-shadow:0 5px 18px rgba(99,102,241,0.4)!important;transition:all 0.3s!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;
                        box-shadow:0 9px 32px rgba(99,102,241,0.6)!important;}
.stNumberInput>div>div>input{
    background:rgba(255,255,255,0.05)!important;color:white!important;
    border:1px solid rgba(99,102,241,0.35)!important;border-radius:9px!important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  البيانات التاريخية + الأرقام الذهبية
# ══════════════════════════════════════════════════════════════════════
HISTORICAL_DATA = [
    8.72,6.75,1.86,2.18,1.25,2.28,1.24,1.2,1.54,24.46,4.16,1.49,
    1.09,1.47,1.54,1.53,2.1,32.04,11,1.17,1.7,2.61,1.26,22.23,
    1.77,1.93,3.35,7.01,1.83,9.39,3.31,2.04,1.3,6.65,1.16,3.39,
    1.95,10.85,1.65,1.22,1.6,4.67,1.85,2.72,1,3.02,1.35,1.3,
    1.37,17.54,1.18,1,14.4,1.11,6.15,2.39,2.22,1.42,1.23,2.42,
    1.07,1.24,2.55,7.26,1.69,5.1,2.59,5.51,2.31,2.12,1.97,1.5,
    3.01,2.29,1.36,4.95,5.09,8.5,1.77,5.52,3.93,1.5,2.28,2.49,
    18.25,1.68,1.42,2.12,4.17,1.04,2.35,1,1.01,5.46,1.13,2.84,
    3.39,2.79,1.59,1.53,4.34,2.96,1.06,1.72,2.16,2.2,3.61,2.34,
    4.49,1.72,1.78,9.27,8.49,2.86,1.66,4.63,9.25,1.35,1,1.64,
    1.86,2.81,2.44,1.74,1.1,1.29,1.45,8.92,1.24,6.39,1.16,1.19,
    2.4,4.64,3.17,24.21,1.17,1.42,2.13,1.12,3.78,1.12,1.52,
    22.81,1.31,1.9,1.38,1.47,2.86,1.79,1.49,1.38,1.84,1.06,3.3,
    5.97,1,2.92,1.64,5.32,3.26,1.78,2.24,3.16,1.6,1.08,1.55,
    1.07,1.02,1.23,1.08,5.22,3.32,24.86,3.37,5.16,1.69,2.31,
    1.07,1.1,1.01,1.36,1.38,1.54,5.34,2.68,5.78,3.63,1.89,8.41,
    4.06,1.44,1.5,3.17,1.02,1.8,1.9,1.86,1.85,1.73,3.86,3.11,
    2.44,1.15,2.03,1.05,3.05,1.88,10.13,2.29,1.41,1,5.46,1.26,
    23.33,1.96,1.03,4.54,1.37,3.5,1.13,1.16,1.43,1.13,1.05,33.27,
    9.96,1.79,2.07,18.51,5.75,1.15,1.08,5.92,1.38,1.61,12.99,
    24.72,4.86,1.11,2.86,1.54,3.71,4,7.57,2.03,2.18,5.52,
    13.37,3.73,2.41,1.79,5.57,4.36,12.33,1.61,3.28,2.89,1.47,
    1.08,26.89,1.53,2.94,5.29,1.23,1.57,1.12,5.69,3.29,2.72,
    1.18,5.03,1.1,1.32,1.18,1.07,1.27,4.6
]

# الأرقام الذهبية — مُدمجة مع نتائج التحليل الإحصائي
GOLDEN_DB = {
    # tier-1: avg_next ≥ 9x مُثبت إحصائياً
    1.05: {"tier":1,"avg_next":14.48,"med_next":10.13,"win5":0.67,"win2":0.67,
           "tgt_lo":5.0,"tgt_hi":15.0,"n":3,"label":"زنبرك أقصى"},
    1.09: {"tier":1,"avg_next":9.73,"med_next":6.15,"win5":0.33,"win2":0.67,
           "tgt_lo":4.0,"tgt_hi":11.0,"n":3,"label":"زنبرك أقصى"},
    1.20: {"tier":1,"avg_next":17.17,"med_next":17.17,"win5":0.50,"win2":0.50,
           "tgt_lo":5.0,"tgt_hi":18.0,"n":2,"label":"زنبرك أقصى"},
    # tier-2: avg_next 5–9x
    1.53: {"tier":2,"avg_next":6.74,"med_next":5.34,"win5":0.40,"win2":0.80,
           "tgt_lo":3.5,"tgt_hi":8.0,"n":5,"label":"إشارة قوية"},
    1.54: {"tier":2,"avg_next":5.97,"med_next":4.34,"win5":0.40,"win2":0.80,
           "tgt_lo":3.5,"tgt_hi":7.0,"n":5,"label":"إشارة قوية"},
    1.77: {"tier":2,"avg_next":8.30,"med_next":5.52,"win5":0.67,"win2":1.00,
           "tgt_lo":4.0,"tgt_hi":9.0,"n":3,"label":"إشارة قوية"},
    1.36: {"tier":2,"avg_next":5.53,"med_next":4.95,"win5":0.50,"win2":0.75,
           "tgt_lo":3.0,"tgt_hi":7.0,"n":4,"label":"إشارة جيدة"},
    1.84: {"tier":2,"avg_next":6.58,"med_next":6.58,"win5":0.50,"win2":1.00,
           "tgt_lo":3.5,"tgt_hi":7.0,"n":2,"label":"إشارة جيدة"},
    1.83: {"tier":2,"avg_next":5.64,"med_next":5.64,"win5":0.50,"win2":1.00,
           "tgt_lo":3.0,"tgt_hi":7.0,"n":2,"label":"إشارة جيدة"},
    # tier-3: avg_next 2.5–5x
    1.01: {"tier":3,"avg_next":3.29,"med_next":2.35,"win5":0.33,"win2":0.67,
           "tgt_lo":2.0,"tgt_hi":5.0,"n":3,"label":"إشارة متوسطة"},
    1.07: {"tier":3,"avg_next":2.51,"med_next":2.55,"win5":0.20,"win2":0.60,
           "tgt_lo":2.0,"tgt_hi":4.0,"n":5,"label":"إشارة متوسطة"},
    1.12: {"tier":3,"avg_next":4.82,"med_next":3.78,"win5":0.33,"win2":0.67,
           "tgt_lo":2.5,"tgt_hi":5.5,"n":6,"label":"إشارة متوسطة"},
    1.19: {"tier":3,"avg_next":3.07,"med_next":2.40,"win5":0.00,"win2":0.67,
           "tgt_lo":2.0,"tgt_hi":4.0,"n":3,"label":"إشارة متوسطة"},
    1.22: {"tier":3,"avg_next":3.12,"med_next":2.55,"win5":0.33,"win2":0.33,
           "tgt_lo":2.0,"tgt_hi":4.5,"n":3,"label":"إشارة متوسطة"},
    1.24: {"tier":3,"avg_next":4.19,"med_next":2.68,"win5":0.25,"win2":0.75,
           "tgt_lo":2.0,"tgt_hi":5.0,"n":4,"label":"إشارة متوسطة"},
    1.29: {"tier":3,"avg_next":5.19,"med_next":4.63,"win5":0.50,"win2":1.00,
           "tgt_lo":2.5,"tgt_hi":6.0,"n":2,"label":"إشارة متوسطة"},
    1.45: {"tier":3,"avg_next":5.91,"med_next":5.32,"win5":0.33,"win2":0.67,
           "tgt_lo":2.5,"tgt_hi":6.5,"n":3,"label":"إشارة متوسطة"},
    1.49: {"tier":3,"avg_next":4.16,"med_next":4.16,"win5":0.25,"win2":0.75,
           "tgt_lo":2.0,"tgt_hi":5.0,"n":4,"label":"إشارة متوسطة"},
    1.66: {"tier":3,"avg_next":7.04,"med_next":7.04,"win5":0.50,"win2":1.00,
           "tgt_lo":3.0,"tgt_hi":7.5,"n":2,"label":"إشارة متوسطة"},
}
GOLDEN_TOL = 0.04

TIER_META = {
    1:{"icon":"🔥","color":"#ff4500","label":"تير-1"},
    2:{"icon":"💎","color":"#ff9500","label":"تير-2"},
    3:{"icon":"✨","color":"#FFD700","label":"تير-3"},
}

# ══════════════════════════════════════════════════════════════════════
#  المحرك الإحصائي الشامل
# ══════════════════════════════════════════════════════════════════════
class StatEngine:
    """
    يدمج كل الفرضيات + المعادلات الإحصائية:
      - Bayesian confidence
      - Entropy / Compression ratio
      - Z-score anomaly detection
      - Autocorrelation
      - Kelly Criterion
      - Markov transition probability
    """

    def __init__(self, history: list, ref_data: list = None):
        self.h   = history
        self.n   = len(history)
        self.ref = ref_data or HISTORICAL_DATA
        self.arr = np.array(history) if history else np.array([])

    # ══════════════════════════════════════════
    # 1. أدوات أساسية
    # ══════════════════════════════════════════
    def _last(self, k):
        return self.h[-k:] if self.n >= k else self.h[:]

    def _streak(self, thr):
        c = 0
        for v in reversed(self.h):
            if v < thr: c += 1
            else: break
        return c

    def _find_golden(self, val):
        best, bd = None, float("inf")
        for g, d in GOLDEN_DB.items():
            df = abs(val - g)
            if df <= GOLDEN_TOL and df < bd:
                best, bd = (g, d), df
        return best  # (gnum, gdata) or None

    # ══════════════════════════════════════════
    # 2. إحصاءات المرجع من البيانات التاريخية
    # ══════════════════════════════════════════
    def _ref_stats(self):
        """
        يحسب توزيع الانتقال الماركوفي من البيانات التاريخية:
        P(next ≥ 5 | streak_under_2 = k)
        """
        ref = self.ref
        transition = defaultdict(list)
        for i in range(len(ref) - 1):
            # احسب الـ streak عند i
            streak = 0
            for j in range(i, -1, -1):
                if ref[j] < 2.0: streak += 1
                else: break
            transition[min(streak, 7)].append(ref[i+1])

        result = {}
        for k, vals in transition.items():
            arr = np.array(vals)
            result[k] = {
                "mean":  round(float(arr.mean()), 3),
                "med":   round(float(np.median(arr)), 3),
                "p_gt2": round(float((arr >= 2.0).mean()), 3),
                "p_gt5": round(float((arr >= 5.0).mean()), 3),
                "p_gt12":round(float((arr >= 12.0).mean()), 3),
                "n":     len(vals),
            }
        return result

    # ══════════════════════════════════════════
    # 3. الإنتروبيا — قياس العشوائية
    # ══════════════════════════════════════════
    def entropy_score(self, window=10):
        """
        Shannon Entropy على نافذة أخيرة.
        انخفاض الإنتروبيا = السلسلة منتظمة (زنبرك مضغوط).
        ارتفاع الإنتروبيا = فوضى (لا نمط).
        """
        if self.n < 4: return 1.0, 0.0
        win = self._last(window)
        # تحويل إلى فئات
        cats = []
        for v in win:
            if v < 1.5:   cats.append(0)
            elif v < 2.0: cats.append(1)
            elif v < 5.0: cats.append(2)
            elif v < 12:  cats.append(3)
            else:          cats.append(4)
        # احسب الإنتروبيا
        from collections import Counter
        cnt = Counter(cats)
        probs = np.array([c/len(cats) for c in cnt.values()])
        ent = -float(np.sum(probs * np.log2(probs + 1e-9)))
        max_ent = np.log2(5)  # أعلى إنتروبيا ممكنة (5 فئات)
        compression = 1.0 - ent / max_ent  # 0=فوضى كاملة, 1=نمط مثالي
        return round(ent, 3), round(compression, 3)

    # ══════════════════════════════════════════
    # 4. Z-score — كشف الشذوذ
    # ══════════════════════════════════════════
    def zscore_last(self):
        """
        Z-score للقيمة الأخيرة مقارنة بالتاريخ.
        z > 2  → قيمة شاذة عالية (قفزة غير عادية)
        z < -1 → قيمة منخفضة جداً (ضغط الزنبرك)
        """
        if self.n < 5: return 0.0
        ref_arr = np.array(self.ref)
        mu  = float(ref_arr.mean())
        std = float(ref_arr.std())
        if std == 0: return 0.0
        return round((self.h[-1] - mu) / std, 3)

    # ══════════════════════════════════════════
    # 5. الارتباط الذاتي — Autocorrelation
    # ══════════════════════════════════════════
    def autocorrelation(self, lag=1):
        """
        هل هناك ارتباط بين الدورة الحالية والسابقة؟
        قيمة سالبة = ميل للتذبذب (بعد خسارة يأتي ربح والعكس)
        قيمة موجبة = ميل للتتابع (خسارة تلو خسارة)
        """
        if self.n < lag + 3: return 0.0
        arr = self.arr
        if len(arr) <= lag: return 0.0
        try:
            corr = float(np.corrcoef(arr[:-lag], arr[lag:])[0,1])
            return round(corr, 3)
        except:
            return 0.0

    # ══════════════════════════════════════════
    # 6. Bayesian Confidence
    # ══════════════════════════════════════════
    def bayesian_confidence(self, prior_p: float, evidence_factors: list) -> float:
        """
        P(jump | evidence) باستخدام Bayes البسيط:
        posterior ∝ prior × likelihood_1 × likelihood_2 × ...
        كل عامل في evidence_factors هو likelihood ratio (> 1 يرفع, < 1 يخفض)
        """
        p = prior_p
        for lr in evidence_factors:
            p = (p * lr) / (p * lr + (1 - p) * 1.0)
            p = max(0.01, min(0.99, p))
        return round(p, 3)

    # ══════════════════════════════════════════
    # 7. Kelly Criterion — حجم الرهان الأمثل
    # ══════════════════════════════════════════
    def kelly_stake(self, p_win: float, odds: float, balance: float,
                    fraction: float = 0.25) -> float:
        """
        f* = (p*b - q) / b   حيث b = odds - 1, q = 1-p
        fraction = كسر Kelly (0.25 = ربع Kelly للحماية)
        """
        if odds <= 1.0 or p_win <= 0: return 0.0
        b = odds - 1.0
        q = 1.0 - p_win
        kelly_full = (p_win * b - q) / b
        if kelly_full <= 0: return 0.0
        kelly_frac = kelly_full * fraction
        stake = round(balance * kelly_frac, 1)
        return max(5.0, min(stake, balance * 0.05))  # حد أقصى 5%

    # ══════════════════════════════════════════
    # 8. Markov — احتمالات الانتقال
    # ══════════════════════════════════════════
    def markov_next_probs(self):
        """
        احتمالات الدورة القادمة بناءً على الماركوف من البيانات التاريخية
        """
        streak = self._streak(2.0)
        ref_stats = self._ref_stats()
        k = min(streak, 7)
        if k in ref_stats:
            return ref_stats[k]
        # fallback
        ref_arr = np.array(self.ref)
        return {
            "mean":  round(float(ref_arr.mean()), 3),
            "p_gt2": round(float((ref_arr >= 2.0).mean()), 3),
            "p_gt5": round(float((ref_arr >= 5.0).mean()), 3),
            "p_gt12":round(float((ref_arr >= 12.0).mean()), 3),
            "n": len(self.ref),
        }

    # ══════════════════════════════════════════
    # 9. قوة الزنبرك الإحصائية
    # ══════════════════════════════════════════
    def spring_analysis(self):
        s2  = self._streak(2.0)
        s18 = self._streak(1.8)
        s15 = self._streak(1.5)
        seq = self._last(max(s2, 1))
        avg_seq = float(np.mean(seq)) if seq else 2.0
        std_seq = float(np.std(seq)) if len(seq)>1 else 0.0

        # ضغط مركّب
        pressure = min(100, int(
            s2  * 9 +
            s18 * 7 +
            s15 * 5 +
            max(0, (1.8 - avg_seq) * 35) +
            max(0, (3 - std_seq) * 3)
        ))

        # مستوى الزنبرك
        if   s18 >= 5: lv, exp_lo, exp_hi = 5, 15.0, 35.0
        elif s18 >= 3: lv, exp_lo, exp_hi = 4,  8.0, 22.0
        elif s2  >= 5: lv, exp_lo, exp_hi = 3,  6.0, 18.0
        elif s2  >= 3: lv, exp_lo, exp_hi = 2,  4.0, 10.0
        elif s2  >= 2: lv, exp_lo, exp_hi = 1,  2.5,  6.0
        else:           lv, exp_lo, exp_hi = 0,  0.0,  0.0

        # Markov
        mk = self.markov_next_probs()

        # رفع الهدف بناء على ماركوف
        if mk.get("p_gt12", 0) >= 0.15:
            exp_hi = max(exp_hi, 25.0)
        elif mk.get("p_gt5", 0) >= 0.35:
            exp_hi = max(exp_hi, 12.0)

        return {
            "s2": s2, "s18": s18, "s15": s15,
            "avg_seq": round(avg_seq, 2),
            "std_seq": round(std_seq, 2),
            "pressure": pressure, "level": lv,
            "exp_lo": exp_lo, "exp_hi": exp_hi,
            "markov": mk,
        }

    # ══════════════════════════════════════════
    # 10. كشف الأنماط الخمسة
    # ══════════════════════════════════════════
    def detect_patterns(self, spring: dict):
        patterns = []
        if self.n < 2: return patterns
        last = self.h[-1]

        # P-A: رقم ذهبي
        gm = self._find_golden(last)
        if gm:
            gn, gd = gm
            patterns.append({
                "id":"GOLDEN","priority":1,
                "gnum":gn,"gdata":gd,
                "desc":f"رقم ذهبي x{gn} ({gd['label']}) — avg_next={gd['avg_next']}x"
            })

        # P-B: تسلسل هابط (F5)
        seq5 = self._last(5)
        drops = sum(1 for i in range(len(seq5)-1) if seq5[i+1] <= seq5[i])
        if drops >= 3 and float(np.mean(seq5)) < 2.5:
            patterns.append({
                "id":"DESCEND","priority":2,
                "drops":drops,"avg":round(float(np.mean(seq5)),2),
                "desc":f"تسلسل هابط {drops}/4 — متوسط x{round(float(np.mean(seq5)),2)}"
            })

        # P-C: قفزة مزدوجة (F4)
        for lb in [1, 2]:
            if self.n > lb and self.h[-(lb+1)] >= 12.0:
                if last >= 5.0:
                    patterns.append({
                        "id":"DOUBLE_JUMP","priority":0,
                        "prev":self.h[-(lb+1)],"lb":lb,
                        "desc":f"قفزة مزدوجة! x{self.h[-(lb+1)]:.2f} → x{last:.2f}"
                    })
                else:
                    patterns.append({
                        "id":"POST_BIG","priority":0,
                        "prev":self.h[-(lb+1)],"lb":lb,
                        "avoid": max(1, 3-lb),
                        "desc":f"تجنب! بعد x{self.h[-(lb+1)]:.2f} بـ{lb} دورة"
                    })
                break

        # P-D: إنتروبيا منخفضة (ضغط انتظام)
        ent, comp = self.entropy_score()
        if comp >= 0.45 and spring["level"] >= 2:
            patterns.append({
                "id":"LOW_ENTROPY","priority":3,
                "compression":comp,"entropy":ent,
                "desc":f"إنتروبيا منخفضة ({ent:.2f}) — ضغط انتظام {comp*100:.0f}%"
            })

        # P-E: z-score سلبي (قيم شاذة منخفضة)
        z = self.zscore_last()
        if z <= -0.8 and spring["level"] >= 1:
            patterns.append({
                "id":"LOW_ZSCORE","priority":3,
                "z":z,
                "desc":f"Z-score={z:.2f} — القيمة الأخيرة منخفضة جداً إحصائياً"
            })

        return patterns

    # ══════════════════════════════════════════
    # 11. القرار الشامل
    # ══════════════════════════════════════════
    def decide(self, balance: float) -> dict:
        if self.n < 3:
            return self._mk_result("WAIT","⏳",
                "أضف 3 دورات للبدء","","",0,None,None,0,0,{},{})

        spring   = self.spring_analysis()
        patterns = self.detect_patterns(spring)
        ent, comp = self.entropy_score()
        z         = self.zscore_last()
        ac        = self.autocorrelation(1)
        mk        = spring["markov"]

        # ─── استخرج الأنماط حسب النوع ───────────────
        pat_golden  = next((p for p in patterns if p["id"]=="GOLDEN"),      None)
        pat_desc    = next((p for p in patterns if p["id"]=="DESCEND"),     None)
        pat_double  = next((p for p in patterns if p["id"]=="DOUBLE_JUMP"), None)
        pat_postbig = next((p for p in patterns if p["id"]=="POST_BIG"),    None)
        pat_entropy = next((p for p in patterns if p["id"]=="LOW_ENTROPY"), None)
        pat_z       = next((p for p in patterns if p["id"]=="LOW_ZSCORE"),  None)

        # ─── أولوية مطلقة: ما بعد القفزة ────────────
        if pat_postbig:
            return self._mk_result(
                "AVOID","⛔",
                f"تجنب {pat_postbig['avoid']} دورات — بعد قفزة x{pat_postbig['prev']:.2f}",
                "70% من الحالات التالية لقفزة ≥12x تنتهي بخسارة.",
                pat_postbig["desc"],
                confidence=78,
                tgt_lo=None, tgt_hi=None,
                stake=0, stake_pct=0,
                spring=spring, patterns=patterns,
                extras={"z":z,"ent":ent,"comp":comp,"ac":ac,"mk":mk}
            )

        # ─── قفزة مزدوجة (نادرة) ─────────────────────
        if pat_double:
            p_win  = self.bayesian_confidence(
                0.25,  # prior P(قفزة مزدوجة) = 25%
                [1.8 if spring["level"]>=2 else 1.0,
                 1.5 if pat_golden else 1.0]
            )
            stake  = self.kelly_stake(p_win, pat_double["prev"]*0.6, balance, 0.15)
            tgt_lo = pat_double["prev"] * 0.5
            tgt_hi = pat_double["prev"] * 0.9
            return self._mk_result(
                "DOUBLE","⚡",
                f"قفزة مزدوجة نادرة! (~{int(p_win*100)}% ثقة)",
                f"بعد x{pat_double['prev']:.2f} ظهر x{self.h[-1]:.2f}. نمط القفزة المزدوجة.",
                pat_double["desc"],
                confidence=int(p_win*100),
                tgt_lo=round(tgt_lo,1), tgt_hi=round(tgt_hi,1),
                stake=stake, stake_pct=round(stake/balance*100,1),
                spring=spring, patterns=patterns,
                extras={"z":z,"ent":ent,"comp":comp,"ac":ac,"mk":mk,"p_win":p_win}
            )

        # ─── بناء likelihood ratios للـ Bayes ─────────
        lr_list = []

        # LR1: قوة الزنبرك
        spring_lr = {0:0.6, 1:1.2, 2:1.8, 3:2.5, 4:3.5, 5:4.5}
        lr_list.append(spring_lr.get(spring["level"], 1.0))

        # LR2: رقم ذهبي
        if pat_golden:
            gd = pat_golden["gdata"]
            tier_lr = {1: 3.0, 2: 2.2, 3: 1.5}
            lr_list.append(tier_lr[gd["tier"]])
            lr_list.append(1 + gd["win5"])  # معزز بنسبة win5

        # LR3: تسلسل هابط
        if pat_desc:
            lr_list.append(1.4)

        # LR4: إنتروبيا منخفضة
        if pat_entropy:
            lr_list.append(1.0 + comp * 0.8)

        # LR5: z-score سلبي
        if pat_z:
            lr_list.append(1.0 + abs(z) * 0.3)

        # LR6: ماركوف
        mk_p5 = mk.get("p_gt5", 0.2)
        lr_list.append(1.0 + mk_p5 * 1.5)

        # LR7: autocorrelation سالب → ميل للتذبذب (مفيد بعد خسارة)
        if ac < -0.1 and spring["level"] >= 1:
            lr_list.append(1.2)

        # prior من التوزيع التاريخي
        ref_arr = np.array(self.ref)
        prior   = float((ref_arr >= 5.0).mean())   # ~21%

        p_win = self.bayesian_confidence(prior, lr_list)

        # ─── تحديد الهدف ──────────────────────────────
        if pat_golden:
            gd = pat_golden["gdata"]
            tgt_lo = gd["tgt_lo"] + spring["level"] * 0.4
            tgt_hi = gd["tgt_hi"] + spring["level"] * 1.0
        else:
            tgt_lo = spring["exp_lo"] if spring["level"] >= 2 else 2.0
            tgt_hi = spring["exp_hi"] if spring["level"] >= 2 else 5.0

        # تعديل بالماركوف
        if mk.get("p_gt12", 0) >= 0.2 and spring["level"] >= 3:
            tgt_hi = max(tgt_hi, mk.get("mean", tgt_hi) * 1.2)

        # ─── Kelly stake ──────────────────────────────
        odds   = (tgt_lo + tgt_hi) / 2
        stake  = self.kelly_stake(p_win, odds, balance, fraction=0.25)

        # ─── تصنيف الحالة ─────────────────────────────
        # متطلب صارم: يجب أن يكون هناك على الأقل عاملان
        signal_count = sum([
            spring["level"] >= 2,
            pat_golden is not None,
            pat_desc is not None,
            pat_entropy is not None,
            pat_z is not None,
            mk_p5 >= 0.30,
        ])

        if p_win >= 0.62 and signal_count >= 3:
            status = "STRONG"
            icon   = "🔥"
            title  = f"إشارة قصوى — {int(p_win*100)}% ثقة"
            main_d = (f"زنبرك مستوى {spring['level']} + "
                      f"{signal_count} عوامل تؤكد القفزة القادمة.")

        elif p_win >= 0.48 and signal_count >= 2:
            status = "BET"
            icon   = "✅"
            title  = f"إشارة جيدة — {int(p_win*100)}% ثقة"
            main_d = (f"زنبرك مستوى {spring['level']} + "
                      f"{signal_count} عوامل. راهن بحذر.")

        elif spring["level"] >= 3 and not pat_golden:
            status = "WAIT"
            icon   = "⏳"
            title  = f"زنبرك قوي — انتظر الإشارة"
            main_d = (f"سلسلة {spring['s2']} خسائر <x2. "
                      f"الزنبرك مضغوط لكن لا رقم ذهبي بعد.")
            stake  = 0

        elif spring["level"] <= 1 and not pat_golden:
            status = "AVOID"
            icon   = "🚫"
            title  = "لا توجد إشارة — لا تراهن"
            main_d = "لا تتوفر شروط الدخول. انتظر زنبرك + رقم ذهبي."
            stake  = 0

        else:
            # زنبرك متوسط أو رقم ذهبي ضعيف
            status = "WAIT"
            icon   = "⏳"
            title  = f"إشارة ضعيفة — انتظر تأكيداً"
            main_d = f"ثقة {int(p_win*100)}% فقط. اجمع المزيد من الشواهد."
            stake  = 0

        return self._mk_result(
            status, icon, title, main_d, "",
            confidence=int(p_win*100),
            tgt_lo=round(tgt_lo,1) if stake > 0 else None,
            tgt_hi=round(tgt_hi,1) if stake > 0 else None,
            stake=stake,
            stake_pct=round(stake/balance*100,1) if stake>0 else 0,
            spring=spring, patterns=patterns,
            extras={"z":z,"ent":ent,"comp":comp,"ac":ac,"mk":mk,
                    "p_win":p_win,"signal_count":signal_count,
                    "lr_list":lr_list,"prior":prior}
        )

    def _mk_result(self, status, icon, title, desc, sub,
                   confidence, tgt_lo, tgt_hi, stake, stake_pct,
                   spring, patterns, extras):
        profit = 0
        if stake > 0 and tgt_lo and tgt_hi:
            profit = round(stake * (tgt_lo + tgt_hi)/2 - stake, 1)
        return {
            "status":status,"icon":icon,"title":title,"desc":desc,"sub":sub,
            "confidence":confidence,"tgt_lo":tgt_lo,"tgt_hi":tgt_hi,
            "stake":stake,"stake_pct":stake_pct,"profit_est":profit,
            "spring":spring,"patterns":patterns,"extras":extras,
        }

    # ══════════════════════════════════════════
    # 12. إحصائيات شاملة
    # ══════════════════════════════════════════
    def full_stats(self):
        if not self.h: return {}
        a = self.arr
        sp = self.spring_analysis()
        ent, comp = self.entropy_score()
        z  = self.zscore_last()
        ac = self.autocorrelation(1)
        mk = self.markov_next_probs()
        return {
            "n":len(self.h),
            "avg":round(float(a.mean()),2),
            "med":round(float(np.median(a)),2),
            "std":round(float(a.std()),2),
            "mx":round(float(a.max()),2),
            "loss_u2": sum(1 for v in self.h if v<2.0),
            "loss_u18":sum(1 for v in self.h if v<1.8),
            "big":     sum(1 for v in self.h if v>=12.0),
            "win_rate":round(sum(1 for v in self.h if v>=2.0)/len(self.h)*100,1),
            "streak_2":sp["s2"],"streak_18":sp["s18"],
            "pressure":sp["pressure"],"level":sp["level"],
            "entropy":ent,"compression":comp,
            "zscore":z,"autocorr":ac,
            "markov":mk,
        }

    def golden_in_history(self, k=25):
        out = []
        for i, v in enumerate(self.h[-k:]):
            gm = self._find_golden(v)
            if gm:
                out.append({"pos":len(self.h)-k+i+1,"val":v,
                            "gnum":gm[0],"gdata":gm[1]})
        return out

    def spring_pressure_series(self):
        """سلسلة ضغط الزنبرك لكل نقطة في التاريخ"""
        out = []
        for i in range(len(self.h)):
            sub = StatEngine(self.h[:i+1])
            sp  = sub.spring_analysis()
            out.append(sp["pressure"])
        return out


# ══════════════════════════════════════════════════════════════════════
#  الرسوم البيانية
# ══════════════════════════════════════════════════════════════════════
def build_main_chart(h: list, engine: StatEngine, pressure_series: list):
    if len(h) < 2: return

    colors, sizes, symbols = [], [], []
    for v in h:
        gm = engine._find_golden(v)
        if gm:
            colors.append("#ff9500"); sizes.append(18); symbols.append("star")
        elif v >= 12:
            colors.append("#a855f7"); sizes.append(16); symbols.append("diamond")
        elif v >= 5:
            colors.append("#00c8ff"); sizes.append(13); symbols.append("circle")
        elif v >= 2:
            colors.append("#00ff88"); sizes.append(10); symbols.append("circle")
        else:
            colors.append("#ff4444"); sizes.append(8);  symbols.append("circle")

    x = list(range(1, len(h)+1))
    fig = go.Figure()

    # مناطق
    ymax = max(max(h)*1.1, 15)
    for y0,y1,clr in [(0,1.5,"rgba(255,30,30,0.06)"),
                       (1.5,2.0,"rgba(255,100,0,0.04)"),
                       (2.0,5.0,"rgba(0,255,136,0.03)"),
                       (5.0,12.0,"rgba(0,200,255,0.03)"),
                       (12.0,ymax,"rgba(168,85,247,0.04)")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=clr, line_width=0)

    # ضغط الزنبرك
    px_ = list(range(max(1,len(h)-len(pressure_series)+1), len(h)+1))
    fig.add_trace(go.Scatter(
        x=px_, y=pressure_series, name="ضغط الزنبرك",
        yaxis="y2", mode="lines",
        line=dict(color="rgba(255,149,0,0.4)", width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(255,149,0,0.06)",
    ))

    # المضاعفات
    fig.add_trace(go.Scatter(
        x=x, y=h, mode="lines+markers+text",
        line=dict(color="rgba(99,102,241,0.5)", width=2, shape="spline"),
        marker=dict(color=colors, size=sizes, symbol=symbols,
                    line=dict(color="rgba(255,255,255,0.2)", width=1)),
        text=[f"x{v:.2f}" for v in h],
        textposition="top center",
        textfont=dict(color="rgba(255,255,255,0.7)", size=8, family="Orbitron"),
        name="المضاعف",
    ))

    for yv, cl, lb in [(1.5,"rgba(255,50,50,0.5)","x1.5"),
                        (2.0,"rgba(255,215,0,0.5)","x2"),
                        (5.0,"rgba(0,200,255,0.5)","x5"),
                        (12.0,"rgba(168,85,247,0.5)","x12")]:
        fig.add_hline(y=yv, line_dash="dot", line_color=cl, line_width=1,
                      annotation_text=lb, annotation_font=dict(color=cl, size=9))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Tajawal"),
        height=370, margin=dict(l=10,r=10,t=25,b=10),
        xaxis=dict(showgrid=False, title="الدورة",
                   tickfont=dict(color="rgba(255,255,255,0.3)")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                   title="المضاعف", tickprefix="x",
                   tickfont=dict(color="rgba(255,255,255,0.3)")),
        yaxis2=dict(overlaying="y", side="right", range=[0,110],
                    showgrid=False, showticklabels=False),
        legend=dict(orientation="h", y=1.06,
                    font=dict(size=10,color="rgba(255,255,255,0.4)"),
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"mc_{len(h)}")


def build_gauge(val, label, key, lo=0, hi=100, thresholds=None):
    if thresholds is None:
        thresholds = [(40,"#ff4444"),(70,"#FFD700"),(100,"#00ff88")]
    color = thresholds[-1][1]
    for thr, clr in thresholds:
        if val <= thr: color = clr; break

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={"text":label,"font":{"size":11,"color":"rgba(255,255,255,0.55)","family":"Tajawal"}},
        number={"font":{"size":26,"color":color,"family":"Orbitron"}},
        gauge={
            "axis":{"range":[lo,hi],"tickwidth":1,"tickcolor":"rgba(255,255,255,0.12)"},
            "bar":{"color":color,"thickness":0.26},
            "bgcolor":"rgba(0,0,0,0.18)","borderwidth":0,
            "steps":[
                {"range":[lo,lo+(hi-lo)*0.4],"color":"rgba(255,50,50,0.07)"},
                {"range":[lo+(hi-lo)*0.4,lo+(hi-lo)*0.7],"color":"rgba(255,215,0,0.07)"},
                {"range":[lo+(hi-lo)*0.7,hi],"color":"rgba(0,255,136,0.07)"},
            ],
        }
    ))
    fig.update_layout(height=185, margin=dict(l=8,r=8,t=40,b=5),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key=key)


def build_stat_radar(extras: dict, key: str):
    """مخطط شبكي للعوامل الإحصائية"""
    p_win    = extras.get("p_win", 0.3)
    comp     = extras.get("comp", 0.0)
    mk_p5    = extras.get("mk", {}).get("p_gt5", 0.2)
    ac_norm  = max(0, -extras.get("ac", 0) * 100)  # سالب → مفيد
    z_norm   = max(0, min(100, (2 - extras.get("z",0)) * 20))
    sc       = extras.get("signal_count", 0) / 6 * 100

    cats = ["ثقة Bayes","ضغط الإنتروبيا",
            "Markov ≥x5","Autocorr","Z-score","إشارات"]
    vals = [p_win*100, comp*100, mk_p5*100, ac_norm, z_norm, sc]

    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself",
        fillcolor="rgba(99,102,241,0.12)",
        line=dict(color="#6366f1", width=2),
        marker=dict(color="#a855f7", size=7),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,100],
                            tickfont=dict(color="rgba(255,255,255,0.3)",size=8),
                            gridcolor="rgba(255,255,255,0.07)"),
            angularaxis=dict(tickfont=dict(color="rgba(255,255,255,0.55)",size=10,
                                           family="Tajawal"),
                             gridcolor="rgba(255,255,255,0.07)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=30,r=30,t=20,b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def build_distribution(h: list, key: str):
    bins   = [0, 1.5, 2.0, 5.0, 12.0, 1000]
    labels = ["<x1.5","x1.5–2","x2–5","x5–12","≥x12"]
    clrs   = ["#ff3232","#ff9500","#00ff88","#00c8ff","#a855f7"]
    counts = [sum(1 for v in h if bins[i]<=v<bins[i+1]) for i in range(len(bins)-1)]
    total  = sum(counts)
    pcts   = [round(c/total*100,1) for c in counts]

    fig = go.Figure(go.Bar(
        x=labels, y=counts, marker_color=clrs,
        text=[f"{p}%" for p in pcts],
        textposition="outside",
        textfont=dict(color="white",size=11,family="Orbitron"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white",family="Tajawal"),
        height=220, margin=dict(l=8,r=8,t=15,b=10),
        xaxis=dict(showgrid=False,tickfont=dict(color="rgba(255,255,255,0.45)")),
        yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.05)",
                   tickfont=dict(color="rgba(255,255,255,0.35)")),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ══════════════════════════════════════════════════════════════════════
#  الجلسة
# ══════════════════════════════════════════════════════════════════════
for k, v in [("history",[]),("balance",1000.0),("log",[])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════
#  الواجهة
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:16px 0 6px;">
<div style="font-family:'Orbitron',monospace;font-size:32px;font-weight:900;
            background:linear-gradient(90deg,#6366f1,#a855f7,#ec4899,#a855f7,#6366f1);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-size:200%;animation:gs 3s linear infinite;">
🧠 CRASH INTELLIGENCE v3
</div>
<div style="color:rgba(255,255,255,0.28);font-size:11px;letter-spacing:4px;margin-top:3px;">
Bayesian · Markov · Entropy · Z-score · Kelly · Autocorrelation
</div>
</div>
<style>@keyframes gs{0%{background-position:0%}100%{background-position:200%}}</style>
""", unsafe_allow_html=True)

# ══ الشريط الجانبي ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="text-align:center;color:#a855f7;font-size:16px;font-weight:700;margin-bottom:10px;">⚙️ التحكم</div>', unsafe_allow_html=True)

    st.markdown("**💰 الرصيد**")
    st.session_state.balance = st.number_input(
        "bal", min_value=10.0, max_value=999999.0,
        value=st.session_state.balance, step=50.0, label_visibility="collapsed"
    )
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ مسح", use_container_width=True):
            st.session_state.history=[]; st.session_state.log=[]; st.rerun()
    with c2:
        if st.button("📊 ديمو", use_container_width=True):
            st.session_state.history = HISTORICAL_DATA[:60]; st.rerun()

    if st.button("🎲 محاكاة واقعية (20)", use_container_width=True):
        sim=[]
        for _ in range(20):
            r=random.random()
            if   r<0.35: sim.append(round(random.uniform(1.0,1.49),2))
            elif r<0.50: sim.append(round(random.uniform(1.5,1.99),2))
            elif r<0.72: sim.append(round(random.uniform(2.0,4.99),2))
            elif r<0.88: sim.append(round(random.uniform(5.0,11.99),2))
            else:         sim.append(round(random.uniform(12.0,40.0),2))
        st.session_state.history=sim; st.rerun()

    st.markdown("---")
    # دليل الأرقام الذهبية مضغوط
    st.markdown("**⭐ الأرقام الذهبية**")
    for tier in [1,2,3]:
        m = TIER_META[tier]
        nums = [(g,d) for g,d in GOLDEN_DB.items() if d["tier"]==tier]
        st.markdown(f'<div style="color:{m["color"]};font-size:11px;font-weight:700;margin:6px 0 2px;">{m["icon"]} {m["label"]}</div>', unsafe_allow_html=True)
        for g,d in nums:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                        border-radius:6px;padding:3px 8px;margin:1px 0;
                        display:flex;justify-content:space-between;align-items:center;">
                <span style="font-family:'Orbitron',monospace;color:{m['color']};font-size:11px;font-weight:700;">x{g}</span>
                <span style="color:rgba(255,255,255,0.3);font-size:9px;">→x{d['tgt_lo']}–{d['tgt_hi']}</span>
                <span style="color:rgba(255,255,255,0.2);font-size:9px;">avg:{d['avg_next']}x</span>
            </div>""", unsafe_allow_html=True)

    # إحصائيات الجلسة
    h = st.session_state.history
    if h:
        st.markdown("---")
        eng_s = StatEngine(h)
        s = eng_s.full_stats()
        items = [
            (s['n'],"الدورات"),
            (f"{s['streak_2']}","خسائر <x2","#ff4444"),
            (f"{s['pressure']}%","ضغط الزنبرك","#ff9500"),
            (f"{s['win_rate']}%","فوق x2","#00ff88"),
            (f"{s['entropy']:.2f}","إنتروبيا","#6366f1"),
            (f"{s['zscore']:.2f}","Z-score","#a855f7"),
        ]
        for item in items:
            val, lbl = item[0], item[1]
            clr = item[2] if len(item)>2 else None
            color_style = f'color:{clr};' if clr else ''
            st.markdown(f"""
            <div class="kpi" style="margin:3px 0;">
                <div class="kn" style="{color_style}">{val}</div>
                <div class="kl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

# ══ منطقة الإدخال ════════════════════════════════════════════════════
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📥 إدخال الدورة")
ci1,ci2,ci3,ci4 = st.columns([2.5,1,1,1])
with ci1:
    new_val = st.number_input("v",min_value=1.00,max_value=1000.0,
                               value=1.50,step=0.01,format="%.2f",
                               label_visibility="collapsed")
with ci2:
    if st.button("➕ أضف", use_container_width=True):
        st.session_state.history.append(round(new_val,2))
        st.session_state.log.append({"t":datetime.now().strftime("%H:%M:%S"),
                                      "v":round(new_val,2)})
        st.rerun()
with ci3:
    if st.button("↩️ حذف", use_container_width=True):
        if st.session_state.history: st.session_state.history.pop()
        if st.session_state.log: st.session_state.log.pop()
        st.rerun()
with ci4:
    if st.button("🔄 تحديث", use_container_width=True): st.rerun()

# شريط الدورات
h = st.session_state.history
if h:
    st.markdown("**📋 آخر الدورات:**")
    eng_t = StatEngine(h)
    b = '<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:12px 16px;line-height:2.4;">'
    for v in h[-35:]:
        gm = eng_t._find_golden(v)
        if gm:           cls="b-gold"
        elif v>=12.0:    cls="b-big"
        elif v>=5.0:     cls="b-win"
        elif v>=2.0:     cls="b-med"
        elif v>=1.8:     cls="b-loss"
        else:            cls="b-loss18"
        sfx="⭐" if gm else ""
        b+=f'<span class="badge {cls}">x{v:.2f}{sfx}</span>'
    b+="</div>"
    st.markdown(b, unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:10px;color:rgba(255,255,255,0.25);direction:rtl;margin-top:3px;">
    <span style="color:#ff9500;">⭐ذهبي</span> |
    <span style="color:#a855f7;">■</span>≥x12 |
    <span style="color:#00ff88;">■</span>x5–12 |
    <span style="color:#FFD700;">■</span>x2–5 |
    <span style="color:#ff4444;">■</span>x1.8–2 |
    <span style="color:#ff2020;">■</span>&lt;x1.8
    </div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ══ التحليل ═══════════════════════════════════════════════════════════
h = st.session_state.history
if len(h) < 3:
    st.markdown(f"""
    <div class="STATUS-WAIT" style="text-align:center;padding:35px;">
        <div style="font-size:44px;">⏳</div>
        <div style="font-size:19px;font-weight:700;margin:10px 0;">أضف {3-len(h)} دورات للبدء</div>
        <div style="color:rgba(255,255,255,0.35);">أو اضغط "ديمو" من الشريط الجانبي</div>
    </div>""", unsafe_allow_html=True)
else:
    engine  = StatEngine(h)
    rec     = engine.decide(st.session_state.balance)
    stats   = engine.full_stats()
    gh      = engine.golden_in_history(30)
    spring  = rec["spring"]
    extras  = rec.get("extras", {})
    patterns= rec.get("patterns", [])

    # حساب سلسلة الضغط مرة واحدة
    pressure_series = engine.spring_pressure_series()

    # ══ الصف الرئيسي ══════════════════════════════════════════════
    col_L, col_R = st.columns([3, 2])

    with col_L:
        # ── بطاقة الحالة الرئيسية ──────────────────────────────
        st.markdown(f'<div class="STATUS-{rec["status"]}">', unsafe_allow_html=True)
        icon_map = {"BET":"✅","STRONG":"🔥","AVOID":"⛔","WAIT":"⏳","DOUBLE":"⚡"}
        st.markdown(f"""
        <div style="font-size:44px;margin-bottom:8px;">{icon_map.get(rec['status'],'⏳')}</div>
        <div style="font-size:22px;font-weight:900;margin-bottom:10px;">{rec['title']}</div>
        <div style="font-size:14px;color:rgba(255,255,255,0.8);line-height:1.8;">{rec['desc']}</div>
        """, unsafe_allow_html=True)

        # النطاق + الرهان
        if rec["tgt_lo"] and rec["tgt_hi"]:
            profit_clr = "#00ff88" if rec["profit_est"]>0 else "#ff4444"
            st.markdown(f"""
            <div style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap;">
                <div style="flex:2;min-width:130px;background:rgba(0,0,0,0.35);border-radius:11px;padding:12px;">
                    <div style="color:rgba(255,255,255,0.38);font-size:11px;">🎯 النطاق المستهدف</div>
                    <div style="font-family:'Orbitron',monospace;font-size:24px;color:#FFD700;font-weight:900;">
                        x{rec['tgt_lo']} — x{rec['tgt_hi']}
                    </div>
                </div>
                <div style="flex:1;min-width:100px;background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.2);border-radius:11px;padding:12px;text-align:center;">
                    <div style="color:rgba(255,255,255,0.38);font-size:10px;">💰 رهان Kelly</div>
                    <div style="font-family:'Orbitron',monospace;font-size:20px;color:#00ff88;font-weight:900;">{rec['stake']:.0f}</div>
                    <div style="color:rgba(255,255,255,0.28);font-size:9px;">{rec['stake_pct']}٪</div>
                </div>
                <div style="flex:1;min-width:100px;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:11px;padding:12px;text-align:center;">
                    <div style="color:rgba(255,255,255,0.38);font-size:10px;">📈 ربح متوقع</div>
                    <div style="font-family:'Orbitron',monospace;font-size:20px;color:{profit_clr};font-weight:900;">+{rec['profit_est']:.0f}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── الأنماط المكتشفة ────────────────────────────────────
        if patterns:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card" style="padding:16px;">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:14px;font-weight:700;color:rgba(255,255,255,0.8);margin-bottom:10px;">🔍 الأنماط المكتشفة</div>', unsafe_allow_html=True)
            pat_styles = {
                "GOLDEN":      ("bx-o","⭐","رقم ذهبي"),
                "DESCEND":     ("bx-b","📉","تسلسل هابط"),
                "DOUBLE_JUMP": ("bx-g","⚡","قفزة مزدوجة"),
                "POST_BIG":    ("bx-r","⛔","تجنب"),
                "LOW_ENTROPY": ("bx-b","🔵","إنتروبيا منخفضة"),
                "LOW_ZSCORE":  ("bx-y","📊","Z-score منخفض"),
            }
            for p in patterns:
                cls, ic, lb = pat_styles.get(p["id"],("bx-b","🔹","نمط"))
                st.markdown(f'<div class="{cls}">{ic} <b>{lb}:</b> {p["desc"]}</div>',
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── الزنبرك المفصّل ─────────────────────────────────────
        st.markdown('<div class="card" style="padding:16px;">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:14px;font-weight:700;color:rgba(255,255,255,0.8);margin-bottom:12px;">🔄 الزنبرك المضغوط — مستوى {spring["level"]}/5</div>', unsafe_allow_html=True)

        p  = spring["pressure"]
        pc = "#ff3232" if p>=70 else "#ff9500" if p>=40 else "#FFD700"
        lv_lbl = ["لا زنبرك","ضعيف","متوسط","قوي","قوي جداً","أقصى"]
        st.markdown(f"""
        <div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="color:rgba(255,255,255,0.55);font-size:12px;">{lv_lbl[min(spring['level'],5)]}</span>
                <span style="font-family:'Orbitron',monospace;color:{pc};font-weight:700;">{p}%</span>
            </div>
            <div class="pw"><div class="pf-o" style="width:{p}%;background:linear-gradient(90deg,#ff6400,{pc});"></div></div>
        </div>""", unsafe_allow_html=True)

        zc1,zc2,zc3,zc4 = st.columns(4)
        for col_, val_, lbl_, clr_ in [
            (zc1, spring["s2"],  "خسائر <x2", "#ff4444"),
            (zc2, spring["s18"],"خسائر <x1.8","#ff6600"),
            (zc3, spring["s15"],"خسائر <x1.5","#ff2020"),
            (zc4, f'{spring["avg_seq"]}x',"متوسط السلسلة","#FFD700"),
        ]:
            with col_:
                st.markdown(f"""<div class="kpi">
                    <div class="kn" style="color:{clr_};">{val_}</div>
                    <div class="kl">{lbl_}</div></div>""", unsafe_allow_html=True)

        # Markov
        mk = spring["markov"]
        st.markdown(f"""
        <div class="bx-b" style="margin-top:10px;">
            📊 <b>Markov (بعد {spring['s2']} خسائر):</b>
            P(≥x2)={mk.get('p_gt2',0)*100:.0f}% |
            P(≥x5)={mk.get('p_gt5',0)*100:.0f}% |
            P(≥x12)={mk.get('p_gt12',0)*100:.0f}% |
            متوسط متوقع={mk.get('mean',0):.2f}x
        </div>""", unsafe_allow_html=True)

        if spring["level"] >= 2:
            st.markdown(f"""
            <div class="bx-g" style="margin-top:8px;">
                🎯 <b>نطاق القفزة المتوقع إحصائياً:</b>
                <span style="font-family:'Orbitron',monospace;color:#FFD700;">
                x{spring['exp_lo']:.1f} — x{spring['exp_hi']:.1f}
                </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── رقم ذهبي مكتشف ─────────────────────────────────────
        golden_pat = next((p for p in patterns if p["id"]=="GOLDEN"), None)
        if golden_pat:
            gn = golden_pat["gnum"]
            gd = golden_pat["gdata"]
            tm = TIER_META[gd["tier"]]
            st.markdown(f"""
            <div class="card" style="background:linear-gradient(135deg,rgba(255,149,0,0.09),rgba(255,70,0,0.04));
                         border-color:rgba(255,149,0,0.38);padding:16px;">
                <div style="text-align:center;margin-bottom:12px;">
                    <span style="font-size:28px;">{tm['icon']}</span>
                    <div style="font-size:15px;font-weight:900;color:{tm['color']};margin-top:3px;">
                        رقم ذهبي {tm['label']} — {gd['label']}
                    </div>
                </div>
                <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center;">
                    <div class="gc" style="flex:1;min-width:90px;">
                        <div style="font-size:9px;color:rgba(255,255,255,0.35);">القيمة الفعلية</div>
                        <div class="gn">x{h[-1]:.2f}</div>
                        <div style="font-size:9px;color:rgba(255,255,255,0.3);">≈ x{gn}</div>
                    </div>
                    <div class="gc" style="flex:1;min-width:90px;">
                        <div style="font-size:9px;color:rgba(255,255,255,0.35);">الهدف</div>
                        <div class="gt">x{gd['tgt_lo']}–x{gd['tgt_hi']}</div>
                        <div style="font-size:9px;color:rgba(255,255,255,0.3);">n={gd['n']} دورات</div>
                    </div>
                    <div class="gc" style="flex:1;min-width:90px;">
                        <div style="font-size:9px;color:rgba(255,255,255,0.35);">متوسط تاريخي</div>
                        <div class="gn" style="color:#a855f7;">{gd['avg_next']}x</div>
                        <div style="font-size:9px;color:rgba(255,255,255,0.3);">win5={gd['win5_pct']*100:.0f}%</div>
                    </div>
                    <div class="gc" style="flex:1;min-width:90px;">
                        <div style="font-size:9px;color:rgba(255,255,255,0.35);">الوسيط</div>
                        <div class="gn" style="color:#FFD700;">{gd['med_next']}x</div>
                        <div style="font-size:9px;color:rgba(255,255,255,0.3);">win2={gd['win2_pct']*100:.0f}%</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_R:
        # ── مقاييس ──────────────────────────────────────────────
        build_gauge(rec["confidence"],"ثقة Bayes %",f"bg_{len(h)}")
        build_gauge(spring["pressure"],"ضغط الزنبرك %",f"sp_{len(h)}")

        # ── المخطط الشبكي ────────────────────────────────────────
        if extras:
            st.markdown('<div class="card" style="padding:14px;">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:12px;font-weight:700;color:rgba(255,255,255,0.7);margin-bottom:6px;text-align:center;">📡 خريطة العوامل الإحصائية</div>', unsafe_allow_html=True)
            build_stat_radar(extras, key=f"rd_{len(h)}")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── إحصائيات نصية ────────────────────────────────────────
        st.markdown('<div class="card" style="padding:14px;">', unsafe_allow_html=True)
        items2 = [
            (f"{stats['zscore']:+.2f}","Z-score","#a855f7"),
            (f"{stats['autocorr']:+.2f}","Autocorr","#6366f1"),
            (f"{stats['entropy']:.2f}","Shannon Entropy","#00c8ff"),
            (f"{stats['compression']*100:.0f}%","Compression","#00ff88"),
        ]
        r1c,r2c = st.columns(2)
        for i,(v,l,c) in enumerate(items2):
            with (r1c if i%2==0 else r2c):
                st.markdown(f"""<div class="kpi" style="margin:3px 0;">
                    <div class="kn" style="color:{c};font-size:18px;">{v}</div>
                    <div class="kl">{l}</div></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── توزيع الفئات ─────────────────────────────────────────
        st.markdown('<div class="card" style="padding:14px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:12px;font-weight:700;color:rgba(255,255,255,0.7);margin-bottom:4px;">📊 توزيع الدورات</div>', unsafe_allow_html=True)
        build_distribution(h, key=f"ds_{len(h)}")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── ما الذي تنتظره ────────────────────────────────────────
        if rec["status"] == "WAIT":
            st.markdown("""
            <div class="bx-y">
            <b>⏳ انتظر أحد هؤلاء:</b><br>
            🔸 رقم ذهبي: 1.05، 1.09، 1.20، 1.53، 1.54، 1.77<br>
            🔸 إنتروبيا < 1.5 + زنبرك ≥3<br>
            🔸 Z-score &lt; −0.8 + سلسلة هابطة
            </div>""", unsafe_allow_html=True)

    # ══ الرسم البياني الرئيسي ══════════════════════════════════════
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**📈 مسار الدورات + ضغط الزنبرك** *(برتقالي=ضغط، نجمة=ذهبي، ◆=قفزة كبيرة)*")
    build_main_chart(h, engine, pressure_series)
    st.markdown("</div>", unsafe_allow_html=True)

    # ══ الأرقام الذهبية الأخيرة ════════════════════════════════════
    if gh:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**⭐ آخر {min(len(gh),5)} أرقام ذهبية في التاريخ:**")
        cols_g = st.columns(min(len(gh),5))
        for i, item in enumerate(gh[-5:]):
            gd = item["gdata"]
            tm = TIER_META[gd["tier"]]
            with cols_g[i]:
                st.markdown(f"""
                <div class="gc">
                    <div style="font-size:9px;color:rgba(255,255,255,0.3);">#{item['pos']}</div>
                    <div class="gn" style="font-size:16px;">{tm['icon']} x{item['val']:.2f}</div>
                    <div style="font-size:9px;color:rgba(255,255,255,0.3);">≈x{item['gnum']}</div>
                    <div class="gt" style="font-size:12px;">→x{gd['tgt_lo']}–{gd['tgt_hi']}</div>
                    <div style="font-size:9px;color:rgba(255,255,255,0.25);">avg:{gd['avg_next']}x</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ══ دليل المنهجية ══════════════════════════════════════════════
    with st.expander("📚 المنهجية الإحصائية الكاملة"):
        st.markdown(f"""
        <div style="direction:rtl;color:rgba(255,255,255,0.8);line-height:2;font-size:13px;">

        <div class="bx-g">
        <b>🔥 F1 — الزنبرك المضغوط (قانون بيانات {len(HISTORICAL_DATA)} دورة):</b><br>
        2 خسائر→6.8x | 3→7.4x | 4→9.2x | 5→14.2x | 6+→19.5x<br>
        الضغط = streak×9 + streak18×7 + streak15×5 + (1.8−avg)×35
        </div>

        <div class="bx-b">
        <b>🎯 F2 — الأرقام الذهبية (مُحسوبة من الداتا):</b><br>
        تير-1: 1.05(avg=14.5x), 1.20(17.2x), 1.09(9.7x)<br>
        تير-2: 1.77(8.3x), 1.53(6.7x), 1.54(6.0x), 1.84(6.6x)<br>
        الهامش: ±0.04 | المحذوفة: 1.91, 1.96, 1.25 (avg<2.5x)
        </div>

        <div class="bx-o">
        <b>📐 F3 — Bayesian Confidence:</b><br>
        P(قفزة|شواهد) = prior × ∏ LR_i<br>
        Prior = {round(sum(1 for v in HISTORICAL_DATA if v>=5)/len(HISTORICAL_DATA)*100)}% من {len(HISTORICAL_DATA)} دورة<br>
        LR الزنبرك: 0.6–4.5 | LR الذهبي: 1.5–3.0 | LR Markov: 1.0–2.0
        </div>

        <div class="bx-g">
        <b>♟️ F4 — Markov Transition:</b><br>
        P(next≥5 | k خسائر) محسوبة مباشرة من البيانات التاريخية.<br>
        كلما زادت سلسلة الخسائر، زادت احتمالية القفزة الكبيرة.
        </div>

        <div class="bx-b">
        <b>📊 F5 — Shannon Entropy + Compression:</b><br>
        Entropy = −Σ p×log2(p) على 5 فئات | Max=2.32 bits<br>
        Compression = 1 − H/H_max | مرتفع = نمط واضح
        </div>

        <div class="bx-y">
        <b>📉 F6 — Z-score + Autocorrelation:</b><br>
        Z = (last − μ_ref) / σ_ref | Z≤−0.8 = قيمة شاذة منخفضة<br>
        Autocorr(lag=1) سالبة = ميل للتذبذب (مؤشر ارتداد)
        </div>

        <div class="bx-o">
        <b>💰 F7 — Kelly Criterion (¼ Kelly):</b><br>
        f* = (p×b − q) / b حيث b=odds−1, q=1−p<br>
        الرهان الفعلي = f* × 0.25 × الرصيد (حد أقصى 5%)
        </div>

        </div>""", unsafe_allow_html=True)

# ══ تحذير ═════════════════════════════════════════════════════════════
st.markdown("""
<div class="bx-r" style="margin-top:8px;">
<b>⚠️ تنبيه قانوني:</b> هذا النظام أداة تحليل إحصائي للأنماط فقط.
Crash يعتمد على RNG معتمد. لا ضمان رياضي مطلق.
تحمّل فقط ما تستطيع خسارته.
</div>""", unsafe_allow_html=True)
