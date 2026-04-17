# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
from datetime import datetime
import random

st.set_page_config(
    page_title="🚀 Crash Intelligence System",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;700;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    * { font-family: 'Tajawal', sans-serif !important; }
    html, body, [data-testid="stAppViewContainer"] { background: #04040f !important; }
    [data-testid="stSidebar"] { background: #06060f !important; border-right: 1px solid rgba(99,102,241,0.2); }
    
    .main-card {
        background: linear-gradient(145deg, rgba(8,8,20,0.98), rgba(12,12,30,0.99));
        border: 1px solid rgba(99,102,241,0.25);
        box-shadow: 0 20px 60px rgba(0,0,0,0.8), inset 0 1px 0 rgba(99,102,241,0.15);
        border-radius: 20px; padding: 26px; margin-bottom: 18px;
        direction: rtl; color: white; position: relative; overflow: hidden;
    }
    .main-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #6366f1, #a855f7, #ec4899, transparent);
    }

    /* ══ حالات الوضع ══ */
    .st-SAFE {
        background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,180,90,0.05));
        border: 2px solid #00ff88; border-radius: 18px; padding: 26px; text-align: center;
        animation: pulse-safe 2s ease-in-out infinite;
    }
    @keyframes pulse-safe {
        0%,100% { box-shadow: 0 0 20px rgba(0,255,136,0.2); }
        50%      { box-shadow: 0 0 55px rgba(0,255,136,0.5); }
    }
    .st-DANGER {
        background: linear-gradient(135deg, rgba(255,40,40,0.12), rgba(180,0,0,0.06));
        border: 2px solid #ff3232; border-radius: 18px; padding: 26px; text-align: center;
        animation: pulse-danger 0.9s ease-in-out infinite;
    }
    @keyframes pulse-danger {
        0%,100% { box-shadow: 0 0 20px rgba(255,50,50,0.3); }
        50%      { box-shadow: 0 0 65px rgba(255,50,50,0.7); }
    }
    .st-WAIT {
        background: linear-gradient(135deg, rgba(255,200,0,0.1), rgba(255,130,0,0.05));
        border: 2px solid #FFD700; border-radius: 18px; padding: 26px; text-align: center;
        box-shadow: 0 0 25px rgba(255,215,0,0.15);
    }
    .st-STRONG {
        background: linear-gradient(135deg, rgba(0,200,255,0.12), rgba(0,130,200,0.06));
        border: 2px solid #00c8ff; border-radius: 18px; padding: 26px; text-align: center;
        animation: pulse-strong 1.6s ease-in-out infinite;
    }
    @keyframes pulse-strong {
        0%,100% { box-shadow: 0 0 25px rgba(0,200,255,0.2); }
        50%      { box-shadow: 0 0 65px rgba(0,200,255,0.55); }
    }
    .st-DOUBLE {
        background: linear-gradient(135deg, rgba(255,100,0,0.12), rgba(200,50,0,0.06));
        border: 2px solid #ff6400; border-radius: 18px; padding: 26px; text-align: center;
        animation: pulse-double 1.2s ease-in-out infinite;
    }
    @keyframes pulse-double {
        0%,100% { box-shadow: 0 0 25px rgba(255,100,0,0.25); }
        50%      { box-shadow: 0 0 60px rgba(255,100,0,0.6); }
    }

    /* ══ شارات ══ */
    .badge {
        display: inline-block; padding: 6px 13px; border-radius: 10px;
        font-size: 15px; font-weight: 900; margin: 3px;
        font-family: 'Orbitron', monospace !important;
    }
    .b-loss  { background:#3d0000; border:1px solid #ff4444; color:#ff7070;
               box-shadow:0 2px 10px rgba(255,50,50,0.2); }
    .b-med   { background:#1a1200; border:1px solid #FFD700; color:#FFD700;
               box-shadow:0 2px 10px rgba(255,215,0,0.15); }
    .b-win   { background:#003d1f; border:1px solid #00ff88; color:#00ff88;
               box-shadow:0 2px 10px rgba(0,255,136,0.15); }
    .b-gold  { background:#2d1a00; border:2px solid #ff9500; color:#ffb84d;
               box-shadow:0 2px 15px rgba(255,149,0,0.35);
               animation: pulse-gold 1.4s ease-in-out infinite; }
    @keyframes pulse-gold {
        0%,100% { box-shadow:0 2px 12px rgba(255,149,0,0.3); }
        50%      { box-shadow:0 2px 25px rgba(255,149,0,0.65); }
    }
    .b-big   { background:#1a0030; border:1px solid #a855f7; color:#c4b5fd;
               box-shadow:0 2px 10px rgba(168,85,247,0.2); }

    /* ══ بطاقات الإحصاء ══ */
    .kpi-box {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 16px; text-align: center; direction: rtl;
        transition: all 0.3s;
    }
    .kpi-box:hover { border-color: rgba(99,102,241,0.4); transform: translateY(-2px); }
    .kpi-num {
        font-family: 'Orbitron', monospace !important;
        font-size: 24px; font-weight: 900;
        background: linear-gradient(90deg,#6366f1,#a855f7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .kpi-lbl { color: rgba(255,255,255,0.35); font-size: 11px; margin-top:4px; letter-spacing:1px; }

    /* ══ بطاقة الرقم الذهبي ══ */
    .gold-card {
        background: linear-gradient(135deg,rgba(255,149,0,0.1),rgba(255,70,0,0.05));
        border: 1px solid rgba(255,149,0,0.45); border-radius: 14px;
        padding: 16px; text-align: center; transition: all 0.3s;
    }
    .gold-card:hover { border-color:#ff9500; transform:translateY(-3px);
                       box-shadow:0 10px 30px rgba(255,149,0,0.3); }
    .gold-num { font-family:'Orbitron',monospace!important; font-size:22px;
                font-weight:900; color:#ff9500; text-shadow:0 0 12px rgba(255,149,0,0.4); }
    .gold-tgt { font-family:'Orbitron',monospace!important; font-size:16px;
                color:#00ff88; margin-top:5px; }

    /* ══ شريط التقدم ══ */
    .prog-wrap { background:rgba(255,255,255,0.05); border-radius:8px; height:8px;
                 margin:6px 0; overflow:hidden; }
    .prog-green  { height:100%; border-radius:8px;
                   background:linear-gradient(90deg,#00c853,#00ff88); transition:width 0.6s; }
    .prog-orange { height:100%; border-radius:8px;
                   background:linear-gradient(90deg,#ff6d00,#ff9500); transition:width 0.6s; }
    .prog-red    { height:100%; border-radius:8px;
                   background:linear-gradient(90deg,#c62828,#ff3232); transition:width 0.6s; }
    .prog-blue   { height:100%; border-radius:8px;
                   background:linear-gradient(90deg,#6366f1,#a855f7); transition:width 0.6s; }

    /* ══ صناديق الرسائل ══ */
    .box-info    { background:rgba(99,102,241,0.07); border:1px solid rgba(99,102,241,0.3);
                   border-right:4px solid #6366f1; border-radius:12px;
                   padding:13px 17px; color:rgba(180,185,255,0.9);
                   font-size:14px; direction:rtl; margin:8px 0; line-height:1.8; }
    .box-warn    { background:rgba(255,100,0,0.07); border:1px solid rgba(255,100,0,0.3);
                   border-right:4px solid #ff6400; border-radius:12px;
                   padding:13px 17px; color:rgba(255,200,150,0.9);
                   font-size:14px; direction:rtl; margin:8px 0; line-height:1.8; }
    .box-success { background:rgba(0,255,136,0.07); border:1px solid rgba(0,255,136,0.3);
                   border-right:4px solid #00ff88; border-radius:12px;
                   padding:13px 17px; color:rgba(150,255,200,0.9);
                   font-size:14px; direction:rtl; margin:8px 0; line-height:1.8; }
    .box-danger  { background:rgba(255,50,50,0.07); border:1px solid rgba(255,50,50,0.3);
                   border-right:4px solid #ff3232; border-radius:12px;
                   padding:13px 17px; color:rgba(255,170,170,0.9);
                   font-size:14px; direction:rtl; margin:8px 0; line-height:1.8; }

    /* ══ تاغ ══ */
    .tag { display:inline-block; padding:3px 11px; border-radius:20px;
           font-size:11px; font-weight:700; letter-spacing:1px; margin:2px; }
    .tag-r { background:rgba(255,50,50,0.2);  color:#ff7070; border:1px solid #ff4444; }
    .tag-g { background:rgba(0,255,136,0.15); color:#00ff88; border:1px solid #00cc70; }
    .tag-y { background:rgba(255,215,0,0.15); color:#FFD700; border:1px solid #cc9900; }
    .tag-b { background:rgba(0,200,255,0.15); color:#00c8ff; border:1px solid #009fc8; }
    .tag-o { background:rgba(255,149,0,0.2);  color:#ffb84d; border:1px solid #ff9500; }
    .tag-p { background:rgba(168,85,247,0.2); color:#c4b5fd; border:1px solid #8b5cf6; }

    /* ══ زنبرك مرئي ══ */
    .spring-bar {
        background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
        border-radius:14px; padding:16px 20px; direction:rtl; margin:8px 0;
    }

    /* ══ أزرار ══ */
    .stButton > button {
        background:linear-gradient(135deg,#6366f1,#8b5cf6,#a855f7) !important;
        color:white !important; border:none !important; font-weight:700 !important;
        font-size:14px !important; border-radius:11px !important; padding:10px 24px !important;
        box-shadow:0 6px 20px rgba(99,102,241,0.4) !important; transition:all 0.3s !important;
    }
    .stButton > button:hover { transform:translateY(-2px) !important;
                                box-shadow:0 10px 35px rgba(99,102,241,0.6) !important; }
    .stNumberInput > div > div > input {
        background:rgba(255,255,255,0.05) !important; color:white !important;
        border:1px solid rgba(99,102,241,0.4) !important; border-radius:10px !important;
    }
    [data-testid="stMetric"] { background:rgba(255,255,255,0.03);
                                 border:1px solid rgba(255,255,255,0.07); border-radius:12px; }
    div[data-testid="stExpander"] { background:rgba(255,255,255,0.02) !important;
                                     border:1px solid rgba(255,255,255,0.08) !important;
                                     border-radius:14px !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ① قاعدة الأرقام الذهبية — مُصفّاة ومُصنّفة
# ══════════════════════════════════════════════════════════════════════
GOLDEN_DB = {
    # ─── تير-1: متوسط تالٍ ≥ 10x ────────────────────────────────
    1.05: {"tier":1, "avg_next":14.48, "win5_pct":67, "tgt_lo":6.0,  "tgt_hi":12.0, "label":"زنبرك أقصى"},
    1.20: {"tier":1, "avg_next":17.17, "win5_pct":50, "tgt_lo":6.0,  "tgt_hi":14.0, "label":"زنبرك أقصى"},
    1.09: {"tier":1, "avg_next": 9.73, "win5_pct":33, "tgt_lo":5.0,  "tgt_hi":10.0, "label":"زنبرك قوي"},
    # ─── تير-2: متوسط تالٍ 5–10x ────────────────────────────────
    1.53: {"tier":2, "avg_next": 6.74, "win5_pct":40, "tgt_lo":4.0,  "tgt_hi": 8.0, "label":"إشارة قوية"},
    1.54: {"tier":2, "avg_next": 5.97, "win5_pct":40, "tgt_lo":4.0,  "tgt_hi": 7.0, "label":"إشارة قوية"},
    1.77: {"tier":2, "avg_next": 8.30, "win5_pct":67, "tgt_lo":5.0,  "tgt_hi": 9.0, "label":"إشارة قوية"},
    1.36: {"tier":2, "avg_next": 5.53, "win5_pct":50, "tgt_lo":3.5,  "tgt_hi": 7.0, "label":"إشارة جيدة"},
    1.66: {"tier":2, "avg_next": 7.04, "win5_pct":50, "tgt_lo":4.0,  "tgt_hi": 7.0, "label":"إشارة جيدة"},
    1.83: {"tier":2, "avg_next": 5.64, "win5_pct":50, "tgt_lo":3.5,  "tgt_hi": 6.0, "label":"إشارة جيدة"},
    1.84: {"tier":2, "avg_next": 6.58, "win5_pct":50, "tgt_lo":4.0,  "tgt_hi": 7.0, "label":"إشارة جيدة"},
    # ─── تير-3: متوسط تالٍ 3–5x ─────────────────────────────────
    1.07: {"tier":3, "avg_next": 2.51, "win5_pct":20, "tgt_lo":2.5,  "tgt_hi": 4.5, "label":"إشارة متوسطة"},
    1.12: {"tier":3, "avg_next": 4.82, "win5_pct":33, "tgt_lo":3.0,  "tgt_hi": 5.0, "label":"إشارة متوسطة"},
    1.19: {"tier":3, "avg_next": 3.07, "win5_pct": 0, "tgt_lo":2.5,  "tgt_hi": 4.0, "label":"إشارة متوسطة"},
    1.22: {"tier":3, "avg_next": 3.12, "win5_pct":33, "tgt_lo":2.5,  "tgt_hi": 4.5, "label":"إشارة متوسطة"},
    1.24: {"tier":3, "avg_next": 4.19, "win5_pct":25, "tgt_lo":2.5,  "tgt_hi": 5.0, "label":"إشارة متوسطة"},
    1.29: {"tier":3, "avg_next": 5.19, "win5_pct":50, "tgt_lo":3.0,  "tgt_hi": 5.5, "label":"إشارة متوسطة"},
    1.45: {"tier":3, "avg_next": 5.91, "win5_pct":33, "tgt_lo":3.0,  "tgt_hi": 6.0, "label":"إشارة متوسطة"},
    1.49: {"tier":3, "avg_next": 4.16, "win5_pct":25, "tgt_lo":2.5,  "tgt_hi": 5.0, "label":"إشارة متوسطة"},
    1.01: {"tier":3, "avg_next": 3.29, "win5_pct":33, "tgt_lo":2.5,  "tgt_hi": 4.5, "label":"إشارة متوسطة"},
    # ─── محذوفة (ضعيفة جداً) ──────────────────────────────────
    # 1.91 avg=1.88 / 1.96 avg=2.01 / 1.25 avg=2.28
}
GOLDEN_TOL = 0.035   # هامش التطابق ±0.035

TIER_META = {
    1: {"icon":"🔥","color":"#ff4500","label":"تير-1 (أقوى)"},
    2: {"icon":"💎","color":"#ff9500","label":"تير-2 (قوي)"},
    3: {"icon":"✨","color":"#FFD700","label":"تير-3 (متوسط)"},
}

# ══════════════════════════════════════════════════════════════════════
# ② محرك التحليل الشامل
# ══════════════════════════════════════════════════════════════════════
class CrashIntelligence:
    """
    يدمج كل الفرضيات:
      F1 — قانون الزنبرك المضغوط
      F2 — الأرقام الذهبية المُصفّاة
      F3 — الزنبرك + الرقم الذهبي = إشارة مركّبة
      F4 — ما بعد القفزة الكبيرة
      F5 — نمط التسلسل الهابط
    """
    def __init__(self, history: list):
        self.h  = history
        self.n  = len(history)

    # ── أدوات مساعدة ──────────────────────────────────────
    def _last(self, k): return self.h[-k:] if self.n >= k else self.h[:]

    def _streak(self, threshold):
        c = 0
        for v in reversed(self.h):
            if v < threshold: c += 1
            else: break
        return c

    def _find_golden(self, val):
        best, bd = None, float("inf")
        for g, d in GOLDEN_DB.items():
            df = abs(val - g)
            if df <= GOLDEN_TOL and df < bd:
                best, bd = (g, d), df
        return best  # (gnum, gdata) or None

    # ── F1: قوة الزنبرك ───────────────────────────────────
    def spring_power(self):
        """
        يُعيد dict مع:
          streak_2, streak_18, pressure (0–100),
          expected_jump_range, spring_level (0-4)
        """
        s2  = self._streak(2.0)
        s18 = self._streak(1.8)
        # متوسط السلسلة الأخيرة
        seq = self._last(max(s2, 1))
        avg_seq = float(np.mean(seq)) if seq else 2.0

        # ضغط الزنبرك (0–100)
        pressure = min(100, int(
            s2  * 10 +          # كل خسارة <x2 تضيف 10
            s18 * 8  +          # كل خسارة <x1.8 تضيف 8 إضافية
            max(0, (1.8 - avg_seq) * 30)  # كلما انخفض المتوسط زاد الضغط
        ))

        # مستوى الزنبرك
        if s18 >= 5:   level, exp_lo, exp_hi = 4, 12.0, 35.0
        elif s18 >= 3: level, exp_lo, exp_hi = 3,  7.0, 20.0
        elif s2  >= 5: level, exp_lo, exp_hi = 3,  6.0, 18.0
        elif s2  >= 3: level, exp_lo, exp_hi = 2,  4.0, 10.0
        elif s2  >= 2: level, exp_lo, exp_hi = 1,  3.0,  7.0
        else:           level, exp_lo, exp_hi = 0,  0.0,  0.0

        return {
            "streak_2": s2, "streak_18": s18,
            "avg_seq": round(avg_seq, 2),
            "pressure": pressure,
            "level": level,
            "exp_lo": exp_lo, "exp_hi": exp_hi,
        }

    # ── F2 + F3: الرقم الذهبي + الزنبرك ──────────────────
    def golden_signal(self, spring: dict):
        """
        يُعيد None أو dict مع تفاصيل الإشارة المركّبة
        """
        if self.n == 0: return None
        last = self.h[-1]
        gm = self._find_golden(last)
        if gm is None: return None
        gnum, gdata = gm

        # حساب قوة الإشارة المركّبة
        base_conf = {1: 70, 2: 60, 3: 50}[gdata["tier"]]
        spring_bonus = spring["level"] * 7   # حتى +28
        # هل قبله سلسلة هابطة؟ (F5)
        seq3 = self._last(4)[:-1]  # 3 دورات قبل الأخيرة
        is_descending = all(seq3[i] >= seq3[i+1] for i in range(len(seq3)-1)) if len(seq3)>=2 else False
        desc_bonus = 8 if is_descending else 0

        confidence = min(94, base_conf + spring_bonus + desc_bonus)

        # تعديل الهدف بناءً على قوة الزنبرك
        tlo = gdata["tgt_lo"] + spring["level"] * 0.5
        thi = gdata["tgt_hi"] + spring["level"] * 1.2

        return {
            "golden_num": gnum,
            "golden_data": gdata,
            "actual_val": last,
            "confidence": confidence,
            "target_lo": round(tlo, 1),
            "target_hi": round(thi, 1),
            "is_descending": is_descending,
            "spring_bonus": spring_bonus,
        }

    # ── F4: ما بعد القفزة الكبيرة ─────────────────────────
    def post_big_jump(self):
        """
        يكشف ما إذا كنا في نافذة ما بعد القفزة:
        - خطر عادي (تجنب 2-3 دورات)
        - "قفزة مزدوجة" (فرصة استثنائية ~25%)
        """
        if self.n < 2: return None
        last = self.h[-1]

        # كشف القفزة المزدوجة: الدورة قبل الأخيرة ≥12x
        for lookback in [1, 2]:
            if self.n > lookback:
                prev = self.h[-(lookback+1)]
                if prev >= 12.0:
                    if last >= 5.0:
                        # قفزة مزدوجة نادرة!
                        return {"type":"double_jump", "prev":prev, "last":last,
                                "lookback":lookback, "confidence":72}
                    else:
                        # بعد قفزة كبيرة، توقع خسائر
                        rounds_to_avoid = max(1, 3 - lookback)
                        return {"type":"post_big", "prev":prev, "last":last,
                                "lookback":lookback, "avoid_rounds":rounds_to_avoid,
                                "confidence":78}
        return None

    # ── F5: نمط التسلسل الهابط ────────────────────────────
    def descending_pattern(self):
        """
        هل آخر 3-5 دورات في تسلسل هابط واضح؟
        مثال: 2.28→1.24→1.20→1.54 → قفزة
        """
        if self.n < 4: return None
        seq = self._last(5)
        # احسب عدد الانحدارات المتتالية من النهاية
        drops = 0
        for i in range(len(seq)-1, 0, -1):
            if seq[i] <= seq[i-1]: drops += 1
            else: break
        # تحقق من أن المتوسط منخفض
        avg = np.mean(seq)
        if drops >= 3 and avg < 2.5:
            return {"drops": drops, "avg": round(avg,2), "seq": seq}
        return None

    # ── التحليل الكامل ─────────────────────────────────────
    def full_analysis(self, balance: float) -> dict:
        if self.n < 3:
            return {"scenario":"INIT", "status":"WAIT",
                    "title":"⏳ أضف المزيد من الدورات",
                    "desc":"تحتاج 3 دورات للبدء",
                    "confidence":0, "spring":{}, "golden":None,
                    "post_big":None, "desc_pat":None,
                    "stake":0, "target_lo":None, "target_hi":None}

        spring   = self.spring_power()
        golden   = self.golden_signal(spring)
        post_big = self.post_big_jump()
        desc_pat = self.descending_pattern()

        # ═══ اتخاذ القرار ════════════════════════════════════
        # P0: قفزة مزدوجة نادرة (أولوية قصوى)
        if post_big and post_big["type"] == "double_jump":
            return self._build(
                scenario="DOUBLE", status="DOUBLE",
                title="⚡ قفزة مزدوجة — فرصة نادرة!",
                desc=(f"قفزة x{post_big['prev']:.2f} → x{post_big['last']:.2f}. "
                      f"نمط القفزة المزدوجة (~25%). ادخل بحذر برهان صغير جداً."),
                confidence=post_big["confidence"],
                target_lo=post_big["last"]*0.8,
                target_hi=post_big["last"]*1.3,
                stake_pct=0.01,
                spring=spring, golden=golden, post_big=post_big, desc_pat=desc_pat,
                balance=balance,
            )

        # P1: تجنب ما بعد القفزة الكبيرة
        if post_big and post_big["type"] == "post_big":
            return self._build(
                scenario="POST_BIG", status="DANGER",
                title=f"⛔ ما بعد القفزة — تجنب {post_big['avoid_rounds']} دورات",
                desc=(f"جاءت x{post_big['prev']:.2f} قبل {post_big['lookback']} دورة. "
                      f"70% من الحالات تليها خسائر. لا تراهن."),
                confidence=post_big["confidence"],
                target_lo=None, target_hi=None, stake_pct=0,
                spring=spring, golden=golden, post_big=post_big, desc_pat=desc_pat,
                balance=balance,
            )

        # P2: إشارة مركّبة قوية (زنبرك ≥2 + رقم ذهبي تير1/2)
        if golden and spring["level"] >= 2 and golden["golden_data"]["tier"] <= 2:
            scenario = "STRONG_A" if spring["level"] >= 3 else "STRONG_B"
            stake_pct = 0.025 if spring["level"] >= 3 else 0.018
            return self._build(
                scenario=scenario, status="STRONG",
                title=("🔥 إشارة قصوى — زنبرك قوي + رقم ذهبي" if spring["level"]>=3
                       else "💎 إشارة قوية — زنبرك + رقم ذهبي"),
                desc=(f"زنبرك مستوى {spring['level']} "
                      f"({spring['streak_2']} خسائر < x2، متوسط x{spring['avg_seq']}) "
                      f"+ رقم ذهبي x{golden['golden_num']} ({golden['golden_data']['label']})."),
                confidence=golden["confidence"],
                target_lo=golden["target_lo"], target_hi=golden["target_hi"],
                stake_pct=stake_pct,
                spring=spring, golden=golden, post_big=post_big, desc_pat=desc_pat,
                balance=balance,
            )

        # P3: رقم ذهبي تير1 وحده (بدون زنبرك كافٍ)
        if golden and golden["golden_data"]["tier"] == 1 and spring["level"] >= 1:
            return self._build(
                scenario="GOLDEN_T1", status="SAFE",
                title=f"⭐ رقم ذهبي تير-1 — {golden['golden_data']['label']}",
                desc=(f"x{golden['actual_val']:.2f} ≈ x{golden['golden_num']} "
                      f"(متوسط تاريخي: {golden['golden_data']['avg_next']}x). "
                      f"زنبرك مستوى {spring['level']}."),
                confidence=golden["confidence"],
                target_lo=golden["target_lo"], target_hi=golden["target_hi"],
                stake_pct=0.015,
                spring=spring, golden=golden, post_big=post_big, desc_pat=desc_pat,
                balance=balance,
            )

        # P4: رقم ذهبي تير2 وحده
        if golden and golden["golden_data"]["tier"] == 2:
            return self._build(
                scenario="GOLDEN_T2", status="SAFE",
                title=f"✨ رقم ذهبي تير-2 — {golden['golden_data']['label']}",
                desc=(f"x{golden['actual_val']:.2f} ≈ x{golden['golden_num']}. "
                      f"الهدف x{golden['target_lo']}–x{golden['target_hi']}."),
                confidence=golden["confidence"],
                target_lo=golden["target_lo"], target_hi=golden["target_hi"],
                stake_pct=0.012,
                spring=spring, golden=golden, post_big=post_big, desc_pat=desc_pat,
                balance=balance,
            )

        # P5: زنبرك قوي بدون رقم ذهبي — انتظار
        if spring["level"] >= 3:
            return self._build(
                scenario="SPRING_WAIT", status="WAIT",
                title=f"⏳ زنبرك مستوى {spring['level']} — انتظر الإشارة",
                desc=(f"سلسلة {spring['streak_2']} خسائر < x2 "
                      f"({spring['streak_18']} منها < x1.8). الزنبرك مضغوط. "
                      f"انتظر رقماً ذهبياً أو x5 للدخول."),
                confidence=55,
                target_lo=spring["exp_lo"], target_hi=spring["exp_hi"],
                stake_pct=0,
                spring=spring, golden=golden, post_big=post_big, desc_pat=desc_pat,
                balance=balance,
            )

        # P6: زنبرك متوسط
        if spring["level"] >= 2:
            return self._build(
                scenario="SPRING_MED", status="WAIT",
                title="⏳ زنبرك متوسط — لا تدخل بعد",
                desc=(f"سلسلة {spring['streak_2']} خسائر. "
                      f"انتظر رقماً ذهبياً أو 3 خسائر إضافية."),
                confidence=45,
                target_lo=None, target_hi=None, stake_pct=0,
                spring=spring, golden=golden, post_big=post_big, desc_pat=desc_pat,
                balance=balance,
            )

        # P7: لا إشارة
        return self._build(
            scenario="NO_SIGNAL", status="DANGER",
            title="🚫 لا توجد إشارة — لا تراهن",
            desc="لا تتوفر شروط الدخول. انتظر: 3+ خسائر متتالية أو رقم ذهبي.",
            confidence=80,
            target_lo=None, target_hi=None, stake_pct=0,
            spring=spring, golden=golden, post_big=post_big, desc_pat=desc_pat,
            balance=balance,
        )

    def _build(self, scenario, status, title, desc, confidence,
               target_lo, target_hi, stake_pct,
               spring, golden, post_big, desc_pat, balance):
        stake = max(5.0, round(balance * stake_pct, 1)) if stake_pct > 0 else 0
        profit = (round(stake * ((target_lo + target_hi)/2) - stake, 1)
                  if (stake > 0 and target_lo and target_hi) else 0)
        return {
            "scenario": scenario, "status": status,
            "title": title, "desc": desc,
            "confidence": confidence,
            "target_lo": target_lo, "target_hi": target_hi,
            "stake": stake, "stake_pct": round(stake_pct*100, 1),
            "profit_est": profit,
            "spring": spring, "golden": golden,
            "post_big": post_big, "desc_pat": desc_pat,
        }

    # ── إحصائيات ──────────────────────────────────────────
    def stats(self):
        if not self.h: return {}
        a = np.array(self.h)
        sp = self.spring_power()
        return {
            "n": len(self.h), "avg": round(a.mean(),2),
            "med": round(float(np.median(a)),2), "mx": round(a.max(),2),
            "loss_u2":  sum(1 for v in self.h if v < 2.0),
            "loss_u18": sum(1 for v in self.h if v < 1.8),
            "med_wins": sum(1 for v in self.h if 2.0<=v<5.0),
            "big_wins": sum(1 for v in self.h if v>=5.0),
            "jumps":    sum(1 for v in self.h if v>=12.0),
            "win_rate": round(sum(1 for v in self.h if v>=2.0)/len(self.h)*100,1),
            "streak_2":  sp["streak_2"],
            "streak_18": sp["streak_18"],
            "pressure":  sp["pressure"],
        }

    def golden_in_history(self, last_k=20):
        found = []
        for i, v in enumerate(self.h[-last_k:]):
            gm = self._find_golden(v)
            if gm:
                found.append({"pos": len(self.h)-last_k+i+1,
                               "val": v, "gnum": gm[0], "gdata": gm[1]})
        return found

    def spring_history(self, last_k=30):
        """يُعيد تاريخ ضغط الزنبرك لكل نقطة"""
        result = []
        for i in range(len(self.h)):
            sub = CrashIntelligence(self.h[:i+1])
            sp  = sub.spring_power()
            result.append(sp["pressure"])
        return result[-last_k:]


# ══════════════════════════════════════════════════════════════════════
# ③ الرسوم البيانية
# ══════════════════════════════════════════════════════════════════════
def chart_history(h: list, engine: CrashIntelligence):
    if len(h) < 2: return
    x = list(range(1, len(h)+1))
    colors, sizes, symbols = [], [], []
    for v in h:
        gm = engine._find_golden(v)
        if gm:
            colors.append("#ff9500"); sizes.append(18); symbols.append("star")
        elif v >= 12.0:
            colors.append("#a855f7"); sizes.append(16); symbols.append("diamond")
        elif v >= 5.0:
            colors.append("#00c8ff"); sizes.append(14); symbols.append("circle")
        elif v >= 2.0:
            colors.append("#00ff88"); sizes.append(11); symbols.append("circle")
        else:
            colors.append("#ff4444"); sizes.append(9);  symbols.append("circle")

    # ضغط الزنبرك
    pressure = engine.spring_history(len(h))

    fig = go.Figure()

    # مناطق ملونة
    fig.add_hrect(y0=0,    y1=1.8,  fillcolor="rgba(255,50,50,0.06)",    line_width=0)
    fig.add_hrect(y0=1.8,  y1=2.0,  fillcolor="rgba(255,150,0,0.04)",    line_width=0)
    fig.add_hrect(y0=2.0,  y1=5.0,  fillcolor="rgba(0,255,136,0.03)",    line_width=0)
    fig.add_hrect(y0=5.0,  y1=12.0, fillcolor="rgba(0,200,255,0.03)",    line_width=0)
    fig.add_hrect(y0=12.0, y1=max(max(h)*1.1, 15),
                  fillcolor="rgba(168,85,247,0.04)", line_width=0)

    # ضغط الزنبرك (محور ثانوي)
    fig.add_trace(go.Scatter(
        x=list(range(max(1, len(h)-len(pressure)+1), len(h)+1)),
        y=pressure, name="ضغط الزنبرك",
        yaxis="y2", mode="lines",
        line=dict(color="rgba(255,149,0,0.35)", width=1.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(255,149,0,0.06)",
    ))

    # خط المضاعفات
    fig.add_trace(go.Scatter(
        x=x, y=h, mode="lines+markers+text",
        line=dict(color="rgba(99,102,241,0.55)", width=2, shape="spline"),
        marker=dict(color=colors, size=sizes, symbol=symbols,
                    line=dict(color="rgba(255,255,255,0.2)", width=1)),
        text=[f"x{v:.2f}" for v in h],
        textposition="top center",
        textfont=dict(color="rgba(255,255,255,0.75)", size=8.5, family="Orbitron"),
        name="المضاعف",
    ))

    # خطوط حدود أفقية
    for yv, clr, lbl in [(1.8,"rgba(255,100,0,0.5)","x1.8"),
                          (2.0,"rgba(255,215,0,0.5)","x2.0"),
                          (5.0,"rgba(0,200,255,0.5)","x5.0"),
                          (12.0,"rgba(168,85,247,0.5)","x12")]:
        fig.add_hline(y=yv, line_dash="dot", line_color=clr, line_width=1,
                      annotation_text=lbl,
                      annotation_font=dict(color=clr, size=10))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Tajawal"),
        height=360, margin=dict(l=15, r=15, t=25, b=15),
        xaxis=dict(showgrid=False, title="الدورة",
                   tickfont=dict(color="rgba(255,255,255,0.3)")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   title="المضاعف", tickprefix="x",
                   tickfont=dict(color="rgba(255,255,255,0.3)")),
        yaxis2=dict(overlaying="y", side="left", range=[0,120],
                    showgrid=False, showticklabels=False),
        legend=dict(orientation="h", y=1.05, font=dict(size=10, color="rgba(255,255,255,0.4)"),
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ch_{len(h)}")


def chart_spring_gauge(pressure: int, key: str):
    color = "#ff3232" if pressure >= 70 else "#ff9500" if pressure >= 40 else "#FFD700"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pressure,
        title={"text":"ضغط الزنبرك","font":{"size":13,"color":"rgba(255,255,255,0.6)","family":"Tajawal"}},
        number={"suffix":"%","font":{"size":30,"color":color,"family":"Orbitron"}},
        delta={"reference":50,"font":{"size":14}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"rgba(255,255,255,0.15)"},
            "bar":{"color":color,"thickness":0.28},
            "bgcolor":"rgba(0,0,0,0.2)", "borderwidth":0,
            "steps":[
                {"range":[0,30], "color":"rgba(255,215,0,0.07)"},
                {"range":[30,60],"color":"rgba(255,149,0,0.07)"},
                {"range":[60,100],"color":"rgba(255,50,50,0.1)"},
            ],
            "threshold":{"line":{"color":"white","width":2},"thickness":0.8,"value":70},
        }
    ))
    fig.update_layout(height=200, margin=dict(l=10,r=10,t=45,b=5),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key=key)


def chart_confidence(val: int, key: str):
    color = "#00ff88" if val>=70 else "#FFD700" if val>=45 else "#ff4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={"text":"الثقة في الإشارة","font":{"size":13,"color":"rgba(255,255,255,0.6)","family":"Tajawal"}},
        number={"suffix":"%","font":{"size":30,"color":color,"family":"Orbitron"}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"rgba(255,255,255,0.15)"},
            "bar":{"color":color,"thickness":0.28},
            "bgcolor":"rgba(0,0,0,0.2)","borderwidth":0,
            "steps":[
                {"range":[0,40],"color":"rgba(255,50,50,0.08)"},
                {"range":[40,70],"color":"rgba(255,215,0,0.08)"},
                {"range":[70,100],"color":"rgba(0,255,136,0.08)"},
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=10,r=10,t=45,b=5),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key=key)


def chart_distribution(h: list, key: str):
    bins = [0,1.8,2.0,5.0,12.0,1000]
    labels = ["< x1.8","x1.8–2","x2–5","x5–12","≥ x12"]
    colors = ["#ff3232","#ff9500","#00ff88","#00c8ff","#a855f7"]
    counts = []
    for i in range(len(bins)-1):
        counts.append(sum(1 for v in h if bins[i]<=v<bins[i+1]))
    total = sum(counts)
    pcts  = [round(c/total*100,1) for c in counts]

    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker_color=colors,
        text=[f"{p}%" for p in pcts],
        textposition="outside",
        textfont=dict(color="white", size=12, family="Orbitron"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Tajawal"),
        height=240, margin=dict(l=10,r=10,t=20,b=10),
        xaxis=dict(showgrid=False, tickfont=dict(color="rgba(255,255,255,0.5)")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   tickfont=dict(color="rgba(255,255,255,0.4)")),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ══════════════════════════════════════════════════════════════════════
# ④ الجلسة
# ══════════════════════════════════════════════════════════════════════
DEMO_DATA = [
    8.72,6.75,1.86,2.18,1.25,2.28,1.24,1.2,1.54,24.46,4.16,1.49,
    1.09,1.47,1.54,1.53,2.1,32.04,11,1.17,1.7,2.61,1.26,22.23,
    1.77,1.93,3.35,7.01,1.83,9.39,3.31,2.04,1.3,6.65,1.16,3.39,
    1.95,10.85,1.65,1.22,1.6,4.67,1.85,2.72,1,3.02,1.35,1.3,
    1.37,17.54,1.18,1,14.4,1.11,6.15,2.39,2.22,1.42,1.23,2.42,
]

for k,v in [("history",[]),("balance",1000.0),("log",[])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════
# ⑤ الواجهة
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:18px 0 8px;">
    <div style="font-family:'Orbitron',monospace;font-size:34px;font-weight:900;
                background:linear-gradient(90deg,#6366f1,#a855f7,#ec4899,#a855f7,#6366f1);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-size:200%;animation:gs 3s linear infinite;">
        🚀 CRASH INTELLIGENCE SYSTEM
    </div>
    <div style="color:rgba(255,255,255,0.3);font-size:12px;letter-spacing:4px;margin-top:4px;">
        الزنبرك المضغوط  •  الأرقام الذهبية  •  تحليل الأنماط
    </div>
</div>
<style>@keyframes gs{0%{background-position:0%}100%{background-position:200%}}</style>
""", unsafe_allow_html=True)

# ══ الشريط الجانبي ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="text-align:center;color:#a855f7;font-size:17px;font-weight:700;margin-bottom:10px;">⚙️ لوحة التحكم</div>', unsafe_allow_html=True)

    st.markdown("**💰 الرصيد**")
    st.session_state.balance = st.number_input(
        "bal", min_value=10.0, max_value=999999.0,
        value=st.session_state.balance, step=50.0, label_visibility="collapsed"
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ مسح", use_container_width=True):
            st.session_state.history = []; st.session_state.log = []; st.rerun()
    with c2:
        if st.button("📊 ديمو", use_container_width=True):
            st.session_state.history = DEMO_DATA.copy(); st.rerun()

    if st.button("🎲 محاكاة واقعية (15)", use_container_width=True):
        sim = []
        for _ in range(15):
            r = random.random()
            if r < 0.48:   sim.append(round(random.uniform(1.0,1.79),2))
            elif r < 0.65: sim.append(round(random.uniform(1.8,1.99),2))
            elif r < 0.82: sim.append(round(random.uniform(2.0,4.99),2))
            elif r < 0.93: sim.append(round(random.uniform(5.0,11.99),2))
            else:           sim.append(round(random.uniform(12.0,40.0),2))
        st.session_state.history = sim; st.rerun()

    st.markdown("---")
    st.markdown("**📌 الأرقام الذهبية**")
    for tier_k in [1,2,3]:
        m = TIER_META[tier_k]
        nums = {g:d for g,d in GOLDEN_DB.items() if d["tier"]==tier_k}
        st.markdown(f'<div style="color:{m["color"]};font-size:12px;font-weight:700;margin:7px 0 3px;">{m["icon"]} {m["label"]}</div>', unsafe_allow_html=True)
        for g, d in nums.items():
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                        border-radius:7px;padding:4px 9px;margin:2px 0;direction:rtl;
                        display:flex;justify-content:space-between;">
                <span style="font-family:'Orbitron',monospace;color:{m['color']};font-size:12px;font-weight:700;">x{g}</span>
                <span style="color:rgba(255,255,255,0.35);font-size:10px;">→ x{d['tgt_lo']}–x{d['tgt_hi']}</span>
            </div>""", unsafe_allow_html=True)

    # إحصائيات الجلسة
    h = st.session_state.history
    if h:
        st.markdown("---")
        eng = CrashIntelligence(h)
        s = eng.stats()
        st.markdown(f"""
        <div class="kpi-box" style="margin:4px 0;">
            <div class="kpi-num">{s['n']}</div>
            <div class="kpi-lbl">الدورات</div>
        </div>
        <div class="kpi-box" style="margin:4px 0;">
            <div class="kpi-num" style="color:#ff4444;">{s['streak_2']}</div>
            <div class="kpi-lbl">خسائر متتالية &lt;x2</div>
        </div>
        <div class="kpi-box" style="margin:4px 0;">
            <div class="kpi-num" style="color:#ff9500;">{s['pressure']}%</div>
            <div class="kpi-lbl">ضغط الزنبرك</div>
        </div>
        <div class="kpi-box" style="margin:4px 0;">
            <div class="kpi-num">{s['win_rate']}%</div>
            <div class="kpi-lbl">معدل فوق x2</div>
        </div>
        """, unsafe_allow_html=True)

# ══ منطقة الإدخال ═══════════════════════════════════════════════════
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown("### 📥 إدخال الدورة")
ci1, ci2, ci3, ci4 = st.columns([2.5, 1, 1, 1])
with ci1:
    new_val = st.number_input(
        "v", min_value=1.00, max_value=1000.0,
        value=1.50, step=0.01, format="%.2f", label_visibility="collapsed"
    )
with ci2:
    if st.button("➕ أضف", use_container_width=True):
        st.session_state.history.append(round(new_val,2))
        st.session_state.log.append({"t":datetime.now().strftime("%H:%M:%S"),
                                      "v":round(new_val,2),"r":len(st.session_state.history)})
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
    eng_tmp = CrashIntelligence(h)
    badges = '<div class="spring-bar">'
    for v in h[-30:]:
        gm = eng_tmp._find_golden(v)
        if gm:        cls = "b-gold"
        elif v>=12.0: cls = "b-big"
        elif v>=5.0:  cls = "b-win"
        elif v>=2.0:  cls = "b-med"
        else:         cls = "b-loss"
        sfx = "⭐" if gm else ""
        badges += f'<span class="badge {cls}">x{v:.2f}{sfx}</span>'
    badges += '</div>'
    st.markdown(badges, unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:rgba(255,255,255,0.3);direction:rtl;margin-top:4px;">
    <span style="color:#ff9500;">⭐ ذهبي</span> &nbsp;|&nbsp;
    <span style="color:#a855f7;">■</span> ≥x12 &nbsp;|&nbsp;
    <span style="color:#00c8ff;">■</span> x5–12 &nbsp;|&nbsp;
    <span style="color:#00ff88;">■</span> x2–5 &nbsp;|&nbsp;
    <span style="color:#ff4444;">■</span> &lt;x2
    </div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ══ التحليل الرئيسي ══════════════════════════════════════════════════
h = st.session_state.history
if len(h) < 3:
    st.markdown(f"""
    <div class="st-WAIT" style="text-align:center;padding:35px;">
        <div style="font-size:46px;">⏳</div>
        <div style="font-size:20px;font-weight:700;margin:10px 0;">أضف {3-len(h)} دورات للبدء</div>
        <div style="color:rgba(255,255,255,0.4);">أو استخدم زر "ديمو" في الشريط الجانبي</div>
    </div>""", unsafe_allow_html=True)
else:
    engine = CrashIntelligence(h)
    rec    = engine.full_analysis(st.session_state.balance)
    stats  = engine.stats()
    golden_hist = engine.golden_in_history(25)
    spring = rec["spring"]

    # ═══ الصف الرئيسي ═══════════════════════════════════════════
    col_main, col_right = st.columns([3, 2])

    with col_main:
        # ── بطاقة الحالة ──────────────────────────────────────
        st.markdown(f'<div class="st-{rec["status"]}">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:46px;margin-bottom:8px;">
            {'🔥' if rec['status']=='STRONG' else '⚡' if rec['status']=='DOUBLE'
             else '✅' if rec['status']=='SAFE' else '⛔' if rec['status']=='DANGER'
             else '⏳'}
        </div>
        <div style="font-size:24px;font-weight:900;margin-bottom:10px;">{rec['title']}</div>
        <div style="font-size:15px;color:rgba(255,255,255,0.8);line-height:1.8;">{rec['desc']}</div>
        """, unsafe_allow_html=True)

        # النطاق المستهدف
        if rec["target_lo"] and rec["target_hi"]:
            st.markdown(f"""
            <div style="margin-top:16px;padding:14px;background:rgba(0,0,0,0.35);border-radius:12px;">
                <div style="color:rgba(255,255,255,0.4);font-size:12px;">🎯 النطاق المستهدف:</div>
                <div style="font-family:'Orbitron',monospace;font-size:28px;color:#FFD700;font-weight:900;margin-top:4px;">
                    x{rec['target_lo']:.1f} — x{rec['target_hi']:.1f}
                </div>
            </div>""", unsafe_allow_html=True)

        # معلومات الرهان
        if rec["stake"] > 0:
            st.markdown(f"""
            <div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap;">
                <div style="flex:1;min-width:110px;background:rgba(0,255,136,0.08);
                            border:1px solid rgba(0,255,136,0.25);border-radius:10px;padding:12px;">
                    <div style="color:rgba(255,255,255,0.4);font-size:11px;">💰 الرهان</div>
                    <div style="font-family:'Orbitron',monospace;font-size:20px;color:#00ff88;font-weight:900;">
                        {rec['stake']:.0f}
                    </div>
                    <div style="color:rgba(255,255,255,0.3);font-size:10px;">{rec['stake_pct']:.1f}٪ من الرصيد</div>
                </div>
                <div style="flex:1;min-width:110px;background:rgba(99,102,241,0.08);
                            border:1px solid rgba(99,102,241,0.25);border-radius:10px;padding:12px;">
                    <div style="color:rgba(255,255,255,0.4);font-size:11px;">📈 ربح متوقع</div>
                    <div style="font-family:'Orbitron',monospace;font-size:20px;color:#a855f7;font-weight:900;">
                        +{rec['profit_est']:.0f}
                    </div>
                    <div style="color:rgba(255,255,255,0.3);font-size:10px;">عند منتصف الهدف</div>
                </div>
                <div style="flex:1;min-width:110px;background:rgba(255,215,0,0.06);
                            border:1px solid rgba(255,215,0,0.2);border-radius:10px;padding:12px;">
                    <div style="color:rgba(255,255,255,0.4);font-size:11px;">🏷️ السيناريو</div>
                    <div style="font-family:'Orbitron',monospace;font-size:14px;color:#FFD700;font-weight:900;">
                        {rec['scenario']}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # end status card

        # ── تفاصيل الزنبرك ────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="main-card" style="padding:18px;">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:16px;font-weight:700;color:rgba(255,255,255,0.85);margin-bottom:14px;">
            🔄 حالة الزنبرك المضغوط
        </div>""", unsafe_allow_html=True)

        # شريط الضغط
        p = spring["pressure"]
        pc = "#ff3232" if p>=70 else "#ff9500" if p>=40 else "#FFD700"
        lv_labels = ["لا زنبرك","ضعيف","متوسط","قوي","قوي جداً"]
        st.markdown(f"""
        <div style="margin-bottom:14px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                <span style="color:rgba(255,255,255,0.6);">مستوى الضغط — {lv_labels[spring['level']]}</span>
                <span style="font-family:'Orbitron',monospace;color:{pc};font-weight:700;">{p}%</span>
            </div>
            <div class="prog-wrap">
                <div class="prog-orange" style="width:{p}%;background:linear-gradient(90deg,#ff6400,{pc});"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        c1s,c2s,c3s = st.columns(3)
        with c1s:
            clr = "#ff4444" if spring["streak_2"]>=3 else "#FFD700"
            st.markdown(f"""<div class="kpi-box">
                <div class="kpi-num" style="color:{clr};">{spring['streak_2']}</div>
                <div class="kpi-lbl">خسائر &lt;x2</div></div>""", unsafe_allow_html=True)
        with c2s:
            clr = "#ff3232" if spring["streak_18"]>=3 else "#ff9500"
            st.markdown(f"""<div class="kpi-box">
                <div class="kpi-num" style="color:{clr};">{spring["streak_18"]}</div>
                <div class="kpi-lbl">خسائر &lt;x1.8</div></div>""", unsafe_allow_html=True)
        with c3s:
            st.markdown(f"""<div class="kpi-box">
                <div class="kpi-num">{spring['avg_seq']}x</div>
                <div class="kpi-lbl">متوسط السلسلة</div></div>""", unsafe_allow_html=True)

        if spring["level"] >= 2:
            st.markdown(f"""
            <div style="margin-top:12px;" class="box-success">
                📊 <b>النطاق المتوقع للقفزة:</b>
                <span style="font-family:'Orbitron',monospace;color:#FFD700;">
                    x{spring['exp_lo']:.1f} — x{spring['exp_hi']:.1f}
                </span>
                (بناءً على بيانات {stats['n']} دورة)
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── رقم ذهبي مكتشف ────────────────────────────────────
        if rec.get("golden"):
            g  = rec["golden"]
            gd = g["golden_data"]
            tm = TIER_META[gd["tier"]]
            st.markdown(f"""
            <div class="main-card" style="background:linear-gradient(135deg,rgba(255,149,0,0.1),rgba(255,70,0,0.05));
                         border-color:rgba(255,149,0,0.4);padding:18px;">
                <div style="text-align:center;margin-bottom:14px;">
                    <span style="font-size:32px;">{tm['icon']}</span>
                    <div style="font-size:17px;font-weight:900;color:{tm['color']};margin-top:4px;">
                        رقم ذهبي مكتشف — {tm['label']}
                    </div>
                </div>
                <div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;">
                    <div class="gold-card" style="flex:1;min-width:100px;">
                        <div style="font-size:11px;color:rgba(255,255,255,0.4);">قيمة الدورة</div>
                        <div class="gold-num">x{g['actual_val']:.2f}</div>
                        <div style="font-size:10px;color:rgba(255,255,255,0.3);">≈ x{g['golden_num']}</div>
                    </div>
                    <div class="gold-card" style="flex:1;min-width:100px;">
                        <div style="font-size:11px;color:rgba(255,255,255,0.4);">الهدف</div>
                        <div class="gold-tgt">x{g['target_lo']}–x{g['target_hi']}</div>
                        <div style="font-size:10px;color:rgba(255,255,255,0.3);">{gd['label']}</div>
                    </div>
                    <div class="gold-card" style="flex:1;min-width:100px;">
                        <div style="font-size:11px;color:rgba(255,255,255,0.4);">متوسط تاريخي</div>
                        <div class="gold-num" style="color:#a855f7;">{gd['avg_next']}x</div>
                        <div style="font-size:10px;color:rgba(255,255,255,0.3);">≥x5: {gd['win5_pct']}%</div>
                    </div>
                </div>
                {'<div class="box-success" style="margin-top:12px;">🔽 نمط تسلسل هابط مكتشف — يُعزز الإشارة</div>' if g['is_descending'] else ''}
            </div>""", unsafe_allow_html=True)

        # ── تحذير القفزة المزدوجة ─────────────────────────────
        if rec.get("post_big") and rec["post_big"]["type"] == "double_jump":
            st.markdown(f"""
            <div class="box-warn">
                ⚡ <b>نمط القفزة المزدوجة:</b>
                x{rec['post_big']['prev']:.2f} → x{rec['post_big']['last']:.2f}
                — هذا النمط يحدث في ~25% من الحالات. ادخل بحذر شديد.
            </div>""", unsafe_allow_html=True)

    with col_right:
        # ── مقياس الثقة والزنبرك ──────────────────────────────
        chart_confidence(rec["confidence"], key=f"conf_{len(h)}")
        chart_spring_gauge(spring["pressure"], key=f"sp_{len(h)}")

        # ── إحصائيات سريعة ────────────────────────────────────
        st.markdown('<div class="main-card" style="padding:16px;">', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f'<div class="kpi-box"><div class="kpi-num">{h[-1]:.2f}x</div><div class="kpi-lbl">آخر دورة</div></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="kpi-box"><div class="kpi-num">{stats["avg"]}x</div><div class="kpi-lbl">المتوسط</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        r3, r4 = st.columns(2)
        with r3:
            st.markdown(f'<div class="kpi-box"><div class="kpi-num">{stats["jumps"]}</div><div class="kpi-lbl">قفزات ≥x12</div></div>', unsafe_allow_html=True)
        with r4:
            st.markdown(f'<div class="kpi-box"><div class="kpi-num">{stats["win_rate"]}%</div><div class="kpi-lbl">فوق x2</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── توزيع الفئات ──────────────────────────────────────
        st.markdown('<div class="main-card" style="padding:16px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:700;color:rgba(255,255,255,0.7);margin-bottom:6px;">📊 توزيع الدورات</div>', unsafe_allow_html=True)
        chart_distribution(h, key=f"dist_{len(h)}")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── ماذا انتظر ─────────────────────────────────────────
        if rec["status"] == "WAIT":
            st.markdown("""
            <div class="box-info">
                <b>⏳ انتظر ظهور أحد:</b><br>
                🔸 رقم ذهبي: 1.05، 1.09، 1.20، 1.53، 1.54، 1.77<br>
                🔸 دورة في نطاق x4.8–x6.5<br>
                🔸 3+ خسائر متتالية &lt; x1.8
            </div>""", unsafe_allow_html=True)

    # ═══ الرسم البياني ══════════════════════════════════════════
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("**📈 مسار الدورات** *(الخط البرتقالي = ضغط الزنبرك، النجوم = أرقام ذهبية)*")
    chart_history(h, engine)
    st.markdown("</div>", unsafe_allow_html=True)

    # ═══ جدول الأرقام الذهبية في التاريخ ═══════════════════════
    if golden_hist:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown("**⭐ الأرقام الذهبية في آخر 25 دورة:**")
        cols_g = st.columns(min(len(golden_hist), 5))
        for i, item in enumerate(golden_hist[-5:]):
            gd = item["gdata"]
            tm = TIER_META[gd["tier"]]
            with cols_g[i]:
                st.markdown(f"""
                <div class="gold-card">
                    <div style="font-size:10px;color:rgba(255,255,255,0.35);">#{item['pos']}</div>
                    <div class="gold-num" style="font-size:18px;">{tm['icon']} x{item['val']:.2f}</div>
                    <div style="font-size:10px;color:rgba(255,255,255,0.3);">≈ x{item['gnum']}</div>
                    <div class="gold-tgt" style="font-size:13px;">→ x{gd['tgt_lo']}–x{gd['tgt_hi']}</div>
                    <div style="font-size:10px;color:rgba(255,255,255,0.35);">تاريخي: {gd['avg_next']}x</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ═══ دليل الفرضيات ══════════════════════════════════════════
    with st.expander("📚 الفرضيات والمنهجية — اضغط للتوسيع"):
        st.markdown("""
        <div style="direction:rtl;color:rgba(255,255,255,0.8);line-height:2;">

        <div class="box-success">
        <b>F1 — قانون الزنبرك المضغوط:</b><br>
        2 خسائر → متوسط قفزة 6.8x &nbsp;|&nbsp;
        3 خسائر → 7.4x &nbsp;|&nbsp;
        4 خسائر → 9.2x &nbsp;|&nbsp;
        5 خسائر → 14.2x &nbsp;|&nbsp;
        6+ خسائر → 19.5x+
        </div>

        <div class="box-info">
        <b>F2 — الأرقام الذهبية المصفّاة:</b><br>
        🔥 تير-1: 1.05 (14.5x)، 1.20 (17.2x)، 1.09 (9.7x)<br>
        💎 تير-2: 1.77 (8.3x)، 1.53 (6.7x)، 1.54 (6.0x)، 1.84 (6.6x)<br>
        ✨ تير-3: 1.12، 1.24، 1.36، 1.45 (متوسط 3–5x)<br>
        ❌ محذوفة: 1.91، 1.96، 1.25 (ضعيفة جداً)
        </div>

        <div class="box-success">
        <b>F3 — الإشارة المركّبة (الأقوى):</b><br>
        زنبرك مستوى ≥2 + رقم ذهبي تير1/2 = ثقة 75–94%<br>
        أمثلة: [1.09,1.47,1.54,1.53] → 32.04x ✅<br>
        [1.43,1.13,1.05] → 33.27x ✅
        </div>

        <div class="box-warn">
        <b>F4 — ما بعد القفزة الكبيرة (≥12x):</b><br>
        70% تليها خسائر — تجنب 2-3 دورات.<br>
        استثناء: "قفزة مزدوجة" (~25% من الحالات) — دخول بحذر شديد.
        </div>

        <div class="box-info">
        <b>F5 — نمط التسلسل الهابط:</b><br>
        3+ دورات هابطة متتالية + رقم ذهبي في النهاية = إشارة معززة (+8% ثقة).<br>
        مثال: 2.28→1.24→1.20→1.54 → 24.46x
        </div>

        </div>""", unsafe_allow_html=True)

# ══ تحذير ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="box-warn" style="margin-top:10px;">
    <b>⚠️ تنبيه:</b> هذا نظام تحليل إحصائي للأنماط فقط. Crash يعتمد على RNG.
    لا يوجد ضمان رياضي مطلق. راهن فقط بما تتحمل خسارته.
</div>""", unsafe_allow_html=True)
