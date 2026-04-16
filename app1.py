# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
import random
from datetime import datetime

st.set_page_config(
    page_title="🚀 Crash Predictor PRO",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;700;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    * { font-family: 'Tajawal', sans-serif !important; }
    body, .main { background: #060610 !important; }
    
    /* ===== بطاقات رئيسية ===== */
    .crash-card {
        background: linear-gradient(145deg, rgba(10,10,25,0.97), rgba(15,15,35,0.99));
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 20px 60px rgba(0,0,0,0.8), inset 0 1px 0 rgba(99,102,241,0.2);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        direction: rtl;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .crash-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #6366f1, #a855f7, #ec4899, transparent);
    }
    
    /* ===== الحالة الرئيسية ===== */
    .status-safe {
        background: linear-gradient(135deg, rgba(0,255,136,0.12), rgba(0,200,100,0.06));
        border: 2px solid #00ff88;
        box-shadow: 0 0 30px rgba(0,255,136,0.3), inset 0 0 20px rgba(0,255,136,0.05);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        animation: safePulse 2s ease-in-out infinite;
    }
    @keyframes safePulse { 0%,100% { box-shadow: 0 0 20px rgba(0,255,136,0.2); } 50% { box-shadow: 0 0 50px rgba(0,255,136,0.5); } }
    
    .status-danger {
        background: linear-gradient(135deg, rgba(255,50,50,0.12), rgba(200,0,0,0.06));
        border: 2px solid #ff3232;
        box-shadow: 0 0 30px rgba(255,50,50,0.3), inset 0 0 20px rgba(255,50,50,0.05);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        animation: dangerPulse 1s ease-in-out infinite;
    }
    @keyframes dangerPulse { 0%,100% { box-shadow: 0 0 20px rgba(255,50,50,0.3); } 50% { box-shadow: 0 0 60px rgba(255,50,50,0.7); } }
    
    .status-caution {
        background: linear-gradient(135deg, rgba(255,200,0,0.12), rgba(255,140,0,0.06));
        border: 2px solid #FFD700;
        box-shadow: 0 0 25px rgba(255,215,0,0.25);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
    }
    .status-recover {
        background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(99,102,241,0.08));
        border: 2px solid #8b5cf6;
        box-shadow: 0 0 30px rgba(139,92,246,0.3);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        animation: recoverPulse 1.5s ease-in-out infinite;
    }
    @keyframes recoverPulse { 0%,100% { box-shadow: 0 0 25px rgba(139,92,246,0.25); } 50% { box-shadow: 0 0 55px rgba(139,92,246,0.55); } }
    
    /* ===== شارات الدورات ===== */
    .round-badge-win {
        display: inline-block;
        background: linear-gradient(135deg, #003d1f, #006b37);
        border: 1px solid #00ff88;
        color: #00ff88;
        padding: 8px 16px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 900;
        margin: 4px;
        cursor: pointer;
        transition: all 0.2s;
        font-family: 'Orbitron', monospace !important;
        box-shadow: 0 4px 15px rgba(0,255,136,0.2);
    }
    .round-badge-loss {
        display: inline-block;
        background: linear-gradient(135deg, #3d0000, #6b0000);
        border: 1px solid #ff4444;
        color: #ff6666;
        padding: 8px 16px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 900;
        margin: 4px;
        font-family: 'Orbitron', monospace !important;
        box-shadow: 0 4px 15px rgba(255,50,50,0.2);
    }
    .round-badge-medium {
        display: inline-block;
        background: linear-gradient(135deg, #1a1200, #2a2000);
        border: 1px solid #FFD700;
        color: #FFD700;
        padding: 8px 16px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 900;
        margin: 4px;
        font-family: 'Orbitron', monospace !important;
        box-shadow: 0 4px 15px rgba(255,215,0,0.2);
    }
    
    /* ===== توصية الرهان ===== */
    .bet-target {
        background: linear-gradient(135deg, #0a0a2e, #12124a);
        border: 2px solid #6366f1;
        border-radius: 14px;
        padding: 18px 22px;
        margin: 8px 0;
        direction: rtl;
        box-shadow: 0 8px 25px rgba(99,102,241,0.2);
        transition: all 0.3s;
    }
    .bet-target:hover {
        border-color: #a855f7;
        box-shadow: 0 12px 35px rgba(168,85,247,0.3);
        transform: translateY(-2px);
    }
    .multiplier-tag {
        font-family: 'Orbitron', monospace !important;
        font-size: 26px;
        font-weight: 900;
        color: #00ff88;
        text-shadow: 0 0 15px rgba(0,255,136,0.5);
    }
    .prob-bar {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        height: 8px;
        margin: 8px 0;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        transition: width 0.5s ease;
    }
    
    /* ===== إحصائيات ===== */
    .stat-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        direction: rtl;
    }
    .stat-number {
        font-family: 'Orbitron', monospace !important;
        font-size: 28px;
        font-weight: 900;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        color: rgba(255,255,255,0.4);
        font-size: 12px;
        margin-top: 5px;
        letter-spacing: 1px;
    }
    
    /* ===== قائمة الاسترداد ===== */
    .recover-item {
        background: rgba(139,92,246,0.08);
        border: 1px solid rgba(139,92,246,0.3);
        border-radius: 10px;
        padding: 12px 18px;
        margin: 5px 0;
        direction: rtl;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* ===== تحذير ===== */
    .warning-box {
        background: rgba(255,100,0,0.08);
        border: 1px solid rgba(255,100,0,0.4);
        border-right: 4px solid #ff6400;
        border-radius: 12px;
        padding: 15px 20px;
        color: rgba(255,200,150,0.9);
        font-size: 14px;
        direction: rtl;
        margin: 10px 0;
    }
    
    /* ===== ليبل السيناريو ===== */
    .scenario-tag {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        margin: 3px;
    }
    .tag-avoid { background: rgba(255,50,50,0.2); color: #ff6666; border: 1px solid #ff4444; }
    .tag-bet { background: rgba(0,255,136,0.15); color: #00ff88; border: 1px solid #00cc70; }
    .tag-wait { background: rgba(255,215,0,0.15); color: #FFD700; border: 1px solid #cc9900; }
    .tag-recover { background: rgba(139,92,246,0.2); color: #c4b5fd; border: 1px solid #8b5cf6; }
    
    /* ===== شريط التاريخ ===== */
    .history-bar {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 15px 20px;
        direction: rtl;
        margin: 10px 0;
    }
    
    /* ===== زر الإدخال ===== */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        border-radius: 12px !important;
        padding: 12px 30px !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
        transition: all 0.3s !important;
        font-family: 'Tajawal' !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px rgba(99,102,241,0.6) !important;
    }
    .stNumberInput > div > div > input, .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        color: white !important;
        border: 1px solid rgba(99,102,241,0.4) !important;
        border-radius: 10px !important;
        font-family: 'Tajawal' !important;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# ثوابت وإعدادات
# ══════════════════════════════════════════════════════
# قائمة الاستراتيجية: نقاط السحب لوضع الاسترداد
RECOVERY_TARGETS = [
    1.01, 1.05, 1.07, 1.09, 1.12, 1.14, 1.19, 1.20, 1.22, 1.24, 1.25, 1.29,
    1.32, 1.36, 1.45, 1.49, 1.50, 1.53, 1.54, 1.57, 1.59, 1.60, 1.66, 1.73,
    1.74, 1.76, 1.77, 1.83, 1.84, 1.91, 1.94, 1.96
]

# حدود الأنماط
PATTERN_THRESHOLDS = {
    "big_crash_after_12": 12.90,
    "crash_after_big_1x9": 1.90,
    "three_losses_limit": 1.80,
    "after_loss_range_low": 2.10,
    "after_loss_range_high": 2.90,
    "rocket_surge_low": 5.00,
    "rocket_surge_high": 6.00,
    "surge_avoid_rounds": 5,
    "medium_rise": 3.00,
    "medium_avoid": 4,
    "double_rise_limit": 2.00,
    "double_consecutive": 2,
    "long_loss_streak": 7,
}

# ══════════════════════════════════════════════════════
# منطق التحليل المتقدم
# ══════════════════════════════════════════════════════
class CrashAnalyzer:
    """
    المحرك الذكي لتحليل أنماط Crash وتوليد التوصيات
    """
    def __init__(self, history: list):
        self.history = history  # قائمة مضاعفات الدورات السابقة (الأحدث في النهاية)
        self.n = len(history)

    # ──────────────────────────────────────────
    # 1. كشف الأنماط
    # ──────────────────────────────────────────
    def _last(self, count: int) -> list:
        """آخر N دورات"""
        return self.history[-count:] if self.n >= count else self.history[:]

    def _is_rise(self, val: float, threshold: float = 2.0) -> bool:
        return val >= threshold

    def detect_patterns(self) -> list:
        """
        يُعيد قائمة بالأنماط المكتشفة
        كل نمط: {id, name, action, rounds_to_avoid, confidence, reason, icon}
        """
        patterns = []
        if self.n == 0:
            return patterns

        last1 = self.history[-1]
        last2 = self._last(2)
        last3 = self._last(3)
        last5 = self._last(5)
        last7 = self._last(7)

        # ── النمط 1: انفجار كبير بعد x12.90 ──────────────────
        # الدورة التالية بعد x≥12.90 → تنفجر بين x1.5 - x1.90
        if last1 >= PATTERN_THRESHOLDS["big_crash_after_12"]:
            patterns.append({
                "id": "P1",
                "icon": "💥",
                "name": "ما بعد الانفجار الكبير",
                "action": "danger",
                "rounds_to_avoid": 1,
                "confidence": 88,
                "reason": f"الدورة الأخيرة كانت x{last1:.2f} (≥12.90). الدورة القادمة ستنفجر في نطاق x1.50-x1.90 غالباً.",
                "predicted_range": (1.50, 1.90),
                "bet_suggestion": None,
            })

        # ── النمط 2: 3 خسائر متتالية تحت x1.80 ──────────────
        if len(last3) == 3 and all(v < PATTERN_THRESHOLDS["three_losses_limit"] for v in last3):
            patterns.append({
                "id": "P2",
                "icon": "🔄",
                "name": "ارتداد بعد 3 خسائر",
                "action": "bet_range",
                "rounds_to_avoid": 0,
                "confidence": 78,
                "reason": f"3 دورات متتالية تحت x1.80: {[f'x{v:.2f}' for v in last3]}. الارتداد متوقع في نطاق x2.10-x2.90.",
                "predicted_range": (2.10, 2.90),
                "bet_suggestion": 2.05,
            })

        # ── النمط 3: صعود x5 في بداية الدورة ────────────────
        if last1 >= PATTERN_THRESHOLDS["rocket_surge_low"] and last1 < 12.0:
            # إذا كان الصعود في منتصف المدى (5-12)
            patterns.append({
                "id": "P3",
                "icon": "🚀",
                "name": "استكمال الصعود العالي",
                "action": "bet_range",
                "rounds_to_avoid": 0,
                "confidence": 72,
                "reason": f"الدورة الأخيرة كانت x{last1:.2f}. الدورة القادمة ستُكمل الصعود وتنفجر في نطاق x5-x6.",
                "predicted_range": (5.0, 6.0),
                "bet_suggestion": 4.80,
            })

        # ── النمط 4: صعود ملحوظ فاتك (تجنب 5-6 دورات) ──────
        # إذا كانت آخر دورة كبيرة جداً (≥5) وأنت لم تدخل
        if self.n >= 2:
            second_last = self.history[-2] if self.n >= 2 else 0
            if second_last >= 5.0 and last1 < 2.0:
                patterns.append({
                    "id": "P4",
                    "icon": "⚠️",
                    "name": "صعود ملحوظ فاتك",
                    "action": "avoid",
                    "rounds_to_avoid": PATTERN_THRESHOLDS["surge_avoid_rounds"],
                    "confidence": 75,
                    "reason": f"صعود كبير x{second_last:.2f} تلاه هبوط x{last1:.2f}. الاحتمالات في 5-6 دورات قادمة للخسارة.",
                    "predicted_range": (1.0, 2.0),
                    "bet_suggestion": None,
                })

        # ── النمط 5: صعود x3 (تجنب 4 دورات بعده) ────────────
        if self.n >= 2:
            recent = self._last(3)
            for i, v in enumerate(recent[:-1]):
                if PATTERN_THRESHOLDS["medium_rise"] <= v < 5.0:
                    rounds_since = len(recent) - 1 - i
                    avoid_remaining = PATTERN_THRESHOLDS["medium_avoid"] - rounds_since
                    if avoid_remaining > 0:
                        patterns.append({
                            "id": "P5",
                            "icon": "📉",
                            "name": "تجنب ما بعد الصعود المتوسط",
                            "action": "avoid",
                            "rounds_to_avoid": avoid_remaining,
                            "confidence": 68,
                            "reason": f"وُجد صعود x{v:.2f} قبل {rounds_since+1} دورات. تجنب {avoid_remaining} دورات أخرى.",
                            "predicted_range": (1.0, 2.0),
                            "bet_suggestion": None,
                        })
                        break

        # ── النمط 6: صعودان متتاليان فوق x2 ────────────────
        if len(last3) >= 3:
            rises = [self._is_rise(v) for v in last3]
            consecutive_rises = 0
            for r in reversed(rises):
                if r:
                    consecutive_rises += 1
                else:
                    break
            if consecutive_rises >= 2:
                patterns.append({
                    "id": "P6",
                    "icon": "🔻",
                    "name": "7 دورات خسارة بعد صعودين",
                    "action": "recover",
                    "rounds_to_avoid": 7,
                    "confidence": 82,
                    "reason": f"صعودان متتاليان فوق x2. توقع 7 دورات لن تتخطى x2. شغّل وضع الاسترداد!",
                    "predicted_range": (1.01, 1.99),
                    "bet_suggestion": None,
                })

        # ── النمط 7: تسلسل 7 خسائر → وضع الاسترداد ─────────
        if len(last7) == 7 and all(v < 2.0 for v in last7):
            patterns.append({
                "id": "P7",
                "icon": "💜",
                "name": "وضع الاسترداد الكامل",
                "action": "recover",
                "rounds_to_avoid": 0,
                "confidence": 91,
                "reason": f"7 دورات متتالية تحت x2. وضع الاسترداد نشط الآن — استخدم قائمة النقاط المنخفضة.",
                "predicted_range": (1.01, 1.96),
                "bet_suggestion": None,
            })

        # ── النمط 8: دورة انفجار مبكر بعد x12.90 ────────────
        if self.n >= 2:
            if self.history[-2] >= 12.90 and last1 < 2.0:
                patterns.append({
                    "id": "P8",
                    "icon": "⚡",
                    "name": "انفجار بعد انفجار كبير",
                    "action": "avoid",
                    "rounds_to_avoid": 2,
                    "confidence": 80,
                    "reason": f"تأكيد النمط 1: x{self.history[-2]:.2f} → x{last1:.2f}. الهشاشة مستمرة لدورتين.",
                    "predicted_range": (1.0, 1.90),
                    "bet_suggestion": None,
                })

        return patterns

    # ──────────────────────────────────────────
    # 2. حساب الاحتمالات
    # ──────────────────────────────────────────
    def calculate_probabilities(self, patterns: list) -> dict:
        """ يحسب احتمالية الأنماط القادمة بناءً على التاريخ والقواعد """
        if self.n == 0:
            return {"danger": 0.5, "safe": 0.3, "medium": 0.2, "recover": 0.0}

        # إحصاء بسيط من التاريخ
        losses = sum(1 for v in self.history if v < 2.0)
        wins_medium = sum(1 for v in self.history if 2.0 <= v < 5.0)
        wins_high = sum(1 for v in self.history if v >= 5.0)

        base_danger = losses / max(self.n, 1)
        base_medium = wins_medium / max(self.n, 1)
        base_high = wins_high / max(self.n, 1)

        # تعديل بناءً على الأنماط
        danger_boost = 0.0
        safe_boost = 0.0
        recover_boost = 0.0

        for p in patterns:
            if p["action"] == "danger":
                danger_boost += 0.25 * (p["confidence"] / 100)
            elif p["action"] == "avoid":
                danger_boost += 0.15 * (p["confidence"] / 100)
            elif p["action"] == "bet_range":
                safe_boost += 0.20 * (p["confidence"] / 100)
            elif p["action"] == "recover":
                recover_boost += 0.30 * (p["confidence"] / 100)

        danger = min(0.95, base_danger + danger_boost)
        safe = min(0.90, base_medium + safe_boost)
        recover = min(0.95, recover_boost)

        total = danger + safe + recover + 0.05
        return {
            "danger": round(danger / total, 3),
            "safe": round(safe / total, 3),
            "medium": round(base_high / total, 3),
            "recover": round(recover / total, 3),
        }

    # ──────────────────────────────────────────
    # 3. التوصية النهائية
    # ──────────────────────────────────────────
    def get_recommendation(self) -> dict:
        """التوصية الشاملة للدورة القادمة"""
        patterns = self.detect_patterns()
        probs = self.calculate_probabilities(patterns)

        # تحديد الحالة
        if not patterns:
            return {
                "status": "neutral",
                "main_action": "انتظر",
                "icon": "⏳",
                "css_class": "status-caution",
                "title": "لا يوجد نمط واضح",
                "description": "أضف المزيد من الدورات لتحليل دقيق",
                "patterns": [],
                "probs": probs,
                "bet_targets": [],
                "recovery_mode": False,
                "avoid_rounds": 0,
                "predicted_range": None,
            }

        # ترتيب حسب الأولوية
        priority_order = {"danger": 0, "avoid": 1, "recover": 2, "bet_range": 3, "neutral": 4}
        sorted_patterns = sorted(patterns, key=lambda p: priority_order.get(p["action"], 5))
        top = sorted_patterns[0]

        status_map = {
            "danger": ("danger", "🚫 لا تراهن", "⛔", "status-danger", "خطر عالٍ — دورة انفجار مبكر"),
            "avoid": ("caution", "⏭️ تجنب", "⚠️", "status-caution", "تجنب هذه الدورات"),
            "recover": ("recover", "💜 استرداد", "💜", "status-recover", "وضع الاسترداد — راهن بنقاط منخفضة"),
            "bet_range": ("safe", "✅ راهن", "🚀", "status-safe", "فرصة مناسبة للرهان"),
        }
        st_key, action, icon, css, title_sfx = status_map.get(
            top["action"], ("neutral", "⏳ انتظر", "⏳", "status-caution", "انتظر إشارة أوضح")
        )

        # قائمة نقاط الرهان
        pred_range = top.get("predicted_range")
        bet_targets = []
        recovery_mode = top["action"] == "recover"

        if recovery_mode:
            bet_targets = RECOVERY_TARGETS
        elif top["action"] == "bet_range" and pred_range:
            low, high = pred_range
            # اختر نقطة سحب آمنة (أقل من المتوقع بهامش 15%)
            safe_exit = round(low * 0.88, 2)
            safe_exit = max(safe_exit, 1.05)
            bet_targets = [safe_exit, round(low * 0.92, 2), round((low + high) / 2 * 0.85, 2)]

        return {
            "status": st_key,
            "main_action": action,
            "icon": icon,
            "css_class": css,
            "title": f"{icon} {title_sfx}",
            "description": top["reason"],
            "patterns": sorted_patterns,
            "probs": probs,
            "bet_targets": bet_targets,
            "recovery_mode": recovery_mode,
            "avoid_rounds": top.get("rounds_to_avoid", 0),
            "predicted_range": pred_range,
            "top_confidence": top["confidence"],
        }


# ══════════════════════════════════════════════════════
# دوال الرسم البياني
# ══════════════════════════════════════════════════════
def render_history_chart(history: list):
    if len(history) < 2:
        return

    x = list(range(1, len(history) + 1))
    colors = []
    for v in history:
        if v >= 5.0:
            colors.append("#a855f7")
        elif v >= 2.0:
            colors.append("#00ff88")
        else:
            colors.append("#ff4444")

    fig = go.Figure()

    # منطقة الخطر
    fig.add_hrect(y0=0, y1=2.0, fillcolor="rgba(255,50,50,0.05)", line_width=0,
                  annotation_text="منطقة الخسارة", annotation_position="left")
    # منطقة الأمان
    fig.add_hrect(y0=2.0, y1=5.0, fillcolor="rgba(0,255,136,0.03)", line_width=0)
    # منطقة الجائزة الكبرى
    fig.add_hrect(y0=5.0, y1=max(max(history)*1.1, 10), fillcolor="rgba(168,85,247,0.05)", line_width=0,
                  annotation_text="منطقة الجائزة", annotation_position="left")

    # خط الرسم
    fig.add_trace(go.Scatter(
        x=x, y=history, mode="lines+markers+text",
        line=dict(color="rgba(99,102,241,0.6)", width=2, shape="spline"),
        marker=dict(color=colors, size=12, line=dict(color="rgba(255,255,255,0.3)", width=1)),
        text=[f"x{v:.2f}" for v in history],
        textposition="top center",
        textfont=dict(color="white", size=10, family="Orbitron"),
    ))

    # خط الإشارة عند 2.0
    fig.add_hline(y=2.0, line_dash="dash", line_color="rgba(255,215,0,0.4)", line_width=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Tajawal"),
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(showgrid=False, title="الدورة", tickfont=dict(color="rgba(255,255,255,0.4)")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="المضاعف",
                   tickprefix="x", tickfont=dict(color="rgba(255,255,255,0.4)")),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"hist_{len(history)}")


def render_probability_chart(probs: dict):
    labels = ["خطر (< x2)", "متوسط (x2-x5)", "عالي (> x5)", "استرداد"]
    values = [probs["danger"]*100, probs["safe"]*100, probs["medium"]*100, probs["recover"]*100]
    colors_pie = ["#ff4444", "#00ff88", "#a855f7", "#6366f1"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.65,
        marker=dict(colors=colors_pie, line=dict(color="rgba(0,0,0,0.5)", width=2)),
        textfont=dict(color="white", size=12, family="Tajawal"),
        textinfo="label+percent",
    ))

    fig.add_annotation(
        text=f"<b>{max(values):.0f}%</b><br>أعلى<br>احتمال",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="white", family="Tajawal"),
        align="center"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Tajawal"),
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.2,
            font=dict(size=11, color="rgba(255,255,255,0.6)"),
            bgcolor="rgba(0,0,0,0)"
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"prob_{sum(v for v in probs.values())}")


def render_confidence_bar(confidence: int, label: str, key: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={"text": label, "font": {"size": 13, "color": "rgba(255,255,255,0.7)", "family": "Tajawal"}},
        number={"suffix": "%", "font": {"size": 30, "color": "#a855f7", "family": "Orbitron"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.2)"},
            "bar": {"color": "#6366f1", "thickness": 0.35},
            "bgcolor": "rgba(0,0,0,0.2)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(255,50,50,0.1)"},
                {"range": [40, 70], "color": "rgba(255,215,0,0.1)"},
                {"range": [70, 100], "color": "rgba(0,255,136,0.1)"},
            ],
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ══════════════════════════════════════════════════════
# إدارة الحالة
# ══════════════════════════════════════════════════════
if "crash_history" not in st.session_state:
    st.session_state.crash_history = []
if "balance" not in st.session_state:
    st.session_state.balance = 1000.0
if "session_log" not in st.session_state:
    st.session_state.session_log = []

# ══════════════════════════════════════════════════════
# الواجهة الرئيسية
# ══════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px;">
    <div style="font-family:'Orbitron',monospace; font-size:42px; font-weight:900; background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899, #a855f7, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-size: 200%; animation: gradientShift 3s linear infinite;">
        🚀 CRASH PREDICTOR PRO
    </div>
    <div style="color:rgba(255,255,255,0.4); font-size:14px; letter-spacing:3px; margin-top:5px;">
        نظام التحليل الذكي للأنماط
    </div>
</div>
<style>
    @keyframes gradientShift { 0%{background-position:0%} 100%{background-position:200%} }
</style>
""", unsafe_allow_html=True)

# ── الشريط الجانبي ──────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align:center; color:#a855f7; font-size:20px; font-weight:700; margin-bottom:15px;">⚙️ لوحة التحكم</div>', unsafe_allow_html=True)

    st.markdown("**💰 الرصيد**")
    st.session_state.balance = st.number_input(
        "رصيدك الحالي", min_value=10.0, max_value=1_000_000.0,
        value=st.session_state.balance, step=50.0, label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**📊 إعدادات التحليل**")
    min_history = st.slider("أقل عدد دورات للتحليل", 3, 10, 5)
    show_all_patterns = st.checkbox("إظهار كل الأنماط", value=True)
    auto_recovery_threshold = st.slider("حد تفعيل الاسترداد (خسائر متتالية)", 3, 7, 5)

    st.markdown("---")
    st.markdown("**🔧 أدوات**")
    if st.button("🗑️ مسح السجل", use_container_width=True):
        st.session_state.crash_history = []
        st.session_state.session_log = []
        st.rerun()

    if st.button("🎲 محاكاة (تجريبي)", use_container_width=True):
        # توليد 10 دورات عشوائية بتوزيع واقعي
        sim = []
        for _ in range(10):
            r = random.random()
            if r < 0.50:
                sim.append(round(random.uniform(1.0, 1.99), 2))
            elif r < 0.75:
                sim.append(round(random.uniform(2.0, 4.99), 2))
            elif r < 0.92:
                sim.append(round(random.uniform(5.0, 12.0), 2))
            else:
                sim.append(round(random.uniform(12.0, 50.0), 2))
        st.session_state.crash_history = sim
        st.rerun()

    st.markdown("---")
    # إحصائيات الجلسة
    h = st.session_state.crash_history
    if h:
        losses = sum(1 for v in h if v < 2.0)
        wins = len(h) - losses
        avg = np.mean(h)
        st.markdown(f"""
        <div class="stat-box" style="margin:5px 0;">
            <div class="stat-number">{len(h)}</div>
            <div class="stat-label">إجمالي الدورات</div>
        </div>
        <div class="stat-box" style="margin:5px 0;">
            <div class="stat-number" style="color:#00ff88;">{wins}</div>
            <div class="stat-label">دورات فوق x2</div>
        </div>
        <div class="stat-box" style="margin:5px 0;">
            <div class="stat-number" style="color:#ff4444;">{losses}</div>
            <div class="stat-label">دورات خسارة</div>
        </div>
        <div class="stat-box" style="margin:5px 0;">
            <div class="stat-number">{avg:.2f}x</div>
            <div class="stat-label">متوسط المضاعف</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# منطقة الإدخال
# ══════════════════════════════════════════════════════
st.markdown('<div class="crash-card">', unsafe_allow_html=True)
st.markdown("### 📥 أدخل نتيجة الدورة الأخيرة")

col_in1, col_in2, col_in3 = st.columns([2, 1, 1])
with col_in1:
    new_value = st.number_input(
        "مضاعف الانفجار (مثال: 1.45 أو 7.20)",
        min_value=1.00, max_value=1000.0, value=1.50, step=0.01,
        format="%.2f", label_visibility="collapsed"
    )
with col_in2:
    if st.button("➕ أضف الدورة", use_container_width=True):
        st.session_state.crash_history.append(round(new_value, 2))
        # حفظ في السجل مع الوقت
        st.session_state.session_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "value": round(new_value, 2),
            "index": len(st.session_state.crash_history),
        })
        st.rerun()
with col_in3:
    if st.button("↩️ حذف آخر دورة", use_container_width=True):
        if st.session_state.crash_history:
            st.session_state.crash_history.pop()
        if st.session_state.session_log:
            st.session_state.session_log.pop()
        st.rerun()

# عرض شريط الدورات
h = st.session_state.crash_history
if h:
    st.markdown("**📋 سجل الدورات:**")
    badges_html = '<div class="history-bar">'
    for i, v in enumerate(h[-20:]):  # آخر 20
        if v >= 5.0:
            cls = "round-badge-win"
        elif v >= 2.0:
            cls = "round-badge-medium"
        else:
            cls = "round-badge-loss"
        idx = max(0, len(h) - 20) + i + 1
        badges_html += f'<span class="{cls}" title="الدورة {idx}">x{v:.2f}</span>'
    badges_html += '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# التحليل والتوصية
# ══════════════════════════════════════════════════════
h = st.session_state.crash_history
if len(h) < min_history:
    st.markdown(f"""
    <div class="status-caution" style="text-align:center; padding:30px;">
        <div style="font-size:50px;">⏳</div>
        <div style="font-size:22px; font-weight:700; margin:10px 0;">أضف المزيد من الدورات</div>
        <div style="color:rgba(255,255,255,0.5);">تحتاج إلى {min_history - len(h)} دورة إضافية لبدء التحليل</div>
    </div>
    """, unsafe_allow_html=True)
else:
    analyzer = CrashAnalyzer(h)
    rec = analyzer.get_recommendation()

    # ── الرسائل الرئيسية ──────────────────────────────
    col_main, col_prob = st.columns([3, 2])

    with col_main:
        # الحالة الرئيسية
        st.markdown(f'<div class="{rec["css_class"]}">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:50px; margin-bottom:10px;">{rec['icon']}</div>
        <div style="font-size:28px; font-weight:900; margin-bottom:12px;">{rec['title']}</div>
        <div style="font-size:16px; color:rgba(255,255,255,0.8); line-height:1.7;">{rec['description']}</div>
        """, unsafe_allow_html=True)

        if rec.get("predicted_range"):
            low, high = rec["predicted_range"]
            st.markdown(f"""
            <div style="margin-top:15px; padding:12px; background:rgba(0,0,0,0.3); border-radius:10px;">
                <span style="color:rgba(255,255,255,0.5); font-size:13px;">النطاق المتوقع للدورة القادمة:</span><br>
                <span style="font-family:'Orbitron',monospace; font-size:24px; color:#FFD700; font-weight:900;">
                    x{low:.2f} — x{high:.2f}
                </span>
            </div>
            """, unsafe_allow_html=True)

        if rec.get("avoid_rounds", 0) > 0:
            st.markdown(f"""
            <div style="margin-top:12px; background:rgba(255,50,50,0.15); border-radius:8px; padding:10px 15px;">
                ⏭️ <b>تجنب {rec['avoid_rounds']} دورة قادمة</b>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # الأنماط المكتشفة
        if rec["patterns"] and show_all_patterns:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**🔍 الأنماط المكتشفة:**")
            for p in rec["patterns"]:
                action_tag = {
                    "danger": '<span class="scenario-tag tag-avoid">⛔ خطر</span>',
                    "avoid": '<span class="scenario-tag tag-wait">⚠️ تجنب</span>',
                    "bet_range": '<span class="scenario-tag tag-bet">✅ راهن</span>',
                    "recover": '<span class="scenario-tag tag-recover">💜 استرداد</span>',
                }.get(p["action"], "")
                st.markdown(f"""
                <div class="bet-target">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            {action_tag}
                            <span style="font-weight:700; font-size:16px; margin-right:8px;">{p['icon']} {p['name']}</span>
                        </div>
                        <div style="text-align:left;">
                            <div style="color:#a855f7; font-size:13px;">ثقة: {p['confidence']}%</div>
                            <div class="prob-bar"><div class="prob-fill" style="width:{p['confidence']}%;"></div></div>
                        </div>
                    </div>
                    <div style="color:rgba(255,255,255,0.6); font-size:14px; margin-top:8px; line-height:1.6;">
                        {p['reason']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_prob:
        # مخطط الاحتمالات
        st.markdown("**📊 توزيع الاحتمالات:**")
        render_probability_chart(rec["probs"])

        # مقياس الثقة
        render_confidence_bar(
            rec.get("top_confidence", 50),
            "مستوى الثقة في التوصية",
            key=f"conf_{len(h)}"
        )

    # ── نقاط الرهان ────────────────────────────────────
    if rec["bet_targets"]:
        st.markdown("<br>", unsafe_allow_html=True)
        if rec["recovery_mode"]:
            st.markdown("""
            <div class="crash-card">
                <div style="text-align:center; font-size:24px; font-weight:900; color:#c4b5fd; margin-bottom:20px;">
                    💜 وضع الاسترداد — اسحب عند هذه النقاط
                </div>
                <div class="warning-box">
                    ⚠️ <b>استراتيجية الاسترداد:</b> راهن بمبلغ صغير (1-2٪ من رصيدك) واسحب عند أول نقطة من القائمة.
                    الهدف استرداد الخسائر بخطوات آمنة، ليس الربح الكبير.
                </div>
            """, unsafe_allow_html=True)
            # عرض قائمة الاسترداد
            cols_rec = st.columns(4)
            for i, target in enumerate(RECOVERY_TARGETS):
                # حساب المبلغ المقترح لكل نقطة (محافظ)
                stake_pct = 0.015  # 1.5٪ من الرصيد
                suggested_stake = max(5.0, st.session_state.balance * stake_pct)
                potential = round(suggested_stake * target, 2)
                profit = round(potential - suggested_stake, 2)
                with cols_rec[i % 4]:
                    st.markdown(f"""
                    <div class="recover-item" style="flex-direction:column; align-items:flex-start;">
                        <div class="multiplier-tag">x{target}</div>
                        <div style="color:rgba(255,255,255,0.5); font-size:11px; margin-top:4px;">
                            رهان: {suggested_stake:.0f} → ربح: +{profit:.1f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # نقاط رهان عادية
            st.markdown(f"""
            <div class="crash-card">
                <div style="text-align:center; font-size:22px; font-weight:900; color:#00ff88; margin-bottom:20px;">
                    🎯 نقاط السحب المقترحة
                </div>
            """, unsafe_allow_html=True)
            cols_bet = st.columns(len(rec["bet_targets"]))
            for i, target in enumerate(rec["bet_targets"]):
                stake_pct = 0.03
                suggested_stake = max(10.0, st.session_state.balance * stake_pct)
                potential = round(suggested_stake * target, 2)
                profit = round(potential - suggested_stake, 2)
                with cols_bet[i]:
                    label = ["✅ نقطة آمنة", "🟡 متوازنة", "🔵 جريئة"][i % 3]
                    st.markdown(f"""
                    <div style="background:rgba(0,255,136,0.08); border:1px solid #00ff88; border-radius:14px; padding:20px; text-align:center;">
                        <div style="font-size:13px; color:#00ff88; margin-bottom:8px;">{label}</div>
                        <div class="multiplier-tag" style="font-size:30px;">x{target:.2f}</div>
                        <div style="margin-top:12px; border-top:1px solid rgba(255,255,255,0.1); padding-top:10px;">
                            <span style="color:rgba(255,255,255,0.5); font-size:12px;">رهان مقترح:</span><br>
                            <span style="color:#FFD700; font-size:20px; font-weight:900;">{suggested_stake:.0f}</span>
                            <span style="color:rgba(255,255,255,0.4); font-size:12px;"> وحدة</span>
                        </div>
                        <div style="color:#00ff88; font-size:14px; margin-top:6px;">
                            ربح متوقع: +{profit:.1f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── الرسم البياني للتاريخ ──────────────────────────
    st.markdown('<div class="crash-card">', unsafe_allow_html=True)
    st.markdown("**📈 مسار الدورات السابقة**")
    render_history_chart(h)

    # إحصائيات سريعة
    c1, c2, c3, c4, c5 = st.columns(5)
    losses_streak = 0
    for v in reversed(h):
        if v < 2.0:
            losses_streak += 1
        else:
            break

    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{h[-1]:.2f}x</div><div class="stat-label">آخر دورة</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{np.mean(h):.2f}x</div><div class="stat-label">المتوسط الكلي</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{max(h):.2f}x</div><div class="stat-label">أعلى مضاعف</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:{"#ff4444" if losses_streak>3 else "#FFD700"};">{losses_streak}</div><div class="stat-label">خسائر متتالية</div></div>', unsafe_allow_html=True)
    with c5:
        win_rate = sum(1 for v in h if v >= 2.0) / len(h) * 100
        st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate:.0f}%</div><div class="stat-label">معدل الفوق x2</div></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── تحذير مسؤول ──────────────────────────────────
st.markdown("""
<div class="warning-box">
    <b>⚠️ تنبيه مهم:</b> هذا النظام أداة تحليل إحصائي للأنماط فقط. ألعاب Crash تعتمد على مولد أرقام عشوائية (RNG)
    ولا يوجد ضمان رياضي مطلق للتوقعات. تحمّل فقط ما تستطيع خسارته. الهدف من النظام: تقليل المخاطر وليس ضمان الربح.
</div>
""", unsafe_allow_html=True)
