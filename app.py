import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

# ============================================================
# إعداد الصفحة
# ============================================================
st.set_page_config(
    page_title="نظام التنبؤ الذكي",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص
st.markdown("""
<style>
    body { direction: rtl; }
    .main { direction: rtl; }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        margin: 5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2.2em;
        font-weight: bold;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.95em;
        color: #b0c4de;
        margin-top: 5px;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #0d2137, #1a4a6b);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: white;
        margin: 10px 0;
    }
    .pred-value {
        font-size: 3em;
        font-weight: bold;
        color: #00ff88;
    }
    .pred-range {
        font-size: 1.1em;
        color: #87ceeb;
        margin-top: 8px;
    }
    
    .confidence-high   { color: #00ff88; font-weight: bold; }
    .confidence-medium { color: #ffa500; font-weight: bold; }
    .confidence-low    { color: #ff4444; font-weight: bold; }
    
    .rec-box {
        border-radius: 10px;
        padding: 15px 20px;
        margin: 8px 0;
        font-size: 1.05em;
    }
    .rec-strong {
        background: rgba(0,255,136,0.15);
        border-left: 4px solid #00ff88;
        color: #00ff88;
    }
    .rec-moderate {
        background: rgba(255,165,0,0.15);
        border-left: 4px solid #ffa500;
        color: #ffa500;
    }
    .rec-weak {
        background: rgba(255,68,68,0.15);
        border-left: 4px solid #ff4444;
        color: #ff4444;
    }
    
    .warning-box {
        background: rgba(255,165,0,0.1);
        border: 1px solid #ffa500;
        border-radius: 8px;
        padding: 12px;
        color: #ffa500;
        margin: 8px 0;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a, #1a3a5c);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# البيانات الأصلية
# ============================================================
DEFAULT_DATA = [
    8.72, 6.75, 1.86, 2.18, 1.25, 2.28, 1.24, 1.2, 1.54, 24.46, 4.16, 1.49,
    1.09, 1.47, 1.54, 1.53, 2.1, 32.04, 11, 1.17, 1.7, 2.61, 1.26, 22.23,
    1.77, 1.93, 3.35, 7.01, 1.83, 9.39, 3.31, 2.04, 1.3, 6.65, 1.16, 3.39,
    1.95, 10.85, 1.65, 1.22, 1.6, 4.67, 1.85, 2.72, 1, 3.02, 1.35, 1.3,
    1.37, 17.54, 1.18, 1, 14.4, 1.11, 6.15, 2.39, 2.22, 1.42, 1.23, 2.42,
    1.07, 1.24, 2.55, 7.26, 1.69, 5.1, 2.59, 5.51, 2.31, 2.12, 1.97, 1.5,
    3.01, 2.29, 1.36, 4.95, 5.09, 8.5, 1.77, 5.52, 3.93, 1.5, 2.28, 2.49,
    18.25, 1.68, 1.42, 2.12, 4.17, 1.04, 2.35, 1, 1.01, 5.46, 1.13, 2.84,
    3.39, 2.79, 1.59, 1.53, 4.34, 2.96, 1.06, 1.72, 2.16, 2.2, 3.61, 2.34,
    4.49, 1.72, 1.78, 9.27, 8.49, 2.86, 1.66, 4.63, 9.25, 1.35, 1, 1.64,
    1.86, 2.81, 2.44, 1.74, 1.1, 1.29, 1.45, 8.92, 1.24, 6.39, 1.16, 1.19,
    2.4, 4.64, 3.17, 24.21, 1.17, 1.42, 2.13, 1.12, 3.78, 1.12, 1.52, 22.81,
    1.31, 1.9, 1.38, 1.47, 2.86, 1.79, 1.49, 1.38, 1.84, 1.06, 3.3, 5.97,
    1, 2.92, 1.64, 5.32, 3.26, 1.78, 2.24, 3.16, 1.6, 1.08, 1.55, 1.07,
    1.02, 1.23, 1.08, 5.22, 3.32, 24.86, 3.37, 5.16, 1.69, 2.31, 1.07, 1.1,
    1.01, 1.36, 1.38, 1.54, 5.34, 2.68, 5.78, 3.63, 1.89, 8.41, 4.06, 1.44,
    1.5, 3.17, 1.02, 1.8, 1.9, 1.86, 1.85, 1.73, 3.86, 3.11, 2.44, 1.15,
    2.03, 1.05, 3.05, 1.88, 10.13, 2.29, 1.41, 1, 5.46, 1.26, 23.33, 1.96,
    1.03, 4.54, 1.37, 3.5, 1.13, 1.16, 1.43, 1.13, 1.05, 33.27, 9.96, 1.79,
    2.07, 18.51, 5.75, 1.15, 1.08, 5.92, 1.38, 1.61, 12.99, 24.72, 4.86,
    1.11, 2.86, 1.54, 3.71, 4, 7.57, 2.03, 2.18, 5.52, 13.37, 3.73, 2.41,
    1.79, 5.57, 4.36, 12.33, 1.61, 3.28, 2.89, 1.47, 1.08, 26.89, 1.53,
    2.94, 5.29, 1.23, 1.57, 1.12, 5.69, 3.29, 2.72, 1.18, 5.03, 1.1, 1.32,
    1.18, 1.07, 1.27, 4.6, 11.68, 1.74, 3.94, 3.63, 1.05, 1.61, 1.62, 2.41,
    6.9, 2.02, 1.01, 3.22, 17.21, 1.95, 8.8, 1.44, 2.76, 3.1, 2.84, 1.35,
    1.84, 1.6, 10.72, 1.17, 3.47, 1.45, 1.29, 1.46, 2.23, 12.3, 3.27, 1.23,
    1.02, 1.66, 3.79, 2.06, 4.55, 7.95, 8.55, 4.08, 2.02, 1.21, 1.19, 1.53,
    4.9, 1.84, 10.51, 1.01, 1.34, 1.5, 1.4, 1.42, 4.18, 7.99, 1.23, 1.67,
    3.16, 1.64, 25.06, 4.52, 1.5, 3.23, 1.09, 1.45, 2.77, 7.42, 7.48, 1.89,
    2.11, 4.1, 1.26, 2.29, 10.12, 1.35, 13.21, 2.36, 22.35, 1.76, 2.22,
    1.04, 1.18, 3.69, 1.47, 10.2, 1.47, 1.68, 2.45, 1.03, 2.04, 1.47, 1.18,
    1.72, 1, 3.25, 1.1, 8.74, 1.01, 1.54, 1.34, 5.22, 5.31, 4.47, 2.78,
    21.37, 3.38, 1.63, 2.21, 2.35, 2.14, 1.46, 1.25, 1.67, 1.08, 3.94, 1.66,
    31.1, 1.73, 2.18, 2.06, 1.08, 1.11, 1, 1.07, 1.31, 1.55, 1.98, 1.75,
    1.23, 1.32, 2.56, 3.21, 1.81, 2.09, 1.34, 3.42, 1.29, 1.36, 1.76, 1.61,
    4.52, 1.08, 1.97, 3.75, 1.8, 6.36, 1.14, 1.72, 2.39, 1.28, 4.22, 2.12,
    1.28, 1.38, 1.42, 28.26, 2.15, 1.31, 1.65, 2.43, 2.76, 1.54, 1.61,
    11.91, 2.93, 8.1, 2.04, 1.84, 1.26, 3.69, 3.97, 3.01, 3.16, 1.3, 7.9,
    1.72, 5.57, 2.42, 1.74, 2.06, 2.86, 1.56, 1.4, 2.35, 2.82, 4.03, 1.28,
    2.21, 1.1, 2.06, 1.14, 1.58, 27.78, 2.04, 1.52, 1.22, 1.4, 1.29, 1.16,
    11.72, 1.33, 1.3, 4.34, 1.02, 1.63, 1.9, 9, 1.42, 3.13, 3.8, 1.02,
    1.25, 2.45, 1.74, 1.06, 1.38, 3.46, 1.08, 1, 1.02, 1.84, 1, 1.77, 3.07,
    5.26, 1.73, 1.07, 3.75, 2.32, 1.6, 1.22, 1.72, 2.01, 1.11, 2.03, 1.17,
    1.98, 2.18, 34.49, 1.2, 10.3, 3.4, 2.58, 2.2, 3.16, 29.22, 4.26, 3.18,
    3.29, 1.09, 2.3, 1.25, 3.05, 2.99, 2.16, 3.02, 2.21, 1.59, 5.74, 1.02,
    1.12, 1.21, 2.25, 4.38, 1.05, 1.05, 1.9, 23.03, 4.93, 1.03, 16.7, 4.08,
    1.68, 2.4, 2.89, 2.85, 2.75, 20.29, 3.57, 9.68, 1.46, 5.73, 4.84, 1.15,
    1.92, 3.71, 3.41, 22.67, 15.65, 1.86, 3.41, 1.89, 1.01, 3.02, 13.81,
    1.55, 1.16, 6.35, 5.6, 2.55, 16.8, 5.48, 1.49, 2.07, 1.05, 1.49, 6.29,
    1.32, 23.22, 1.07, 1.65, 20.07, 1.14, 1.1, 18.38, 4.34, 3.8, 6.17, 2.27,
    1.69, 1.07, 3.74, 1.6, 1.02, 1.45, 1.86, 5.13, 1.57, 6.93, 15.82, 1,
    1.16, 4.14, 1.08, 2.35, 2.15, 13.52, 10.87, 9.85, 1.97, 1, 3.46, 1.31,
    3.28, 2.74, 1.98, 2.22, 1, 9.95, 1.41, 1.43, 2.13, 4.6, 2.68, 4.13,
    1.61, 1.46, 1.23, 9.57, 1.14, 1.17, 14.27, 4.01, 5.55, 1.95, 2.48, 1.78,
    2.21, 1.65, 1.08, 2.63, 8.53, 2.2, 1.33, 21.72, 1.3, 1.43, 6.37, 1.09,
    3.94, 1.88, 3.38, 1.66, 1.41, 22.99, 1.55, 7.5, 25.48, 2.21, 3.62, 1.68,
    9.92, 3.4, 2.66, 1.03, 4.63, 1.89, 1.77, 1.9, 1.01, 1.81, 32.39, 2.1,
    1.23, 6.26, 9.06, 1.17, 2.41, 2.52, 1.63, 5.61, 1, 2.63, 1.88, 1.5,
    23.8, 5.65, 1.05, 1.07, 2.05, 1.7, 2.4, 18.27, 3.68, 13.17, 4.99, 20.81,
    1.51, 6.33, 9.85, 10.15, 17.05, 27.6, 4.65, 3.18, 2.54, 3.92, 4.74,
    1.81, 1.91, 4.42, 1.57, 2.17, 1.25, 1.03, 1.15, 1.19, 13.97, 2.39, 1.34,
    2.52, 1.47, 2.91, 2.31, 1.29, 1.61, 4.13, 1.83, 2.96, 1.08, 1.28, 13.53,
    1.15, 1.51, 1.31, 3.45, 9.32, 5.42, 3.27, 2.56, 2.07, 1.83, 14.1, 15.36,
    1.93, 1.47, 16.96, 1.61, 2.38, 2.66, 1.28, 1.46, 3.09, 6.73, 1.12, 1.85,
    3.21, 1.15, 3.71, 1.64, 4.88, 11.09, 3.82, 2.49, 21.23, 2.01, 2.47,
    2.47, 2.19, 2.14, 1, 2.09, 1.03, 5.22, 1.65, 1.13, 14.43, 1.68, 1.86,
    1.21, 1.14, 1.47, 1.26, 3.44, 23.9, 2.53, 2.72, 1, 1.13, 3.34, 1.43, 1,
    2.48, 2.01, 2.22, 6.43, 1.81, 2.12, 1.3, 4.02, 1.79, 3.9, 1.3, 5.04,
    1.77, 6.67, 2.21, 1.58, 5.38, 2.79, 6.12, 2.95, 1.14, 1.19, 1.19, 10.23,
    17.96, 10.1, 2.4, 9.29, 1.28, 4.07, 1.64, 2.1, 2.67, 1.08, 16.82, 2.83,
    24.42, 1.01, 3.24, 5.05, 3.24, 1.56, 2.32, 1.23, 1.72, 3.39, 1.96, 1.18,
    3.21, 23.95, 9.46, 23.12, 1.45, 3.22, 5, 2.04, 2.73, 6.28, 1.21, 14.3,
    1.48, 3.3, 3.73, 4.09, 2.88, 8.83, 1.15, 4.58, 4.23, 2.34, 2, 11.38,
    1.81, 1.03, 1.76, 2.41, 2.5, 5.82, 2.18, 10.19, 2.08, 18.19, 4.22, 7.78,
    1.96, 1.43, 1.08, 2.38, 1.37, 1.21, 4.48, 1.64, 1.62, 21.24, 1.22, 7.99,
    1.13, 1.29, 2.36, 3.94, 1.08, 1.41, 1.97, 1.41, 1.95, 1.28, 4.56, 3.35,
    1.37, 1.18, 1.03, 3.67, 1.43, 1.8, 2.48, 11.95, 1.5, 3.52, 2.03, 1,
    1.1, 10.13, 1.44, 14.19, 2.1, 8.46, 1.06, 1.66, 1.2, 7.22, 1.75, 1.78,
    3.76, 2.21, 1, 25.19, 5.96, 5.42, 2.67, 1.37, 1.39, 15.95, 2.8, 1.76,
    1.7, 2.81, 8.87, 1.48, 1.03, 1.14, 1.05, 10.29, 1.71, 23.98, 2.34, 1.97,
    1.33, 24.02, 2.01, 13.74, 2.5, 1.33, 1.02, 1.76, 1.37, 8.97, 1.27, 1.38,
    4.47, 1.38, 3.02, 17, 13.35, 1.07, 1.38, 5.74, 6.68, 24.72, 1.47, 1.25,
    4.51, 4.47, 1.99, 1.15, 4.03, 1.17, 3.42, 6.46, 1.31, 1.46, 6.67, 3.79,
    1.56, 3.98, 1.62, 2.13, 1.07, 4.88, 1.62, 1.5, 6.11, 1.31, 1.85, 1.93,
    1.09, 1.49, 1.41, 1.24, 1.05, 6.99, 1.33, 1.73, 10.76, 21.77, 1.18,
    1.06, 5.36, 1.45, 1.16, 6.43, 2.1, 4.15, 1.14, 2.21, 33.48, 2.88, 1,
    4.7, 1.27, 5.75, 4.97, 1.11, 3.51, 21.47, 1.21, 1.98, 1.11, 1.46, 1.77,
    1.22, 2.65, 1.66, 5.29, 1.58, 2.03, 5.86, 1.1, 1.68, 1.35, 1.72, 1.15,
    2.69, 2.81, 3.46, 1.58, 1.07, 7.18, 2.35, 6.05, 1.24, 5.69, 5.46, 1,
    3.04, 4.76, 1.56, 1.41, 2.43, 7.97, 1.22, 1.94, 1.51, 21.71, 3.03, 1.43,
    5.07, 1.87, 1.12, 1, 1.32, 1, 1.08, 1.1, 1.04, 1, 1.09, 1.97, 2.97,
    1.21, 1.61, 5.94, 2.55, 4.48, 1.14, 2.73, 1.34, 1.33, 1.29, 1.25, 5.44,
    1.77, 2.18, 2.52, 1.28, 22.25, 1.04, 3.57, 6.53, 1.34, 5.75, 1.61, 3.89,
    1.07, 2.13, 5.05, 1.53, 3.53, 8.31, 2.15, 1.39, 1.23, 1.68, 17.14, 1.23,
    2.38, 1, 2.02, 19.48, 1.22, 1.42, 6.26, 16.11, 2.05, 3.51, 3.53, 1.83,
    6.86, 1.24, 27.78, 2.33, 3.43, 2.92, 1.26, 15.11, 24.58, 1.12, 2.46,
    5.61, 9.79, 2.33, 1.34, 7.86, 1.1, 2.61, 2.34, 4.5, 1.79, 1.75, 18,
    8.66, 1.92, 11.5, 1.35, 2.53, 1.79, 1.14, 1.58, 1.84, 1.35, 6.44, 4.49,
    3.02, 3.16, 1.12, 1.42, 9.14, 1.26, 1.19, 2.47, 1.2, 3.88, 1.03, 1.85,
    1.07, 1.03, 1.13, 4.87, 1.03, 1.8, 1.29, 6.11, 1.73, 30.16, 2.99, 2.34,
    1.56, 4.33, 1.23, 7.39, 1.57, 3.16, 2.73, 1.46, 1.01, 8.24, 1.61, 2.28,
    1.91, 1.49, 5.12, 3.53, 20.05, 3.26, 2.25, 6.61, 1.35, 4.32, 1, 2.13,
    1.83, 1.26, 2.27, 1.21, 1.64, 1.77, 1.06, 1.05, 1.98, 3.1, 3.74, 22.09,
    2.17, 2.97, 1.26, 1.83, 4.44, 1.08, 2.22, 1.24, 1.7, 20.14, 16.56, 1.72,
    1.37, 1.06, 1.65, 2.42, 3.84, 1, 1.56, 1.93, 1.03, 1.47, 1.76, 12.64,
    1.12, 1.32, 1.89, 1.64, 1.2, 3.15, 1.88, 1.12, 1.01, 1.45, 1.71, 1.65,
    1.65, 5.16, 1.48, 1.73
]

# ============================================================
# دوال التحليل الأساسية
# ============================================================

@st.cache_data
def find_threshold(data):
    """إيجاد العتبة الطبيعية الفاصلة"""
    sorted_d = sorted(data)
    gaps = []
    for i in range(len(sorted_d) - 1):
        if 4.0 <= sorted_d[i] <= 15.0:
            gap = sorted_d[i+1] - sorted_d[i]
            gaps.append((gap, sorted_d[i], sorted_d[i+1]))
    if not gaps:
        return 6.0
    gaps.sort(reverse=True)
    return round((gaps[0][1] + gaps[0][2]) / 2, 2)


@st.cache_data
def build_cycles(data, threshold):
    """بناء الدورات من البيانات"""
    cycles = []
    acc    = 0.0
    vals   = []

    for i, val in enumerate(data):
        if val >= threshold:
            cycles.append({
                'end_idx'   : i,
                'jump'      : val,
                'acc_before': round(acc, 4),
                'n_charges' : len(vals),
                'charges'   : vals.copy()
            })
            acc  = 0.0
            vals = []
        else:
            acc = round(acc + val, 4)
            vals.append(val)

    remaining = {
        'current_acc' : round(acc, 4),
        'current_vals': vals,
        'n_values'    : len(vals)
    }
    return cycles, remaining


@st.cache_data
def compute_model(cycles):
    """حساب معاملات النموذج"""
    valid = [(c['acc_before'], c['jump'])
             for c in cycles if c['acc_before'] > 2.0]
    if len(valid) < 5:
        return None

    x = np.array([v[0] for v in valid])
    y = np.array([v[1] for v in valid])

    # Least Squares
    denom  = np.mean(x**2) - np.mean(x)**2
    slope  = (np.mean(x*y) - np.mean(x)*np.mean(y)) / (denom + 1e-10)
    inter  = np.mean(y) - slope * np.mean(x)

    # R²
    y_pred = slope * x + inter
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # وسيط النسب
    slope_med = float(np.median(y / x))

    # معامل الدورات الأخيرة (آخر 10)
    recent = valid[-10:] if len(valid) >= 10 else valid
    xr = np.array([v[0] for v in recent])
    yr = np.array([v[1] for v in recent])
    dr = np.mean(xr**2) - np.mean(xr)**2
    slope_r = (np.mean(xr*yr) - np.mean(xr)*np.mean(yr)) / (dr + 1e-10)
    inter_r = np.mean(yr) - slope_r * np.mean(xr)

    return {
        'slope'       : round(float(slope),     4),
        'intercept'   : round(float(inter),     4),
        'slope_med'   : round(slope_med,         4),
        'slope_recent': round(float(slope_r),    4),
        'inter_recent': round(float(inter_r),    4),
        'r2'          : round(r2,                4),
        'x'           : x,
        'y'           : y,
        'n_points'    : len(valid)
    }


def predict(acc, model, ceiling, recent_weight=0.6):
    """
    التوقع مع نطاق الثقة
    يجمع بين المعامل التاريخي والأخير
    """
    s_hist = model['slope']
    i_hist = model['intercept']
    s_rec  = model['slope_recent']
    i_rec  = model['inter_recent']
    r2     = model['r2']

    p_hist = s_hist * acc + i_hist
    p_rec  = s_rec  * acc + i_rec
    p_final = (1 - recent_weight) * p_hist + recent_weight * p_rec

    # نطاق الثقة بناءً على R² والانحراف التاريخي
    y_all  = model['y']
    resid_std = float(np.std(y_all - (s_hist * model['x'] + i_hist)))
    margin = resid_std * (1.0 + (1 - r2))

    lo = max(1.0,    p_final - margin)
    hi = min(ceiling, p_final + margin)

    # مستوى الثقة
    x_all = model['x']
    z = abs(acc - float(np.mean(x_all))) / (float(np.std(x_all)) + 1e-10)
    conf = r2 * 100 - min(15, z * 4) - min(10, abs(s_hist - s_rec) * 15)
    conf = max(10.0, min(92.0, conf))

    return {
        'point'  : round(p_final, 2),
        'lo'     : round(lo,      2),
        'hi'     : round(hi,      2),
        'conf'   : round(conf,    1),
        'p_hist' : round(p_hist,  2),
        'p_rec'  : round(p_rec,   2),
        'margin' : round(margin,  2)
    }


def backtest(cycles, model, n=20):
    """اختبار خلفي على آخر N دورة"""
    if len(cycles) < n + 5:
        n = max(5, len(cycles) // 4)

    train  = cycles[:-n]
    test   = cycles[-n:]

    valid  = [(c['acc_before'], c['jump'])
              for c in train if c['acc_before'] > 2.0]
    if len(valid) < 5:
        return None, n

    x = np.array([v[0] for v in valid])
    y = np.array([v[1] for v in valid])
    d = np.mean(x**2) - np.mean(x)**2
    s = (np.mean(x*y) - np.mean(x)*np.mean(y)) / (d + 1e-10)
    b = np.mean(y) - s * np.mean(x)

    rows = []
    for c in test:
        acc  = c['acc_before']
        real = c['jump']
        pred = s * acc + b
        err  = abs(pred - real)
        pct  = 100 * err / real if real > 0 else 0
        rows.append({
            'العداد قبل القفزة': round(acc,  2),
            'القفزة الفعلية'   : round(real, 2),
            'التوقع'           : round(pred, 2),
            'الخطأ المطلق'     : round(err,  2),
            'الخطأ %'          : round(pct,  1),
            'الحكم'            : '✅' if pct < 30 else ('⚠️' if pct < 55 else '❌')
        })

    df  = pd.DataFrame(rows)
    acc = (df['الخطأ %'] < 30).sum() / len(df) * 100
    return df, acc


# ============================================================
# الشريط الجانبي
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ إعدادات النظام")
    st.markdown("---")

    data_source = st.radio(
        "مصدر البيانات",
        ["البيانات الافتراضية", "إدخال يدوي"],
        index=0
    )

    if data_source == "إدخال يدوي":
        raw_input = st.text_area(
            "أدخل القيم (مفصولة بفاصلة أو سطر جديد)",
            height=200,
            placeholder="1.5, 2.3, 8.7, 1.1, ..."
        )
        try:
            import re
            nums = re.findall(r"[\d.]+", raw_input)
            user_data = [float(n) for n in nums if float(n) > 0]
            if len(user_data) < 20:
                st.warning("يحتاج النموذج 20 قيمة على الأقل")
                user_data = DEFAULT_DATA
        except Exception:
            user_data = DEFAULT_DATA
    else:
        user_data = DEFAULT_DATA

    st.markdown("---")
    st.markdown("### 🎛️ ضبط النموذج")

    auto_thresh = st.checkbox("عتبة تلقائية", value=True)
    if not auto_thresh:
        manual_thresh = st.slider(
            "العتبة الفاصلة", 3.0, 15.0, 6.0, 0.5
        )

    recent_w = st.slider(
        "وزن الدورات الأخيرة", 0.0, 1.0, 0.6, 0.05,
        help="0 = تاريخي كامل | 1 = أخير 10 فقط"
    )

    n_backtest = st.slider("دورات الاختبار الخلفي", 10, 40, 20, 5)

    st.markdown("---")
    st.markdown("### 📌 معلومات")
    st.info(
        "النموذج يعتمد على:\n"
        "- العداد التراكمي\n"
        "- انحدار خطي تكيفي\n"
        "- نافذة متحركة (آخر 10)\n"
        "- اختبار خلفي تلقائي"
    )

# ============================================================
# تشغيل التحليل
# ============================================================
data      = user_data
threshold = (find_threshold(data) if auto_thresh else manual_thresh)
cycles, remaining = build_cycles(data, threshold)
model     = compute_model(cycles)

# ============================================================
# العنوان الرئيسي
# ============================================================
st.markdown(
    "<h1 style='text-align:center; color:#00d4ff;'>"
    "🎯 نظام التنبؤ الذكي بالأنماط</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#87ceeb; font-size:1.1em;'>"
    "تحليل متقدم للأنماط الخفية مع نظام توصيات ذكي</p>",
    unsafe_allow_html=True
)
st.markdown("---")

if model is None:
    st.error("البيانات غير كافية لبناء النموذج. أضف المزيد من القيم.")
    st.stop()

# ============================================================
# بطاقات الإحصاءات العليا
# ============================================================
all_jumps = [c['jump'] for c in cycles]
acc_now   = remaining['current_acc']
ceiling   = max(all_jumps)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(data)}</div>
        <div class="metric-label">إجمالي القيم</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(cycles)}</div>
        <div class="metric-label">دورات مكتملة</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{threshold}</div>
        <div class="metric-label">العتبة الفاصلة</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{acc_now:.2f}</div>
        <div class="metric-label">العداد الحالي</div>
    </div>""", unsafe_allow_html=True)

with col5:
    r2_pct = model['r2'] * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{r2_pct:.1f}%</div>
        <div class="metric-label">جودة النموذج R²</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# التوقع الرئيسي
# ============================================================
pred_result = predict(acc_now, model, ceiling, recent_w)
p      = pred_result['point']
lo     = pred_result['lo']
hi     = pred_result['hi']
conf   = pred_result['conf']
margin = pred_result['margin']

# هل يتجاوز السقف؟
split_mode = p > ceiling
if split_mode:
    s  = model['slope_recent']
    b  = model['inter_recent']
    r_acc  = max(0.0, acc_now - (ceiling - b) / (s + 1e-10))
    p2_res = predict(r_acc, model, ceiling, recent_w)
    p2     = p2_res['point']

st.markdown("## 🔮 التوقع القادم")

pred_col, conf_col = st.columns([2, 1])

with pred_col:
    if split_mode:
        st.markdown(f"""
        <div class="prediction-box">
            <div style="color:#ffa500; font-size:1em; margin-bottom:8px;">
                ⚡ وضع التقسيم - العداد يتجاوز السقف
            </div>
            <div style="display:flex; justify-content:space-around;">
                <div>
                    <div style="color:#aaa; font-size:0.9em;">القفزة الأولى</div>
                    <div class="pred-value">{ceiling:.1f}</div>
                    <div style="color:#87ceeb;">عند السقف</div>
                </div>
                <div style="color:#555; font-size:2em; padding-top:10px;">→</div>
                <div>
                    <div style="color:#aaa; font-size:0.9em;">القفزة الثانية</div>
                    <div class="pred-value">{p2:.1f}</div>
                    <div style="color:#87ceeb;">العداد المتبقي: {r_acc:.2f}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box">
            <div style="color:#aaa; font-size:0.9em; margin-bottom:5px;">
                القفزة المتوقعة التالية
            </div>
            <div class="pred-value">{p:.2f}</div>
            <div class="pred-range">
                نطاق الثقة: [{lo:.2f} &nbsp;—&nbsp; {hi:.2f}]
            </div>
            <div style="color:#87ceeb; margin-top:8px; font-size:0.9em;">
                بالمعامل التاريخي: {pred_result['p_hist']:.2f} &nbsp;|&nbsp;
                بالمعامل الأخير: {pred_result['p_rec']:.2f}
            </div>
        </div>""", unsafe_allow_html=True)

with conf_col:
    conf_color = (
        "#00ff88" if conf >= 65 else
        "#ffa500" if conf >= 45 else
        "#ff4444"
    )
    conf_label = (
        "ثقة عالية"    if conf >= 65 else
        "ثقة متوسطة"  if conf >= 45 else
        "ثقة منخفضة"
    )
    # شريط الثقة بـ Plotly
    fig_conf = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = conf,
        title = {'text': "مستوى الثقة %",
                 'font': {'color': 'white', 'size': 14}},
        number= {'suffix': "%", 'font': {'color': conf_color, 'size': 28}},
        gauge = {
            'axis' : {'range': [0, 100],
                      'tickcolor': 'gray'},
            'bar'  : {'color': conf_color},
            'steps': [
                {'range': [0,  45], 'color': 'rgba(255,68,68,0.15)'},
                {'range': [45, 65], 'color': 'rgba(255,165,0,0.15)'},
                {'range': [65,100], 'color': 'rgba(0,255,136,0.15)'},
            ],
            'threshold': {
                'line' : {'color': 'white', 'width': 2},
                'value': conf
            },
            'bgcolor': 'rgba(0,0,0,0)'
        }
    ))
    fig_conf.update_layout(
        height=220,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_conf, use_container_width=True)
    st.markdown(
        f"<div style='text-align:center; color:{conf_color}; "
        f"font-weight:bold;'>{conf_label}</div>",
        unsafe_allow_html=True
    )

# ============================================================
# نظام التوصيات
# ============================================================
st.markdown("---")
st.markdown("## 📋 نظام التوصيات")

rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    st.markdown("### 🚦 التوصية الرئيسية")

    # تحديد قوة الإشارة
    pct_of_ceiling = p / ceiling * 100 if not split_mode else 100

    if split_mode:
        rec_class = "rec-strong"
        rec_icon  = "⚡"
        rec_title = "إشارة قوية جداً - وضع تقسيم"
        rec_body  = (
            f"العداد ({acc_now:.2f}) يتجاوز حد التقسيم. "
            f"توقع قفزة أولى عند السقف ({ceiling:.1f}) "
            f"تليها قفزة ثانية (~{p2:.1f})."
        )
    elif pct_of_ceiling >= 75:
        rec_class = "rec-strong"
        rec_icon  = "🟢"
        rec_title = "إشارة قوية - قفزة كبيرة محتملة"
        rec_body  = (
            f"العداد ({acc_now:.2f}) في مرحلة متقدمة. "
            f"التوقع: قفزة بين {lo:.1f} و {hi:.1f}."
        )
    elif pct_of_ceiling >= 45:
        rec_class = "rec-moderate"
        rec_icon  = "🟡"
        rec_title = "إشارة متوسطة - قفزة معتدلة"
        rec_body  = (
            f"العداد ({acc_now:.2f}) في منطقة وسطية. "
            f"التوقع: {lo:.1f} - {hi:.1f}. "
            f"انتظر تأكيداً إضافياً."
        )
    else:
        rec_class = "rec-weak"
        rec_icon  = "🔴"
        rec_title = "إشارة ضعيفة - لا تزال في مرحلة الشحن"
        rec_body  = (
            f"العداد ({acc_now:.2f}) منخفض نسبياً. "
            f"النظام لا يزال يتراكم. "
            f"توقع قفزة صغيرة ({lo:.1f} - {hi:.1f}) أو استمرار الشحن."
        )

    st.markdown(f"""
    <div class="rec-box {rec_class}">
        <strong>{rec_icon} {rec_title}</strong><br>
        <span style="font-size:0.95em;">{rec_body}</span>
    </div>""", unsafe_allow_html=True)

    # تفاصيل إضافية
    st.markdown("### 📐 تفاصيل النموذج")
    details = {
        "المعامل (الوزن) التاريخي": f"{model['slope']:.4f}",
        "المعامل الأخير (10 دورات)": f"{model['slope_recent']:.4f}",
        "القاطع": f"{model['intercept']:.4f}",
        "جودة النموذج R²": f"{model['r2']:.4f}",
        "السقف المرصود": f"{ceiling:.2f}",
        "نقاط التدريب": f"{model['n_points']}",
        "هامش الخطأ المتوقع": f"±{margin:.2f}",
    }
    for k, v in details.items():
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; "
            f"padding:4px 0; border-bottom:1px solid #1a3a5c;'>"
            f"<span style='color:#87ceeb;'>{k}</span>"
            f"<span style='color:#00d4ff; font-weight:bold;'>{v}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

with rec_col2:
    st.markdown("### 🎰 السيناريوهات المحتملة")

    # بناء سيناريوهات بناءً على البيانات الفعلية
    s     = model['slope_recent']
    b_int = model['inter_recent']
    r2    = model['r2']
    std_j = float(np.std(all_jumps))

    scenarios = []

    # سيناريو متفائل (عداد أعلى بانحراف واحد)
    acc_opt  = acc_now + float(np.std(model['x'])) * 0.3
    p_opt    = min(ceiling, s * acc_opt + b_int)
    scenarios.append(("متفائل  🟢", p_opt,
                      p_opt - margin * 0.5,
                      min(ceiling, p_opt + margin * 0.5),
                      35))

    # سيناريو أساسي
    scenarios.append(("أساسي   🟡", p,
                      lo, hi, 50))

    # سيناريو متشائم (عداد أقل)
    acc_pes = max(0, acc_now - float(np.std(model['x'])) * 0.3)
    p_pes   = max(1.0, s * acc_pes + b_int)
    scenarios.append(("متشائم  🔴", p_pes,
                      max(1, p_pes - margin * 0.5),
                      p_pes + margin * 0.5,
                      15))

    for name, point, slo, shi, prob in scenarios:
        shi = min(ceiling, shi)
        slo = max(1.0,     slo)
        color = (
            "#00ff88" if "متفائل" in name else
            "#ffa500" if "أساسي"  in name else
            "#ff4444"
        )
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);
                    border-radius:10px; padding:12px;
                    margin:8px 0; border-left:3px solid {color};">
            <div style="display:flex; justify-content:space-between;
                        align-items:center;">
                <span style="color:{color}; font-weight:bold;">{name}</span>
                <span style="color:{color}; font-size:1.4em;
                             font-weight:bold;">{point:.1f}</span>
            </div>
            <div style="color:#87ceeb; font-size:0.85em; margin-top:4px;">
                النطاق: [{slo:.1f} — {shi:.1f}]
                &nbsp;|&nbsp; الاحتمال: ~{prob}%
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>تحذير:</strong> هذا نموذج إحصائي.
        الثقة مبنية على R² الفعلي من البيانات.
        لا توجد ضمانات مطلقة في أي نظام تنبؤ.
    </div>""", unsafe_allow_html=True)

# ============================================================
# الرسوم البيانية
# ============================================================
st.markdown("---")
st.markdown("## 📊 التحليل البياني")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 السلسلة الزمنية",
    "🔗 العداد مقابل القفزة",
    "📊 توزيع القفزات",
    "🧪 الاختبار الخلفي"
])

# --- التبويب 1: السلسلة الزمنية ---
with tab1:
    n_show = st.slider("عدد القيم المعروضة", 50, len(data), 200, 50)
    data_show = data[-n_show:]
    idx_show  = list(range(len(data) - n_show, len(data)))

    fig1 = go.Figure()

    # القيم الكاملة
    fig1.add_trace(go.Scatter(
        x=idx_show, y=data_show,
        mode='lines',
        name='القيم',
        line=dict(color='#4a9eff', width=1),
        opacity=0.7
    ))

    # تمييز القفزات
    jump_x = [i for i in idx_show if data[i] >= threshold]
    jump_y = [data[i] for i in jump_x]
    fig1.add_trace(go.Scatter(
        x=jump_x, y=jump_y,
        mode='markers',
        name='قفزات',
        marker=dict(color='#ff6b6b', size=8, symbol='circle')
    ))

    # خط العتبة
    fig1.add_hline(
        y=threshold, line_dash='dash',
        line_color='yellow', opacity=0.5,
        annotation_text=f"العتبة: {threshold}"
    )

    # خط السقف
    fig1.add_hline(
        y=ceiling, line_dash='dot',
        line_color='#ff4444', opacity=0.7,
        annotation_text=f"السقف: {ceiling:.2f}"
    )

    fig1.update_layout(
        title='السلسلة الزمنية مع تمييز القفزات',
        xaxis_title='الموضع',
        yaxis_title='القيمة',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,25,45,0.8)',
        font_color='white',
        legend=dict(bgcolor='rgba(0,0,0,0.3)'),
        height=420
    )
    st.plotly_chart(fig1, use_container_width=True)

    # العداد التراكمي
    acc_trace = []
    acc_running = 0.0
    for v in data[-n_show:]:
        if v >= threshold:
            acc_trace.append(acc_running)
            acc_running = 0.0
        else:
            acc_running += v
            acc_trace.append(acc_running)

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=idx_show, y=acc_trace,
        mode='lines',
        fill='tozeroy',
        name='العداد',
        line=dict(color='#00d4ff', width=1.5),
        fillcolor='rgba(0,212,255,0.1)'
    ))
    fig_acc.update_layout(
        title='العداد التراكمي عبر الزمن',
        xaxis_title='الموضع',
        yaxis_title='قيمة العداد',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,25,45,0.8)',
        font_color='white',
        height=280
    )
    st.plotly_chart(fig_acc, use_container_width=True)

# --- التبويب 2: العداد مقابل القفزة ---
with tab2:
    x_pts = model['x']
    y_pts = model['y']

    s_ls = model['slope']
    b_ls = model['intercept']
    x_line = np.linspace(float(x_pts.min()), float(x_pts.max()), 100)
    y_line = s_ls * x_line + b_ls

    fig2 = go.Figure()

    # نقاط البيانات
    fig2.add_trace(go.Scatter(
        x=x_pts, y=y_pts,
        mode='markers',
        name='دورات فعلية',
        marker=dict(
            color=y_pts,
            colorscale='Viridis',
            size=8,
            showscale=True,
            colorbar=dict(title='قيمة القفزة')
        )
    ))

    # خط الانحدار
    fig2.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode='lines',
        name=f'y = {s_ls:.3f}x + {b_ls:.3f}  (R²={model["r2"]:.3f})',
        line=dict(color='#ff6b6b', width=2, dash='dash')
    ))

    # نقطة التوقع الحالية
    fig2.add_trace(go.Scatter(
        x=[acc_now], y=[p],
        mode='markers',
        name=f'التوقع الحالي ({p:.2f})',
        marker=dict(color='#00ff88', size=14, symbol='star')
    ))

    # نطاق الثقة
    y_lo = s_ls * x_line + b_ls - margin
    y_hi = s_ls * x_line + b_ls + margin
    fig2.add_trace(go.Scatter(
        x=np.concatenate([x_line, x_line[::-1]]),
        y=np.concatenate([y_hi, y_lo[::-1]]),
        fill='toself',
        fillcolor='rgba(255,107,107,0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='نطاق الثقة'
    ))

    fig2.update_layout(
        title='العلاقة بين العداد والقفزة (اكتشاف المعامل الخفي)',
        xaxis_title='العداد قبل القفزة',
        yaxis_title='قيمة القفزة',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,25,45,0.8)',
        font_color='white',
        legend=dict(bgcolor='rgba(0,0,0,0.3)'),
        height=480
    )
    st.plotly_chart(fig2, use_container_width=True)

    # إحصاءات الارتباط
    corr = float(np.corrcoef(x_pts, y_pts)[0, 1])
    c1, c2, c3 = st.columns(3)
    c1.metric("معامل الارتباط r", f"{corr:.4f}")
    c2.metric("R²", f"{model['r2']:.4f}")
    c3.metric("عدد النقاط", model['n_points'])

# --- التبويب 3: توزيع القفزات ---
with tab3:
    fig3 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'توزيع قيم القفزات',
            'توزيع أطوال الدورات',
            'القفزات عبر الزمن (آخر 50)',
            'العداد مقابل التوقع'
        ]
    )

    # هيستوغرام القفزات
    fig3.add_trace(
        go.Histogram(
            x=all_jumps, nbinsx=20,
            name='القفزات',
            marker_color='#4a9eff',
            opacity=0.8
        ),
        row=1, col=1
    )

    # هيستوغرام أطوال الدورات
    lengths = [c['n_charges'] for c in cycles]
    fig3.add_trace(
        go.Histogram(
            x=lengths, nbinsx=15,
            name='الأطوال',
            marker_color='#ff6b6b',
            opacity=0.8
        ),
        row=1, col=2
    )

    # آخر 50 قفزة
    last50_j = all_jumps[-50:]
    last50_i = list(range(len(all_jumps) - 50, len(all_jumps)))
    fig3.add_trace(
        go.Bar(
            x=last50_i, y=last50_j,
            name='آخر قفزة',
            marker_color=[
                '#00ff88' if j >= ceiling * 0.75 else
                '#ffa500' if j >= ceiling * 0.40 else '#4a9eff'
                for j in last50_j
            ]
        ),
        row=2, col=1
    )

    # العداد قبل القفزة مقابل التوقع
    acc_vals  = [c['acc_before'] for c in cycles if c['acc_before'] > 2]
    pred_vals = [model['slope'] * a + model['intercept'] for a in acc_vals]
    real_vals = [c['jump'] for c in cycles if c['acc_before'] > 2]

    fig3.add_trace(
        go.Scatter(
            x=acc_vals, y=real_vals,
            mode='markers', name='فعلي',
            marker=dict(color='#4a9eff', size=6)
        ),
        row=2, col=2
    )
    fig3.add_trace(
        go.Scatter(
            x=acc_vals, y=pred_vals,
            mode='markers', name='متوقع',
            marker=dict(color='#ff6b6b', size=6, symbol='x')
        ),
        row=2, col=2
    )

    fig3.update_layout(
        height=620,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,25,45,0.8)',
        font_color='white',
        showlegend=False
    )
    st.plotly_chart(fig3, use_container_width=True)

    # إحصاءات القفزات
    st.markdown("### 📋 إحصاءات شاملة")
    stats_col1, stats_col2 = st.columns(2)

    with stats_col1:
        st.markdown("**إحصاءات القفزات**")
        jstats = {
            "العدد"               : len(all_jumps),
            "المتوسط"             : f"{np.mean(all_jumps):.3f}",
            "الوسيط"              : f"{np.median(all_jumps):.3f}",
            "الانحراف المعياري"   : f"{np.std(all_jumps):.3f}",
            "الحد الأدنى"         : f"{min(all_jumps):.2f}",
            "الحد الأقصى (السقف)": f"{max(all_jumps):.2f}",
        }
        for k, v in jstats.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:3px 0;border-bottom:1px solid #1a3a5c;'>"
                f"<span style='color:#87ceeb;'>{k}</span>"
                f"<span style='color:#00d4ff;font-weight:bold;'>{v}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    with stats_col2:
        st.markdown("**إحصاءات الدورات**")
        cstats = {
            "عدد الدورات"         : len(cycles),
            "متوسط الطول"         : f"{np.mean(lengths):.1f}",
            "وسيط الطول"          : f"{np.median(lengths):.1f}",
            "أقصر دورة"           : min(lengths),
            "أطول دورة"           : max(lengths),
            "العداد الحالي"       : f"{acc_now:.4f}",
        }
        for k, v in cstats.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:3px 0;border-bottom:1px solid #1a3a5c;'>"
                f"<span style='color:#87ceeb;'>{k}</span>"
                f"<span style='color:#00d4ff;font-weight:bold;'>{v}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

# --- التبويب 4: الاختبار الخلفي ---
with tab4:
    bt_df, bt_acc = backtest(cycles, model, n=n_backtest)

    if bt_df is not None:
        # مقياس الدقة
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d2137,#1a4a6b);
                    border-radius:12px; padding:20px; text-align:center;
                    margin-bottom:20px;">
            <div style="color:#aaa; font-size:0.9em;">
                دقة النموذج (خطأ &lt; 30%)
            </div>
            <div style="font-size:2.5em; font-weight:bold;
                        color:{'#00ff88' if bt_acc>=65 else
                               '#ffa500' if bt_acc>=45 else '#ff4444'};">
                {bt_acc:.1f}%
            </div>
            <div style="color:#87ceeb; font-size:0.85em;">
                على آخر {n_backtest} دورة
            </div>
        </div>""", unsafe_allow_html=True)

        # رسم الفعلي مقابل المتوقع
        fig_bt = go.Figure()
        x_bt = list(range(len(bt_df)))

        fig_bt.add_trace(go.Bar(
            x=x_bt, y=bt_df['القفزة الفعلية'],
            name='فعلي', marker_color='#4a9eff', opacity=0.8
        ))
        fig_bt.add_trace(go.Scatter(
            x=x_bt, y=bt_df['التوقع'],
            mode='lines+markers', name='متوقع',
            line=dict(color='#ff6b6b', width=2),
            marker=dict(size=8)
        ))
        fig_bt.update_layout(
            title='الفعلي مقابل المتوقع (آخر دورات)',
            xaxis_title='رقم الدورة',
            yaxis_title='القيمة',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,25,45,0.8)',
            font_color='white',
            legend=dict(bgcolor='rgba(0,0,0,0.3)'),
            height=350
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # رسم توزيع الخطأ %
        fig_err = go.Figure()
        err_colors = [
            '#00ff88' if e < 30 else
            '#ffa500' if e < 55 else '#ff4444'
            for e in bt_df['الخطأ %']
        ]
        fig_err.add_trace(go.Bar(
            x=x_bt, y=bt_df['الخطأ %'],
            marker_color=err_colors,
            name='الخطأ %'
        ))
        fig_err.add_hline(
            y=30, line_dash='dash',
            line_color='#ffa500',
            annotation_text="حد القبول 30%"
        )
        fig_err.update_layout(
            title='توزيع نسبة الخطأ لكل دورة',
            xaxis_title='رقم الدورة',
            yaxis_title='الخطأ %',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,25,45,0.8)',
            font_color='white',
            height=280
        )
        st.plotly_chart(fig_err, use_container_width=True)

        # جدول التفاصيل
        with st.expander("📋 جدول تفاصيل الاختبار الخلفي"):
            st.dataframe(
                bt_df.style.applymap(
                    lambda v: 'color: #00ff88' if v == '✅' else
                              'color: #ffa500' if v == '⚠️' else
                              'color: #ff4444',
                    subset=['الحكم']
                ),
                use_container_width=True,
                height=400
            )
    else:
        st.warning("بيانات غير كافية للاختبار الخلفي")

# ============================================================
# الفوتر
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a6a8a; font-size:0.85em;">
    نظام التنبؤ الذكي | يعتمد على تحليل الأنماط الإحصائية
    <br>
    ⚠️ للأغراض التحليلية فقط — ليس ضماناً للنتائج المستقبلية
</div>
""", unsafe_allow_html=True)
