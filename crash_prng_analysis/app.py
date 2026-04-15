# app_academic.py
"""
تطبيق أكاديمي متكامل لتحليل عشوائية Crash
مشروع تخرج - تحليل PRNG
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# استيراد الوحدات السابقة
from randomness_tests import RandomnessTestSuite
from prng_analysis import PRNGAnalyzer

st.set_page_config(
    page_title="🎓 تحليل PRNG - مشروع تخرج",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 تحليل عشوائية لعبة Crash — مشروع تخرج أكاديمي")
st.caption(
    "تطبيق اختبارات NIST SP 800-22 لتحليل مولدات الأرقام الزائفة"
)

# ============================ إدخال البيانات ============================
st.header("📥 إدخال البيانات")

col1, col2 = st.columns([2, 1])

with col1:
    input_method = st.radio(
        "طريقة الإدخال:",
        ["إدخال يدوي", "رفع ملف CSV", "استخدام بيانات نموذجية"]
    )

with col2:
    st.info(
        "📌 لجمع البيانات:\n"
        "1. العب أو راقب اللعبة\n"
        "2. سجّل نتيجة كل جولة\n"
        "3. أدخلها هنا للتحليل",
        icon="ℹ️"
    )

raw_data = None

if input_method == "إدخال يدوي":
    user_text = st.text_area(
        "أدخل القيم مفصولة بمسافات أو أسطر:",
        height=150,
        placeholder="مثال: 1.23 4.56 2.10 1.05 8.92..."
    )
    if user_text.strip():
        try:
            raw_data = [
                float(x) for x in 
                user_text.replace('\n', ' ').split() 
                if x.strip()
            ]
            st.success(f"✅ تم تحميل {len(raw_data)} قيمة")
        except:
            st.error("❌ تنسيق خاطئ")

elif input_method == "رفع ملف CSV":
    uploaded = st.file_uploader(
        "ارفع ملف CSV (عمود 'crash_point')", 
        type=['csv']
    )
    if uploaded:
        df_up = pd.read_csv(uploaded)
        if 'crash_point' in df_up.columns:
            raw_data = df_up['crash_point'].dropna().tolist()
            st.success(f"✅ تم تحميل {len(raw_data)} قيمة")
        else:
            st.error("❌ لم يُوجد عمود 'crash_point'")

else:  # بيانات نموذجية
    raw_data = [
        8.72, 6.75, 1.86, 2.18, 1.25, 2.28, 1.24, 1.2, 1.54,
        24.46, 4.16, 1.49, 1.09, 1.47, 1.54, 1.53, 2.1, 32.04,
        11.0, 1.17, 1.7, 2.61, 1.26, 22.23, 1.77, 1.93, 3.35,
        7.01, 1.83, 9.39, 3.31, 2.04, 1.3, 6.65, 1.16, 3.39,
        1.95, 10.85, 1.65, 1.22, 1.6, 4.67, 1.85, 2.72, 1.0,
        3.02, 1.35, 1.3, 1.37, 17.54, 1.18, 1.0, 14.4, 1.11,
        6.15, 2.39, 2.22, 1.42, 1.23, 2.42, 1.07, 1.24, 2.55,
        7.26, 1.69, 5.1, 2.59, 5.51, 2.31, 2.12, 1.97, 1.5,
        3.01, 2.29, 1.36, 4.95, 5.09, 8.5, 1.77, 5.52, 3.93,
        1.5, 2.28, 2.49, 18.25, 1.68, 1.42, 2.12, 4.17, 1.04,
        2.35, 1.0, 1.01, 5.46, 1.13, 2.84, 3.39, 2.79, 1.59,
        1.53, 4.34, 2.96
    ]
    st.info(f"📊 استخدام {len(raw_data)} قيمة نموذجية")

# ============================ التحليل ============================
if raw_data and len(raw_data) >= 30:
    st.markdown("---")
    
    if st.button("🚀 تشغيل التحليل الكامل", type="primary"):
        
        with st.spinner("⏳ جارٍ تشغيل اختبارات NIST..."):
            suite = RandomnessTestSuite(raw_data)
            all_results = suite.run_all_tests()
            
            analyzer = PRNGAnalyzer(raw_data)
            prng_results = analyzer.run_full_analysis()
        
        # ==== لوحة النتائج الرئيسية ====
        st.header("📊 نتائج اختبارات العشوائية")
        
        # ملخص
        passed = all_results['passed_tests']
        total = all_results['total_tests']
        verdict = all_results['verdict']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("اختبارات نجحت", f"{passed}/{total}")
        col2.metric("معدل النجاح", f"{passed/total*100:.0f}%")
        col3.metric(
            "الحكم", 
            "عشوائي" if passed >= total*0.8 else "يحتوي أنماط",
            delta="✅" if passed >= total*0.8 else "🔴"
        )
        
        # تفاصيل الاختبارات
        st.subheader("🔬 تفاصيل الاختبارات")
        
        test_data = []
        for key, result in suite.results.items():
            test_data.append({
                'الاختبار': result['test_name'],
                'P-Value': result.get('p_value', 'N/A'),
                'النتيجة': '✅ نجح' if result.get('passed') else '❌ فشل',
                'التفسير': result.get('interpretation', '')
            })
        
        st.dataframe(
            pd.DataFrame(test_data), 
            use_container_width=True
        )
        
        # ==== تحليل PRNG ====
        st.header("🔐 تحليل PRNG المتقدم")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📡 التحليل الطيفي",
            "🔄 الدوريات", 
            "📊 التوزيع",
            "🔢 الارتباط الذاتي"
        ])
        
        with tab1:
            spectral = prng_results['spectral']
            st.write(f"**{spectral['interpretation']}**")
            st.write(f"نسبة الهيمنة: {spectral['dominance_ratio']}")
            
            if spectral['dominant_frequencies']:
                df_freq = pd.DataFrame(spectral['dominant_frequencies'])
                fig_fft = px.bar(
                    df_freq, x='period', y='relative_power',
                    title="أقوى الترددات (FFT)",
                    labels={
                        'period': 'الدورة (جولات)', 
                        'relative_power': 'القوة النسبية'
                    }
                )
                st.plotly_chart(fig_fft, use_container_width=True)
        
        with tab2:
            cycle = prng_results['cycle']
            st.write(f"**{cycle['interpretation']}**")
            st.write(f"أفضل فترة: {cycle['best_period']} جولة")
            st.write(f"أقوى ارتباط: {cycle['best_correlation']}")
        
        with tab3:
            arr = np.array(raw_data)
            
            # مخطط التوزيع
            fig_dist = make_subplots(rows=1, cols=2)
            
            # توزيع فعلي
            bins = [1, 1.5, 2, 3, 5, 10, 20, 100]
            labels = ['<1.5', '1.5-2', '2-3', '3-5', '5-10', '10-20', '>20']
            counts_actual = [
                ((arr >= bins[i]) & (arr < bins[i+1])).sum() 
                for i in range(len(bins)-1)
            ]
            
            # توزيع نظري
            n = len(arr)
            house_edge = 0.99
            counts_theoretical = []
            for i in range(len(bins)-1):
                p_l = min(house_edge/bins[i], 1.0)
                p_h = (
                    min(house_edge/bins[i+1], 1.0) 
                    if bins[i+1] < 100 else 0.0
                )
                counts_theoretical.append((p_l - p_h) * n)
            
            fig_dist.add_trace(
                go.Bar(
                    x=labels, y=counts_actual, 
                    name='فعلي', marker_color='blue'
                ), 
                row=1, col=1
            )
            fig_dist.add_trace(
                go.Bar(
                    x=labels, y=counts_theoretical, 
                    name='نظري (Power Law)', marker_color='red'
                ), 
                row=1, col=2
            )
            
            fig_dist.update_layout(title="مقارنة التوزيع الفعلي بالنظري")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with tab4:
            # مخطط الارتباط الذاتي
            acf_data = suite.results.get('autocorrelation', {})
            if 'autocorrelations' in acf_data:
                acf_df = pd.DataFrame(acf_data['autocorrelations'])
                bound = acf_data.get('significance_bound', 0.1)
                
                fig_acf = go.Figure()
                fig_acf.add_trace(go.Bar(
                    x=acf_df['lag'],
                    y=acf_df['autocorrelation'],
                    marker_color=[
                        'red' if sig else 'blue' 
                        for sig in acf_df['significant']
                    ],
                    name='ACF'
                ))
                fig_acf.add_hline(
                    y=bound, line_dash="dash", 
                    line_color="red",
                    annotation_text="حد الدلالة (95%)"
                )
                fig_acf.add_hline(
                    y=-bound, line_dash="dash", 
                    line_color="red"
                )
                fig_acf.update_layout(
                    title="الارتباط الذاتي (ACF) - الأحمر = دال إحصائياً",
                    xaxis_title="Lag",
                    yaxis_title="Autocorrelation"
                )
                st.plotly_chart(fig_acf, use_container_width=True)
        
        # ==== الاستنتاج الأكاديمي ====
        st.header("📝 الاستنتاج الأكاديمي")
        
        conclusion_color = (
            "success" if passed >= total * 0.8 
            else "warning"
        )
        
        if passed >= total * 0.8:
            st.success(
                f"""
                ### نتيجة التحليل: البيانات تُظهر خصائص عشوائية
                
                - **{passed}/{total}** اختبار نجح
                - المولد يستوفي معايير NIST SP 800-22
                - **الاستنتاج:** PRNG قوي - صعب كسره بالطرق الإحصائية
                - **التوصية:** تحليل الخوارزمية على مستوى الكود أكثر جدوى
                """,
                icon="📊"
            )
        else:
            st.warning(
                f"""
                ### نتيجة التحليل: اكتُشفت أنماط إحصائية!
                
                - **{total - passed}/{total}** اختبار فشل
                - يوجد انحراف عن العشوائية التامة
                - **الاستنتاج:** PRNG يحتوي نقاط ضعف قابلة للدراسة
                - **التوصية:** تعميق التحليل على الأنماط المكتشفة
                """,
                icon="🔍"
            )
        
        # تصدير النتائج
        st.download_button(
            "📥 تحميل التقرير الكامل (JSON)",
            data=json.dumps({
                'randomness_tests': all_results,
                'prng_analysis': str(prng_results)
            }, ensure_ascii=False, indent=2),
            file_name="crash_analysis_report.json",
            mime="application/json"
        )

elif raw_data:
    st.warning(f"⚠️ أدخل 30 قيمة على الأقل (لديك {len(raw_data)})")

st.markdown("---")
st.caption(
    "🎓 هذا التطبيق لأغراض بحثية أكاديمية بحتة | "
    "تحليل PRNG | مشروع تخرج"
)
