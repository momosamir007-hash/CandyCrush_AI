# app.py
"""
╔══════════════════════════════════════════════════╗
║ 🍬 Candy Crush AI v2 - Fixed for Cloud ║
╚══════════════════════════════════════════════════╝
"""
import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
import time
from io import BytesIO
from candy_elements import (
    ALL_ELEMENTS,
    ElementCategory,
    get_elements_by_category,
)
from classifier_v2 import CandyCrushClassifierV2
from move_engine import CandyEngine
from grid_visualizer import (
    draw_arrows_on_image,
    draw_grid_overlay,
    create_grid_image,
    create_move_diagram,
    highlight_matches,
)

# ═══════════════════════════════════════
# إعداد الصفحة
# ═══════════════════════════════════════
st.set_page_config(
    page_title="🍬 Candy Crush AI v2",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════
# CSS
# ═══════════════════════════════════════
st.markdown("""
<style>
.main-title {
    text-align: center;
    background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff, #9b59b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: 900;
}
.subtitle {
    text-align: center;
    color: #888;
    font-size: 1em;
}
.element-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 10px;
    margin: 5px 0;
    border-left: 4px solid #ffd93d;
    color: #ddd;
    font-size: 0.85em;
}
.priority-high {
    border-left-color: #ff4444;
}
.priority-med {
    border-left-color: #ffaa00;
}
.priority-low {
    border-left-color: #44ff44;
}
.grid-display {
    font-size: 1.3em;
    letter-spacing: 3px;
    line-height: 2;
    font-family: monospace;
    text-align: center;
    background: #1a1a2e;
    padding: 15px;
    border-radius: 10px;
}
.legend-item {
    display: inline-block;
    margin: 2px 6px;
    font-size: 0.9em;
}
.warning-box {
    background: #3a1a1a;
    border: 1px solid #ff4444;
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# تحميل النموذج — مع معالجة الأخطاء
# ═══════════════════════════════════════
@st.cache_resource
def load_classifier():
    """تحميل المصنف مرة واحدة فقط"""
    try:
        classifier = CandyCrushClassifierV2(
            model_name="openai/clip-vit-base-patch32",
            device="cpu",  # CPU أكثر توافقاً على Cloud
            active_categories=[
                ElementCategory.BASIC_CANDY,
                ElementCategory.SPECIAL_CANDY,
                ElementCategory.BLOCKER,
                ElementCategory.COVER,
                ElementCategory.BOARD,
                ElementCategory.INGREDIENT,
            ]
        )
        return classifier
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        return None

# ═══════════════════════════════════════
# الشريط الجانبي
# ═══════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ الإعدادات")
    st.markdown("### 📐 الشبكة")
    grid_rows = st.slider("صفوف", 5, 12, 9)
    grid_cols = st.slider("أعمدة", 5, 12, 9)

    st.markdown("### 🎯 الدقة")
    cell_padding = st.slider("تقليص الحواف", 0.0, 0.3, 0.12, 0.02)

    st.markdown("### 🏆 العرض")
    top_moves = st.slider("عدد الحركات", 1, 10, 3)
    show_overlay = st.checkbox("شبكة ملونة", True)
    show_legend = st.checkbox("دليل الرموز", True)

    st.markdown("---")

    # دليل العناصر
    st.markdown("### 📖 دليل العناصر")
    cat_filter = st.selectbox(
        "الفئة",
        [
            "الكل",
            "📦 حلوى أساسية",
            "⭐ حلوى خاصة",
            "🧱 عوائق",
            "🧊 أغطية",
            "🎯 عناصر اللوحة",
            "🍒 مكونات",
        ]
    )

    cat_map = {
        "📦 حلوى أساسية": ElementCategory.BASIC_CANDY,
        "⭐ حلوى خاصة": ElementCategory.SPECIAL_CANDY,
        "🧱 عوائق": ElementCategory.BLOCKER,
        "🧊 أغطية": ElementCategory.COVER,
        "🎯 عناصر اللوحة": ElementCategory.BOARD,
        "🍒 مكونات": ElementCategory.INGREDIENT,
    }

    if cat_filter == "الكل":
        display_elems = ALL_ELEMENTS
    else:
        cat_e = cat_map.get(cat_filter)
        if cat_e:
            display_elems = get_elements_by_category(cat_e)
        else:
            display_elems = ALL_ELEMENTS

    for eid, elem in display_elems.items():
        flags = []
        if not elem.is_movable:
            flags.append("ثابت")
        if elem.spreads:
            flags.append("ينتشر⚠️")
        if elem.has_timer:
            flags.append("مؤقت💣")
        if elem.layers > 1:
            flags.append(f"{elem.layers}ط")

        pc = "priority-low"
        if elem.priority_score >= 50:
            pc = "priority-high"
        elif elem.priority_score >= 20:
            pc = "priority-med"

        flag_str = " | ".join(flags)
        card_html = (
            f'<div class="element-card {pc}">'
            f'{elem.emoji} <b>{elem.name_ar}</b><br>'
            f'<small>{elem.name_en}</small><br>'
            f'<small>📝 {elem.special_behavior}</small>'
        )
        if flags:
            card_html += f'<br><small>🏷️ {flag_str}</small>'
        card_html += '</div>'
        st.markdown(card_html, unsafe_allow_html=True)

# ═══════════════════════════════════════
# المحتوى الرئيسي
# ═══════════════════════════════════════
st.markdown('<p class="main-title">🍬 Candy Crush AI v2</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">'
    'يتعرف على 50+ عنصر: حلوى · ثلج · سجن · شوكولاتة · '
    'فشار · قنابل · جيلي · مربى · وأكثر!'
    '</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ═══ رفع الصورة ═══
col_up, col_tips = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader(
        "📸 ارفع لقطة شاشة للوحة",
        type=["jpg", "jpeg", "png", "webp"]
    )
with col_tips:
    st.markdown("""
    **📋 نصائح:**
    - ✅ قص اللوحة فقط
    - ✅ صورة واضحة
    - ❌ لا ترفع أثناء الحركة
    """)

# ═══════════════════════════════════════
# التحليل
# ═══════════════════════════════════════
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    st.image(img, caption="📸 الصورة المرفوعة", width=400)

    if st.button("🚀 ابدأ التحليل الذكي", type="primary", use_container_width=True):
        # ═══ تحميل المصنف ═══
        with st.spinner("🔄 تحميل نموذج الذكاء الاصطناعي..."):
            classifier = load_classifier()
        if classifier is None:
            st.error("❌ فشل تحميل النموذج!")
            st.stop()

        # ═══ مرحلة 1: التصنيف ═══
        st.markdown("---")
        st.markdown("### 🔍 المرحلة 1: قراءة اللوحة")
        progress = st.progress(0, "🔍 بدء التحليل...")
        start_time = time.time()

        h, w = img_np.shape[:2]
        cell_h = h // grid_rows
        cell_w = w // grid_cols
        pad_y = int(cell_h * cell_padding)
        pad_x = int(cell_w * cell_padding)

        # ═══ قص الخلايا ═══
        cells_pil = []
        total_cells = grid_rows * grid_cols
        count = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                y1 = r * cell_h + pad_y
                y2 = (r + 1) * cell_h - pad_y
                x1 = c * cell_w + pad_x
                x2 = (c + 1) * cell_w - pad_x
                # حماية الحدود
                y1 = max(0, y1)
                y2 = min(h, y2)
                x1 = max(0, x1)
                x2 = min(w, x2)
                cell = img_np[y1:y2, x1:x2]
                if cell.size > 0 and cell.shape[0] > 5 and cell.shape[1] > 5:
                    cell_resized = cv2.resize(cell, (72, 72))
                    cell_pil = Image.fromarray(cell_resized)
                else:
                    cell_pil = Image.new("RGB", (72, 72), (0, 0, 0))
                cells_pil.append(cell_pil)
                count += 1
                progress.progress(
                    count / total_cells * 0.3,
                    f"📐 قص الخلايا... {count}/{total_cells}"
                )

        # ═══ تصنيف دفعي ═══
        progress.progress(0.3, "🧠 CLIP يحلل...")
        results = classifier.classify_batch(cells_pil, batch_size=16)

        # ═══ ملء الشبكة ═══
        grid = np.full((grid_rows, grid_cols), "empty", dtype=object)
        conf_grid = np.zeros((grid_rows, grid_cols))
        idx = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                if idx < len(results):
                    elem_id, confidence = results[idx]
                    grid[r, c] = elem_id
                    conf_grid[r, c] = confidence
                idx += 1

        elapsed = time.time() - start_time
        progress.progress(1.0, f"✅ تم في {elapsed:.1f} ثانية")

        # ═══════════════════════════════════
        # عرض الشبكة
        # ═══════════════════════════════════
        st.markdown("### 🎮 الشبكة المكتشفة")
        grid_text_lines = []
        for r in range(grid_rows):
            row_emojis = []
            for c in range(grid_cols):
                elem = ALL_ELEMENTS.get(grid[r, c])
                emoji = elem.emoji if elem else '❓'
                row_emojis.append(emoji)
            grid_text_lines.append(" ".join(row_emojis))
        grid_display = "\n".join(grid_text_lines)
        st.markdown(f'<div class="grid-display">{grid_display}</div>', unsafe_allow_html=True)

        # ═══ دليل الرموز ═══
        if show_legend:
            unique_ids = set(grid.flatten())
            legend_parts = []
            for eid in sorted(unique_ids):
                elem = ALL_ELEMENTS.get(eid)
                if elem and eid != 'empty':
                    legend_parts.append(f'<span class="legend-item">{elem.emoji}={elem.name_ar}</span>')
            if legend_parts:
                st.markdown("**📖 الرموز:** " + " ".join(legend_parts), unsafe_allow_html=True)

        # ═══════════════════════════════════
        # إحصائيات
        # ═══════════════════════════════════
        st.markdown("### 📊 إحصائيات")
        stats = {}
        cat_stats = {}
        for r in range(grid_rows):
            for c in range(grid_cols):
                eid = grid[r, c]
                stats[eid] = stats.get(eid, 0) + 1
                elem = ALL_ELEMENTS.get(eid)
                if elem:
                    cat = elem.category.value
                    cat_stats[cat] = cat_stats.get(cat, 0) + 1

        scols = st.columns(min(6, len(stats)))
        sorted_stats = sorted(stats.items(), key=lambda x: -x[1])
        for idx_s, (eid, cnt) in enumerate(sorted_stats[:6]):
            if eid == 'empty':
                continue
            elem = ALL_ELEMENTS.get(eid)
            if elem:
                with scols[idx_s % len(scols)]:
                    pct = cnt / total_cells * 100
                    st.metric(f"{elem.emoji} {elem.name_ar}", cnt, f"{pct:.0f}%")

        # ═══ تحذيرات العناصر الخطرة ═══
        dangers = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                elem = ALL_ELEMENTS.get(grid[r, c])
                if elem and (elem.has_timer or elem.spreads or elem.priority_score >= 30):
                    dangers.append((r, c, elem))

        if dangers:
            st.markdown("### ⚠️ تنبيهات!")
            for r, c, elem in dangers:
                if elem.has_timer:
                    st.error(f"💣 **{elem.name_ar}** ({r},{c}) — {elem.special_behavior}")
                elif elem.spreads:
                    st.warning(f"🦠 **{elem.name_ar}** ({r},{c}) — {elem.special_behavior}")

        # ═══════════════════════════════════
        # مرحلة 2: الحركات
        # ═══════════════════════════════════
        st.markdown("---")
        st.markdown("### 🧠 المرحلة 2: تحليل الحركات")
        engine = CandyEngine(grid)
        moves = engine.find_all_moves()

        if not moves:
            st.warning("⚠️ لا توجد حركات صالحة!")
        else:
            # عدادات
            mc = st.columns(4)
            with mc[0]:
                st.metric("🎯 إجمالي", len(moves))
            with mc[1]:
                st.metric("🏆 أعلى نقاط", moves[0]['score'])
            with mc[2]:
                mx_chain = max(m.get('chain_depth', 1) for m in moves)
                st.metric("⛓️ سلسلة", f"x{mx_chain}")
            with mc[3]:
                n_specials = sum(len(m.get('special_candies', [])) for m in moves)
                st.metric("⭐ خاصة", n_specials)

            # ═══ صورة مع الأسهم ═══
            st.markdown("### 🏹 الحركات على اللوحة")
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            if show_overlay:
                img_bgr = draw_grid_overlay(img_bgr, grid, opacity=0.2)

            img_arrows = draw_arrows_on_image(img_bgr, grid, moves, top_n=top_moves)
            img_result = cv2.cvtColor(img_arrows, cv2.COLOR_BGR2RGB)
            st.image(
                img_result,
                caption=f"🏹 أفضل {min(top_moves, len(moves))} حركات",
                use_container_width=True
            )

            # ═══ تفاصيل الحركات ═══
            st.markdown("### 📋 تفاصيل الحركات")
            medals = ['🥇', '🥈', '🥉'] + ['🎯'] * 20
            for i, move in enumerate(moves[:top_moves]):
                r1, c1 = move['pos1']
                r2, c2 = move['pos2']
                e1 = ALL_ELEMENTS.get(grid[r1, c1])
                e2 = ALL_ELEMENTS.get(grid[r2, c2])
                em1 = e1.emoji if e1 else '❓'
                em2 = e2.emoji if e2 else '❓'
                n1 = e1.name_ar if e1 else grid[r1, c1]
                n2 = e2.name_ar if e2 else grid[r2, c2]

                with st.expander(
                    f"{medals[i]} {em1}({r1},{c1}) ↔ {em2}({r2},{c2}) — {move['score']} نقطة",
                    expanded=(i == 0)
                ):
                    dc = st.columns(3)
                    with dc[0]:
                        st.markdown(
                            f"**من:** {em1} {n1}\n\n"
                            f"**موقع:** ({r1},{c1})\n\n"
                            f"**الاتجاه:** {move['direction']}"
                        )
                    with dc[1]:
                        st.markdown(
                            f"**إلى:** {em2} {n2}\n\n"
                            f"**موقع:** ({r2},{c2})\n\n"
                            f"**النقاط:** {move['score']}"
                        )
                    with dc[2]:
                        ch = move.get('chain_depth', 1)
                        pr = move.get('priority', 0)
                        spc = move.get('special_candies', [])
                        st.markdown(
                            f"**السلسلة:** x{ch}\n\n"
                            f"**الأولوية:** {pr}\n\n"
                            f"**خاصة:** {', '.join(spc) if spc else 'لا'}"
                        )
                    details = move.get('details', '')
                    if details:
                        st.code(details)

                    # رسم تخطيطي
                    try:
                        dia = create_move_diagram(grid, move, cell_size=40)
                        dia_rgb = cv2.cvtColor(dia, cv2.COLOR_BGR2RGB)
                        st.image(dia_rgb, width=400)
                    except Exception:
                        pass

            # ═══ جدول كل الحركات ═══
            if len(moves) > top_moves:
                with st.expander(f"📊 كل الحركات ({len(moves)})"):
                    table = []
                    for i, m in enumerate(moves):
                        table.append({
                            '#': i + 1,
                            'من': str(m['pos1']),
                            'إلى': str(m['pos2']),
                            'اتجاه': m['direction'],
                            'نقاط': m['score'],
                            'سلسلة': m.get('chain_depth', 1),
                            'أولوية': m.get('priority', 0),
                        })
                    st.dataframe(table, use_container_width=True, hide_index=True)

            # ═══ تحميل ═══
            st.markdown("---")
            dl = st.columns(3)
            with dl[0]:
                buf = BytesIO()
                Image.fromarray(img_result).save(buf, format="PNG")
                st.download_button(
                    "📥 صورة الأسهم",
                    buf.getvalue(),
                    "moves.png",
                    "image/png",
                    use_container_width=True
                )
            with dl[1]:
                csv_str = ""
                for r in range(grid_rows):
                    csv_str += ",".join(str(grid[r, c]) for c in range(grid_cols)) + "\n"
                st.download_button(
                    "📥 الشبكة CSV",
                    csv_str,
                    "grid.csv",
                    "text/csv",
                    use_container_width=True
                )
            with dl[2]:
                rpt = "=== Candy Crush AI Report ===\n\n"
                rpt += f"Grid: {grid_rows}x{grid_cols}\n"
                rpt += f"Total moves: {len(moves)}\n\n"
                for i, m in enumerate(moves[:top_moves]):
                    rpt += f"#{i+1}: {m['pos1']}→{m['pos2']} Score:{m['score']} {m.get('details','')}\n"
                st.download_button(
                    "📥 التقرير",
                    rpt,
                    "report.txt",
                    "text/plain",
                    use_container_width=True
                )

# ═══ الفوتر ═══
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#666;font-size:0.8em;">'
    '🍬 Candy Crush AI v2 | 50+ عنصر | CLIP Zero-Shot | Streamlit'
    '</div>',
    unsafe_allow_html=True
        )
