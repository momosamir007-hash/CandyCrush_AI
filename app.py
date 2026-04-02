# app_v2.py
"""
╔══════════════════════════════════════════════════╗
║ 🍬 Candy Crush AI v2 - يتعرف على 50+ عنصر ║
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
    print_catalog
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

# ═══ إعداد الصفحة ═══
st.set_page_config(
    page_title="🍬 Candy Crush AI v2",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══ CSS ═══
st.markdown("""
<style>
.main-title {
    text-align: center;
    background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff, #9b59b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8em;
    font-weight: 900;
}
.element-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 12px;
    margin: 6px 0;
    border-left: 4px solid #ffd93d;
    color: white;
    font-size: 0.9em;
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
    font-size: 1.4em;
    letter-spacing: 4px;
    line-height: 2;
    font-family: monospace;
    text-align: center;
}
.stat-metric {
    background: #16213e;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    color: white;
    margin: 5px;
}
.legend-item {
    display: inline-block;
    margin: 3px 8px;
    font-size: 0.95em;
}
</style>
""", unsafe_allow_html=True)

# ═══ تحميل النموذج ═══
@st.cache_resource
def load_classifier():
    return CandyCrushClassifierV2(
        model_name="openai/clip-vit-base-patch32",
        active_categories=[
            ElementCategory.BASIC_CANDY,
            ElementCategory.SPECIAL_CANDY,
            ElementCategory.BLOCKER,
            ElementCategory.COVER,
            ElementCategory.BOARD,
            ElementCategory.INGREDIENT,
        ]
    )

# ═══ الشريط الجانبي ═══
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
    show_legend = st.checkbox("دليل العناصر", True)
    show_details = st.checkbox("تفاصيل التعرف", False)

    st.markdown("---")

    # ═══ دليل العناصر ═══
    st.markdown("### 📖 دليل العناصر الكامل")
    cat_filter = st.selectbox(
        "اختر الفئة",
        [
            "الكل",
            "📦 حلوى أساسية",
            "⭐ حلوى خاصة",
            "🧱 عوائق وحواجز",
            "🧊 أغطية وطبقات",
            "🎯 عناصر اللوحة",
            "🍒 مكونات",
        ]
    )
    cat_map = {
        "📦 حلوى أساسية": ElementCategory.BASIC_CANDY,
        "⭐ حلوى خاصة":   ElementCategory.SPECIAL_CANDY,
        "🧱 عوائق وحواجز": ElementCategory.BLOCKER,
        "🧊 أغطية وطبقات": ElementCategory.COVER,
        "🎯 عناصر اللوحة": ElementCategory.BOARD,
        "🍒 مكونات":      ElementCategory.INGREDIENT,
    }

    if cat_filter == "الكل":
        display_elements = ALL_ELEMENTS
    else:
        cat_enum = cat_map.get(cat_filter)
        display_elements = get_elements_by_category(cat_enum)

    for elem_id, elem in display_elements.items():
        flags = []
        if not elem.is_movable:
            flags.append("ثابت")
        if elem.spreads:
            flags.append("ينتشر⚠️")
        if elem.has_timer:
            flags.append("مؤقت💣")
        if elem.layers > 1:
            flags.append(f"{elem.layers}طبقات")

        priority_class = ""
        if elem.priority_score >= 50:
            priority_class = "priority-high"
        elif elem.priority_score >= 20:
            priority_class = "priority-med"
        else:
            priority_class = "priority-low"

        flag_str = " | ".join(flags)
        st.markdown(
            f'<div class="element-card {priority_class}">'
            f'{elem.emoji} <b>{elem.name_ar}</b><br>'
            f'<small>{elem.name_en}</small><br>'
            f'<small>📝 {elem.special_behavior}</small>'
            + (f'<br><small>🏷️ {flag_str}</small>' if flags else '')
            + f'</div>',
            unsafe_allow_html=True
        )

# ═══ العنوان ═══
st.markdown('<p class="main-title">🍬 Candy Crush AI v2</p>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#888;">'
    'يتعرف على 50+ عنصر: حلوى، عوائق، ثلج، سجن، شوكولاتة، '
    'فشار، قنابل، وأكثر!'
    '</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ═══ رفع الصورة ═══
uploaded = st.file_uploader(
    "📸 ارفع لقطة شاشة للوحة اللعب",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    col_img, col_info = st.columns([2, 1])
    with col_img:
        st.image(img, caption="الصورة المرفوعة", width=450)
    with col_info:
        st.info(
            f"📐 الأبعاد: {img_np.shape[1]}×{img_np.shape[0]}\n\n"
            f"🔢 الشبكة: {grid_rows}×{grid_cols}\n\n"
            f"📊 إجمالي الخلايا: {grid_rows * grid_cols}"
        )

    if st.button("🚀 تحليل شامل (50+ عنصر)", type="primary", use_container_width=True):
        # ═══ تحميل المصنف ═══
        classifier = load_classifier()

        # ═══ مرحلة 1: التصنيف ═══
        st.markdown("### 🔍 المرحلة 1: قراءة اللوحة")
        progress = st.progress(0, "بدء التحليل...")
        start = time.time()

        # تقسيم وتصنيف
        h, w = img_np.shape[:2]
        cell_h = h // grid_rows
        cell_w = w // grid_cols
        pad_y = int(cell_h * cell_padding)
        pad_x = int(cell_w * cell_padding)

        grid = np.full((grid_rows, grid_cols), "empty", dtype=object)
        conf_grid = np.zeros((grid_rows, grid_cols))

        cells = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                y1 = r * cell_h + pad_y
                y2 = (r + 1) * cell_h - pad_y
                x1 = c * cell_w + pad_x
                x2 = (c + 1) * cell_w - pad_x
                cell = img_np[y1:y2, x1:x2]
                if cell.size > 0:
                    cell = cv2.resize(cell, (72, 72))
                    cells.append(cell)

        # تصنيف دفعي
        progress.progress(0.3, "🧠 CLIP يحلل الخلايا...")
        results = classifier.classify_batch([Image.fromarray(c) for c in cells])

        # ملء الشبكة
        idx = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                elem_id, confidence = results[idx]
                grid[r, c] = elem_id
                conf_grid[r, c] = confidence
                idx += 1

        elapsed = time.time() - start
        progress.progress(1.0, f"✅ تم في {elapsed:.1f}s")

        # ═══ عرض الشبكة ═══
        st.markdown("### 🎮 الشبكة المكتشفة")
        grid_text = ""
        for r in range(grid_rows):
            row_emojis = []
            for c in range(grid_cols):
                elem = ALL_ELEMENTS.get(grid[r, c])
                row_emojis.append(elem.emoji if elem else '❓')
            grid_text += " ".join(row_emojis) + "\n"
        st.markdown(f'<div class="grid-display">{grid_text}</div>', unsafe_allow_html=True)

        # ═══ دليل الرموز ═══
        if show_legend:
            st.markdown("**📖 دليل الرموز:**")
            unique_elements = set(grid.flatten())
            legend_html = ""
            for eid in sorted(unique_elements):
                elem = ALL_ELEMENTS.get(eid)
                if elem:
                    legend_html += (
                        f'<span class="legend-item">'
                        f'{elem.emoji} = {elem.name_ar}'
                        f'</span>'
                    )
            st.markdown(legend_html, unsafe_allow_html=True)

        # ═══ إحصائيات متقدمة ═══
        st.markdown("### 📊 تحليل اللوحة")
        stats = {}
        category_stats = {}
        for r in range(grid_rows):
            for c in range(grid_cols):
                eid = grid[r, c]
                stats[eid] = stats.get(eid, 0) + 1
                elem = ALL_ELEMENTS.get(eid)
                if elem:
                    cat = elem.category.value
                    category_stats[cat] = category_stats.get(cat, 0) + 1

        # عدادات حسب الفئة
        cat_cols = st.columns(4)
        cat_display = {
            'basic_candy':   ('📦', 'حلوى أساسية'),
            'special_candy': ('⭐', 'حلوى خاصة'),
            'blocker':       ('🧱', 'عوائق'),
            'cover':         ('🧊', 'أغطية'),
        }
        for idx, (cat_key, (emoji, name)) in enumerate(cat_display.items()):
            with cat_cols[idx]:
                count = category_stats.get(cat_key, 0)
                st.metric(f"{emoji} {name}", count)

        # تحذيرات العناصر الخطرة
        danger_elements = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                elem = ALL_ELEMENTS.get(grid[r, c])
                if elem and (elem.has_timer or elem.spreads or elem.priority_score >= 30):
                    danger_elements.append((r, c, elem))

        if danger_elements:
            st.markdown("### ⚠️ تنبيهات مهمة!")
            for r, c, elem in danger_elements:
                if elem.has_timer:
                    st.error(f"💣 **{elem.name_ar}** في ({r},{c}) — {elem.special_behavior}")
                elif elem.spreads:
                    st.warning(f"🦠 **{elem.name_ar}** في ({r},{c}) — {elem.special_behavior}")
                elif elem.priority_score >= 50:
                    st.info(f"⭐ **{elem.name_ar}** في ({r},{c}) — {elem.special_behavior}")

        # ═══ مرحلة 2: الحركات ═══
        st.markdown("---")
        st.markdown("### 🧠 المرحلة 2: تحليل الحركات")
        engine = CandyEngine(grid)
        moves = engine.find_all_moves()

        if not moves:
            st.warning("⚠️ لا توجد حركات صالحة!")
        else:
            # عدادات
            m_cols = st.columns(4)
            with m_cols[0]:
                st.metric("🎯 إجمالي", len(moves))
            with m_cols[1]:
                st.metric("🏆 أعلى نقاط", moves[0]['score'])
            with m_cols[2]:
                max_chain = max(m.get('chain_depth', 1) for m in moves)
                st.metric("⛓️ أطول سلسلة", f"x{max_chain}")
            with m_cols[3]:
                specials = sum(len(m.get('special_candies', [])) for m in moves)
                st.metric("⭐ خاصة", specials)

            # ═══ الصورة مع الأسهم ═══
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
                    f"{medals[i]} {em1} ({r1},{c1}) ↔ {em2} ({r2},{c2}) — {move['score']} نقطة",
                    expanded=(i == 0)
                ):
                    dc = st.columns(3)
                    with dc[0]:
                        st.markdown(
                            f"**من:** {em1} {n1} ({r1},{c1})\n\n"
                            f"**إلى:** {em2} {n2} ({r2},{c2})\n\n"
                            f"**الاتجاه:** {move['direction']}"
                        )
                    with dc[1]:
                        st.markdown(
                            f"**النقاط:** {move['score']}\n\n"
                            f"**السلسلة:** x{move.get('chain_depth', 1)}\n\n"
                            f"**الأولوية:** {move.get('priority', 0)}"
                        )
                    with dc[2]:
                        spc = move.get('special_candies', [])
                        if spc:
                            st.markdown("**⭐ حلوى خاصة:**")
                            for s in spc:
                                st.markdown(f" • {s}")
                        else:
                            st.markdown("لا حلوى خاصة")
                        st.code(move.get('details', ''))

                    # رسم تخطيطي
                    dia = create_move_diagram(grid, move, cell_size=40)
                    dia_rgb = cv2.cvtColor(dia, cv2.COLOR_BGR2RGB)
                    st.image(dia_rgb, width=400)

            # ═══ التحميل ═══
            st.markdown("---")
            dl = st.columns(3)
            with dl[0]:
                buf = BytesIO()
                Image.fromarray(img_result).save(buf, "PNG")
                st.download_button(
                    "📥 صورة الأسهم",
                    buf.getvalue(),
                    "moves.png",
                    "image/png",
                    use_container_width=True
                )
            with dl[1]:
                csv = ""
                for r in range(grid_rows):
                    csv += ",".join(grid[r]) + "\n"
                st.download_button(
                    "📥 الشبكة CSV",
                    csv,
                    "grid.csv",
                    "text/csv",
                    use_container_width=True
                )
            with dl[2]:
                report = "=== Candy Crush AI Report ===\n\n"
                report += f"Grid: {grid_rows}x{grid_cols}\n"
                report += f"Elements found: {len(stats)}\n"
                report += f"Moves found: {len(moves)}\n\n"
                for i, m in enumerate(moves[:top_moves]):
                    report += (
                        f"Move {i+1}: {m['pos1']}→{m['pos2']} "
                        f"Score:{m['score']} "
                        f"{m['details']}\n"
                    )
                st.download_button(
                    "📥 التقرير",
                    report,
                    "report.txt",
                    "text/plain",
                    use_container_width=True
                )

# ═══ الفوتر ═══
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#666;font-size:0.8em;">'
    '🍬 Candy Crush AI v2 | '
    '50+ عنصر | CLIP Zero-Shot | Streamlit<br>'
    'يكتشف: حلوى · ثلج · سجن · شوكولاتة · فشار · '
    'قنابل · جيلي · مربى · كريمة · وأكثر!'
    '</div>',
    unsafe_allow_html=True
    )
