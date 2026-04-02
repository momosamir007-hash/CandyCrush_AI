# app.py
"""
╔═══════════════════════════════════════════════════╗
║ 🍬 Candy Crush AI Assistant - Streamlit App ║
║ النسخة المطورة مع الأسهم والتحليل المتقدم ║
╚═══════════════════════════════════════════════════╝
"""
import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
import time
from transformers import CLIPProcessor, CLIPModel
from move_engine import CandyEngine
from grid_visualizer import (
    draw_arrows_on_image,
    draw_grid_overlay,
    create_grid_image,
    create_move_diagram,
    highlight_matches,
    CANDY_EMOJI,
    CANDY_COLORS_RGB
)

# ═══════════════════════════════════════
# إعداد الصفحة
# ═══════════════════════════════════════
st.set_page_config(
    page_title="🍬 Candy Crush AI",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════
# CSS مخصص
# ═══════════════════════════════════════
st.markdown("""
<style>
.main-header {
    text-align: center;
    background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff, #9b59b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 0;
}
.sub-header {
    text-align: center;
    color: #888;
    margin-top: 0;
}
.move-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    border-left: 4px solid #ffd93d;
    color: white;
}
.score-badge {
    background: #ffd93d;
    color: #1a1a2e;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
}
.chain-badge {
    background: #ff6b6b;
    color: white;
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 0.85em;
}
.special-badge {
    background: #9b59b6;
    color: white;
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 0.85em;
}
.stat-box {
    background: #16213e;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    color: white;
}
.stat-number {
    font-size: 2em;
    font-weight: bold;
    color: #ffd93d;
}
.grid-emoji {
    font-size: 1.5em;
    letter-spacing: 5px;
    line-height: 1.8;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# تحميل النموذج
# ═══════════════════════════════════════
@st.cache_resource
def load_clip_model():
    """تحميل CLIP مرة واحدة فقط"""
    with st.spinner("🔄 جاري تحميل نموذج الذكاء الاصطناعي..."):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_clip_model()

# ═══════════════════════════════════════
# تعريف أوصاف الحلوى لـ CLIP
# ═══════════════════════════════════════
CANDY_PROMPTS = {
    'red': [
        "a red candy piece",
        "red jellybean candy",
        "bright red round candy",
    ],
    'blue': [
        "a blue candy piece",
        "blue lozenge candy",
        "bright blue diamond candy",
    ],
    'green': [
        "a green candy piece",
        "green square candy",
        "bright green candy",
    ],
    'yellow': [
        "a yellow candy piece",
        "yellow drop candy",
        "bright yellow candy",
    ],
    'orange': [
        "a orange candy piece",
        "orange diamond candy",
        "bright orange candy",
    ],
    'purple': [
        "a purple candy piece",
        "purple round candy",
        "bright purple candy",
    ],
}

# قائمة مسطحة للتصنيف
ALL_LABELS = []
LABEL_TO_COLOR = {}
for color, prompts in CANDY_PROMPTS.items():
    for prompt in prompts:
        ALL_LABELS.append(prompt)
        LABEL_TO_COLOR[prompt] = color

# ═══════════════════════════════════════
# دالة تصنيف الخلية
# ═══════════════════════════════════════
def classify_cell(cell_image: Image.Image) -> tuple:
    """تصنيف خلية واحدة باستخدام CLIP"""
    inputs = processor(
        text=ALL_LABELS,
        images=cell_image,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=0)

    # تجميع الاحتمالات حسب اللون
    color_scores = {}
    for idx, label in enumerate(ALL_LABELS):
        color = LABEL_TO_COLOR[label]
        if color not in color_scores:
            color_scores[color] = 0
        color_scores[color] += probs[idx].item()

    # أفضل لون
    best_color = max(color_scores, key=color_scores.get)
    confidence = color_scores[best_color] / sum(color_scores.values())
    return best_color, confidence

def classify_board(
    img_np: np.ndarray,
    rows: int = 9,
    cols: int = 9,
    padding: float = 0.1,
    progress_bar=None
) -> tuple:
    """تصنيف اللوحة الكاملة"""
    h, w = img_np.shape[:2]
    cell_h = h // rows
    cell_w = w // cols
    grid = np.full((rows, cols), "empty", dtype=object)
    confidence_grid = np.zeros((rows, cols))
    total = rows * cols
    count = 0

    for r in range(rows):
        for c in range(cols):
            # حدود الخلية مع padding
            pad_y = int(cell_h * padding)
            pad_x = int(cell_w * padding)
            y1 = r * cell_h + pad_y
            y2 = (r + 1) * cell_h - pad_y
            x1 = c * cell_w + pad_x
            x2 = (c + 1) * cell_w - pad_x
            cell = img_np[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            cell_pil = Image.fromarray(cell)
            color, conf = classify_cell(cell_pil)
            grid[r, c] = color
            confidence_grid[r, c] = conf
            count += 1
            if progress_bar:
                progress_bar.progress(
                    count / total,
                    text=f"🔍 تحليل الخلية ({r},{c})... {count}/{total}"
                )

    return grid, confidence_grid

# ═══════════════════════════════════════
# الشريط الجانبي
# ═══════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ الإعدادات")
    st.markdown("### 📐 أبعاد الشبكة")
    grid_rows = st.slider("عدد الصفوف", 5, 12, 9)
    grid_cols = st.slider("عدد الأعمدة", 5, 12, 9)

    st.markdown("### 🎯 دقة التحليل")
    cell_padding = st.slider(
        "تقليص الحواف (Padding)",
        0.0, 0.3, 0.12, 0.02,
        help="نسبة القص من حواف كل خلية لتجنب التداخل"
    )

    st.markdown("### 🏆 عرض الحركات")
    top_moves = st.slider("عدد أفضل الحركات", 1, 10, 3)
    show_grid_overlay = st.checkbox("عرض الشبكة الملونة", True)
    show_match_highlight = st.checkbox("تمييز الحلوى المتطابقة", True)
    show_confidence = st.checkbox("عرض نسب الثقة", False)

    st.markdown("---")
    st.markdown("### 📊 عن المشروع")
    st.info(
        "يستخدم نموذج **CLIP** من OpenAI\n\n"
        "للتعرف على ألوان الحلوى بدون\n"
        "تدريب مسبق (Zero-Shot)"
    )

# ═══════════════════════════════════════
# العنوان الرئيسي
# ═══════════════════════════════════════
st.markdown(
    '<p class="main-header">🍬 Candy Crush AI Assistant</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">'
    'مساعد ذكي يحلل لوحة اللعب ويقترح أفضل الحركات بالأسهم'
    '</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ═══════════════════════════════════════
# رفع الصورة
# ═══════════════════════════════════════
col_upload, col_info = st.columns([2, 1])
with col_upload:
    uploaded_file = st.file_uploader(
        "📸 ارفع لقطة شاشة للوحة اللعب",
        type=["jpg", "jpeg", "png", "webp"],
        help="قص الصورة لتحتوي على لوحة اللعب فقط (مربعة قدر الإمكان)"
    )
with col_info:
    st.markdown("""
    **📋 نصائح للصورة المثالية:**
    - ✅ قص اللوحة فقط (بدون أشرطة)
    - ✅ صورة واضحة بدون ضبابية
    - ✅ اللوحة كاملة ظاهرة
    - ❌ لا ترفع أثناء حركة الحلوى
    """)

# ═══════════════════════════════════════
# التحليل الرئيسي
# ═══════════════════════════════════════
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.markdown("### 📸 الصورة المرفوعة")
    st.image(img, use_container_width=False, width=400)

    if st.button("🚀 ابدأ التحليل الذكي", type="primary", use_container_width=True):
        # ═══ مرحلة 1: تصنيف الشبكة ═══
        st.markdown("---")
        st.markdown("### 🔍 المرحلة 1: قراءة اللوحة")
        progress = st.progress(0, text="🔍 بدء التحليل...")
        start_time = time.time()

        grid, conf_grid = classify_board(
            img_np,
            rows=grid_rows,
            cols=grid_cols,
            padding=cell_padding,
            progress_bar=progress
        )
        analysis_time = time.time() - start_time
        progress.progress(1.0, text=f"✅ تم التحليل في {analysis_time:.1f} ثانية")

        # ═══ عرض الشبكة ═══
        st.markdown("### 🎮 الشبكة المكتشفة")
        grid_text = ""
        for r in range(grid_rows):
            row_emojis = []
            for c in range(grid_cols):
                row_emojis.append(CANDY_EMOJI.get(grid[r, c], '❓'))
            grid_text += " ".join(row_emojis) + "\n"
        st.markdown(
            f'<div class="grid-emoji">{grid_text}</div>',
            unsafe_allow_html=True
        )

        # ═══ إحصائيات ═══
        stats = {}
        for r in range(grid_rows):
            for c in range(grid_cols):
                candy = grid[r, c]
                stats[candy] = stats.get(candy, 0) + 1

        st.markdown("### 📊 إحصائيات اللوحة")
        stat_cols = st.columns(min(len(stats), 6))
        for idx, (candy, count) in enumerate(
            sorted(stats.items(), key=lambda x: -x[1])
        ):
            if candy == 'empty':
                continue
            with stat_cols[idx % len(stat_cols)]:
                emoji = CANDY_EMOJI.get(candy, '❓')
                st.metric(
                    label=f"{emoji} {candy}",
                    value=count,
                    delta=f"{count/(grid_rows*grid_cols)*100:.0f}%"
                )

        # نسبة الثقة
        if show_confidence:
            st.markdown("### 🎯 خريطة الثقة")
            conf_text = ""
            for r in range(grid_rows):
                row_parts = []
                for c in range(grid_cols):
                    conf = conf_grid[r, c]
                    if conf >= 0.7:
                        row_parts.append(f"🟢{conf:.0%}")
                    elif conf >= 0.4:
                        row_parts.append(f"🟡{conf:.0%}")
                    else:
                        row_parts.append(f"🔴{conf:.0%}")
                conf_text += " | ".join(row_parts) + "\n"
            st.code(conf_text)

        # ═══ مرحلة 2: تحليل الحركات ═══
        st.markdown("---")
        st.markdown("### 🧠 المرحلة 2: تحليل الحركات")
        engine = CandyEngine(grid)
        moves = engine.find_all_moves()

        if not moves:
            st.warning("⚠️ لا توجد حركات صالحة! قد تحتاج اللوحة لإعادة التحليل.")
        else:
            # ═══ عدادات سريعة ═══
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("🎯 إجمالي الحركات", len(moves))
            with metric_cols[1]:
                st.metric("🏆 أعلى نقاط", moves[0]['score'])
            with metric_cols[2]:
                max_chain = max(m.get('chain_depth', 1) for m in moves)
                st.metric("⛓️ أطول سلسلة", f"x{max_chain}")
            with metric_cols[3]:
                specials_count = sum(len(m.get('special_candies', [])) for m in moves)
                st.metric("⭐ حلوى خاصة", specials_count)

            # ═══ الصورة مع الأسهم ═══
            st.markdown("### 🏹 الحركات المقترحة على اللوحة")
            # تحويل لـ BGR لـ OpenCV
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # رسم الشبكة الملونة (اختياري)
            if show_grid_overlay:
                img_bgr = draw_grid_overlay(img_bgr, grid, opacity=0.25)

            # رسم الأسهم
            img_with_arrows = draw_arrows_on_image(
                img_bgr, grid, moves, top_n=top_moves
            )
            # تحويل للعرض
            img_result_rgb = cv2.cvtColor(img_with_arrows, cv2.COLOR_BGR2RGB)
            st.image(
                img_result_rgb,
                caption=f"🏹 أفضل {min(top_moves, len(moves))} حركات مقترحة",
                use_container_width=True
            )

            # ═══ تفاصيل كل حركة ═══
            st.markdown("### 📋 تفاصيل الحركات")
            medals = ['🥇', '🥈', '🥉'] + ['🎯'] * 20
            for i, move in enumerate(moves[:top_moves]):
                r1, c1 = move['pos1']
                r2, c2 = move['pos2']
                candy1 = move.get('candy1', grid[r1, c1])
                candy2 = move.get('candy2', grid[r2, c2])
                emoji1 = CANDY_EMOJI.get(candy1, '❓')
                emoji2 = CANDY_EMOJI.get(candy2, '❓')

                with st.expander(
                    f"{medals[i]} الحركة {i+1}: "
                    f"{emoji1} ({r1},{c1}) ↔ {emoji2} ({r2},{c2}) "
                    f"— {move['score']} نقطة",
                    expanded=(i == 0)
                ):
                    detail_cols = st.columns([1, 1, 1])
                    with detail_cols[0]:
                        st.markdown(f"""
                        **📍 من:** صف {r1}, عمود {c1}
                        **📍 إلى:** صف {r2}, عمود {c2}
                        **↔️ الاتجاه:** {move['direction']}
                        """)
                    with detail_cols[1]:
                        st.markdown(f"""
                        **💰 النقاط:** {move['score']}
                        **⛓️ السلسلة:** x{move.get('chain_depth', 1)}
                        **🏆 الأولوية:** {move.get('priority', 0)}
                        """)
                    with detail_cols[2]:
                        specials = move.get('special_candies', [])
                        if specials:
                            st.markdown("**⭐ حلوى خاصة:**")
                            for s in specials:
                                st.markdown(f" • {s}")
                        else:
                            st.markdown("**⭐ حلوى خاصة:** لا يوجد")
                        st.markdown(f"**📝 التفاصيل:** `{move['details']}`")

                    # ═══ رسم تخطيطي للحركة ═══
                    diagram_col1, diagram_col2 = st.columns(2)
                    with diagram_col1:
                        st.markdown("**🎯 الحركة:**")
                        diagram = create_move_diagram(grid, move, cell_size=45)
                        diagram_rgb = cv2.cvtColor(diagram, cv2.COLOR_BGR2RGB)
                        st.image(diagram_rgb, use_container_width=True)
                    with diagram_col2:
                        if show_match_highlight:
                            st.markdown("**💥 النتيجة المتوقعة:**")
                            highlighted = highlight_matches(
                                img_bgr, grid, move, engine
                            )
                            highlighted_rgb = cv2.cvtColor(
                                highlighted, cv2.COLOR_BGR2RGB
                            )
                            st.image(highlighted_rgb, use_container_width=True)

            # ═══ جدول كل الحركات ═══
            if len(moves) > top_moves:
                with st.expander(
                    f"📊 عرض جميع الحركات ({len(moves)} حركة)", expanded=False
                ):
                    table_data = []
                    for i, m in enumerate(moves):
                        table_data.append({
                            '#': i + 1,
                            'من': f"({m['pos1'][0]},{m['pos1'][1]})",
                            'إلى': f"({m['pos2'][0]},{m['pos2'][1]})",
                            'الاتجاه': m['direction'],
                            'النقاط': m['score'],
                            'السلسلة': f"x{m.get('chain_depth', 1)}",
                            'الأولوية': m.get('priority', 0),
                            'التفاصيل': m['details'][:40],
                        })
                    st.dataframe(
                        table_data,
                        use_container_width=True,
                        hide_index=True
                    )

            # ═══ تحميل النتيجة ═══
            st.markdown("---")
            st.markdown("### 💾 تحميل النتائج")
            dl_cols = st.columns(3)
            with dl_cols[0]:
                # صورة مع الأسهم
                result_pil = Image.fromarray(img_result_rgb)
                from io import BytesIO
                buf = BytesIO()
                result_pil.save(buf, format="PNG")
                st.download_button(
                    "📥 تحميل صورة الأسهم",
                    buf.getvalue(),
                    "candy_crush_moves.png",
                    "image/png",
                    use_container_width=True
                )
            with dl_cols[1]:
                # الشبكة كنص
                grid_export = ""
                for r in range(grid_rows):
                    grid_export += ",".join(grid[r]) + "\n"
                st.download_button(
                    "📥 تحميل الشبكة (CSV)",
                    grid_export,
                    "candy_grid.csv",
                    "text/csv",
                    use_container_width=True
                )
            with dl_cols[2]:
                # تقرير الحركات
                report = "Candy Crush AI - Move Report\n"
                report += "=" * 40 + "\n\n"
                for i, m in enumerate(moves[:top_moves]):
                    report += f"Move {i+1}: {m['pos1']} → {m['pos2']}\n"
                    report += f" Direction: {m['direction']}\n"
                    report += f" Score: {m['score']}\n"
                    report += f" Chain: x{m.get('chain_depth',1)}\n"
                    report += f" Details: {m['details']}\n\n"
                st.download_button(
                    "📥 تحميل التقرير",
                    report,
                    "move_report.txt",
                    "text/plain",
                    use_container_width=True
                )

# ═══════════════════════════════════════
# الفوتر
# ═══════════════════════════════════════
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#666; font-size:0.85em;">
    🍬 Candy Crush AI Assistant | Powered by <b>CLIP</b> (OpenAI) | Built with <b>Streamlit</b><br>
    Zero-Shot Detection — لا يحتاج تدريب مسبق
    </div>
    """,
    unsafe_allow_html=True
    )
