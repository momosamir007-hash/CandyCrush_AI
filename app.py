import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from move_engine import CandyEngine 
import os

st.set_page_config(page_title="Candy Crush AI", layout="centered")
st.title("🍬 العرّاف: مساعد كاندي كراش")
st.markdown("---")

# 1. تحميل النموذج
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)

model = load_model()

if model is None:
    st.error("❌ ملف `best.pt` غير موجود! تأكد من رفعه إلى مستودع GitHub.")
    st.stop()

st.sidebar.success("✅ النموذج جاهز للعمل!")

# خريطة الألوان بناءً على الكود الذي استخدمناه في التدريب
class_map = {
    0: 'red', 1: 'blue', 2: 'green', 
    3: 'yellow', 4: 'orange', 5: 'purple', 
    6: 'blocker'
}

st.info("💡 نصيحة: ارفع صورة مقصوصة تحتوي على لوحة اللعب فقط (9x9) للحصول على أعلى دقة.")
uploaded_file = st.file_uploader("📷 ارفع صورة اللوحة...", type=["jpg", "png", "jpeg"])

if uploaded_file and st.button("🚀 تحليل ورسم أفضل حركة"):
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    h, w, _ = img_np.shape

    with st.spinner("الذكاء الاصطناعي يقرأ اللوحة..."):
        # التوقع باستخدام YOLO
        results = model.predict(img_np, conf=0.35)
        boxes = results[0].boxes
        
        # بناء الشبكة 9x9 (مصفوفة فارغة)
        grid = np.full((9, 9), 'empty', dtype=object)
        cell_w, cell_h = w / 9, h / 9
        
        # توزيع الحلوى المكتشفة على خلايا الشبكة
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].item())
            
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            col, row = int(cx // cell_w), int(cy // cell_h)
            
            if 0 <= row < 9 and 0 <= col < 9:
                grid[row, col] = class_map.get(cls, 'empty')

        # تشغيل المحرك الاستراتيجي
        engine = CandyEngine(grid)
        moves = engine.find_all_moves()

        st.markdown("---")
        st.subheader("💡 الحل البصري المقترح:")

        if moves:
            best_move = moves[0]
            img_with_arrow = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            def get_pixel_coords(pos, img_w, img_h):
                r, c = pos
                x = int((c + 0.5) * (img_w / 9))
                y = int((r + 0.5) * (img_h / 9))
                return (x, y)

            p1_pixel = get_pixel_coords(best_move['pos1'], w, h)
            p2_pixel = get_pixel_coords(best_move['pos2'], w, h)

            # رسم دائرة شفافة وسهم أخضر
            overlay = img_with_arrow.copy()
            radius = int(min(cell_w, cell_h) * 0.4)
            cv2.circle(overlay, p1_pixel, radius, (0, 255, 0), -1) 
            img_with_arrow = cv2.addWeighted(img_with_arrow, 0.6, overlay, 0.4, 0)
            cv2.arrowedLine(img_with_arrow, p1_pixel, p2_pixel, (0, 255, 0), 10, tipLength=0.4, line_type=cv2.LINE_AA)

            result_img_pil = Image.fromarray(cv2.cvtColor(img_with_arrow, cv2.COLOR_BGR2RGB))
            st.image(result_img_pil, caption="🥇 هذه هي الحركة الأفضل!", use_container_width=True)
            
            st.success(f"حرك من الصف {best_move['pos1'][0]} العمود {best_move['pos1'][1]} ➜ {best_move['direction']}")
            st.info(f"✨ النتيجة المتوقعة: {best_move['score']} نقطة ({best_move['details']})")
        else:
            st.image(image, use_container_width=True)
            st.warning("⚠️ لم يتم العثور على حركات مطابقة في هذه اللوحة.")
