import streamlit as st
from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from move_engine import CandyEngine 

# إعداد الصفحة
st.set_page_config(page_title="Candy Crush AI", layout="wide")
st.title("🍬 مساعد كاندي كراش الذكي")

# تحميل النموذج (مرة واحدة)
@st.cache_resource
def load_models():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_models()

# تعريف الألوان
candy_labels = ['red candy', 'blue candy', 'green candy', 'yellow candy', 'orange candy', 'purple candy']
mapping = {label: label.split()[0] for label in candy_labels}

uploaded_file = st.file_uploader("ارفع صورة اللوحة (مقصوصة مربعة)", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, width=400)
    
    if st.button("🚀 ابدأ التحليل"):
        with st.spinner("جاري قراءة اللوحة..."):
            img_np = np.array(img)
            h, w, _ = img_np.shape
            grid = np.full((9, 9), "empty", dtype=object)
            
            # تقسيم وتحليل (تبسيط لـ 9x9)
            ch, cw = h//9, w//9
            for r in range(9):
                for c in range(9):
                    cell = Image.fromarray(img_np[r*ch:(r+1)*ch, c*cw:(c+1)*cw])
                    inputs = processor(text=candy_labels, images=cell, return_tensors="pt", padding=True)
                    outputs = model(**inputs)
                    idx = outputs.logits_per_image.argmax().item()
                    grid[r, c] = mapping[candy_labels[idx]]
            
            # تشغيل المحرك
            engine = CandyEngine(grid)
            moves = engine.find_all_moves()
            
            st.subheader("💡 أفضل الحركات:")
            if moves:
                for m in moves[:3]:
                    st.success(f"حرك من {m['pos1']} إلى {m['pos2']} ({m['direction']}) ➜ {m['score']} نقطة")
            else:
                st.warning("لا توجد حركات واضحة!")
