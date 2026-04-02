import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="مساعد كاندي كراش", layout="centered")
st.title("🍬 محلل لوحة كاندي كراش (النسخة الخاصة)")

# دالة لتحميل نموذجك الخاص لمرة واحدة فقط لتسريع الأداء
@st.cache_resource
def load_custom_model():
    # هنا نقوم بتحميل الملف الذي قمت بتدريبه
    return YOLO("best.pt")

try:
    model = load_custom_model()
    st.sidebar.success("✅ تم تحميل نموذج best.pt بنجاح!")
except Exception as e:
    st.sidebar.error("❌ لم يتم العثور على ملف best.pt. تأكد من وضعه في نفس المجلد مع ملف app.py.")
    st.stop()

# واجهة رفع الصور
uploaded_file = st.file_uploader("قم برفع إحدى لقطات الشاشة التي صورتها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة الأصلية
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة الأصلية", use_container_width=True)
    
    if st.button("🔍 تحليل اللوحة"):
        with st.spinner("الذكاء الاصطناعي يقوم بعمله..."):
            # تحويل الصورة لتناسب مكتبة YOLO
            img_array = np.array(image)
            
            # تنفيذ التوقع (Inference)
            # وضعنا نسبة الثقة 0.25، يمكنك رفعها لاحقاً إذا ظهرت مربعات خاطئة
            results = model.predict(source=img_array, conf=0.25)
            
            # استخراج الصورة مع المربعات المرسومة حول الحلوى
            res_plotted = results[0].plot()
            
            # عرض النتيجة
            st.image(res_plotted, caption="اللوحة بعد التحليل", use_container_width=True)
            
            # طباعة عدد العناصر التي وجدها
            st.success(f"تم اكتشاف {len(results[0].boxes)} عنصر في اللوحة!")
