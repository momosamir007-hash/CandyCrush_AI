import streamlit as st
from PIL import Image, ImageDraw
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

st.set_page_config(page_title="كاندي كراش - Zero Shot", layout="centered")
st.title("🍬 محلل كاندي كراش (بدون تدريب مسبق)")

# 1. تحميل النموذج المتقدم (سيأخذ وقتاً قليلاً في المرة الأولى للتحميل)
@st.cache_resource
def load_zero_shot_model():
    # نموذج OWL-ViT من جوجل المتخصص في البحث بالنصوص
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    return processor, model

with st.spinner("جاري تحميل العقل المدبر OWL-ViT..."):
    processor, model = load_zero_shot_model()
    st.sidebar.success("✅ النموذج جاهز للعمل!")

# 2. الكلمات التي نريد من الذكاء الاصطناعي البحث عنها في الصورة
# يمكنك تعديل هذه الكلمات أو إضافتها باللغة الإنجليزية
queries = [
    "red candy", "blue candy", "green candy", 
    "yellow candy", "orange candy", "purple candy", 
    "black licorice", "ice block", "jelly"
]

st.sidebar.markdown("**العناصر التي نبحث عنها:**")
for q in queries:
    st.sidebar.markdown(f"- {q}")

# 3. واجهة رفع الصور
uploaded_file = st.file_uploader("ارفع صورة لوحة كاندي كراش...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # فتح الصورة وتحويلها للصيغة المناسبة
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة الأصلية", use_container_width=True)

    # شريط التحكم بحساسية النموذج
    threshold = st.slider("نسبة الثقة (Threshold)", min_value=0.01, max_value=0.50, value=0.10, step=0.01)

    if st.button("🔍 ابحث عن العناصر"):
        with st.spinner("الذكاء الاصطناعي يبحث الآن..."):
            
            # تجهيز الصورة والنصوص للنموذج
            inputs = processor(text=[queries], images=image, return_tensors="pt")
            
            # تنفيذ البحث (بدون حسابات التدريب لتسريع العملية)
            with torch.no_grad():
                outputs = model(**inputs)

            # معالجة النتائج وتحديد حجم الصورة لضبط المربعات
            target_sizes = torch.tensor([image.size[::-1]])
            
            # تم تصحيح هذا السطر بإضافة image_processor
            results = processor.image_processor.post_process_object_detection(
                outputs=outputs, 
                target_sizes=target_sizes, 
                threshold=threshold
            )
            
            # أخذ نتائج أول صورة (لأننا رفعنا صورة واحدة)
            i = 0
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
            
            # رسم المربعات على الصورة
            draw = ImageDraw.Draw(image)
            count = 0
            
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                label_name = queries[label.item()]
                confidence = round(score.item(), 2)
                
                # رسم المربع بخط عريض (أحمر)
                draw.rectangle(box, outline="red", width=4)
                
                # كتابة اسم العنصر ونسبة الثقة فوقه
                draw.text((box[0], box[1] - 15), f"{label_name}: {confidence}", fill="white")
                count += 1
                
            # عرض النتيجة النهائية
            st.image(image, caption="اللوحة بعد التحليل النصي", use_container_width=True)
            st.success(f"تم اكتشاف {count} عنصر بناءً على الوصف!")
