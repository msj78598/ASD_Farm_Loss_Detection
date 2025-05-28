import streamlit as st
import os, math, requests, io, base64
import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO
import joblib

# إعدادات Streamlit
st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد الزراعي",
    layout="wide",
    page_icon="🌾"
)

# إعدادات المسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
DETECTED_DIR = os.path.join(BASE_DIR, "DETECTED_FIELDS")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

MODEL_PATH = os.path.join(BASE_DIR, "models", "last.pt")
ML_MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "isolation_scaler.joblib")
API_KEY = "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"
ZOOM = 15  # تعديل الزوم حسب طلبك
IMG_SIZE = 640
CALIBRATION_FACTOR = 0.6695

for folder in [IMG_DIR, DETECTED_DIR, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# تحميل النماذج
@st.cache_resource
def load_models():
    model_yolo = YOLO(MODEL_PATH)
    model_ml = joblib.load(ML_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model_yolo, model_ml, scaler

model_yolo, model_ml, scaler = load_models()

# دوال المساعدة
def download_image(lat, lon, meter_id):
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={ZOOM}&size={IMG_SIZE}x{IMG_SIZE}&maptype=satellite&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        with open(img_path, "wb") as f:
            f.write(response.content)
        return img_path
    return None

def detect_field(img_path, lat, meter_id):
    image = Image.open(img_path).convert("RGB")
    results = model_yolo.predict(source=image, imgsz=IMG_SIZE, conf=0.3)[0]
    if len(results.boxes) == 0:
        return None, None, None, None
    box = results.boxes.xyxy[0].cpu().numpy()
    conf = float(results.boxes.conf[0].cpu())
    center_x, center_y = (box[0]+box[2])/2, (box[1]+box[3])/2
    dist_px = math.hypot(center_x - IMG_SIZE/2, center_y - IMG_SIZE/2)
    scale = 156543.03392 * math.cos(math.radians(lat)) / (2 ** ZOOM)
    real_dist = dist_px * scale
    area = abs(box[2]-box[0]) * abs(box[3]-box[1]) * (scale**2) * CALIBRATION_FACTOR
    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="lime", width=3)
    out_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
    image.save(out_path)
    return conf, out_path, int(area), int(real_dist)

def determine_priority(consumption, area, breaker):
    if area > 50000 and (breaker < 200 or consumption < 10000):
        return "قصوى", "high"
    elif consumption < 10000:
        return "عالية", "medium"
    else:
        return "منخفضة", "low"

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output) as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# تصميم CSS
st.markdown("""
<style>
.high{background:#ffebee;border-left:5px solid #f44336;}
.medium{background:#fff3e0;border-left:5px solid #ff9800;}
.low{background:#e8f5e9;border-left:5px solid #4caf50;}
</style>
""", unsafe_allow_html=True)

# واجهة المستخدم
st.title("🌾 نظام اكتشاف حالات الفاقد الزراعي")

uploaded_file = st.file_uploader("📁 رفع ملف البيانات Excel", type=["xlsx"])

sort_col = st.sidebar.selectbox("فرز حسب:", ["بدون", "consumption", "Breaker"])
sort_order = st.sidebar.radio("نوع الفرز:", ["تصاعدي", "تنازلي"], horizontal=True)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if sort_col != "بدون":
        df.sort_values(by=sort_col, ascending=(sort_order=="تصاعدي"), inplace=True)

    tab1, tab2 = st.tabs(["🎯 النتائج", "📊 البيانات الخام"])

    results = []
    with tab1:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, row in df.iterrows():
            progress_bar.progress((idx+1)/len(df))
            status_text.text(f"معالجة الحالة رقم {idx+1} من {len(df)}")
            meter_id, lat, lon = row["Subscription"], row["y"], row["x"]
            breaker, consumption = row["Breaker"], row["consumption"]
            office_number = row["Office"]

            img_path = download_image(lat, lon, meter_id)
            if not img_path:
                continue

            conf, img_detected, area, dist = detect_field(img_path, lat, meter_id)
            if not img_detected:
                continue

            pri, css_class = determine_priority(consumption, area, breaker)
            with open(img_detected, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            # البطاقة الجديدة المعدلة بشكل احترافي
            st.markdown(f"""
            <div style='display:flex;align-items:center;border-radius:12px;padding:10px;box-shadow:0 4px 8px rgba(0,0,0,0.1);background-color:#fff;margin-bottom:15px;border-left:5px solid {{"high":"#f44336","medium":"#ff9800","low":"#4caf50"}[css_class]};'>
                <img src='data:image/png;base64,{img_b64}' width='180' style='border-radius:8px;margin-left:15px;'/>
                <div style='flex:1;'>
                    <h4 style='margin:5px 0;font-size:16px;'>🔢 عداد: {meter_id}</h4>
                    <p style='margin:3px;'>🏢 المكتب: {office_number}</p>
                    <p style='margin:3px;'>📐 المساحة: {area:,} م² | 📍 بعد العداد: {dist} م</p>
                    <p style='margin:3px;'>📊 الثقة: {conf*100:.1f}%</p>
                    <p style='margin:3px;'>💡 الاستهلاك: {consumption:,} ك.و.س | ⚡ القاطع: {breaker} أمبير</p>
                    <p style='margin:3px;font-weight:bold;'>🚨 الأولوية: {pri}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            results.append({
                "عداد": meter_id, 
                "مكتب": office_number,
                "مساحة": area, 
                "مسافة (م)": dist,
                "ثقة": conf, 
                "استهلاك": consumption, 
                "قاطع": breaker, 
                "أولوية": pri
            })

        status_text.text("✅ اكتملت المعالجة!")

    with tab2:
        st.dataframe(df)

    if results:
        df_results = pd.DataFrame(results)
        st.sidebar.download_button("📥 تصدير النتائج Excel",
                                   data=to_excel(df_results),
                                   file_name="نتائج_الفحص.xlsx")
        st.sidebar.metric("🔴 حالات قصوى", (df_results["أولوية"]=="قصوى").sum())
        st.sidebar.metric("🟠 حالات عالية", (df_results["أولوية"]=="عالية").sum())
        st.sidebar.metric("🟢 حالات منخفضة", (df_results["أولوية"]=="منخفضة").sum())
else:
    st.info("⬆️ الرجاء رفع ملف البيانات للبدء")
