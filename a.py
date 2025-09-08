import os
import math
import base64
import requests
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import joblib
from ultralytics import YOLO
from geopy.distance import geodesic

# إعدادات عامة
st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد للفئة الزراعية",
    layout="wide",
    page_icon="🌾"
)

# المسارات الرئيسية
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
DETECTED_DIR = os.path.join(BASE_DIR, "DETECTED_FIELDS")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
ML_MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "isolation_scaler.joblib")
FORM_PATH = os.path.join(BASE_DIR, "TEMPLATE.xlsx")
CALIBRATION_FACTOR = 0.6695

for path in [IMG_DIR, DETECTED_DIR, OUTPUT_FOLDER]:
    os.makedirs(path, exist_ok=True)

@st.cache_resource
def load_models():
    model_yolo = YOLO(MODEL_PATH)
    model_ml = joblib.load(ML_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model_yolo, model_ml, scaler

def download_image(lat, lon, meter_id):
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": 15,
        "size": "640x640",
        "maptype": "satellite",
        "markers": f"color:red|label:X|{lat},{lon}",
        "key": "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"
    }
    response = requests.get(url, params=params, timeout=15)
    if response.status_code == 200:
        with open(img_path, "wb") as f:
            f.write(response.content)
        return img_path
    return None

def detect_field(img_path, lat, lon, meter_id, model_yolo):
    image = Image.open(img_path).convert("RGB")
    results = model_yolo.predict(source=image, imgsz=640, conf=0.5)[0]
    if not results.boxes:
        return None, None, None, None
    box = results.boxes[0].xyxy[0].cpu().numpy()
    conf = float(results.boxes[0].conf.cpu().numpy())
    if conf < 0.9:
        return None, None, None, None
    scale = 156543.03392 * math.cos(math.radians(lat)) / (2 ** 16)
    area = abs(box[2] - box[0]) * abs(box[3] - box[1]) * (scale ** 2)
    corrected_area = area * CALIBRATION_FACTOR
    if corrected_area < 5000:
        return None, None, None, None

    img_center_pixel = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    dx = (img_center_pixel[0] - 320) * scale
    dy = (img_center_pixel[1] - 320) * scale

    field_lat = lat - (dy / 111320)
    field_lon = lon + (dx / (40075000 * math.cos(math.radians(lat)) / 360))

    distance = geodesic((lat, lon), (field_lat, field_lon)).meters
    if distance > 400:
        return None, None, None, None

    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    draw.line([(320, 320), img_center_pixel], fill="yellow", width=2)
    out_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
    image.save(out_path)
    return round(conf * 100, 2), out_path, int(corrected_area), round(distance, 2)

st.title("🌾111 نظام اكتشاف حالات الفاقد الكهربائي للفئة الزراعية")
st.download_button("📥 تحميل نموذج البيانات (TEMPLATE.xlsx)", open(FORM_PATH, "rb"), file_name="TEMPLATE.xlsx")

uploaded_file = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.dropna(subset=["Subscription", "Office", "Breaker", "consumption", "x", "y"], inplace=True)

    breaker_options = ["الكل"] + sorted(df["Breaker"].unique().tolist())
    breaker_filter = st.sidebar.selectbox("سعة القاطع", breaker_options)
    sort_order = st.sidebar.radio("ترتيب حسب الاستهلاك", ["بدون ترتيب", "تصاعدي", "تنازلي"])

    if breaker_filter != "الكل":
        df = df[df["Breaker"] == breaker_filter]

    if sort_order == "تصاعدي":
        df = df.sort_values(by="consumption", ascending=True)
    elif sort_order == "تنازلي":
        df = df.sort_values(by="consumption", ascending=False)

    model_yolo, model_ml, scaler = load_models()

    progress_bar = st.progress(0)
    progress_text = st.empty()

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        meter_id, lat, lon = row["Subscription"], row["y"], row["x"]
        breaker, consumption, office = row["Breaker"], row["consumption"], row["Office"]

        progress_text.text(f"جاري تحليل العداد رقم {meter_id} ({i}/{total})")

        img_path = download_image(lat, lon, meter_id)
        if not img_path:
            continue

        conf, img_detected, area, distance = detect_field(img_path, lat, lon, meter_id, model_yolo)
        if conf is None:
            continue

        anomaly = model_ml.predict(scaler.transform([[breaker, consumption, lon, lat]]))[0]
        confidence = (breaker < area * 0.006) * 0.4 + (consumption < area * 0.4) * 0.4 + (anomaly == 1) * 0.2
        priority = "قصوى" if confidence >= 0.7 else "متوسطة" if confidence >= 0.4 else "منخفضة"
        color = {"قصوى": "crimson", "متوسطة": "orange", "منخفضة": "green"}[priority]

        with open(img_detected, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode()

        map_link = f"https://www.google.com/maps?q={lat},{lon}"

        st.markdown(f"""
        <div style='border:2px solid {color};padding:10px;border-radius:10px;display:flex;align-items:center;margin-bottom:10px;'>
            <img src="data:image/png;base64,{encoded_img}" width="300px" style="border-radius:10px;margin-left:15px;">
            <div style="padding-right:20px;text-align:right;">
                <h4 style='color:{color};'>عداد: {meter_id} ({priority})</h4>
                نسبة الثقة: {confidence*100:.2f}%<br>
                المسافة: {distance} متر<br>
                المساحة: {area} م²<br>
                الاستهلاك: {consumption} ك.و.س<br>
                القاطع: {breaker} أمبير<br>
                المكتب: {office}<br>
                <a href="{map_link}" target="_blank">📍 عرض الموقع</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        progress_bar.progress(i / total)

    progress_text.text("✅ تم الانتهاء من التحليل.")
else:
    st.warning("يرجى رفع ملف Excel يحتوي على البيانات المطلوبة.")

    st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس | 00966553339838 | ")
st.markdown("👨‍💻 **تحديث :**  8-9-2025 |  | ")
