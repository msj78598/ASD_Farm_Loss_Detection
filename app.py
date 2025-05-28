import os
import math
import io
import pandas as pd
import requests
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import joblib
import streamlit as st
import urllib.parse
import base64

# ------------------------- إعدادات Streamlit -------------------------
st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية",
    layout="wide",
    page_icon="🌾"
)

# ------------------------- المسارات الرئيسية -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
DETECTED_DIR = os.path.join(BASE_DIR, "DETECTED_FIELDS")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

MODEL_PATH = os.path.join(BASE_DIR, "models", "last.pt")
ML_MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "isolation_scaler.joblib")

API_KEY = "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"
ZOOM = 18
IMG_SIZE = 640

# إنشاء المجلدات اللازمة
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DETECTED_DIR, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------------- تحميل النماذج -------------------------
@st.cache_resource
def load_models():
    model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False)
    model_ml = joblib.load(ML_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model_yolo, model_ml, scaler

model_yolo, model_ml, scaler = load_models()

# ------------------------- دوال المساعدة -------------------------
def download_image(lat, lon, meter_id):
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM,
        "size": f"{IMG_SIZE}x{IMG_SIZE}",
        "maptype": "satellite",
        "key": API_KEY
    }
    response = requests.get(url)
    if response.status_code == 200:
        with open(img_path, "wb") as f:
            f.write(response.content)
        return img_path
    return None

def detect_field(img_path, lat, meter_id):
    image = Image.open(img_path).convert("RGB")
    results = model_yolo(image)
    df_result = results.pandas().xyxy[0]
    fields = df_result[(df_result["confidence"] >= 0.5)]
    if fields.empty:
        return None, None, None
    nearest_field = fields.iloc[0]
    box = [nearest_field["xmin"], nearest_field["ymin"], nearest_field["xmax"], nearest_field["ymax"]]
    scale = 156543.03392 * abs(math.cos(math.radians(lat))) / (2 ** ZOOM)
    area = abs(box[2]-box[0]) * abs(box[3]-box[1]) * scale**2 * 0.6695
    if area < 5000:
        return None, None, None
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="green", width=3)
    out_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
    image.save(out_path)
    confidence = round(nearest_field["confidence"] * 100, 2)
    return confidence, out_path, int(area)

def generate_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

def generate_whatsapp_share_link(meter_id, area, consumption, location_link):
    message = f"عداد: {meter_id}\nمساحة: {area:,} م²\nاستهلاك: {consumption:,} ك.و.س\n{location_link}"
    return f"https://wa.me/?text={urllib.parse.quote(message)}"

# ------------------------- واجهة Streamlit -------------------------
st.markdown("<h1 style='text-align:center;'>🌾 نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    tab1, tab2 = st.tabs(["🎯 النتائج المباشرة", "📊 البيانات الخام"])

    results = []
    with tab1:
        progress_bar = st.progress(0)
        for idx, row in df.iterrows():
            meter_id, lat, lon = row["Subscription"], row["y"], row["x"]
            breaker, consumption = row["Breaker"], row["consumption"]

            img_path = download_image(lat, lon, meter_id)
            if not img_path:
                continue

            conf, img_detected, area = detect_field(img_path, lat, meter_id)
            if conf is None:
                continue

            location_link = generate_google_maps_link(lat, lon)
            whatsapp_link = generate_whatsapp_share_link(meter_id, area, consumption, location_link)

            results.append({
                "meter_id": meter_id,
                "confidence": conf,
                "area": area,
                "consumption": consumption,
                "breaker": breaker,
                "location_link": location_link,
                "whatsapp_link": whatsapp_link
            })

            with open(img_detected, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()

            st.markdown(f"""
            <div style='border:1px solid #ddd;padding:15px;border-radius:10px;margin-bottom:15px;'>
                <img src='data:image/png;base64,{img_base64}' width='300'>
                <p><b>عداد:</b> {meter_id}</p>
                <p><b>الثقة:</b> {conf}%</p>
                <p><b>المساحة:</b> {area:,} م²</p>
                <p><b>الاستهلاك:</b> {consumption:,} ك.و.س</p>
                <a href='{whatsapp_link}' target='_blank'>📤 واتساب</a> |
                <a href='{location_link}' target='_blank'>📍 خريطة</a>
            </div>
            """, unsafe_allow_html=True)

            progress_bar.progress((idx + 1) / len(df))

    with tab2:
        st.dataframe(df)

    if results:
        df_results = pd.DataFrame(results)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_results.to_excel(writer, index=False)
        st.sidebar.download_button("📥 تصدير النتائج كملف Excel", data=buffer.getvalue(), file_name="نتائج_الفحص.xlsx")

        st.sidebar.markdown("### 📊 إحصائيات")
        st.sidebar.metric("🔢 عدد الحالات", len(results))

else:
    st.info("يرجى رفع ملف البيانات للبدء.")

