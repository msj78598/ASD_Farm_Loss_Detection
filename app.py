import os
import math
import io
import pandas as pd
import requests
from pathlib import Path
from PIL import Image, ImageDraw
import streamlit as st
import urllib.parse
import base64
import numpy as np
import joblib
from ultralytics import YOLO

# إعدادات عامة
st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد للفئة الزراعية",
    layout="wide",
    page_icon="🌾"
)

# المسارات الرئيسية (احتفظ بنفس مسارات النظام الحالي)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
DETECTED_DIR = os.path.join(BASE_DIR, "DETECTED_FIELDS")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(BASE_DIR, "models", "last.pt")
ML_MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "isolation_scaler.joblib")
CALIBRATION_FACTOR = 0.6695

Path(IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(DETECTED_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# تحميل النماذج
@st.cache_resource
def load_models():
    model_yolo = YOLO(MODEL_PATH)
    model_ml = joblib.load(ML_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model_yolo, model_ml, scaler

# تحميل صورة
ZOOM = 18
IMG_SIZE = 640
API_KEY = "API_KEY"

def download_image(lat, lon, meter_id):
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM,
        "size": "640x640",
        "maptype": "satellite",
        "markers": f"color:red|label:X|{lat},{lon}",
        "key": API_KEY
    }
    response = requests.get(url)
    if response.status_code == 200:
        with open(img_path, "wb") as f:
            f.write(response.content)
        return img_path
    return None

# دالة اكتشاف الحقول مع التحقق من المسافة
def detect_field(img_path, lat, meter_id, model_yolo):
    image = Image.open(img_path).convert("RGB")
    results = model_yolo.predict(source=image, imgsz=640, conf=0.5)[0]
    if not results.boxes:
        return None, None, None
    box = results.boxes[0].xyxy[0].cpu().numpy()
    conf = float(results.boxes[0].conf.cpu().numpy())

    box_center_x = (box[0] + box[2]) / 2
    box_center_y = (box[1] + box[3]) / 2
    dist = math.sqrt((box_center_x - IMG_SIZE/2)**2 + (box_center_y - IMG_SIZE/2)**2)
    scale = 156543.03392 * math.cos(math.radians(lat)) / (2 ** ZOOM)
    real_distance = dist * scale
    if real_distance > 500:
        return None, None, None

    area = abs(box[2] - box[0]) * abs(box[3] - box[1]) * (scale ** 2) * CALIBRATION_FACTOR
    if area < 5000:
        return None, None, None

    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    out_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
    image.save(out_path)
    return round(conf * 100, 2), out_path, int(area)

# واجهة المستخدم
st.title("🌾 نظام اكتشاف حالات الفاقد الكهربائي للفئة الزراعية")
uploaded_file = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    model_yolo, model_ml, scaler = load_models()
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, row in df.iterrows():
        progress_bar.progress((idx + 1) / len(df))
        status_text.text(f"جاري معالجة الحالة رقم {idx + 1} من {len(df)}...")

        meter_id, lat, lon = row["Subscription"], row["y"], row["x"]
        breaker, consumption = row["Breaker"], row["consumption"]

        img_path = download_image(lat, lon, meter_id)
        if not img_path:
            continue

        conf, img_detected, area = detect_field(img_path, lat, meter_id, model_yolo)
        if conf is None:
            continue

        location_link = f"https://www.google.com/maps?q={lat},{lon}"

        with open(img_detected, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()

        st.markdown(f"""
        <div style="border:2px solid #ddd;padding:15px;border-radius:12px;margin-bottom:20px;">
            <img src="data:image/png;base64,{img_base64}" width="300px"/>
            <p>🔢 العداد: {meter_id}</p>
            <p>📊 نسبة الثقة: {conf}%</p>
            <p>📐 المساحة: {area:,} م²</p>
            <p>💡 الاستهلاك: {consumption:,}</p>
            <p>⚡ القاطع: {breaker}</p>
            <a href="{location_link}">📍 عرض الموقع</a>
        </div>
        """, unsafe_allow_html=True)

    status_text.text("✅ اكتملت معالجة جميع الحالات بنجاح!")
