import os
import math
import io
import base64
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import streamlit as st
import joblib
from ultralytics import YOLO
import urllib.parse
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
CALIBRATION_FACTOR = 0.6695

for path in [IMG_DIR, DETECTED_DIR, OUTPUT_FOLDER]:
    os.makedirs(path, exist_ok=True)

@st.cache_resource
def load_models():
    model_yolo = YOLO(MODEL_PATH)
    model_ml = joblib.load(ML_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model_yolo, model_ml, scaler

# تحميل الصورة
def download_image(lat, lon, meter_id):
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path

    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": 16,
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

# كشف الحقول
def detect_field(img_path, lat, meter_id, model_yolo):
    image = Image.open(img_path).convert("RGB")
    results = model_yolo.predict(source=image, imgsz=640, conf=0.5)[0]

    if not results.boxes:
        return None, None, None

    box = results.boxes[0].xyxy[0].cpu().numpy()
    conf = float(results.boxes[0].conf.cpu().numpy())
    if conf < 0.9:
        return None, None, None

    scale = 156543.03392 * math.cos(math.radians(lat)) / (2 ** 16)
    area = abs(box[2] - box[0]) * abs(box[3] - box[1]) * (scale ** 2)
    corrected_area = area * CALIBRATION_FACTOR

    if corrected_area < 5000:
        return None, None, None

    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    out_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
    image.save(out_path)

    return round(conf * 100, 2), out_path, int(corrected_area)

# اختيار العداد الأقرب للحقل
def filter_closest(df, lat, lon):
    distances = df.apply(lambda row: geodesic((lat, lon), (row['y'], row['x'])).meters, axis=1)
    min_distance = distances.min()
    if min_distance <= 500:
        return df.iloc[distances.idxmin()]
    return None

# الواجهة
st.title("🌾 نظام اكتشاف حالات الفاقد الكهربائي للفئة الزراعية")
uploaded_file = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = df.dropna(subset=["Subscription", "Office", "Breaker", "consumption", "x", "y"])

    breaker_filter = st.sidebar.selectbox("عرض حسب سعة القاطع", ["الكل"] + sorted(df["Breaker"].unique().tolist()))
    min_consumption = st.sidebar.number_input("أقل استهلاك (ك.و.س)", min_value=0, value=0)

    if breaker_filter != "الكل":
        df = df[df["Breaker"] == breaker_filter]

    df = df[df["consumption"] >= min_consumption]
    st.sidebar.write(f"🔢 عدد العدادات: {len(df)}")

    model_yolo, model_ml, scaler = load_models()

    progress = st.progress(0)
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        meter_id, lat, lon = row["Subscription"], row["y"], row["x"]
        breaker, consumption, office = row["Breaker"], row["consumption"], row["Office"]

        img_path = download_image(lat, lon, meter_id)
        if not img_path:
            continue

        conf, img_detected, area = detect_field(img_path, lat, meter_id, model_yolo)
        if conf is None:
            continue

        anomaly = model_ml.predict(scaler.transform([[breaker, consumption, lon, lat]]))[0]

        confidence = (breaker < area * 0.006) * 0.4 + (consumption < area * 0.4) * 0.4 + (anomaly == 1) * 0.2
        priority = "عالية" if confidence >= 0.7 else "متوسطة" if confidence >= 0.4 else "منخفضة"
        color = {"عالية": "crimson", "متوسطة": "orange", "منخفضة": "green"}[priority]

        st.markdown(f"""
        <div style='border:2px solid {color}; padding:10px; margin-bottom:10px; border-radius:10px;'>
            <h4 style='color:{color};'>عداد: {meter_id}</h4>
            نسبة الثقة: {confidence * 100:.2f}%<br>
            المساحة: {area} م²<br>
            الاستهلاك: {consumption} ك.و.س<br>
            القاطع: {breaker} أمبير<br>
            المكتب: {office}
        </div>
        """, unsafe_allow_html=True)

        progress.progress(i / total)
else:
    st.warning("يرجى رفع ملف Excel يحتوي على البيانات المطلوبة.")
