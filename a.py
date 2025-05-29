import os
os.environ["YOLO_VERBOSE"] = "False"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

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
import torch

# ------------------------- إعدادات عامة -------------------------
st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد للفئة الزراعية",
    layout="wide",
    page_icon="🌾"
)

# ------------------------- المسارات الرئيسية -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
DETECTED_DIR = os.path.join(BASE_DIR, "DETECTED_FIELDS")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
CALIBRATION_FACTOR = 0.6695

for path in [IMG_DIR, DETECTED_DIR, OUTPUT_FOLDER]:
    os.makedirs(path, exist_ok=True)

# تحميل النموذج بنفس الطريقة السابقة
@st.cache_resource
def load_yolo():
    model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
    return model_yolo

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
        "key": "YOUR_API_KEY"
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        with open(img_path, "wb") as f:
            f.write(r.content)
        return img_path
    return None

def detect_field(img_path, lat, meter_id, model_yolo):
    image = Image.open(img_path).convert("RGB")
    results = model_yolo(image)
    detections = results.xyxy[0].cpu().numpy()

    if len(detections) == 0:
        return None, None, None

    box = detections[0][:4]
    conf = detections[0][4]

    if conf < 0.5:
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

# المعادلات والشروط الجديدة (متفق عليها)
def compute_final_confidence(area, breaker, consumption, yolo_confidence):
    expected_consumption = area * 1
    expected_breaker = area / 500

    consumption_ratio = consumption / expected_consumption
    breaker_ratio = breaker / expected_breaker

    consumption_risk = max(0, (1 - consumption_ratio)) * 100
    breaker_risk = max(0, (1 - breaker_ratio)) * 100

    case_risk = (consumption_risk * 0.7) + (breaker_risk * 0.3)

    final_confidence = (case_risk * 0.6) + (100 - case_risk) * (yolo_confidence / 100) * 0.4

    if final_confidence >= 80:
        priority = "🔴 قصوى"
        color = "crimson"
    elif final_confidence >= 60:
        priority = "🟠 عالية"
        color = "orange"
    elif final_confidence >= 40:
        priority = "🟡 تنبيه"
        color = "gold"
    else:
        priority = "🟢 طبيعي"
        color = "green"

    return round(final_confidence, 2), priority, color

# =============================
# واجهة المستخدم
# =============================
st.title("🌾 نظام اكتشاف حالات الفاقد الكهربائي للفئة الزراعية")
uploaded_file = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.lower().str.strip()

    model_yolo = load_yolo()

    for idx, row in df.iterrows():
        meter_id, lat, lon = row["subscription"], row["y"], row["x"]
        breaker, consumption = row["breaker"], row["consumption"]
        
        img_path = download_image(lat, lon, meter_id)
        if not img_path:
            continue

        yolo_conf, img_detected, area = detect_field(img_path, lat, meter_id, model_yolo)
        if yolo_conf is None:
            continue

        final_confidence, priority, color = compute_final_confidence(area, breaker, consumption, yolo_conf)

        with open(img_detected, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()

        st.markdown(f'''
        <div style="border:2px solid {color};padding:15px;border-radius:10px;margin-bottom:20px;">
            <img src="data:image/png;base64,{img_data}" width="300" />
            <p>🔢 العداد: {meter_id}</p>
            <p>📐 المساحة: {area:,} م²</p>
            <p>💡 الاستهلاك: {consumption:,} ك.و.س</p>
            <p>⚡ القاطع: {breaker} أمبير</p>
            <p>📊 نسبة الثقة: {final_confidence}%</p>
            <p>🚨 الأولوية: {priority}</p>
        </div>
        ''', unsafe_allow_html=True)
