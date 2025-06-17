# 🔧 نظام اكتشاف حالات الفاقد الكهربائي للفئة الزراعية - نسخة محدثة
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
from ultralytics import YOLO
import urllib.parse
from geopy.distance import geodesic


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
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(r.content)
            return img_path
    except:
        pass
    return None

def detect_field(img_path, lat, lon, meter_id, model_yolo):
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

    img_center_pixel = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    dx = (img_center_pixel[0] - 320) * scale
    dy = (img_center_pixel[1] - 320) * scale

    field_lat = lat - (dy / 111320)
    field_lon = lon + (dx / (40075000 * math.cos(math.radians(lat)) / 360))

    distance = geodesic((lat, lon), (field_lat, field_lon)).meters
    if distance > 400:
        return None, None, None

    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    out_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
    image.save(out_path)
    return round(conf * 100, 2), out_path, int(corrected_area)



def predict_anomaly(row, model_ml, scaler):
    X = [[row["breaker"], row["consumption"], row["x"], row["y"]]]
    X_scaled = scaler.transform(X)
    return model_ml.predict(X_scaled)[0]

def evaluate_case(area, breaker, consumption, anomaly):
    expected_amp = (area / 10000) * 85
    min_kwh_per_m2 = 0.4
    min_expected_kwh = area * min_kwh_per_m2
    messages = []
    if breaker < expected_amp * 0.6:
        messages.append("⚠️ احتمال توصيلة مباشرة")
    if consumption < min_expected_kwh:
        messages.append("⚠️ فاقد محتمل")
    if anomaly == 1:
        messages.append("🔺 حالة شاذة")
    return " / ".join(messages) if messages else "✅ طبيعية"

def compute_confidence(area, breaker, consumption, anomaly):
    score = 0
    if breaker < (area / 10000) * 85 * 0.6:
        score += 0.4
    if consumption < area * 0.4:
        score += 0.4
    if anomaly == 1:
        score += 0.2
    return round(score * 100, 2)

def generate_map_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

def generate_whatsapp_link(meter_id, conf, area, link, consumption, breaker, office, priority):
    msg = f"تقرير حالة عداد زراعي\\n\\nرقم العداد: {meter_id}\\nالمكتب: {office}\\nالتقييم: {priority}\\nنسبة الثقة: {conf}%\\nالمساحة: {area:,} م²\\nالاستهلاك: {consumption:,} ك.و.س\\nالقاطع: {breaker} أمبير\\nالموقع: {link}"
    return f"https://wa.me/?text={urllib.parse.quote(msg)}"

# =============================
# واجهة المستخدم
# =============================
st.title("🌾 نظام اكتشاف حالات الفاقد الكهربائي للفئة الزراعية")
uploaded_file = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df["subscription"] = df["Subscription"].astype(str).str.strip()
    df["office"] = df["Office"].astype(str)
    df["breaker"] = pd.to_numeric(df["Breaker"], errors="coerce")
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # ✅ فلترة مسبقة
    st.sidebar.header("⚙️ خيارات العرض والتحليل")
    breaker_filter = st.sidebar.selectbox("عرض حسب سعة القاطع", options=["الكل"] + sorted(df["breaker"].dropna().unique().astype(int).tolist()))
    consumption_min = st.sidebar.number_input("أقل استهلاك (ك.و.س)", min_value=0, value=0)

    if breaker_filter != "الكل":
        df = df[df["breaker"] == int(breaker_filter)]
    df = df[df["consumption"] >= consumption_min]

    st.sidebar.markdown("### 📊 الإحصائيات")
    st.sidebar.markdown(f"🔢 عدد العدادات: **{len(df)}**")

    model_yolo, model_ml, scaler = load_models()
    severe_count = 0
    medium_count = 0
    normal_count = 0

    for idx, row in df.iterrows():
        meter_id, lat, lon = row["subscription"], row["y"], row["x"]
        breaker, consumption, office = row["breaker"], row["consumption"], row["office"]
        img_path = download_image(lat, lon, meter_id)
        if not img_path:
            continue
        conf, img_detected, area = detect_field(img_path, lat, meter_id, model_yolo)
        if conf is None or area is None:
            continue

        anomaly = predict_anomaly(row, model_ml, scaler)
        evaluation = evaluate_case(area, breaker, consumption, anomaly)
        confidence_score = compute_confidence(area, breaker, consumption, anomaly)
        map_link = generate_map_link(lat, lon)
        wa_link = generate_whatsapp_link(meter_id, confidence_score, area, map_link, consumption, breaker, office, evaluation)

        if "فاقد" in evaluation or "شاذة" in evaluation:
            severe_count += 1
            color = "crimson"
        elif "توصيلة" in evaluation:
            medium_count += 1
            color = "orange"
        else:
            normal_count += 1
            color = "green"

        with open(img_detected, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()

        st.markdown(f'''
        <div style="display:flex; gap:20px; border:2px solid #ddd; padding:15px; border-radius:12px; margin-bottom:20px;">
            <div><img src="data:image/png;base64,{img_data}" width="300px" style="border-radius:10px; border:1px solid #aaa"/></div>
            <div style="flex:1;">
                <h4>🔢 العداد: {meter_id}</h4>
                <p>📊 <b>نسبة الثقة:</b> {confidence_score}%</p>
                <p>📐 <b>المساحة:</b> {area:,} م²</p>
                <p>💡 <b>الاستهلاك:</b> {consumption:,} ك.و.س</p>
                <p>⚡ <b>القاطع:</b> {breaker} أمبير</p>
                <p>🏢 <b>المكتب:</b> {office}</p>
                <p>🧠 <b>الأولوية:</b> <span style='color:{color}; font-weight:bold'>{evaluation}</span></p>
                <a href="{map_link}" target="_blank">📍 عرض الموقع</a> |
                <a href="{wa_link}" target="_blank">📤 مشاركة واتساب</a>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.sidebar.markdown(f"🚨 حالات حرجة: **{severe_count}**")
    st.sidebar.markdown(f"🟠 أولويات متوسطة: **{medium_count}**")
    st.sidebar.markdown(f"✅ طبيعية: **{normal_count}**")
else:
    st.warning("يرجى رفع ملف يحتوي على الأعمدة: Subscription, Office, Breaker, consumption, x, y")
