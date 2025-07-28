import os
import math
import base64
import requests
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import joblib
import sys
import time
from io import BytesIO
sys.modules['cv2'] = __import__('cv2')
from ultralytics import YOLO
from geopy.distance import geodesic

st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد للفئة الزراعية",
    layout="wide",
    page_icon="🌾"
)

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

    # تعديل لحساب المسافة من حافة الحقل إلى العداد
    width_px = abs(box[2] - box[0])
    height_px = abs(box[3] - box[1])
    radius_px = max(width_px, height_px) / 2
    radius_m = radius_px * scale
    center_distance = geodesic((lat, lon), (field_lat, field_lon)).meters
    edge_distance = max(center_distance - radius_m, 0)

    if edge_distance > 100:
        return None, None, None, None

    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    draw.line([(320, 320), img_center_pixel], fill="yellow", width=2)
    out_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
    image.save(out_path)
    return round(conf * 100, 2), out_path, int(corrected_area), round(edge_distance, 2)

st.title("🌾 نظام اكتشاف حالات الفاقد الكهربائي للفئة الزراعية")
st.download_button("📥 تحميل نموذج البيانات (TEMPLATE.xlsx)", open(FORM_PATH, "rb"), file_name="TEMPLATE.xlsx")

uploaded_file = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.dropna(subset=["Subscription", "Office", "Breaker", "consumption", "x", "y"], inplace=True)

    breaker_filter = st.sidebar.selectbox("سعة القاطع", ["الكل"] + sorted(df["Breaker"].unique().tolist()))
    sort_order = st.sidebar.radio("ترتيب حسب الاستهلاك", ["بدون ترتيب", "تصاعدي", "تنازلي"])

    if breaker_filter != "الكل":
        df = df[df["Breaker"] == breaker_filter]

    if sort_order == "تصاعدي":
        df = df.sort_values(by="consumption", ascending=True)
    elif sort_order == "تنازلي":
        df = df.sort_values(by="consumption", ascending=False)

    st.sidebar.info(f"🔢 عدد الحالات في الملف: {len(df)}")

    if st.sidebar.button("🚀 بدء التحليل"):
        model_yolo, model_ml, scaler = load_models()
        progress_bar = st.sidebar.progress(0)
        start_time = time.time()

        colors = {"قصوى": "#ff4d4d", "متوسطة": "#ffa500", "منخفضة": "#4CAF50"}
        results = []
        cols = st.columns(3)
        col_index = 0

        for i, (_, row) in enumerate(df.iterrows(), 1):
            meter_id, lat, lon = row["Subscription"], row["y"], row["x"]
            breaker, consumption, office = row["Breaker"], row["consumption"], row["Office"]
            img_path = download_image(lat, lon, meter_id)
            if not img_path:
                continue

            conf, img_detected, area, distance = detect_field(img_path, lat, lon, meter_id, model_yolo)
            if conf is None:
                continue

            anomaly = model_ml.predict(scaler.transform([[breaker, consumption, lon, lat]]))[0]
            confidence = (breaker < area * 0.006) * 0.4 + (consumption < area * 0.4) * 0.4 + (anomaly == 1) * 0.2
            priority = "قصوى" if confidence >= 0.7 else "متوسطة" if confidence >= 0.4 else "منخفضة"
            border_color = colors.get(priority, "#cccccc")

            results.append([meter_id, priority, confidence, distance, area, consumption, breaker, office, lat, lon])

            with open(img_detected, "rb") as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode()

            cols[col_index % 3].markdown(f"""
            <div style="border:4px solid {border_color};padding:10px;border-radius:10px;margin:5px;text-align:center;">
                <img src="data:image/png;base64,{img_b64}" width="250" style="border-radius:8px;"><br>
                <strong>عداد {meter_id} ({priority})</strong><br>
                الثقة:{conf}% | المسافة:{distance}م | المساحة:{area}م²<br>
                الاستهلاك:{consumption} | القاطع:{breaker} | المكتب:{office}<br>
                <a href="https://maps.google.com?q={lat},{lon}">📍 الموقع</a>
                <a href="https://wa.me/?text=عداد:{meter_id}%20الموقع:{lat},{lon}">📲 واتساب</a>
            </div>
            """, unsafe_allow_html=True)
            col_index += 1
            progress_bar.progress(i / len(df))

        results_df = pd.DataFrame(results)
        buffer = BytesIO()
        results_df.to_excel(buffer, index=False)
        buffer.seek(0)
        st.sidebar.download_button("📥 تحميل النتائج Excel", buffer, file_name="results.xlsx")

        html_results = "<html><head><meta charset='UTF-8'></head><body><div style='display:flex;flex-wrap:wrap;'>"
        for res in results:
            meter_id, priority, confidence, distance, area, consumption, breaker, office, lat, lon = res
            border_color = colors.get(priority, "#cccccc")
            img_detected = os.path.join(DETECTED_DIR, f"{meter_id}.png")
            with open(img_detected, "rb") as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode()

            html_results += f"""
            <div style='border:4px solid {border_color};padding:10px;border-radius:10px;margin:5px;text-align:center;'>
                <img src='data:image/png;base64,{img_b64}' width='250' style='border-radius:8px;'><br>
                <strong>عداد {meter_id} ({priority})</strong><br>
                الثقة: {confidence*100:.1f}% | المسافة: {distance}م | المساحة: {area}م²<br>
                الاستهلاك: {consumption} | القاطع: {breaker} | المكتب: {office}<br>
                <a href='https://maps.google.com?q={lat},{lon}'>📍 الموقع</a>
                <a href='https://wa.me/?text=عداد:{meter_id}%20الموقع:{lat},{lon}'>📲 واتساب</a>
            </div>
            """

        html_results += "</div></body></html>"

        st.sidebar.download_button(
            label="📥 تحميل التقرير الكامل HTML",
            data=html_results.encode('utf-8'),
            file_name='report.html',
            mime='text/html'
        )
        duration = time.time() - start_time
        st.sidebar.success(f"⏱️ اكتمل التحليل في {round(duration,2)} ثانية")
