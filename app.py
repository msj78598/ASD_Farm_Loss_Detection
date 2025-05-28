# الدفعة الأولى: الإعدادات العامة وتحميل النماذج ودوال المساعدة

import os
import math
import requests
from pathlib import Path
from PIL import Image, ImageDraw
import streamlit as st
import joblib
from ultralytics import YOLO

# ---------------------- إعدادات عامة ----------------------
st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية",
    layout="wide",
    page_icon="🌾"
)

# ---------------------- إعدادات المسارات ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
DETECTED_DIR = os.path.join(BASE_DIR, "DETECTED_FIELDS")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

MODEL_PATH = os.path.join(BASE_DIR, "models", "last.pt")
ML_MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "isolation_scaler.joblib")

CALIBRATION_FACTOR = 0.6695
ZOOM = 18
IMG_SIZE = 640
API_KEY = "API_KEY"

# إنشاء المجلدات الضرورية
for path in [IMG_DIR, DETECTED_DIR, OUTPUT_FOLDER]:
    os.makedirs(path, exist_ok=True)

# ---------------------- تحميل النماذج ----------------------
@st.cache_resource
def load_models():
    model_yolo = YOLO(MODEL_PATH)
    model_ml = joblib.load(ML_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model_yolo, model_ml, scaler

# ---------------------- دوال المساعدة ----------------------
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

def detect_field(img_path, lat, meter_id, model_yolo):
    image = Image.open(img_path).convert("RGB")
    results = model_yolo.predict(source=image, imgsz=IMG_SIZE, conf=0.5)[0]
    if not results.boxes:
        return None, None, None
    box = results.boxes[0].xyxy[0].cpu().numpy()
    conf = float(results.boxes[0].conf.cpu().numpy())
    box_center_x = (box[0] + box[2]) / 2
    box_center_y = (box[1] + box[3]) / 2
    dist = math.sqrt((box_center_x - IMG_SIZE/2)**2 + (box_center_y - IMG_SIZE/2)**2)
    scale = 156543.03392 * math.cos(math.radians(lat)) / (2 ** ZOOM)
    real_dist = dist * scale
    if real_dist > 500:
        return None, None, None
    area = abs(box[2] - box[0]) * abs(box[3] - box[1]) * (scale**2) * CALIBRATION_FACTOR
    if area < 5000:
        return None, None, None
    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    out_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
    image.save(out_path)
    return round(conf * 100, 2), out_path, int(area)

def generate_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

def generate_whatsapp_share_link(meter_id, area, consumption, location_link):
    message = f"عداد: {meter_id}\\nمساحة: {area:,} م²\\nاستهلاك: {consumption:,} ك.و.س\\n{location_link}"
    return f"https://wa.me/?text={requests.utils.quote(message)}"

# الدفعة الأولى: الإعدادات العامة وتحميل النماذج ودوال المساعدة

import os
import math
import requests
from pathlib import Path
from PIL import Image, ImageDraw
import streamlit as st
import joblib
from ultralytics import YOLO

# ---------------------- إعدادات عامة ----------------------
st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية",
    layout="wide",
    page_icon="🌾"
)

# ---------------------- إعدادات المسارات ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
DETECTED_DIR = os.path.join(BASE_DIR, "DETECTED_FIELDS")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

MODEL_PATH = os.path.join(BASE_DIR, "models", "last.pt")
ML_MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "isolation_scaler.joblib")

CALIBRATION_FACTOR = 0.6695
ZOOM = 18
IMG_SIZE = 640
API_KEY = "API_KEY"

# إنشاء المجلدات الضرورية
for path in [IMG_DIR, DETECTED_DIR, OUTPUT_FOLDER]:
    os.makedirs(path, exist_ok=True)

# ---------------------- تحميل النماذج ----------------------
@st.cache_resource
def load_models():
    model_yolo = YOLO(MODEL_PATH)
    model_ml = joblib.load(ML_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model_yolo, model_ml, scaler

# ---------------------- دوال المساعدة ----------------------
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

def detect_field(img_path, lat, meter_id, model_yolo):
    image = Image.open(img_path).convert("RGB")
    results = model_yolo.predict(source=image, imgsz=IMG_SIZE, conf=0.5)[0]
    if not results.boxes:
        return None, None, None
    box = results.boxes[0].xyxy[0].cpu().numpy()
    conf = float(results.boxes[0].conf.cpu().numpy())
    box_center_x = (box[0] + box[2]) / 2
    box_center_y = (box[1] + box[3]) / 2
    dist = math.sqrt((box_center_x - IMG_SIZE/2)**2 + (box_center_y - IMG_SIZE/2)**2)
    scale = 156543.03392 * math.cos(math.radians(lat)) / (2 ** ZOOM)
    real_dist = dist * scale
    if real_dist > 500:
        return None, None, None
    area = abs(box[2] - box[0]) * abs(box[3] - box[1]) * (scale**2) * CALIBRATION_FACTOR
    if area < 5000:
        return None, None, None
    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    out_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
    image.save(out_path)
    return round(conf * 100, 2), out_path, int(area)

def generate_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

def generate_whatsapp_share_link(meter_id, area, consumption, location_link):
    message = f"عداد: {meter_id}\\nمساحة: {area:,} م²\\nاستهلاك: {consumption:,} ك.و.س\\n{location_link}"
    return f"https://wa.me/?text={requests.utils.quote(message)}"

# الدفعة الثالثة: حلقة معالجة البيانات وعرض النتائج بشكل متقدم، مع ميزة التصدير وإحصائيات الشريط الجانبي

import streamlit as st
import pandas as pd
import numpy as np
import base64
import io

# افترض أن الدوال التالية تم تعريفها من الدفعة الأولى:
# load_models, download_image, detect_field, generate_google_maps_link, generate_whatsapp_share_link

model_yolo, model_ml, scaler = load_models()
results = []

if uploaded_file:
    progress_bar = st.progress(0)
    status_text = st.empty()

    with tab1:
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

            location_link = generate_google_maps_link(lat, lon)
            whatsapp_link = generate_whatsapp_share_link(meter_id, area, consumption, location_link)

            results.append({
                "meter_id": meter_id,
                "confidence": conf,
                "area": area,
                "consumption": consumption,
                "breaker": breaker,
                "location_link": location_link,
                "whatsapp_link": whatsapp_link,
                "img_detected": img_detected
            })

            with open(img_detected, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()

            priority_class = "priority-high" if consumption < 10000 else "priority-medium" if consumption < 20000 else "priority-low"

            st.markdown(f"""
            <div class="card {priority_class}">
                <img src="data:image/png;base64,{img_base64}"/>
                <div class="details">
                    <h4>🔢 العداد: {meter_id}</h4>
                    <p>📊 نسبة الثقة: {conf}%</p>
                    <p>📐 المساحة: {area:,} م²</p>
                    <p>💡 الاستهلاك: {consumption:,} ك.و.س</p>
                    <p>⚡ القاطع: {breaker} أمبير</p>
                    <a href="{whatsapp_link}" class="action-btn whatsapp" target="_blank">📤 مشاركة واتساب</a>
                    <a href="{location_link}" class="action-btn map" target="_blank">📍 عرض الموقع</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

        status_text.text("✅ اكتملت معالجة جميع الحالات بنجاح!")

        # تصدير النتائج
        if results:
            df_results = pd.DataFrame(results)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df_results.to_excel(writer, index=False)
            st.sidebar.download_button("📥 تصدير النتائج كملف Excel", data=buffer.getvalue(), file_name="نتائج_الفحص.xlsx")

        # إحصائيات الشريط الجانبي
        st.sidebar.markdown("### 📊 إحصائيات سريعة")
        high_priority = sum(df_results["consumption"] < 10000)
        medium_priority = sum((df_results["consumption"] >= 10000) & (df_results["consumption"] < 20000))
        low_priority = sum(df_results["consumption"] >= 20000)

        st.sidebar.metric("🔴 حالات قصوى", high_priority)
        st.sidebar.metric("🟠 حالات متوسطة", medium_priority)
        st.sidebar.metric("🟢 حالات منخفضة", low_priority)

