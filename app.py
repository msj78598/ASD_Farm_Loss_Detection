# استيراد مكتبة Streamlit أولًا
import streamlit as st

# إعداد الصفحة يجب أن يكون أول أمر Streamlit
st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية",
    layout="wide",
    page_icon="🌾"
)

# استيراد المكتبات بعد set_page_config
import os
import math
import requests
from pathlib import Path
from PIL import Image, ImageDraw
import joblib
import pandas as pd
import numpy as np
import base64
import io
from ultralytics import YOLO

# ---------------- إعدادات المسارات ----------------
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

# ---------------- تحميل النماذج ----------------
@st.cache_resource
def load_models():
    model_yolo = YOLO(MODEL_PATH)
    model_ml = joblib.load(ML_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model_yolo, model_ml, scaler

model_yolo, model_ml, scaler = load_models()

# ---------------- دوال المساعدة ----------------
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
    message = f"عداد: {meter_id}\nمساحة: {area:,} م²\nاستهلاك: {consumption:,} ك.و.س\n{location_link}"
    return f"https://wa.me/?text={requests.utils.quote(message)}"

# ---------------- إعدادات CSS ----------------
st.markdown("""
<style>
.main {direction: rtl; text-align: right; font-family: Arial, sans-serif;}
.header {background-color: #2c3e50; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 30px;}
.card {border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 25px; border-left: 5px solid; display: flex; gap: 25px;}
.priority-high {border-color: #ff0000; background-color: #ffebee;}
.priority-medium {border-color: #ffa500; background-color: #fff3e0;}
.priority-low {border-color: #008000; background-color: #e8f5e9;}
.card img {border-radius: 8px; border: 1px solid #ddd; width: 300px;}
.details {flex: 1;}
.action-btn {padding: 8px 15px; border-radius: 5px; font-weight: bold;}
.whatsapp {background-color: #25D366; color: white;}
.map {background-color: #4285F4; color: white;}
</style>
""", unsafe_allow_html=True)

# ---------------- واجهة المستخدم ----------------
st.markdown("""
<div class="header">
    <h1>🌾 نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية</h1>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.dataframe(df.head())  # مثال للعرض الأولي، تابع إضافة باقي الكود للمعالجة والعرض بشكل كامل.
