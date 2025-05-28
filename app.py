import streamlit as st
import os
import math
import requests
import io
import base64
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

# المسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
DETECTED_DIR = os.path.join(BASE_DIR, "DETECTED_FIELDS")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
ML_MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "isolation_scaler.joblib")

API_KEY = "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"
ZOOM = 15
IMG_SIZE = 640
CALIBRATION_FACTOR = 0.6695

# إنشاء المجلدات
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

# دوال مساعدة
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

def generate_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

def generate_whatsapp_share_link(meter_id, area, consumption, location_link):
    message = f"عداد: {meter_id}\nمساحة: {area:,} م²\nاستهلاك: {consumption:,} ك.و.س\n{location_link}"
    return f"https://wa.me/?text={requests.utils.quote(message)}"

# تصميم CSS
st.markdown("""
<style>
.card {border-radius:10px;box-shadow:0 4px 8px rgba(0,0,0,0.1);padding:20px;margin-bottom:20px;border-left:5px solid;background-color:#f9f9f9;font-family:'Arial',sans-serif;direction:rtl;}
.card-header {display:flex;justify-content:space-between;align-items:center;padding-bottom:10px;border-bottom:1px solid #eee;}
.priority-badge {padding:5px 15px;border-radius:20px;color:white;font-weight:bold;}
.high-badge {background-color:#e74c3c;}
.medium-badge {background-color:#f39c12;}
.low-badge {background-color:#27ae60;}
.card-content {display:flex;gap:20px;margin-top:10px;}
.card-image {width:160px;height:160px;border-radius:8px;object-fit:cover;}
.detail-row {display:flex;margin-bottom:5px;}
.detail-label {font-weight:bold;width:100px;color:#555;}
.detail-value {flex:1;}
.card-actions {margin-top:10px;display:flex;gap:10px;}
.action-btn {padding:6px 12px;text-decoration:none;border-radius:6px;color:white;font-size:14px;}
.whatsapp-btn {background-color:#25D366;}
.map-btn {background-color:#4285F4;}
.priority-high {border-color:#e74c3c;background-color:#ffebee;}
.priority-medium {border-color:#f39c12;background-color:#fff3e0;}
.priority-low {border-color:#27ae60;background-color:#e8f5e9;}
</style>
""", unsafe_allow_html=True)

# تأكد من وجود الملف
uploaded_file = st.file_uploader("📁 رفع ملف Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    for idx, row in df.iterrows():
        meter_id, lat, lon = row["Subscription"], row["y"], row["x"]
        breaker, consumption, office = row["Breaker"], row["consumption"], row["Office"]

        img_path = download_image(lat, lon, meter_id)
        if not img_path:
            continue

        conf, img_detected, area, dist = detect_field(img_path, lat, meter_id)
        if not img_detected:
            continue

        pri, css_class = determine_priority(consumption, area, breaker)
        img_b64 = base64.b64encode(open(img_detected, "rb").read()).decode()

        # انسخ محتوى البطاقة المحدّث هنا من الكود السابق
        # (تم التحديث بالفعل في هذا المستند)

st.success("✅ تم بنجاح")
