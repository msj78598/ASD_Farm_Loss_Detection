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
from concurrent.futures import ThreadPoolExecutor

# إعدادات عامة
st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد الزراعي",
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
    return YOLO(MODEL_PATH), joblib.load(ML_MODEL_PATH), joblib.load(SCALER_PATH)

model_yolo, model_ml, scaler = load_models()

# تحميل صورة من Google Maps
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

# اكتشاف الحقول باستخدام YOLO
@st.cache_data
def detect_field(img_path, lat, lon, meter_id):
    image = Image.open(img_path).convert("RGB")
    results = model_yolo.predict(source=image, imgsz=640, conf=0.5)[0]
    if not results.boxes:
        return None

    box = results.boxes[0].xyxy[0].cpu().numpy()
    conf = float(results.boxes[0].conf.cpu().numpy())
    if conf < 0.9:
        return None

    scale = 156543.03392 * math.cos(math.radians(lat)) / (2 ** 16)
    area = abs(box[2] - box[0]) * abs(box[3] - box[1]) * (scale ** 2)
    corrected_area = area * CALIBRATION_FACTOR

    if corrected_area < 5000:
        return None

    img_center_pixel = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    dx = (img_center_pixel[0] - 320) * scale
    dy = (img_center_pixel[1] - 320) * scale

    field_lat = lat - (dy / 111320)
    field_lon = lon + (dx / (40075000 * math.cos(math.radians(lat)) / 360))

    distance = geodesic((lat, lon), (field_lat, field_lon)).meters
    if distance > 400:
        return None

    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    out_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
    image.save(out_path)

    return round(conf * 100, 2), out_path, int(corrected_area), round(distance, 2)

# تحميل نموذج Excel
st.title("🌾 نظام اكتشاف حالات الفاقد الزراعي")
st.download_button("📥 تحميل نموذج البيانات", open(FORM_PATH, "rb"), file_name="TEMPLATE.xlsx")

uploaded_file = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.dropna(subset=["Subscription", "Office", "Breaker", "consumption", "x", "y"], inplace=True)

    breaker_filter = st.sidebar.selectbox("سعة القاطع", ["الكل"] + sorted(df["Breaker"].unique()))
    sort_order = st.sidebar.radio("ترتيب حسب الاستهلاك", ["بدون ترتيب", "تصاعدي", "تنازلي"])

    if breaker_filter != "الكل":
        df = df[df["Breaker"] == breaker_filter]

    if sort_order == "تصاعدي":
        df = df.sort_values(by="consumption", ascending=True)
    elif sort_order == "تنازلي":
        df = df.sort_values(by="consumption", ascending=False)

    if st.button("🚀 بدء التحليل"):
        results = []
        progress_bar = st.progress(0)

        def process_row(row):
            meter_id, lat, lon = row["Subscription"], row["y"], row["x"]
            breaker, consumption, office = row["Breaker"], row["consumption"], row["Office"]

            img_path = download_image(lat, lon, meter_id)
            if not img_path:
                return

            detection = detect_field(img_path, lat, lon, meter_id)
            if detection:
                conf, img_detected, area, distance = detection
                anomaly = model_ml.predict(scaler.transform([[breaker, consumption, lon, lat]]))[0]
                priority = "قصوى" if anomaly else "منخفضة"
                return meter_id, conf, img_detected, area, distance, consumption, breaker, office, priority, lat, lon

        with ThreadPoolExecutor(max_workers=5) as executor:
            for i, res in enumerate(executor.map(process_row, [row for _, row in df.iterrows()]), 1):
                if res:
                    results.append(res)
                progress_bar.progress(i / len(df))

        df_results = pd.DataFrame(results, columns=["Subscription", "Confidence", "Image", "Area", "Distance", "Consumption", "Breaker", "Office", "Priority", "Lat", "Lon"])
        st.dataframe(df_results)

        df_results.to_excel(os.path.join(OUTPUT_FOLDER, "results.xlsx"), index=False)
        with open(os.path.join(OUTPUT_FOLDER, "results.xlsx"), "rb") as file:
            st.download_button("📥 تحميل النتائج Excel", file, "results.xlsx")
else:
    st.warning("يرجى رفع ملف Excel يحتوي على البيانات المطلوبة.")
