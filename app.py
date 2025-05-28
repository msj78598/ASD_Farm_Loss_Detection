import streamlit as st
import os, math, requests, io, base64
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

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output) as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# تصميم CSS
st.markdown(f"""
<div class="card priority-{css_class}">
    <div class="card-header">
        <h3>🔢 العداد: {meter_id}</h3>
        <span class="priority-badge {css_class}-badge">{pri}</span>
    </div>
    <div class="card-content">
        <img class="card-image" src="data:image/png;base64,{img_b64}">
        <div class="card-details">
            <div class="detail-row"><span class="detail-label">المكتب:</span><span class="detail-value">{office}</span></div>
            <div class="detail-row"><span class="detail-label">ثقة الكشف:</span><span class="detail-value">{conf*100:.1f}%</span></div>
            <div class="detail-row"><span class="detail-label">المساحة:</span><span class="detail-value">{area:,} م²</span></div>
            <div class="detail-row"><span class="detail-label">الاستهلاك:</span><span class="detail-value">{consumption:,}</span></div>
            <div class="detail-row"><span class="detail-label">القاطع:</span><span class="detail-value">{breaker}</span></div>
        </div>
    </div>
    <div class="card-actions">
        <a class="action-btn whatsapp-btn" target="_blank" href="{generate_whatsapp_share_link(meter_id, area, consumption, generate_google_maps_link(lat,lon))}">📱 واتساب</a>
        <a class="action-btn map-btn" target="_blank" href="{generate_google_maps_link(lat,lon)}">📍 خريطة الموقع</a>
    </div>
</div>
""", unsafe_allow_html=True)

# واجهة المستخدم
st.title("🌾 نظام اكتشاف حالات الفاقد الزراعي")
uploaded_file = st.file_uploader("📁 رفع ملف Excel", type=["xlsx"])

sort_col = st.sidebar.selectbox("فرز حسب:", ["بدون", "consumption", "Breaker"])
sort_order = st.sidebar.radio("نوع الفرز:", ["تصاعدي", "تنازلي"], horizontal=True)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if sort_col != "بدون":
        df.sort_values(by=sort_col, ascending=(sort_order=="تصاعدي"), inplace=True)

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, row in df.iterrows():
        progress_bar.progress((idx+1)/len(df))
        status_text.text(f"معالجة الحالة رقم {idx+1} من {len(df)}")
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

        st.markdown(f"""
<div class="card priority-{css_class}">
    <div class="card-header">
        <h3>🔢 العداد: {meter_id}</h3>
        <span class="priority-badge {css_class}-badge">{pri}</span>
    </div>
    <div class="card-content">
        <img class="card-image" src="data:image/png;base64,{img_b64}">
        <div class="card-details">
            <div class="detail-row"><span class="detail-label">المكتب:</span><span>{office}</span></div>
            <div class="detail-row"><span class="detail-label">ثقة الكشف:</span><span>{conf*100:.1f}%</span></div>
            <div class="detail-row"><span class="detail-label">المساحة:</span><span>{area:,} م²</span></div>
            <div class="detail-row"><span class="detail-label">الاستهلاك:</span><span>{consumption:,}</span></div>
            <div class="detail-row"><span class="detail-label">القاطع:</span><span>{breaker}</span></div>
        </div>
    </div>
    <div class="card-actions">
        <a class="action-btn whatsapp-btn" href="{generate_whatsapp_share_link(meter_id, area, consumption, generate_google_maps_link(lat,lon))}">📱 واتساب</a>
        <a class="action-btn map-btn" href="{generate_google_maps_link(lat,lon)}">📍 خريطة الموقع</a>
    </div>
</div>
""", unsafe_allow_html=True)


    status_text.text("✅ تم بنجاح")
