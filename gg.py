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

# === إضافات من أجل Copernicus / Sentinel-2 ===
from datetime import datetime, timedelta
from pystac_client import Client
from odc.stac import stac_load

# ==============================================

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

CDSE_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# -------- الحصول على توكن تلقائياً (اختياري) ----------
_token_cache = {"access_token": None, "exp": 0.0}

def get_cdse_token():
    """يجلب/يجدّد Access Token من CDSE باستعمال CLIENT_ID/SECRET أو يرجع CDSE_TOKEN إن كان موجوداً."""
    # إن كان لديك توكن جاهز في البيئة نستخدمه
    tok = os.getenv("CDSE_TOKEN")
    if tok:
        return tok
    # غير ذلك نجلبه عبر OAuth Client Credentials
    if _token_cache["access_token"] and time.time() < _token_cache["exp"] - 60:
        return _token_cache["access_token"]
    client_id = os.getenv("CDSE_CLIENT_ID")
    client_secret = os.getenv("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("يرجى ضبط CDSE_CLIENT_ID و CDSE_CLIENT_SECRET أو CDSE_TOKEN في متغيرات البيئة.")
    resp = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials",
              "client_id": client_id,
              "client_secret": client_secret},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache["access_token"] = data["access_token"]
    _token_cache["exp"] = time.time() + int(data.get("expires_in", 3600))
    return _token_cache["access_token"]
# ------------------------------------------------------

@st.cache_resource
def load_models():
    model_yolo = YOLO(MODEL_PATH)
    model_ml = joblib.load(ML_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model_yolo, model_ml, scaler

# ====== الدالة الجديدة: جلب صورة Sentinel-2 بدل Google ======
def download_image(lat, lon, meter_id,
                   days_back=30, max_cloud=10,
                   out_px=640, emulate_zoom=16):
    """
    تُرجع (img_path, mpp) حيث mpp = أمتار/بكسل للصورة (10م غالباً).
    تُقص الصورة على مربع يحاكي تقريبا عرض (zoom=16, 640px) الذي كنت تستخدمه مع خرائط جوجل.
    """
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")
    mpp_sidecar = img_path.replace(".png", ".mpp.txt")
    if os.path.exists(img_path) and os.path.exists(mpp_sidecar):
        try:
            mpp = float(open(mpp_sidecar).read())
            return img_path, mpp
        except Exception:
            pass

    # 1) نحسب مقياس البكسل التقريبي الذي كان في Google عند zoom=16
    gmap_mpp = 156543.03392 * math.cos(math.radians(lat)) / (2 ** emulate_zoom)
    half_size_m = gmap_mpp * (out_px / 2.0)

    # نحول نصف العرض بالمتر إلى درجات (تقريب جيد للمديات الصغيرة)
    dlat = half_size_m / 111320.0
    dlon = half_size_m / (111320.0 * math.cos(math.radians(lat)))
    minx, miny, maxx, maxy = lon - dlon, lat - dlat, lon + dlon, lat + dlat
    bbox_4326 = (minx, miny, maxx, maxy)

    # 2) نبحث أحدث مشهد Sentinel-2 L2A ضمن نافذة زمنية وبحد غيوم
    headers = {}
    try:
        headers["Authorization"] = f"Bearer {get_cdse_token()}"
    except Exception:
        pass  # قد يعمل عامة بدون توكن، لكن يفضّل وجوده

    client = Client.open(CDSE_STAC_URL, headers=headers)
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)

    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_4326,
        datetime=f"{start.isoformat()}Z/{end.isoformat()}Z",
        query={"eo:cloud_cover": {"lt": max_cloud}}
    )
    items = list(search.get_items())
    if not items:
        return None, None
    # الأحدث أولاً
    items.sort(key=lambda it: it.properties.get("datetime"), reverse=True)
    item = items[0]

    # 3) التحميل والقص على الـ bbox بدقة 10م (True Color: B04,B03,B02)
    ds = stac_load(
        [item],
        bands=["B04", "B03", "B02"],
        bbox=bbox_4326,
        resolution=10,   # متر/بكسل
        chunks={},       # تحميل مباشر
    )
    # (T, Y, X)
    r = ds["B04"].isel(time=0).data
    g = ds["B03"].isel(time=0).data
    b = ds["B02"].isel(time=0).data

    # تحويل إلى uint8 [0..255]
    def to_uint8(x):
        x = x.astype(np.float32)
        x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-6)
        return (x * 255.0).clip(0, 255).astype(np.uint8)

    rgb = np.dstack([to_uint8(r), to_uint8(g), to_uint8(b)])
    Image.fromarray(rgb).save(img_path)

    # بما أننا طلبنا resolution=10 → مقياس البكسل 10 متر/بكسل
    mpp = 10.0
    with open(mpp_sidecar, "w") as f:
        f.write(str(mpp))

    return img_path, mpp
# ============================================================

def detect_field(img_path, lat, lon, meter_id, model_yolo, mpp):
    image = Image.open(img_path).convert("RGB")
    results = model_yolo.predict(source=image, imgsz=640, conf=0.1)[0]
    if not results.boxes:
        return None, None, None, None
    box = results.boxes[0].xyxy[0].cpu().numpy()
    conf = float(results.boxes[0].conf.cpu().numpy())
    if conf < 0.1:
        return None, None, None, None

    # مقياس البكسل الحقيقي للصورة (10م عادةً ل Sentinel-2)
    scale = mpp
    area = abs(box[2] - box[0]) * abs(box[3] - box[1]) * (scale ** 2)
    corrected_area = area * CALIBRATION_FACTOR
    if corrected_area < 1000:
        return None, None, None, None

    img_center_pixel = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    # نفس حساباتك السابقة لكن بمقياس البكسل الحقيقي
    dx = (img_center_pixel[0] - 320) * scale
    dy = (img_center_pixel[1] - 320) * scale

    field_lat = lat - (dy / 111320)
    field_lon = lon + (dx / (40075000 * math.cos(math.radians(lat)) / 360))

    width_px = abs(box[2] - box[0])
    height_px = abs(box[3] - box[1])
    radius_px = max(width_px, height_px) / 2
    radius_m = radius_px * scale
    center_distance = geodesic((lat, lon), (field_lat, field_lon)).meters
    edge_distance = max(center_distance - radius_m, 0)

    if edge_distance > 200:
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
    
    ignore_anomalies = st.sidebar.checkbox("🔍 عرض كل الحقول المكتشفة فقط (تجاهل الشذوذ والاستهلاك)")

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

        colors = {"قصوى": "#ff4d4d", "متوسطة": "#ffa500", "منخفضة": "#4CAF50", "مكتشف": "#1E90FF"}
        results = []
        cols = st.columns(3)
        col_index = 0

        for i, (_, row) in enumerate(df.iterrows(), 1):
            meter_id, lat, lon = row["Subscription"], row["y"], row["x"]
            breaker, consumption, office = row["Breaker"], row["consumption"], row["Office"]

            # ======= استبدال جلب الصورة =======
            img_mpp = download_image(lat, lon, meter_id)
            if not img_mpp or img_mpp[0] is None:
                # إذا ما لقى مشهد مناسب ممكن تتخطى أو تضع لوج
                continue
            img_path, mpp = img_mpp
            # ==================================

            conf, img_detected, area, distance = detect_field(img_path, lat, lon, meter_id, model_yolo, mpp)
            if conf is None:
                continue

            if ignore_anomalies:
                confidence = 1.0
                priority = "مكتشف"
            else:
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

st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس | 00966553339838 | ")
