# -*- coding: utf-8 -*-
"""
نظام اكتشاف حالات الفاقد للفئة الزراعية (Streamlit + YOLO + Isolation Forest + Copernicus Sentinel-2)
"""

import os, time, base64, math, traceback, io
from dataclasses import dataclass
from typing import Optional, Tuple, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from geopy.distance import geodesic
import joblib

# -----------------------------
# 1) الإعدادات العامة AppConfig
# -----------------------------
@dataclass
class AppConfig:
    zoom: int = 15
    map_size: Tuple[int, int] = (640, 640)
    calibration_factor: float = 0.6695
    yolo_conf_threshold: float = 0.5
    min_confidence_accept: float = 0.9
    min_area_m2: float = 5000.0
    max_edge_distance_m: float = 100.0
    risk_low: float = 0.40
    risk_high: float = 0.70
    request_timeout_s: int = 30
    max_retries: int = 3
    retry_backoff_s: float = 1.0
    save_checkpoint_every: int = 20
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    output_dir: str = "output"
    models_dir: str = "models"
    form_template_path: str = "TEMPLATE.xlsx"
    page_title: str = "🌾 نظام اكتشاف حالات الفاقد للفئة الزراعية"
    page_icon: str = "🌾"

# -----------------------------
# 2) أدوات Utilities
# -----------------------------
def meters_per_pixel(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# -----------------------------
# 3) إدخال/إخراج البيانات
# -----------------------------
def read_excel(file_obj) -> pd.DataFrame:
    df = pd.read_excel(file_obj)
    return df.dropna(subset=["Subscription", "Office", "Breaker", "consumption", "x", "y"])

def save_results_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf.read()

def save_results_html(rows: List[List], colors: dict, detected_dir: str) -> bytes:
    html = ["<html><head><meta charset='UTF-8'></head><body><div style='display:flex;flex-wrap:wrap;'>"]
    for r in rows:
        meter_id, priority, risk, distance, area, consumption, breaker, office, lat, lon = r
        border = colors.get(priority, "#ccc")
        img_path = os.path.join(detected_dir, f"{meter_id}.png")
        img_tag = ""
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
                img_tag = f"<img src='data:image/png;base64,{img_b64}' width='250' style='border-radius:8px;'>"
        html.append(f"""
<div style='border:4px solid {border};padding:10px;border-radius:10px;margin:6px;text-align:center;'>
  {img_tag}<br>
  <strong>عداد {meter_id} ({priority})</strong><br>
  درجة الخطر: {risk*100:.1f}% | المسافة: {distance:.1f}م | المساحة: {area}م²<br>
  الاستهلاك: {consumption} | القاطع: {breaker} | المكتب: {office}<br>
  <a href='https://maps.google.com?q={lat},{lon}'>📍 الموقع</a>
  <a href='https://wa.me/?text=عداد:{meter_id}%20الموقع:{lat},{lon}'>📲 واتساب</a>
</div>""")
    html.append("</div></body></html>")
    return "\n".join(html).encode("utf-8")

# -----------------------------
# 4) Vision (YOLO)
# -----------------------------
@dataclass
class FieldDetection:
    bbox_xyxy: Tuple[float, float, float, float]
    conf: float
    area_m2: float
    center_latlon: Tuple[float, float]
    edge_distance_m: float
    out_img_path: str

@st.cache_resource
def load_yolo(model_path: str) -> YOLO:
    return YOLO(model_path)

def detect_best_box(image: Image.Image, model: YOLO, min_conf=0.5):
    results = model.predict(source=image, imgsz=640, conf=min_conf, verbose=False)[0]
    if not results or not results.boxes or len(results.boxes) == 0:
        return None, None
    confs = results.boxes.conf.cpu().numpy()
    idx = int(confs.argmax())
    return results.boxes.xyxy[idx].cpu().numpy(), float(confs[idx])

def detect_field(img_path, lat, lon, meter_id, model_yolo,
                 zoom, calibration_factor, min_conf_accept,
                 min_area_m2, max_edge_distance_m, detected_dir) -> Optional[FieldDetection]:
    image = Image.open(img_path).convert("RGB")
    box, conf = detect_best_box(image, model_yolo, min_conf=min_conf_accept)
    if box is None or conf < min_conf_accept:
        return None

    res = meters_per_pixel(lat, zoom)
    width_px = abs(box[2]-box[0])
    height_px = abs(box[3]-box[1])
    area = width_px * height_px * (res**2)
    corrected_area = area * calibration_factor
    if corrected_area < min_area_m2:
        return None

    img_cx, img_cy = image.width/2, image.height/2
    bx_cx, bx_cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    dx_m = (bx_cx - img_cx) * res
    dy_m = (bx_cy - img_cy) * res
    dlat = -(dy_m / 111320.0)
    dlon = dx_m / (40075000.0 * math.cos(math.radians(lat)) / 360.0)
    field_lat = lat + dlat
    field_lon = lon + dlon

    radius_px = max(width_px, height_px) / 2
    radius_m = radius_px * res
    center_distance = geodesic((lat, lon), (field_lat, field_lon)).meters
    edge_distance = max(center_distance - radius_m, 0.0)
    if edge_distance > max_edge_distance_m:
        return None

    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    draw.line([(img_cx, img_cy), (bx_cx, bx_cy)], fill="yellow", width=2)
    os.makedirs(detected_dir, exist_ok=True)
    out_path = os.path.join(detected_dir, f"{meter_id}.png")
    image.save(out_path)
    return FieldDetection(tuple(box.tolist()), conf, int(corrected_area),
                          (field_lat, field_lon), round(edge_distance,2), out_path)

# -----------------------------
# 5) Risk Model
# -----------------------------
class RiskModel:
    def __init__(self, model_path, scaler_path, low_thr, high_thr):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.low_thr = low_thr
        self.high_thr = high_thr

    def compute(self, breaker, consumption, lon, lat, area_m2):
        X = np.array([[breaker, consumption, lon, lat]], dtype=float)
        Xs = self.scaler.transform(X)
        anomaly = self.model.predict(Xs)[0]
        r1 = 1.0 if breaker < area_m2 * 0.006 else 0.0
        r2 = 1.0 if consumption < area_m2 * 0.4 else 0.0
        r3 = 1.0 if anomaly == 1 else 0.0
        score = 0.4*r1 + 0.4*r2 + 0.2*r3
        if score >= self.high_thr:
            priority = "قصوى"
        elif score >= self.low_thr:
            priority = "متوسطة"
        else:
            priority = "منخفضة"
        return score, priority

# -----------------------------
# 6) واجهة Streamlit
# -----------------------------
cfg = AppConfig()
st.set_page_config(page_title=cfg.page_title, page_icon=cfg.page_icon, layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, cfg.images_dir)
DETECTED_DIR = os.path.join(BASE_DIR, cfg.detected_dir)
OUTPUT_DIR = os.path.join(BASE_DIR, cfg.output_dir)
MODELS_DIR = os.path.join(BASE_DIR, cfg.models_dir)
FORM_PATH = os.path.join(BASE_DIR, cfg.form_template_path)
MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
ML_MODEL_PATH = os.path.join(MODELS_DIR, "isolation_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "isolation_scaler.joblib")

ensure_dirs(IMG_DIR, DETECTED_DIR, OUTPUT_DIR, MODELS_DIR)

@st.cache_resource
def _load_models_cached():
    yolo = load_yolo(MODEL_PATH)
    risk = RiskModel(ML_MODEL_PATH, SCALER_PATH, cfg.risk_low, cfg.risk_high)
    return yolo, risk

# ✅ Copernicus Sentinel-2 image downloader
@st.cache_data(show_spinner=False, ttl=24*3600)
def download_image(lat: float, lon: float, meter_id: str, zoom: int, size: Tuple[int,int],
                   map_type: str, timeout: int) -> Optional[str]:
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path

    token = st.secrets.get("COPERNICUS_TOKEN", "")
    if not token:
        st.error("❌ لم يتم العثور على COPERNICUS_TOKEN في secrets.toml")
        return None

    bbox = [lon - 0.0008, lat - 0.0008, lon + 0.0008, lat + 0.0008]
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"

    payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {"mosaickingOrder": "mostRecent"}}],
        },
        "output": {"width": 640, "height": 640,
                   "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
        "evalscript": """
            //VERSION=3
            function setup() {
                return { input: ["B04","B03","B02"], output: { bands: 3 } };
            }
            function evaluatePixel(sample) {
                return [sample.B04*2.5, sample.B03*2.5, sample.B02*2.5];
            }
        """
    }

    headers = {"Authorization": f"Bearer {token}"}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(r.content)
            return img_path
        else:
            st.warning(f"⚠️ فشل تحميل صورة من Copernicus (status {r.status_code}) للعداد {meter_id}")
            return None
    except Exception as e:
        st.error(f"❌ خطأ أثناء الاتصال بـ Copernicus: {e}")
        return None

# -------------------------------------------------------
# واجهة المستخدم
# -------------------------------------------------------
st.title(cfg.page_title)
if os.path.exists(FORM_PATH):
    st.download_button("📥 تحميل نموذج البيانات (TEMPLATE.xlsx)", open(FORM_PATH, "rb"), file_name="TEMPLATE.xlsx")

uploaded_file = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

colors = {"قصوى": "#ff4d4d", "متوسطة": "#ffa500", "منخفضة": "#4CAF50"}

if uploaded_file:
    df = read_excel(uploaded_file)
    st.sidebar.info(f"🔢 عدد الحالات في الملف: {len(df)}")
    if st.sidebar.button("🚀 بدء التحليل"):
        model_yolo, risk_model = _load_models_cached()
        progress_bar = st.sidebar.progress(0)
        start_time = time.time()
        cols = st.columns(3)
        col_index = 0
        results_rows = []

        for i, (_, row) in enumerate(df.iterrows(), 1):
            try:
                meter_id, lat, lon = str(row["Subscription"]), float(row["y"]), float(row["x"])
                breaker, consumption, office = float(row["Breaker"]), float(row["consumption"]), str(row["Office"])
                img_path = download_image(lat, lon, meter_id, cfg.zoom, cfg.map_size, "satellite", cfg.request_timeout_s)
                if not img_path:
                    continue
                det = detect_field(img_path, lat, lon, meter_id, model_yolo,
                                   cfg.zoom, cfg.calibration_factor, cfg.min_confidence_accept,
                                   cfg.min_area_m2, cfg.max_edge_distance_m, DETECTED_DIR)
                if det is None:
                    continue

                score, priority = risk_model.compute(breaker, consumption, lon, lat, det.area_m2)
                results_rows.append([meter_id, priority, score, det.edge_distance_m,
                                     det.area_m2, consumption, breaker, office, lat, lon])

                with open(det.out_img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                cols[col_index % 3].markdown(f"""
                <div style="border:4px solid {colors.get(priority, '#ccc')};padding:10px;border-radius:12px;margin:6px;text-align:center;">
                    <img src="data:image/png;base64,{img_b64}" width="260" style="border-radius:8px;"><br>
                    <strong>عداد {meter_id} ({priority})</strong><br>
                    درجة الخطر:{score*100:.1f}% | المسافة:{det.edge_distance_m:.1f}م | المساحة:{det.area_m2}م²<br>
                    الاستهلاك:{consumption} | القاطع:{breaker} | المكتب:{office}<br>
                    <a href="https://maps.google.com?q={lat},{lon}">📍 الموقع</a>
                    <a href="https://wa.me/?text=عداد:{meter_id}%20الموقع:{lat},{lon}">📲 واتساب</a>
                </div>
                """, unsafe_allow_html=True)
                col_index += 1
                progress_bar.progress(i / len(df))
            except Exception as e:
                st.warning(f"⚠️ خطأ عند العداد {row.get('Subscription','?')}: {e}")
                continue

        if results_rows:
            results_df = pd.DataFrame(results_rows, columns=[
                "Subscription","priority","risk_score","edge_distance_m","area_m2",
                "consumption","breaker","office","lat","lon"
            ])
            excel_bytes = save_results_excel(results_df)
            html_bytes = save_results_html(results_rows, colors, DETECTED_DIR)
            st.sidebar.download_button("📥 تحميل النتائج Excel", data=excel_bytes, file_name="results.xlsx")
            st.sidebar.download_button("📥 تحميل التقرير الكامل HTML", data=html_bytes, file_name="report.html", mime="text/html")

        duration = time.time() - start_time
        st.sidebar.success(f"⏱️ اكتمل التحليل في {round(duration,2)} ثانية")

st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس | 00966553339838 | ")
