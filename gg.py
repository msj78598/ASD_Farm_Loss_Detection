# -*- coding: utf-8 -*-
"""
نظام اكتشاف حالات الفاقد للفئة الزراعية (Streamlit + YOLO + Isolation Forest + Copernicus)
- يدعم وضع معاينة الصور فقط قبل التمرير على النموذج.
"""

import os, io, time, base64, math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from geopy.distance import geodesic
from ultralytics import YOLO
import joblib

# ======================= الإعدادات =======================
@dataclass
class AppConfig:
    zoom: int = 15
    map_size: Tuple[int, int] = (640, 640)
    calibration_factor: float = 0.6695
    min_confidence_accept: float = 0.9
    min_area_m2: float = 5000.0
    max_edge_distance_m: float = 100.0
    risk_low: float = 0.40
    risk_high: float = 0.70
    request_timeout_s: int = 30
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    output_dir: str = "output"
    models_dir: str = "models"
    page_title: str = "🌾 نظام اكتشاف حالات الفاقد للفئة الزراعية"
    page_icon: str = "🌾"

cfg = AppConfig()

# ======================= دوال مساعدة =======================
def meters_per_pixel(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

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
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
                img_tag = f"<img src='data:image/png;base64,{img_b64}' width='250' style='border-radius:8px;'>"
        else:
            img_tag = ""
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

# ======================= تحميل النماذج =======================
@st.cache_resource
def load_yolo(model_path: str):
    return YOLO(model_path)

class RiskModel:
    def __init__(self, model_path, scaler_path, low_thr, high_thr):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.low_thr, self.high_thr = low_thr, high_thr

    def compute(self, breaker, consumption, lon, lat, area_m2):
        X = np.array([[breaker, consumption, lon, lat]], dtype=float)
        Xs = self.scaler.transform(X)
        anomaly = self.model.predict(Xs)[0]
        r1 = 1.0 if breaker < area_m2 * 0.006 else 0.0
        r2 = 1.0 if consumption < area_m2 * 0.4 else 0.0
        r3 = 1.0 if anomaly == 1 else 0.0
        score = 0.4*r1 + 0.4*r2 + 0.2*r3
        if score >= self.high_thr:
            pr = "قصوى"
        elif score >= self.low_thr:
            pr = "متوسطة"
        else:
            pr = "منخفضة"
        return score, pr

# ======================= كشف الحقول =======================
@dataclass
class FieldDetection:
    bbox_xyxy: Tuple[float, float, float, float]
    conf: float
    area_m2: float
    center_latlon: Tuple[float, float]
    edge_distance_m: float
    out_img_path: str

def detect_best_box(image: Image.Image, model: YOLO, min_conf=0.5):
    results = model.predict(source=image, imgsz=640, conf=min_conf, verbose=False)[0]
    if not results or not results.boxes or len(results.boxes) == 0:
        return None, None
    confs = results.boxes.conf.cpu().numpy()
    idx = int(confs.argmax())
    return results.boxes.xyxy[idx].cpu().numpy(), float(confs[idx])

def detect_field(img_path, lat, lon, meter_id, model_yolo,
                 zoom, calibration_factor, min_conf_accept,
                 min_area_m2, max_edge_distance_m, detected_dir):
    image = Image.open(img_path).convert("RGB")
    box, conf = detect_best_box(image, model_yolo, min_conf=min_conf_accept)
    if box is None or conf < min_conf_accept:
        return None

    res = meters_per_pixel(lat, zoom)
    w_px, h_px = abs(box[2]-box[0]), abs(box[3]-box[1])
    area = w_px * h_px * (res**2)
    corrected = area * calibration_factor
    if corrected < min_area_m2:
        return None

    cx_img, cy_img = image.width/2, image.height/2
    bx, by = (box[0]+box[2])/2, (box[1]+box[3])/2
    dx_m, dy_m = (bx-cx_img)*res, (by-cy_img)*res
    dlat = -(dy_m / 111320.0)
    dlon = dx_m / (40075000.0 * math.cos(math.radians(lat)) / 360.0)
    flat, flon = lat+dlat, lon+dlon

    radius_px = max(w_px, h_px)/2
    radius_m = radius_px * res
    dist = geodesic((lat, lon), (flat, flon)).meters
    edge = max(dist - radius_m, 0)
    if edge > max_edge_distance_m:
        return None

    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    draw.line([(cx_img, cy_img), (bx, by)], fill="yellow", width=2)
    os.makedirs(detected_dir, exist_ok=True)
    out_path = os.path.join(detected_dir, f"{meter_id}.png")
    image.save(out_path)
    return FieldDetection(tuple(box.tolist()), conf, int(corrected), (flat, flon), round(edge,2), out_path)

# ======================= تنزيل الصور من Copernicus =======================
@st.cache_data(show_spinner=False, ttl=24*3600)
def download_image(lat, lon, meter_id, timeout=30):
    img_path = os.path.join(cfg.images_dir, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path

    token = st.secrets.get("COPERNICUS_TOKEN", "")
    if not token:
        st.error("❌ لم يتم ضبط COPERNICUS_TOKEN في secrets.toml")
        return None

    # صندوق صغير حول النقطة (حوالي ~180م)
    bbox = [lon-0.0008, lat-0.0008, lon+0.0008, lat+0.0008]

    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a"}]
        },
        "output": {"width": cfg.map_size[0], "height": cfg.map_size[1],
                   "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
        "evalscript": """
        //VERSION=3
        function setup(){return {input:["B04","B03","B02"],output:{bands:3}};}
        function evaluatePixel(s){return [s.B04*2.5,s.B03*2.5,s.B02*2.5];}
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
            st.warning(f"⚠️ Copernicus status {r.status_code} للعداد {meter_id}")
            return None
    except Exception as e:
        st.error(f"❌ فشل تحميل صورة من Copernicus: {e}")
        return None

# ======================= الواجهة =======================
st.set_page_config(page_title=cfg.page_title, page_icon=cfg.page_icon, layout="wide")
ensure_dirs(cfg.images_dir, cfg.detected_dir, cfg.output_dir, cfg.models_dir)

MODEL_PATH = os.path.join(cfg.models_dir, "best.pt")
ML_MODEL_PATH = os.path.join(cfg.models_dir, "isolation_model.joblib")
SCALER_PATH = os.path.join(cfg.models_dir, "isolation_scaler.joblib")

st.title(cfg.page_title)
uploaded = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])
colors = {"قصوى": "#ff4d4d", "متوسطة": "#ffa500", "منخفضة": "#4CAF50"}

if uploaded:
    df = read_excel(uploaded)
    st.sidebar.info(f"🔢 عدد الحالات: {len(df)}")

    # فلاتر
    breaker_filter = st.sidebar.selectbox("سعة القاطع", ["الكل"] + sorted(df["Breaker"].unique().tolist()))
    sort_order = st.sidebar.radio("ترتيب حسب الاستهلاك", ["بدون ترتيب", "تصاعدي", "تنازلي"])
    if breaker_filter != "الكل":
        df = df[df["Breaker"] == breaker_filter]
    if sort_order == "تصاعدي":
        df = df.sort_values(by="consumption", ascending=True)
    elif sort_order == "تنازلي":
        df = df.sort_values(by="consumption", ascending=False)

    # عناصر وضع المعاينة
    preview_only = st.sidebar.checkbox("🖼️ عرض الصور فقط (بدون تشغيل النموذج)")
    btn_preview = st.sidebar.button("📥 تنزيل/عرض الصور")

    # --- وضع معاينة الصور فقط ---
    if btn_preview:
        progress = st.sidebar.progress(0)
        cols = st.columns(4)  # شبكة 4 أعمدة
        shown = 0
        n = len(df)
        t0 = time.time()

        for i, (_, row) in enumerate(df.iterrows(), 1):
            meter = str(row["Subscription"])
            lat, lon = float(row["y"]), float(row["x"])
            img_path = download_image(lat, lon, meter)
            if not img_path:
                progress.progress(i / max(n, 1))
                continue
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            cols[shown % 4].markdown(
                f"""
                <div style="border:1px solid #ddd;border-radius:8px;padding:6px;margin:6px;text-align:center">
                  <img src="data:image/png;base64,{b64}" width="230" style="border-radius:6px"><br>
                  <small>عداد {meter}<br>Lat {lat:.6f}, Lon {lon:.6f}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
            shown += 1
            progress.progress(i / max(n, 1))

        st.sidebar.success(f"✅ تم عرض {shown} صورة خلال {time.time()-t0:.1f} ثانية")
        st.stop()  # لا نكمل للتشغيل

    # زر بدء التحليل (التشغيل الكامل)
    if st.sidebar.button("🚀 بدء التحليل"):
        model_yolo = load_yolo(MODEL_PATH)
        risk_model = RiskModel(ML_MODEL_PATH, SCALER_PATH, cfg.risk_low, cfg.risk_high)
        progress = st.sidebar.progress(0)
        results, cols, col_i = [], st.columns(3), 0
        t0 = time.time()

        for i, (_, row) in enumerate(df.iterrows(), 1):
            try:
                meter = str(row["Subscription"])
                lat, lon = float(row["y"]), float(row["x"])
                br, cons, off = float(row["Breaker"]), float(row["consumption"]), str(row["Office"])

                img = download_image(lat, lon, meter)
                if not img:
                    progress.progress(i / len(df))
                    continue

                det = detect_field(img, lat, lon, meter, model_yolo,
                                   cfg.zoom, cfg.calibration_factor,
                                   cfg.min_confidence_accept, cfg.min_area_m2,
                                   cfg.max_edge_distance_m, cfg.detected_dir)
                if det is None:
                    progress.progress(i / len(df))
                    continue

                score, pr = risk_model.compute(br, cons, lon, lat, det.area_m2)
                results.append([meter, pr, score, det.edge_distance_m, det.area_m2, cons, br, off, lat, lon])

                with open(det.out_img_path, "rb") as f:
                    img64 = base64.b64encode(f.read()).decode()
                cols[col_i % 3].markdown(f"""
                <div style="border:4px solid {colors.get(pr,'#ccc')};padding:10px;border-radius:12px;margin:6px;text-align:center;">
                  <img src="data:image/png;base64,{img64}" width="260"><br>
                  <strong>عداد {meter} ({pr})</strong><br>
                  خطر:{score*100:.1f}% | مسافة:{det.edge_distance_m:.1f}م | مساحة:{det.area_m2}م²<br>
                  استهلاك:{cons} | قاطع:{br} | مكتب:{off}<br>
                  <a href="https://maps.google.com?q={lat},{lon}">📍 الموقع</a>
                </div>""", unsafe_allow_html=True)
                col_i += 1
                progress.progress(i / len(df))

            except Exception as e:
                st.warning(f"⚠️ خطأ في العداد {row.get('Subscription','?')}: {e}")
                progress.progress(i / len(df))
                continue

        if results:
            res_df = pd.DataFrame(results, columns=[
                "Subscription","priority","risk_score","edge_distance_m","area_m2",
                "consumption","breaker","office","lat","lon"
            ])
            excel_bytes = save_results_excel(res_df)
            html_bytes = save_results_html(results, colors, cfg.detected_dir)
            st.sidebar.download_button("📥 نتائج Excel", data=excel_bytes, file_name="results.xlsx")
            st.sidebar.download_button("📥 تقرير HTML", data=html_bytes, file_name="report.html", mime="text/html")

        st.sidebar.success(f"⏱️ اكتمل التحليل خلال {round(time.time()-t0,1)} ثانية")

st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس | 00966553339838 | ")
