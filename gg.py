# -*- coding: utf-8 -*-
"""
"""

import os, io, time, base64, math, re
from dataclasses import dataclass
from typing import Tuple, List, Optional

import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from geopy.distance import geodesic
from ultralytics import YOLO
import joblib

# ======================= إعدادات ثابتة =======================
@dataclass
class AppConfig:
    map_size: Tuple[int, int] = (640, 640)   # أبعاد الصورة الناتجة
    scene_size_m: int = 2500                 # عرض/ارتفاع المشهد بالأمتار (ثابت)
    calibration_factor: float = 0.6695
    min_confidence_accept: float = 0.45
    min_area_m2: float = 5000.0

    # ✅ Progressive search by EDGE distance
    r_start_m: int = 50
    r_step_m: int = 10
    r_max_m: int = 200

    risk_low: float = 0.40
    risk_high: float = 0.70
    request_timeout_s: int = 30
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    output_dir: str = "output"
    models_dir: str = "models"
    page_title: str = "🌾 نظام اكتشاف حالات الفاقد للفئة الزراعية"
    page_icon: str = "🌾"

    # ====== إعدادات فلترة “الخضرة” ======
    green_ratio_min: float = 0.0
    green_dominance: float = 1.1
    green_min_value: int = 60

cfg = AppConfig()

# ======================= تحسينات الواجهة (CSS فقط) =======================
st.set_page_config(page_title=cfg.page_title, page_icon=cfg.page_icon, layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-card {
        background: white; border-radius: 12px; padding: 15px; margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-right: 8px solid #ccc;
    }
    .status-قصوى { border-right-color: #ff4d4d; }
    .status-متوسطة { border-right-color: #ffa500; }
    .status-منخفضة { border-right-color: #4CAF50; }
    .btn-link {
        display: inline-block; padding: 6px 12px; border-radius: 5px;
        text-decoration: none; font-size: 14px; margin: 2px; color: white !important;
    }
    .google-btn { background-color: #4285F4; }
    .wa-btn { background-color: #25D366; }
    </style>
    """, unsafe_allow_html=True)

# ======================= أدوات عامة =======================
def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def clean_meter_id(val) -> str:
    """إرجاع رقم العداد كنص نظيف بدون .0 أو صيغة علمية أو مسافات."""
    if pd.isna(val):
        return ""
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
        s = str(val).strip()
        return re.sub(r"\.0+$", "", s)
    except Exception:
        s = str(val).strip()
        try:
            if re.fullmatch(r"[0-9]+(\.[0-9]+)?[eE][\+\-]?\d+", s):
                return str(int(float(s)))
        except Exception:
            pass
        s = re.sub(r"\.0+$", "", s)
        return s

def read_excel(file_obj) -> pd.DataFrame:
    df = pd.read_excel(file_obj, dtype={"Subscription": str})
    df["Subscription"] = df["Subscription"].apply(clean_meter_id)
    return df.dropna(subset=["Subscription", "Office", "Breaker", "consumption", "x", "y"])

def save_results_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf.read()

def save_results_html(rows: List[List], colors: dict, detected_dir: str) -> bytes:
    html = ["<html><head><meta charset='UTF-8'></head><body><div style='display:flex;flex-wrap:wrap;'>"]
    for r in rows:
        meter_id, priority, risk, edge_d, center_d, area, consumption, breaker, office, lat, lon = r
        border = colors.get(priority, "#ccc")
        pth = os.path.join(detected_dir, f"{meter_id}.png")
        img_tag = ""
        if os.path.exists(pth):
            with open(pth, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            img_tag = f"<img src='data:image/png;base64,{img_b64}' width='250' style='border-radius:8px;'>"
        html.append(f"""
<div style='border:4px solid {border};padding:10px;border-radius:10px;margin:6px;text-align:center;'>
  {img_tag}<br>
  <strong>عداد {meter_id} ({priority})</strong><br>
  خطر: {risk*100:.1f}% | حافة: {edge_d:.1f}م | مركز: {center_d:.1f}م | مساحة: {area}م²<br>
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

    def compute(self, breaker, consumption, lon, lat, area_m2, green_ratio):
        effective_area = area_m2 * green_ratio
        X = np.array([[breaker, consumption, lon, lat]], dtype=float)
        Xs = self.scaler.transform(X)
        anomaly = self.model.predict(Xs)[0]
        r1 = 1.0 if breaker < (effective_area * 0.0013) else 0.0
        r2 = 1.0 if consumption < (effective_area * 0.20) else 0.0
        r3 = 1.0 if anomaly == 1 else 0.0
        score = 0.4 * r1 + 0.4 * r2 + 0.2 * r3
        if score >= self.high_thr:
            pr = "قصوى"
        elif score >= self.low_thr:
            pr = "متوسطة"
        else:
            pr = "منخفضة"
        return score, pr

# ======================= دالة تقدير الخضرة =======================
def estimate_green_ratio(image: Image.Image, box_xyxy: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = image.crop((x1, y1, x2, y2))
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.size == 0:
        return 0.0
    R = arr[..., 0].astype(np.float32)
    G = arr[..., 1].astype(np.float32)
    B = arr[..., 2].astype(np.float32)
    dominance_mask = (G > R * cfg.green_dominance) & (G > B * cfg.green_dominance) & (G > cfg.green_min_value)
    Rn = R / 255.0; Gn = G / 255.0; Bn = B / 255.0
    exg = 2.0 * Gn - Rn - Bn
    exg_mask = exg > 0.08
    hsv = crop.convert("HSV")
    H = np.asarray(hsv.getchannel(0), dtype=np.uint8)
    S = np.asarray(hsv.getchannel(1), dtype=np.uint8)
    V = np.asarray(hsv.getchannel(2), dtype=np.uint8)
    hsv_mask = (H >= 25) & (H <= 67) & (S >= 60) & (V >= 50)
    green_mask = dominance_mask | exg_mask | hsv_mask
    return float(green_mask.mean())

# ======================= الكشف =======================
@dataclass
class FieldDetection:
    bbox_xyxy: Tuple[float, float, float, float]
    conf: float
    area_m2: int
    center_latlon: Tuple[float, float]
    edge_distance_m: float
    center_distance_m: float
    out_img_path: str
    green_ratio: float

def detect_boxes(image: Image.Image, model: YOLO, min_conf=0.5):
    res = model.predict(source=image, imgsz=640, conf=min_conf, verbose=False)[0]
    if not res or not res.boxes or len(res.boxes) == 0:
        return []
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    idxs = np.argsort(-confs)
    return [(boxes[i], float(confs[i])) for i in idxs]

def detect_field_progressive(img_path, lat, lon, meter_id, model_yolo, calibration_factor, min_conf_accept, min_area_m2, detected_dir, r_start=50, r_step=10, r_max=200):
    image = Image.open(img_path).convert("RGB")
    boxes = detect_boxes(image, model_yolo, min_conf=min_conf_accept)
    if not boxes:
        return None
    m_per_px = cfg.scene_size_m / float(cfg.map_size[0])
    cx, cy = image.width / 2, image.height / 2
    candidates = []
    for box, conf in boxes:
        w_px, h_px = abs(box[2]-box[0]), abs(box[3]-box[1])
        area = w_px * h_px * (m_per_px**2)
        corrected = area * calibration_factor
        if corrected < min_area_m2:
            continue
        bx, by = (box[0]+box[2])/2, (box[1]+box[3])/2
        dx_m, dy_m = (bx-cx)*m_per_px, (by-cy)*m_per_px
        dlat = -(dy_m / 111320.0)
        dlon = dx_m / (40075000.0 * math.cos(math.radians(lat)) / 360.0)
        flat, flon = lat + dlat, lon + dlon
        center_dist = geodesic((lat, lon), (flat, flon)).meters
        green_ratio = estimate_green_ratio(image, tuple(box.tolist()))
        if green_ratio < cfg.green_ratio_min:
            continue
        radius_px = max(w_px, h_px) / 2
        radius_m = radius_px * m_per_px
        edge_dist = max(center_dist - radius_m, 0.0)
        candidates.append((edge_dist, center_dist, box, conf, int(corrected), (flat, flon), green_ratio))
    if not candidates:
        return None
    chosen = None
    chosen_R = None
    for R in range(r_start, r_max + 1, r_step):
        within = [c for c in candidates if c[0] <= R]
        if within:
            chosen = min(within, key=lambda x: (x[0], x[1]))
            chosen_R = R
            break
    if chosen is None:
        return None
    edge_dist, center_dist, box, conf, area_m2, (flat, flon), green_ratio = chosen
    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    draw.line([(cx, cy), ((box[0]+box[2])/2, (box[1]+box[3])/2)], fill="yellow", width=2)
    draw.text((int(box[0]) + 4, int(box[1]) + 4), f"R<= {chosen_R}m | Edge:{edge_dist:.1f}m | Green {green_ratio*100:.0f}%", fill="white")
    os.makedirs(detected_dir, exist_ok=True)
    out_path = os.path.join(detected_dir, f"{meter_id}.png")
    image.save(out_path)
    return FieldDetection(bbox_xyxy=tuple(box.tolist()), conf=float(conf), area_m2=int(area_m2), center_latlon=(flat, flon), edge_distance_m=float(edge_dist), center_distance_m=float(center_dist), out_img_path=out_path, green_ratio=float(green_ratio))

# ======================= CDSE Token & Download =======================
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
def get_cdse_token():
    tok = st.session_state.get("_cdse_token")
    exp = st.session_state.get("_cdse_token_exp", 0)
    if tok and time.time() < exp - 60:
        return tok
    cid = st.secrets.get("CDSE_CLIENT_ID")
    csec = st.secrets.get("CDSE_CLIENT_SECRET")
    if not cid or not csec:
        raise RuntimeError("CDSE_CLIENT_ID / CDSE_CLIENT_SECRET غير موجودة في secrets.toml")
    data = {"grant_type":"client_credentials","client_id":cid,"client_secret":csec}
    r = requests.post(TOKEN_URL, data=data, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"CDSE token error {r.status_code}")
    js = r.json()
    access = js["access_token"]
    expires = int(js.get("expires_in", 3600))
    st.session_state["_cdse_token"] = access
    st.session_state["_cdse_token_exp"] = time.time() + expires
    return access

def bbox_from_meters(lat: float, lon: float, size_m: float):
    half = size_m / 2.0
    dlat = half / 111320.0
    dlon = half / (111320.0 * math.cos(math.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]

@st.cache_data(show_spinner=False, ttl=24*3600)
def download_image(lat, lon, meter_id, timeout=30):
    img_path = os.path.join(cfg.images_dir, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path
    def _request(token):
        bbox = bbox_from_meters(lat, lon, cfg.scene_size_m)
        url = "https://sh.dataspace.copernicus.eu/api/v1/process"
        payload = {
            "input": {"bounds": {"bbox": bbox}, "data": [{"type": "sentinel-2-l2a"}]},
            "output": {"width": cfg.map_size[0], "height": cfg.map_size[1], "responses": [{"format":{"type":"image/png"}}]},
            "evalscript": "//VERSION=3\nfunction setup(){return {input:['B04','B03','B02'],output:{bands:3}}}\nfunction evaluatePixel(s){return [s.B04*1.8, s.B03*1.8, s.B02*1.8]}"
        }
        headers = {"Authorization": f"Bearer {token}"}
        return requests.post(url, headers=headers, json=payload, timeout=timeout)
    token = get_cdse_token()
    r = _request(token)
    if r.status_code == 200:
        with open(img_path, "wb") as f:
            f.write(r.content)
        return img_path
    return None

# ======================= واجهة Streamlit =======================
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
    breaker_filter = st.sidebar.selectbox("سعة القاطع", ["الكل"] + sorted(df["Breaker"].unique().tolist()))
    sort_order = st.sidebar.radio("ترتيب حسب الاستهلاك", ["بدون ترتيب", "تصاعدي", "تنازلي"])
    if breaker_filter != "الكل":
        df = df[df["Breaker"] == breaker_filter]
    if sort_order == "تصاعدي":
        df = df.sort_values(by="consumption", ascending=True)
    elif sort_order == "تنازلي":
        df = df.sort_values(by="consumption", ascending=False)

    if st.sidebar.button("📥 تنزيل/عرض الصور"):
        progress = st.sidebar.progress(0)
        cols = st.columns(4)
        shown, n = 0, len(df)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            meter = clean_meter_id(row["Subscription"])
            p = download_image(float(row["y"]), float(row["x"]), meter)
            if p:
                with open(p, "rb") as f: b64 = base64.b64encode(f.read()).decode()
                cols[shown % 4].markdown(f"<div class='main-card'><img src='data:image/png;base64,{b64}' width='100%'><br><small>عداد {meter}</small></div>", unsafe_allow_html=True)
                shown += 1
            progress.progress(i / n)
        st.stop()

    if st.sidebar.button("🚀 بدء التحليل"):
        model_yolo = load_yolo(MODEL_PATH)
        risk_model = RiskModel(ML_MODEL_PATH, SCALER_PATH, cfg.risk_low, cfg.risk_high)
        progress = st.sidebar.progress(0)
        results, col_idx, ui_cols = [], 0, st.columns(3)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            try:
                meter, lat, lon = clean_meter_id(row["Subscription"]), float(row["y"]), float(row["x"])
                img_path = download_image(lat, lon, meter)
                if not img_path: continue
                det = detect_field_progressive(img_path, lat, lon, meter, model_yolo, cfg.calibration_factor, cfg.min_confidence_accept, cfg.min_area_m2, cfg.detected_dir)
                if det:
                    score, pr = risk_model.compute(float(row["Breaker"]), float(row["consumption"]), lon, lat, det.area_m2, det.green_ratio)
                    results.append([meter, pr, score, det.edge_distance_m, det.center_distance_m, det.area_m2, float(row["consumption"]), float(row["Breaker"]), str(row["Office"]), lat, lon])
                    with open(det.out_img_path, "rb") as f: img64 = base64.b64encode(f.read()).decode()
                    ui_cols[col_idx % 3].markdown(f"""
                        <div class="main-card status-{pr}">
                            <img src="data:image/png;base64,{img64}" width="100%"><br>
                            <strong>عداد {meter} ({pr})</strong><br>
                            خطر:{score*100:.1f}% | مساحة:{det.area_m2}م²<br>
                            <a class="btn-link google-btn" href="https://maps.google.com?q={lat},{lon}">📍 الموقع</a>
                            <a class="btn-link wa-btn" href="https://wa.me/?text=عداد:{meter}">📲 واتساب</a>
                        </div>""", unsafe_allow_html=True)
                    col_idx += 1
            except: pass
            progress.progress(i / len(df))
        if results:
            res_df = pd.DataFrame(results, columns=["Subscription","priority","risk_score","edge_distance_m","center_distance_m","area_m2","consumption","breaker","office","lat","lon"])
            st.sidebar.download_button("📥 نتائج Excel", data=save_results_excel(res_df), file_name="results.xlsx")

st.markdown("<br><hr><center>👨‍💻 تطوير: مشهور العباس 2026 | 00966553339838</center>", unsafe_allow_html=True)
