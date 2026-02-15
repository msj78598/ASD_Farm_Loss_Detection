# -*- coding: utf-8 -*-
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
    map_size: Tuple[int, int] = (640, 640)
    scene_size_m: int = 2500
    calibration_factor: float = 0.6695
    min_confidence_accept: float = 0.45
    min_area_m2: float = 5000.0
    r_start_m: int = 50
    r_step_m: int = 10
    r_max_m: int = 200
    risk_low: float = 0.40
    risk_high: float = 0.70
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    output_dir: str = "output"
    models_dir: str = "models"
    page_title: str = "🌾 نظام اكتشاف حالات الفاقد للفئة الزراعية"
    page_icon: str = "🌾"
    green_ratio_min: float = 0.0
    green_dominance: float = 1.1
    green_min_value: int = 60

cfg = AppConfig()

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

# ======================= التنسيق الجمالي (CSS) =======================
st.set_page_config(page_title=cfg.page_title, page_icon=cfg.page_icon, layout="wide")

st.markdown("""
    <style>
    .risk-card {
        background-color: white; border-radius: 12px; padding: 15px; margin-bottom: 15px;
        border-right: 10px solid #ccc; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .priority-قصوى { border-right-color: #ff4d4d !important; }
    .priority-متوسطة { border-right-color: #ffa500 !important; }
    .priority-منخفضة { border-right-color: #4CAF50 !important; }
    .stat-box { text-align: center; padding: 10px; border-radius: 8px; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ======================= الدوال الأساسية (من كودك بدون تغيير) =======================
def ensure_dirs(*paths):
    for p in paths: os.makedirs(p, exist_ok=True)

def clean_meter_id(val) -> str:
    if pd.isna(val): return ""
    s = str(val).strip()
    return re.sub(r"\.0+$", "", s)

def estimate_green_ratio(image: Image.Image, box_xyxy: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    if x2 <= x1 or y2 <= y1: return 0.0
    crop = image.crop((x1, y1, x2, y2))
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.size == 0: return 0.0
    R, G, B = arr[..., 0].astype(np.float32), arr[..., 1].astype(np.float32), arr[..., 2].astype(np.float32)
    dom_mask = (G > R * cfg.green_dominance) & (G > B * cfg.green_dominance) & (G > cfg.green_min_value)
    exg = 2.0 * (G/255.0) - (R/255.0) - (B/255.0)
    hsv = crop.convert("HSV")
    H, S, V = np.asarray(hsv.getchannel(0)), np.asarray(hsv.getchannel(1)), np.asarray(hsv.getchannel(2))
    hsv_mask = (H >= 25) & (H <= 67) & (S >= 60) & (V >= 50)
    return float((dom_mask | (exg > 0.08) | hsv_mask).mean())

def detect_field_progressive(img_path, lat, lon, meter_id, model_yolo) -> Optional[FieldDetection]:
    image = Image.open(img_path).convert("RGB")
    res = model_yolo.predict(source=image, imgsz=640, conf=cfg.min_confidence_accept, verbose=False)[0]
    if not res.boxes: return None
    
    m_per_px = cfg.scene_size_m / 640.0
    cx, cy = 320, 320
    candidates = []
    
    for box in res.boxes:
        xyxy = box.xyxy.cpu().numpy()[0]
        conf = float(box.conf.cpu().numpy()[0])
        w_px, h_px = abs(xyxy[2]-xyxy[0]), abs(xyxy[3]-xyxy[1])
        area = w_px * h_px * (m_per_px**2) * cfg.calibration_factor
        if area < cfg.min_area_m2: continue
        
        bx, by = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
        center_dist = geodesic((lat, lon), (lat - ((by-cy)*m_per_px/111320.0), lon + ((bx-cx)*m_per_px/(111320.0*math.cos(math.radians(lat)))))).meters
        green_r = estimate_green_ratio(image, tuple(xyxy))
        radius_m = (max(w_px, h_px) / 2) * m_per_px
        edge_dist = max(center_dist - radius_m, 0.0)
        candidates.append((edge_dist, center_dist, xyxy, conf, int(area), green_r))

    for R in range(cfg.r_start_m, cfg.r_max_m + 1, cfg.r_step_m):
        within = [c for c in candidates if c[0] <= R]
        if within:
            edge, cent, box, cf, ar, gr = min(within, key=lambda x: (x[0], x[1]))
            draw = ImageDraw.Draw(image)
            draw.rectangle(box.tolist(), outline="green", width=4)
            out_p = os.path.join(cfg.detected_dir, f"{meter_id}.png")
            image.save(out_p)
            return FieldDetection(tuple(box), cf, ar, (0,0), edge, cent, out_p, gr)
    return None

class RiskModel:
    def __init__(self, m_path, s_path, low, high):
        self.model = joblib.load(m_path)
        self.scaler = joblib.load(s_path)
        self.low, self.high = low, high

    def compute(self, breaker, consumption, lon, lat, area_m2, green_ratio):
        eff_area = area_m2 * green_ratio
        X = self.scaler.transform([[breaker, consumption, lon, lat]])
        anomaly = self.model.predict(X)[0]
        r1 = 1.0 if breaker < (eff_area * 0.0013) else 0.0
        r2 = 1.0 if consumption < (eff_area * 0.20) else 0.0
        r3 = 1.0 if anomaly == 1 else 0.0
        score = 0.4 * r1 + 0.4 * r2 + 0.2 * r3
        pr = "قصوى" if score >= self.high else "متوسطة" if score >= self.low else "منخفضة"
        return score, pr

# (دوال التحميل و Token الأقمار الصناعية تبقى كما هي في كودك)
def get_cdse_token():
    cid, csec = st.secrets["CDSE_CLIENT_ID"], st.secrets["CDSE_CLIENT_SECRET"]
    r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token", 
                      data={"grant_type":"client_credentials","client_id":cid,"client_secret":csec})
    return r.json()["access_token"]

def download_image(lat, lon, meter_id):
    path = os.path.join(cfg.images_dir, f"{meter_id}.png")
    if os.path.exists(path): return path
    try:
        token = get_cdse_token()
        d = (cfg.scene_size_m/2)/111320.0
        payload = {
            "input": {"bounds": {"bbox": [lon-d, lat-d, lon+d, lat+d]}, "data": [{"type": "sentinel-2-l2a"}]},
            "output": {"width": 640, "height": 640, "responses": [{"format": {"type": "image/png"}}]},
            "evalscript": "//VERSION=3\nfunction setup(){return{input:['B04','B03','B02'],output:{bands:3}}}\nfunction evaluatePixel(s){return[s.B04*1.8,s.B03*1.8,s.B02*1.8]}"
        }
        r = requests.post("https://sh.dataspace.copernicus.eu/api/v1/process", headers={"Authorization": f"Bearer {token}"}, json=payload)
        if r.status_code == 200:
            with open(path, "wb") as f: f.write(r.content)
            return path
    except: return None
    return None

# ======================= واجهة التطبيق الرئيسية =======================
ensure_dirs(cfg.images_dir, cfg.detected_dir, cfg.output_dir, cfg.models_dir)
st.title(cfg.page_title)

uploaded = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded)
    df["Subscription"] = df["Subscription"].apply(clean_meter_id)
    
    if st.sidebar.button("🚀 بدء التحليل الاحترافي"):
        yolo = YOLO(os.path.join(cfg.models_dir, "best.pt"))
        risk_engine = RiskModel(os.path.join(cfg.models_dir, "isolation_model.joblib"), 
                                os.path.join(cfg.models_dir, "isolation_scaler.joblib"), cfg.risk_low, cfg.risk_high)
        
        final_results = []
        prog = st.progress(0)
        
        for i, (_, row) in enumerate(df.iterrows()):
            m_id = row["Subscription"]
            img_p = download_image(row["y"], row["x"], m_id)
            if img_p:
                det = detect_field_progressive(img_p, row["y"], row["x"], m_id, yolo)
                if det:
                    score, pr = risk_engine.compute(row["Breaker"], row["consumption"], row["x"], row["y"], det.area_m2, det.green_ratio)
                    final_results.append({
                        "id": m_id, "pr": pr, "score": score, "det": det, "row": row
                    })
            prog.progress((i+1)/len(df))
        
        if final_results:
            # ✅ الترتيب حسب الخطورة (Score) تنازلياً
            final_results.sort(key=lambda x: x["score"], reverse=True)
            
            # عرض إحصائيات سريعة
            cols = st.columns(3)
            with cols[0]: st.markdown('<div class="stat-box" style="background:#ff4d4d">قصوى: '+str(len([x for x in final_results if x['pr']=="قصوى"]))+'</div>', unsafe_allow_html=True)
            with cols[1]: st.markdown('<div class="stat-box" style="background:#ffa500">متوسطة: '+str(len([x for x in final_results if x['pr']=="متوسطة"]))+'</div>', unsafe_allow_html=True)
            with cols[2]: st.markdown('<div class="stat-box" style="background:#4CAF50">منخفضة: '+str(len([x for x in final_results if x['pr']=="منخفضة"]))+'</div>', unsafe_allow_html=True)
            st.write("---")

            # عرض النتائج في بطاقات احترافية
            for item in final_results:
                m_id, pr, score, det, row = item["id"], item["pr"], item["score"], item["det"], item["row"]
                with st.container():
                    st.markdown(f"""
                    <div class="risk-card priority-{pr}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="flex: 2;">
                                <h3 style="margin:0; color:#333;">عداد: {m_id} <small>({pr})</small></h3>
                                <p style="font-size:14px; color:#666;">
                                    <b>درجة الخطورة:</b> {score*100:.1f}% | <b>المساحة المكتشفة:</b> {det.area_m2} م²<br>
                                    <b>الاستهلاك:</b> {row['consumption']} | <b>القاطع:</b> {row['Breaker']} أمبير | <b>المكتب:</b> {row['Office']}
                                </p>
                                <a href="https://www.google.com/maps?q={row['y']},{row['x']}" target="_blank">📍 عرض الموقع على الخريطة</a>
                            </div>
                            <div style="flex: 1; text-align: right;">
                                <img src="data:image/png;base64,{base64.b64encode(open(det.out_img_path, 'rb').read()).decode()}" width="200" style="border-radius:8px;">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # زر تحميل النتائج
            res_df = pd.DataFrame([{
                "Subscription": x["id"], "priority": x["pr"], "risk_score": x["score"], 
                "area_m2": x["det"].area_m2, "consumption": x["row"]["consumption"]
            } for x in final_results])
            st.sidebar.download_button("📥 تحميل النتائج Excel", data=res_df.to_csv().encode('utf-8-sig'), file_name="results.csv")
