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

# ======================= الإعدادات والـ Dataclasses =======================
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
    models_dir: str = "models"
    page_title: str = "🌾 نظام رصد الفاقد الزراعي الذكي"

cfg = AppConfig()

@dataclass
class FieldDetection:
    bbox_xyxy: tuple
    conf: float
    area_m2: int
    center_latlon: tuple
    edge_distance_m: float
    center_distance_m: float
    out_img_path: str
    green_ratio: float

# ======================= تنسيق الواجهة الاحترافي =======================
st.set_page_config(page_title=cfg.page_title, layout="wide")

st.markdown("""
    <style>
    .risk-card {
        background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;
        border-right: 10px solid #ccc; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        display: flex; justify-content: space-between; align-items: center;
    }
    .priority-قصوى { border-right-color: #ff4d4d; background-color: #fff5f5; }
    .priority-متوسطة { border-right-color: #ffa500; background-color: #fffaf0; }
    .priority-منخفضة { border-right-color: #4CAF50; background-color: #f7fff7; }
    .stat-card { background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #ddd; }
    </style>
""", unsafe_allow_html=True)

# ======================= منطق الحسابات (من كودك الأصلي) =======================
def estimate_green_ratio(image, box_xyxy):
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    crop = image.crop((x1, y1, x2, y2))
    arr = np.asarray(crop)
    if arr.size == 0: return 0.0
    R, G, B = arr[..., 0].astype(float), arr[..., 1].astype(float), arr[..., 2].astype(float)
    dom = (G > R * 1.1) & (G > B * 1.1) & (G > 60)
    exg = 2.0*(G/255.0) - (R/255.0) - (B/255.0)
    hsv = np.asarray(crop.convert("HSV"))
    hsv_m = (hsv[...,0] >= 25) & (hsv[...,0] <= 67) & (hsv[...,1] >= 60)
    return float((dom | (exg > 0.08) | hsv_m).mean())

def download_image(lat, lon, m_id):
    path = os.path.join(cfg.images_dir, f"{m_id}.png")
    if os.path.exists(path): return path
    try:
        cid, csec = st.secrets["CDSE_CLIENT_ID"], st.secrets["CDSE_CLIENT_SECRET"]
        t_res = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token", 
                             data={"grant_type":"client_credentials","client_id":cid,"client_secret":csec}, timeout=15)
        token = t_res.json()["access_token"]
        d = (cfg.scene_size_m/2)/111320.0
        payload = {
            "input": {"bounds": {"bbox": [lon-d, lat-d, lon+d, lat+d]}, "data": [{"type": "sentinel-2-l2a"}]},
            "output": {"width": 640, "height": 640, "responses": [{"format": {"type": "image/png"}}]},
            "evalscript": "//VERSION=3\nfunction setup(){return{input:['B04','B03','B02'],output:{bands:3}}}\nfunction evaluatePixel(s){return[s.B04*1.8,s.B03*1.8,s.B02*1.8]}"
        }
        r = requests.post("https://sh.dataspace.copernicus.eu/api/v1/process", headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=30)
        if r.status_code == 200:
            with open(path, "wb") as f: f.write(r.content)
            return path
    except: return None
    return None

def detect_field_progressive(img_path, lat, lon, m_id, model_yolo):
    image = Image.open(img_path).convert("RGB")
    res = model_yolo.predict(image, imgsz=640, conf=cfg.min_confidence_accept, verbose=False)[0]
    if not res.boxes: return None
    
    m_per_px = cfg.scene_size_m / 640.0
    candidates = []
    for box in res.boxes:
        xyxy = box.xyxy.cpu().numpy()[0]
        # ✅ الحساب الفعلي للمساحة بناءً على الأبعاد المكتشفة
        w_px, h_px = abs(xyxy[2]-xyxy[0]), abs(xyxy[3]-xyxy[1])
        real_area = w_px * h_px * (m_per_px**2) * cfg.calibration_factor
        if real_area < cfg.min_area_m2: continue
        
        bx, by = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
        dist = geodesic((lat, lon), (lat - ((by-320)*m_per_px/111320.0), lon + ((bx-320)*m_per_px/(111320.0*math.cos(math.radians(lat)))))).meters
        green = estimate_green_ratio(image, tuple(xyxy))
        radius = (max(w_px, h_px)/2) * m_per_px
        edge_d = max(dist - radius, 0.0)
        candidates.append((edge_d, dist, xyxy, int(real_area), green))

    for R in range(cfg.r_start_m, cfg.r_max_m + 1, cfg.r_step_m):
        within = [c for c in candidates if c[0] <= R]
        if within:
            edge, dist, xy, ar, gr = min(within, key=lambda x: (x[0], x[1]))
            draw = ImageDraw.Draw(image)
            draw.rectangle(xy.tolist(), outline="#00ff00", width=5)
            out_p = os.path.join(cfg.detected_dir, f"{m_id}.png")
            image.save(out_p)
            return FieldDetection(tuple(xy), 0.0, ar, (0,0), edge, dist, out_p, gr)
    return None

# ======================= إدارة حالة الجلسة (Session State) =======================
if 'results' not in st.session_state:
    st.session_state.results = []
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# ======================= واجهة التطبيق =======================
os.makedirs(cfg.images_dir, exist_ok=True)
os.makedirs(cfg.detected_dir, exist_ok=True)

st.title(cfg.page_title)
uploaded = st.file_uploader("📁 ارفع ملف المزارع (Excel)", type=["xlsx"])

if uploaded and not st.session_state.analyzed:
    df = pd.read_excel(uploaded)
    if st.button("🚀 بدء التحليل الفعلي"):
        yolo_model = YOLO(os.path.join(cfg.models_dir, "best.pt"))
        # استدعاء نماذج المخاطر (تأكد من مسارات الملفات لديك)
        risk_model = joblib.load(os.path.join(cfg.models_dir, "isolation_model.joblib"))
        risk_scaler = joblib.load(os.path.join(cfg.models_dir, "isolation_scaler.joblib"))
        
        results_list = []
        progress_bar = st.progress(0)
        
        for i, (_, row) in enumerate(df.iterrows()):
            m_id = str(row["Subscription"]).split('.')[0]
            img_p = download_image(row["y"], row["x"], m_id)
            if img_p:
                det = detect_field_progressive(img_p, row["y"], row["x"], m_id, yolo_model)
                if det:
                    # حساب المخاطر الفعلي
                    eff_area = det.area_m2 * det.green_ratio
                    X = risk_scaler.transform([[row["Breaker"], row["consumption"], row["x"], row["y"]]])
                    anomaly = risk_model.predict(X)[0]
                    r1 = 1.0 if row["Breaker"] < (eff_area * 0.0013) else 0.0
                    r2 = 1.0 if row["consumption"] < (eff_area * 0.20) else 0.0
                    score = (0.4 * r1) + (0.4 * r2) + (0.2 * (1.0 if anomaly == 1 else 0.0))
                    pr = "قصوى" if score >= cfg.risk_high else "متوسطة" if score >= cfg.risk_low else "منخفضة"
                    
                    results_list.append({
                        "m_id": m_id, "pr": pr, "score": score, "area": det.area_m2,
                        "cons": row["consumption"], "br": row["Breaker"], "off": row["Office"],
                        "lat": row["y"], "lon": row["x"], "img": det.out_img_path
                    })
            progress_bar.progress((i + 1) / len(df))
        
        # ترتيب وتخزين النتائج
        results_list.sort(key=lambda x: x['score'], reverse=True)
        st.session_state.results = results_list
        st.session_state.analyzed = True
        st.rerun()

# عرض النتائج الثابتة
if st.session_state.analyzed:
    res = st.session_state.results
    
    # ملخص سريع
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="stat-card">🔴 قصوى<br><b>{len([x for x in res if x["pr"]=="قصوى"])}</b></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card">🟠 متوسطة<br><b>{len([x for x in res if x["pr"]=="متوسطة"])}</b></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-card">🟢 منخفضة<br><b>{len([x for x in res if x["pr"]=="منخفضة"])}</b></div>', unsafe_allow_html=True)
    
    st.write("---")

    for item in res:
        with st.container():
            st.markdown(f"""
            <div class="risk-card priority-{item['pr']}">
                <div style="flex: 2;">
                    <h3 style="margin:0;">عداد: {item['m_id']}</h3>
                    <p style="margin:10px 0;">
                        <b>أولوية:</b> {item['pr']} | <b>المخاطرة:</b> {item['score']*100:.1f}%<br>
                        <b>المساحة:</b> {item['area']:,} م² | <b>الاستهلاك:</b> {item['cons']}<br>
                        <b>المكتب:</b> {item['off']} | <b>القاطع:</b> {item['br']} أمبير
                    </p>
                    <a href="https://www.google.com/maps?q={item['lat']},{item['lon']}" target="_blank">📍 عرض على الخريطة</a>
                </div>
                <div style="flex: 1; text-align: right;">
                    <img src="data:image/png;base64,{base64.b64encode(open(item['img'], 'rb').read()).decode()}" width="200" style="border-radius:10px; border: 2px solid #ddd;">
                </div>
            </div>
            """, unsafe_allow_html=True)

    if st.sidebar.button("🗑️ تحليل ملف جديد"):
        st.session_state.analyzed = False
        st.session_state.results = []
        st.rerun()
