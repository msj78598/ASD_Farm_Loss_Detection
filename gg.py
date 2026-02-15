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

# ======================= الإعدادات الاحترافية =======================
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
    page_title: str = "🌾 نظام الذكاء الاصطناعي لرصد الهدر الزراعي"

cfg = AppConfig()

# ======================= نظام تخزين الجلسة (لحل مشكلة الالتفاف) =======================
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = []
if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False

# ======================= واجهة المستخدم CSS =======================
st.set_page_config(page_title=cfg.page_title, layout="wide")
st.markdown("""
    <style>
    .main-card {
        background: #ffffff; border-radius: 15px; padding: 20px; margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); display: flex; align-items: center;
        border-right: 12px solid #ddd; transition: 0.3s;
    }
    .status-قصوى { border-right-color: #ff4d4d !important; background: #fff5f5; }
    .status-متوسطة { border-right-color: #ffa500 !important; background: #fffaf0; }
    .status-منخفضة { border-right-color: #2ecc71 !important; background: #f7fff9; }
    .badge { padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: bold; color: white; }
    .badge-قصوى { background: #ff4d4d; }
    .badge-متوسطة { background: #ffa500; }
    .badge-منخفضة { background: #2ecc71; }
    </style>
""", unsafe_allow_html=True)

# ======================= وظائف المعالجة الأساسية =======================
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

def download_image(lat, lon, m_id):
    path = os.path.join(cfg.images_dir, f"{m_id}.png")
    if os.path.exists(path): return path
    try:
        cid, csec = st.secrets["CDSE_CLIENT_ID"], st.secrets["CDSE_CLIENT_SECRET"]
        res = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token", 
                           data={"grant_type":"client_credentials","client_id":cid,"client_secret":csec}, timeout=10)
        token = res.json()["access_token"]
        d = (cfg.scene_size_m/2)/111320.0
        payload = {
            "input": {"bounds": {"bbox": [lon-d, lat-d, lon+d, lat+d]}, "data": [{"type": "sentinel-2-l2a"}]},
            "output": {"width": 640, "height": 640, "responses": [{"format": {"type": "image/png"}}]},
            "evalscript": "//VERSION=3\nfunction setup(){return{input:['B04','B03','B02'],output:{bands:3}}}\nfunction evaluatePixel(s){return[s.B04*1.8,s.B03*1.8,s.B02*1.8]}"
        }
        r = requests.post("https://sh.dataspace.copernicus.eu/api/v1/process", headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=20)
        if r.status_code == 200:
            with open(path, "wb") as f: f.write(r.content)
            return path
    except: return None

# ======================= واجهة التطبيق =======================
os.makedirs(cfg.images_dir, exist_ok=True)
os.makedirs(cfg.detected_dir, exist_ok=True)

st.title(cfg.page_title)
file = st.file_uploader("📂 ارفع ملف الإكسل لبدء المعالجة", type=["xlsx"])

if file and not st.session_state.is_analyzing:
    df = pd.read_excel(file)
    if st.sidebar.button("🚀 بدء التحليل الذكي"):
        yolo = YOLO(os.path.join(cfg.models_dir, "best.pt"))
        risk_mod = joblib.load(os.path.join(cfg.models_dir, "isolation_model.joblib"))
        scaler = joblib.load(os.path.join(cfg.models_dir, "isolation_scaler.joblib"))
        
        results = []
        bar = st.progress(0)
        for i, (_, row) in enumerate(df.iterrows()):
            m_id = str(row["Subscription"]).split('.')[0]
            img_path = download_image(row["y"], row["x"], m_id)
            if img_path:
                # التحليل باستخدام YOLO
                pred = yolo.predict(img_path, imgsz=640, conf=cfg.min_confidence_accept, verbose=False)[0]
                if pred.boxes:
                    # ✅ حساب المساحة الحقيقي بناءً على حجم البوكس المكتشف
                    box = pred.boxes[0].xyxy.cpu().numpy()[0]
                    w, h = abs(box[2]-box[0]), abs(box[3]-box[1])
                    m_px = cfg.scene_size_m / 640.0
                    calc_area = int(w * h * (m_px**2) * cfg.calibration_factor)
                    
                    # حساب الـ Risk
                    X = scaler.transform([[row["Breaker"], row["consumption"], row["x"], row["y"]]])
                    anom = risk_mod.predict(X)[0]
                    score = (0.4 * (1.0 if row["consumption"] < (calc_area*0.2) else 0.0)) + (0.4 * (1.0 if row["Breaker"] < (calc_area*0.0013) else 0.0)) + (0.2 * (1.0 if anom == 1 else 0.0))
                    pr = "قصوى" if score >= cfg.risk_high else "متوسطة" if score >= cfg.risk_low else "منخفضة"
                    
                    # حفظ الصورة
                    out_p = os.path.join(cfg.detected_dir, f"{m_id}.png")
                    Image.open(img_path).save(out_p) # يمكن إضافة رسم المربع هنا
                    
                    results.append({
                        "m_id": m_id, "pr": pr, "score": score, "area": calc_area,
                        "cons": row["consumption"], "br": row["Breaker"], "lat": row["y"], "lon": row["x"], "img_path": out_p
                    })
            bar.progress((i+1)/len(df))
        
        # ✅ الترتيب النهائي وتخزين الجلسة
        results.sort(key=lambda x: x['score'], reverse=True)
        st.session_state.analysis_data = results
        st.session_state.is_analyzing = True
        st.rerun()

# --- عرض النتائج الثابتة ---
if st.session_state.is_analyzing:
    res = st.session_state.analysis_data
    st.sidebar.success(f"✅ اكتمل التحليل: {len(res)} حالة")
    
    # بطاقات النتائج
    for item in res:
        b64 = get_image_base64(item['img_path'])
        st.markdown(f"""
        <div class="main-card status-{item['pr']}">
            <div style="flex: 2; padding-right: 20px;">
                <h3 style="margin:0;">عداد: {item['m_id']} <span class="badge badge-{item['pr']}">{item['pr']}</span></h3>
                <p style="margin: 10px 0; color: #555;">
                    <b>خطورة:</b> {item['score']*100:.1f}% | <b>المساحة الفعالية:</b> {item['area']:,} م²<br>
                    <b>الاستهلاك:</b> {item['cons']} | <b>القاطع:</b> {item['br']} أمبير
                </p>
                <a href="https://maps.google.com/?q={item['lat']},{item['lon']}" target="_blank" style="color:#007bff; text-decoration:none;">📍 موقع المزرعة</a>
            </div>
            <div style="flex: 1; text-align: left;">
                <img src="data:image/png;base64,{b64}" width="220" style="border-radius:12px; border:3px solid #eee;">
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.sidebar.button("🗑️ مسح وإعادة تحميل"):
        st.session_state.is_analyzing = False
        st.session_state.analysis_data = []
        st.rerun()
