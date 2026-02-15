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
    models_dir: str = "models"
    page_title: str = "🌾 نظام رصد الفاقد الزراعي المطور"

cfg = AppConfig()

@dataclass
class FieldDetection:
    bbox_xyxy: tuple; conf: float; area_m2: int; center_latlon: tuple
    edge_distance_m: float; center_distance_m: float; out_img_path: str; green_ratio: float

# ======================= تحسينات الواجهة =======================
st.set_page_config(page_title=cfg.page_title, layout="wide")

st.markdown("""
    <style>
    .risk-card {
        background-color: white; border-radius: 15px; padding: 20px; margin-bottom: 15px;
        border-right: 12px solid #ccc; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        display: flex; justify-content: space-between; align-items: center;
    }
    .priority-قصوى { border-right-color: #e74c3c !important; background-color: #fdf2f2; }
    .priority-متوسطة { border-right-color: #f39c12 !important; background-color: #fef9e7; }
    .priority-منخفضة { border-right-color: #27ae60 !important; background-color: #f4faf6; }
    .badge { padding: 5px 12px; border-radius: 20px; color: white; font-size: 0.8em; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ======================= منطق العمل الأساسي =======================
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

def detect_field_progressive(img_path, lat, lon, m_id, model_yolo):
    img = Image.open(img_path).convert("RGB")
    res = model_yolo.predict(img, imgsz=640, conf=cfg.min_confidence_accept, verbose=False)[0]
    if not res.boxes: return None
    m_px = cfg.scene_size_m / 640.0
    cands = []
    for b in res.boxes:
        xy = b.xyxy.cpu().numpy()[0]
        area = (abs(xy[2]-xy[0])*abs(xy[3]-xy[1])) * (m_px**2) * cfg.calibration_factor
        if area < cfg.min_area_m2: continue
        dist = geodesic((lat, lon), (lat - (( (xy[1]+xy[3])/2 -320)*m_px/111320), lon + (( (xy[0]+xy[2])/2 -320)*m_px/(111320*math.cos(math.radians(lat)))))).meters
        gr = estimate_green_ratio(img, xy)
        edge = max(dist - (max(abs(xy[2]-xy[0]), abs(xy[3]-xy[1]))/2 * m_px), 0)
        cands.append((edge, dist, xy, area, gr))
    
    for R in range(cfg.r_start_m, cfg.r_max_m+1, cfg.r_step_m):
        within = [c for c in cands if c[0] <= R]
        if within:
            edge, dist, xy, ar, gr = min(within, key=lambda x: (x[0], x[1]))
            draw = ImageDraw.Draw(img)
            draw.rectangle(xy.tolist(), outline="#27ae60", width=5)
            out = os.path.join(cfg.detected_dir, f"{m_id}.png")
            img.save(out)
            return FieldDetection(tuple(xy), 0.0, int(ar), (0,0), edge, dist, out, gr)
    return None

# ======================= واجهة التطبيق =======================
os.makedirs(cfg.images_dir, exist_ok=True)
os.makedirs(cfg.detected_dir, exist_ok=True)

st.title(cfg.page_title)

# حماية البيانات من الاختفاء باستخدام session_state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

uploaded = st.file_uploader("📁 ارفع ملف الاكسل", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded)
    if st.sidebar.button("🚀 بدء التحليل (لن تختفي النتائج)"):
        with st.spinner("جاري معالجة البيانات واستدعاء الأقمار الصناعية..."):
            yolo = YOLO(os.path.join(cfg.models_dir, "best.pt"))
            # نموذج Risk (تبسيط للمثال، استخدم منطقك المسجل)
            results = []
            prog = st.progress(0)
            
            # --- حلقة المعالجة ---
            for i, (_, row) in enumerate(df.iterrows()):
                m_id = str(row["Subscription"]).split('.')[0]
                # استدعاء الصور والتحليل (تم اختصارها هنا لسرعة الرد)
                # ملاحظة: دالة download_image و compute_risk تضاف هنا كما في كودك
                # بعد الحصول على Score و Priority و Detection:
                
                # تجربة وهمية للمثال (استبدلها بمنطقك):
                results.append({
                    "id": m_id, "pr": "قصوى" if i%3==0 else "متوسطة", 
                    "score": 0.85 if i%3==0 else 0.50, 
                    "area": 12000, "cons": row["consumption"], "br": row["Breaker"],
                    "lat": row["y"], "lon": row["x"], "img": f"{cfg.detected_dir}/{m_id}.png"
                })
                prog.progress((i+1)/len(df))
            
            # حفظ النتائج في "الخزنة"
            results.sort(key=lambda x: x['score'], reverse=True)
            st.session_state.analysis_results = results

# عرض النتائج من "الخزنة" (حتى لو ضغطت أي شيء آخر ستبقى هنا)
if st.session_state.analysis_results:
    res = st.session_state.analysis_results
    st.success(f"✅ تم تحليل {len(res)} حالة. النتائج مرتبة حسب الخطورة:")
    
    for item in res:
        color = "#e74c3c" if item['pr'] == "قصوى" else "#f39c12"
        st.markdown(f"""
        <div class="risk-card priority-{item['pr']}">
            <div style="flex: 2;">
                <h3 style="margin:0;">عداد: {item['id']} <span class="badge" style="background:{color}">{item['pr']}</span></h3>
                <p>خطورة: {item['score']*100:.1f}% | مساحة: {item['area']} م² | استهلاك: {item['cons']}</p>
                <a href="https://maps.google.com/?q={item['lat']},{item['lon']}" target="_blank">📍 موقع المزرعة</a>
            </div>
            <div style="flex: 1; text-align: right;">
                <img src="data:image/png;base64,{base64.b64encode(open(item['img'], 'rb').read()).decode() if os.path.exists(item['img']) else ''}" width="180" style="border-radius:10px;">
            </div>
        </div>
        """, unsafe_allow_html=True)

    # زر التحميل الآن يعمل دون مسح النتائج
    res_df = pd.DataFrame(res).drop(columns=['img'])
    st.sidebar.download_button("📥 تحميل التقرير النهائي", data=res_df.to_csv().encode('utf-8-sig'), file_name="results.csv")

if st.sidebar.button("🗑️ مسح النتائج والبدء من جديد"):
    st.session_state.analysis_results = None
    st.rerun()
