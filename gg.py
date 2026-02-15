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
    request_timeout_s: int = 30
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    output_dir: str = "output"
    models_dir: str = "models"
    page_title: str = "🌾 نظام الذكاء الاصطناعي لرصد الفاقد الزراعي"
    page_icon: str = "🌾"
    green_ratio_min: float = 0.0
    green_dominance: float = 1.1
    green_min_value: int = 60

cfg = AppConfig()

# ======================= التنسيق الجمالي (CSS) =======================
def local_css():
    st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #007bff; color: white; }
    .report-card {
        background: white; border-radius: 15px; padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
        border-right: 8px solid #ccc; transition: transform 0.2s;
    }
    .report-card:hover { transform: scale(1.02); }
    .badge {
        padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; color: white;
    }
    .status-high { background-color: #dc3545; border-right-color: #dc3545 !important; }
    .status-med { background-color: #ffc107; color: #212529; border-right-color: #ffc107 !important; }
    .status-low { background-color: #28a745; border-right-color: #28a745 !important; }
    </style>
    """, unsafe_allow_html=True)

# ======================= أدوات عامة =======================
def ensure_dirs(*paths):
    for p in paths: os.makedirs(p, exist_ok=True)

def clean_meter_id(val) -> str:
    if pd.isna(val): return ""
    try:
        f = float(val)
        if f.is_integer(): return str(int(f))
        return re.sub(r"\.0+$", "", str(val).strip())
    except: return str(val).strip()

def save_results_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()

def save_results_html(rows: List[List], detected_dir: str) -> bytes:
    colors = {"قصوى": "#dc3545", "متوسطة": "#ffc107", "منخفضة": "#28a745"}
    html = ["""
    <html><head><meta charset='UTF-8'>
    <style>
        body { font-family: 'Segoe UI', Tahoma; direction: rtl; background: #f4f7f6; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; padding: 20px; }
        .card { background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-top: 5px solid; }
        .info { padding: 15px; text-align: center; }
        .btn { display: inline-block; padding: 5px 15px; margin: 5px; border-radius: 5px; text-decoration: none; color: white; background: #007bff; font-size: 13px; }
    </style></head><body><div class='grid'>"""]
    
    for r in rows:
        meter_id, pr, score, edge_d, center_d, area, consumption, breaker, office, lat, lon = r
        border = colors.get(pr, "#ccc")
        pth = os.path.join(detected_dir, f"{meter_id}.png")
        img_b64 = ""
        if os.path.exists(pth):
            with open(pth, "rb") as f: img_b64 = base64.b64encode(f.read()).decode()
        
        html.append(f"""
        <div class='card' style='border-top-color: {border}'>
            <img src='data:image/png;base64,{img_b64}' width='100%'>
            <div class='info'>
                <h3 style='margin:5px 0;'>عداد {meter_id}</h3>
                <span style='color:{border}; font-weight:bold;'>أولوية {pr} ({score*100:.1f}%)</span><br>
                <small>مساحة: {area}م² | استهلاك: {consumption}</small><br>
                <a class='btn' href='https://www.google.com/maps?q={lat},{lon}'>📍 الموقع</a>
                <a class='btn' style='background:#25d366' href='https://wa.me/?text=عداد:{meter_id}%20الموقع:{lat},{lon}'>📲 واتساب</a>
            </div>
        </div>""")
    html.append("</div></body></html>")
    return "\n".join(html).encode("utf-8")

# ======================= المحرك الحسابي =======================
@st.cache_resource
def load_yolo(model_path: str): return YOLO(model_path)

class RiskModel:
    def __init__(self, model_path, scaler_path, low_thr, high_thr):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.low_thr, self.high_thr = low_thr, high_thr

    def compute(self, breaker, consumption, lon, lat, area_m2, green_ratio):
        effective_area = area_m2 * green_ratio
        X = self.scaler.transform(np.array([[breaker, consumption, lon, lat]], dtype=float))
        anomaly = self.model.predict(X)[0]
        
        r1 = 1.0 if breaker < (effective_area * 0.0013) else 0.0
        r2 = 1.0 if consumption < (effective_area * 0.20) else 0.0
        r3 = 1.0 if anomaly == 1 else 0.0
        
        score = 0.4 * r1 + 0.4 * r2 + 0.2 * r3
        pr = "قصوى" if score >= self.high_thr else "متوسطة" if score >= self.low_thr else "منخفضة"
        return score, pr

def estimate_green_ratio(image: Image.Image, box_xyxy: tuple) -> float:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    crop = image.crop((x1, y1, x2, y2))
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.size == 0: return 0.0
    R, G, B = arr[..., 0].astype(float), arr[..., 1].astype(float), arr[..., 2].astype(float)
    
    dom_mask = (G > R * cfg.green_dominance) & (G > B * cfg.green_dominance) & (G > cfg.green_min_value)
    exg = 2.0*(G/255.0) - (R/255.0) - (B/255.0)
    hsv = np.asarray(crop.convert("HSV"))
    hsv_mask = (hsv[...,0] >= 25) & (hsv[...,0] <= 67) & (hsv[...,1] >= 60)
    
    return float((dom_mask | (exg > 0.08) | hsv_mask).mean())

def detect_field_progressive(img_path, lat, lon, meter_id, model_yolo) -> Optional[FieldDetection]:
    image = Image.open(img_path).convert("RGB")
    res = model_yolo.predict(image, imgsz=640, conf=cfg.min_confidence_accept, verbose=False)[0]
    if not res.boxes: return None
    
    m_per_px = cfg.scene_size_m / 640.0
    candidates = []
    for box in res.boxes:
        xyxy = box.xyxy.cpu().numpy()[0]
        area = abs(xyxy[2]-xyxy[0]) * abs(xyxy[3]-xyxy[1]) * (m_per_px**2) * cfg.calibration_factor
        if area < cfg.min_area_m2: continue
        
        bx, by = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
        dist = geodesic((lat, lon), (lat - ((by-320)*m_per_px/111320.0), lon + ((bx-320)*m_per_px/(111320.0*math.cos(math.radians(lat)))))).meters
        
        green = estimate_green_ratio(image, tuple(xyxy))
        radius = (max(abs(xyxy[2]-xyxy[0]), abs(xyxy[3]-xyxy[1]))/2) * m_per_px
        candidates.append((max(dist-radius, 0), dist, xyxy, area, green))

    for R in range(cfg.r_start_m, cfg.r_max_m + 1, cfg.r_step_m):
        within = [c for c in candidates if c[0] <= R]
        if within:
            edge_d, cent_d, box, area, green = min(within, key=lambda x: (x[0], x[1]))
            draw = ImageDraw.Draw(image)
            draw.rectangle(box.tolist(), outline="#00ff00", width=4)
            out_p = os.path.join(cfg.detected_dir, f"{meter_id}.png")
            image.save(out_p)
            return FieldDetection(tuple(box), 0.0, int(area), (0,0), edge_d, cent_d, out_p, green)
    return None

# (دوال الـ Token والتحميل تبقى كما هي في كودك الأصلي)
def get_cdse_token():
    # ... (نفس منطق الكود الأصلي لضمان عمل الحساب) ...
    cid, csec = st.secrets["CDSE_CLIENT_ID"], st.secrets["CDSE_CLIENT_SECRET"]
    r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token", 
                      data={"grant_type":"client_credentials","client_id":cid,"client_secret":csec})
    return r.json()["access_token"]

def download_image(lat, lon, meter_id):
    path = os.path.join(cfg.images_dir, f"{meter_id}.png")
    if os.path.exists(path): return path
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
    return None

# ======================= واجهة المستعرض =======================
local_css()
ensure_dirs(cfg.images_dir, cfg.detected_dir, cfg.output_dir, cfg.models_dir)

st.title(cfg.page_title)
uploaded = st.file_uploader("📁 ارفع ملف المزارع (Excel)", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded)
    df["Subscription"] = df["Subscription"].apply(clean_meter_id)
    
    st.sidebar.header("⚙️ التحكم والفرز")
    st.sidebar.metric("إجمالي الحالات", len(df))
    
    if st.sidebar.button("🚀 بدء التحليل الذكي"):
        yolo_model = load_yolo(os.path.join(cfg.models_dir, "best.pt"))
        risk_model = RiskModel(os.path.join(cfg.models_dir, "isolation_model.joblib"), 
                               os.path.join(cfg.models_dir, "isolation_scaler.joblib"), cfg.risk_low, cfg.risk_high)
        
        results = []
        prog_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (_, row) in enumerate(df.iterrows(), 1):
            m_id = row["Subscription"]
            status_text.text(f"جاري معالجة العداد: {m_id}")
            try:
                img_p = download_image(row["y"], row["x"], m_id)
                if img_p:
                    det = detect_field_progressive(img_p, row["y"], row["x"], m_id, yolo_model)
                    if det:
                        score, pr = risk_model.compute(row["Breaker"], row["consumption"], row["x"], row["y"], det.area_m2, det.green_ratio)
                        results.append([m_id, pr, score, det.edge_distance_m, det.center_distance_m, det.area_m2, row["consumption"], row["Breaker"], row["Office"], row["y"], row["x"]])
            except: pass
            prog_bar.progress(i/len(df))
        
        if results:
            # ✅ فرز النتائج حسب الخطورة (Score) تنازلياً
            results.sort(key=lambda x: x[2], reverse=True)
            
            # عرض الإحصائيات (KPIs)
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 قصوى", len([r for r in results if r[1]=="قصوى"]))
            c2.metric("🟠 متوسطة", len([r for r in results if r[1]=="متوسطة"]))
            c3.metric("🟢 منخفضة", len([r for r in results if r[1]=="منخفضة"]))
            
            # عرض البطاقات
            for r in results:
                m_id, pr, score, edge_d, cent_d, area, cons, br, off, lat, lon = r
                st_class = "status-high" if pr=="قصوى" else "status-med" if pr=="متوسطة" else "status-low"
                
                with st.container():
                    st.markdown(f"""
                    <div class="report-card {st_class}">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <h3 style="margin:0;">عداد: {m_id} <span class="badge {st_class}">{pr}</span></h3>
                                <p style="margin:10px 0;">
                                    <b>درجة الخطورة:</b> {score*100:.1f}% | <b>المساحة:</b> {area} م² | <b>الاستهلاك:</b> {cons}<br>
                                    <b>المكتب:</b> {off} | <b>القاطع:</b> {br} أمبير
                                </p>
                            </div>
                            <img src="data:image/png;base64,{base64.b64encode(open(os.path.join(cfg.detected_dir, f'{m_id}.png'),'rb').read()).decode()}" width="150" style="border-radius:10px">
                        </div>
                        <hr>
                        <a href="https://www.google.com/maps?q={lat},{lon}" target="_blank">📍 عرض على الخريطة</a>
                    </div>
                    """, unsafe_allow_html=True)
            
            # أزرار التحميل
            res_df = pd.DataFrame(results, columns=["Subscription","priority","risk_score","edge_dist","center_dist","area_m2","consumption","breaker","office","lat","lon"])
            st.sidebar.download_button("📥 تحميل Excel", data=save_results_excel(res_df), file_name="النتائج.xlsx")
            st.sidebar.download_button("📥 تحميل تقرير HTML", data=save_results_html(results, cfg.detected_dir), file_name="تقرير_مفصل.html")

st.markdown(f"<div style='text-align:center; color:gray; padding:20px;'>تطوير: مشهور العباس 2026 | {cfg.page_icon}</div>", unsafe_allow_html=True)
