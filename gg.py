# -*- coding: utf-8 -*-
"""
نظام اكتشاف حالات الفاقد للفئة الزراعية - النسخة المطورة 2026
تطوير: مشهور العباس
تحديث: منطق الخطورة المتدرج ونسبة الخضرة الفعلية
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

# ======================= إعدادات التطبيق =======================
@dataclass
class AppConfig:
    map_size: Tuple[int, int] = (640, 640)
    scene_size_m: int = 2500
    calibration_factor: float = 0.6695
    min_confidence_accept: float = 0.45
    min_area_m2: float = 5000.0

    # إعدادات البحث التدريجي
    r_start_m: int = 50
    r_step_m: int = 10
    r_max_m: int = 250  # تمت زيادتها لضمان الشمولية

    # عتبات التصنيف الجديد
    thr_critical: float = 0.80  # حرجة جداً
    thr_high: float = 0.60      # قصوى
    thr_medium: float = 0.35    # متوسطة
    
    request_timeout_s: int = 30
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    output_dir: str = "output"
    models_dir: str = "models"
    page_title: str = "🌾 نظام اكتشاف حالات الفاقد للفئة الزراعية"
    page_icon: str = "🌾"

    # فلترة الخضرة
    green_ratio_min: float = 0.0  # تم ضبطها لصفر بناءً على طلبك لمنع القفز
    green_dominance: float = 1.1
    green_min_value: int = 60

cfg = AppConfig()

# ======================= أدوات عامة =======================
def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def clean_meter_id(val) -> str:
    if pd.isna(val): return ""
    try:
        f = float(val)
        if f.is_integer(): return str(int(f))
        s = str(val).strip()
        return re.sub(r"\.0+$", "", s)
    except:
        return str(val).strip()

def save_results_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf.read()

def get_risk_color(score: float) -> str:
    """تحديد لون متدرج حسب مستوى الخطورة"""
    if score >= cfg.thr_critical: return "#8B0000" # أحمر داكن
    if score >= cfg.thr_high: return "#FF0000"     # أحمر
    if score >= cfg.thr_medium: return "#FF8C00"   # برتقالي
    return "#2E8B57"                               # أخضر

def save_results_html(rows: List[List], detected_dir: str) -> bytes:
    html = ["<html><head><meta charset='UTF-8'><style>body{font-family:'Segoe UI',tahoma;direction:rtl;background:#f8f9fa;padding:20px;} .grid{display:flex;flex-wrap:wrap;justify-content:center;} .card{background:white;border-radius:15px;box-shadow:0 10px 20px rgba(0,0,0,0.1);width:320px;margin:15px;overflow:hidden;border-top:8px solid #ccc;transition:0.3s;} .card:hover{transform:translateY(-5px);}</style></head><body><div class='grid'>"]
    for r in rows:
        meter_id, priority, score, edge_d, center_d, area, consumption, breaker, office, lat, lon, g_ratio = r
        color = get_risk_color(score)
        pth = os.path.join(detected_dir, f"{meter_id}.png")
        img_tag = ""
        if os.path.exists(pth):
            with open(pth, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            img_tag = f"<img src='data:image/png;base64,{img_b64}' style='width:100%;height:200px;object-fit:cover;'>"
        
        html.append(f"""
<div class='card' style='border-top-color:{color};'>
  {img_tag}
  <div style='padding:15px;text-align:center;'>
    <h3 style='margin:0 0 10px 0;'>عداد {meter_id}</h3>
    <div style='background:{color};color:white;padding:5px 10px;border-radius:20px;display:inline-block;font-weight:bold;margin-bottom:10px;'>{priority} ({score*100:.1f}%)</div>
    <p style='font-size:13px;color:#444;line-height:1.6;'>
        <b>مكتب:</b> {office}<br>
        <b>مساحة فعلية:</b> {int(area * g_ratio)} م² (خضرة {g_ratio*100:.0f}%)<br>
        <b>الاستهلاك:</b> {consumption} | <b>القاطع:</b> {breaker}<br>
        <b>المسافة:</b> {edge_d:.1f} متر
    </p>
    <div style='margin-top:15px;'>
        <a href='https://www.google.com/maps?q={lat},{lon}' target='_blank' style='text-decoration:none;color:#007bff;font-weight:bold;'>📍 فتح الموقع</a>
    </div>
  </div>
</div>""")
    html.append("</div></body></html>")
    return "\n".join(html).encode("utf-8")

# ======================= نماذج الذكاء الاصطناعي =======================
class RiskModel:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def compute(self, breaker, consumption, lon, lat, area_m2, green_ratio):
        # 1. حساب المساحة الخضراء الفعلية
        effective_area = area_m2 * green_ratio
        
        # 2. الاحتياج المتوقع (المعايير المرجعية)
        target_cons = effective_area * 0.20
        target_br = effective_area * 0.0013

        # 3. حساب نسبة العجز المتدرجة (Ratio-based)
        # كلما زاد الفرق بين المتوقع والفعلي، زادت الخطورة
        cons_risk = max(0, min(1, (target_cons - consumption) / target_cons)) if target_cons > 0 else 0
        br_risk = max(0, min(1, (target_br - breaker) / target_br)) if target_br > 0 else 0

        # 4. عامل الشذوذ من Isolation Forest
        X = self.scaler.transform([[breaker, consumption, lon, lat]])
        anomaly = 1.0 if self.model.predict(X)[0] == 1 else 0.0

        # 5. الجمع الموزون (الأوزان متدرجة الآن)
        score = (0.50 * cons_risk) + (0.30 * br_risk) + (0.20 * anomaly)

        # 6. التصنيف
        if score >= cfg.thr_critical: pr = "حرجة جداً"
        elif score >= cfg.thr_high: pr = "قصوى"
        elif score >= cfg.thr_medium: pr = "متوسطة"
        else: pr = "منخفضة"

        return score, pr

# ======================= دوال المعالجة والتحليل =======================
def estimate_green_ratio(image: Image.Image, box_xyxy) -> float:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    crop = image.crop((x1, y1, x2, y2))
    arr = np.asarray(crop)
    if arr.size == 0: return 0.0
    R, G, B = arr[...,0].astype(float), arr[...,1].astype(float), arr[...,2].astype(float)
    # قناع الخضرة: تداخل ExG و Dominance
    mask = (G > R * cfg.green_dominance) & (G > B * cfg.green_dominance) & (G > cfg.green_min_value)
    return float(mask.mean())

def detect_field_progressive(img_path, lat, lon, meter_id, model_yolo) -> Optional[dict]:
    image = Image.open(img_path).convert("RGB")
    res = model_yolo.predict(source=image, imgsz=640, conf=cfg.min_confidence_accept, verbose=False)[0]
    if not res or not res.boxes: return None

    m_per_px = cfg.scene_size_m / 640.0
    cx, cy = image.width / 2, image.height / 2
    candidates = []

    for box in res.boxes:
        xyxy = box.xyxy.cpu().numpy()[0]
        w, h = abs(xyxy[2]-xyxy[0]), abs(xyxy[3]-xyxy[1])
        area = (w * h * (m_per_px**2)) * cfg.calibration_factor
        if area < cfg.min_area_m2: continue

        bx, by = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
        dist_m = geodesic((lat, lon), (lat - ((by-cy)*m_per_px/111320), lon + ((bx-cx)*m_per_px/(111320*math.cos(math.radians(lat)))))).meters
        
        g_ratio = estimate_green_ratio(image, xyxy)
        # حساب المسافة من حافة الحقل
        radius_m = (max(w, h) / 2) * m_per_px
        edge_dist = max(dist_m - radius_m, 0.0)
        
        candidates.append({'edge': edge_dist, 'center': dist_m, 'box': xyxy, 'area': int(area), 'green': g_ratio})

    if not candidates: return None

    # تطبيق منطق البحث التدريجي
    for R in range(cfg.r_start_m, cfg.r_max_m + 1, cfg.r_step_m):
        within = [c for c in candidates if c['edge'] <= R]
        if within:
            best = min(within, key=lambda x: (x['edge'], x['center']))
            # رسم الكشف
            draw = ImageDraw.Draw(image)
            draw.rectangle(best['box'].tolist(), outline="green", width=5)
            draw.line([(cx, cy), ((best['box'][0]+best['box'][2])/2, (best['box'][1]+best['box'][3])/2)], fill="yellow", width=2)
            
            out_p = os.path.join(cfg.detected_dir, f"{meter_id}.png")
            image.save(out_p)
            best['path'] = out_p
            return best
    return None

# ======================= وظائف اتصال CDSE =======================
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

def get_cdse_token():
    cid = st.secrets.get("CDSE_CLIENT_ID")
    csec = st.secrets.get("CDSE_CLIENT_SECRET")
    data = {"grant_type":"client_credentials","client_id":cid,"client_secret":csec}
    r = requests.post(TOKEN_URL, data=data, timeout=20)
    return r.json()["access_token"]

@st.cache_data(ttl=3600*24)
def download_image(lat, lon, meter_id):
    img_path = os.path.join(cfg.images_dir, f"{meter_id}.png")
    if os.path.exists(img_path): return img_path
    
    try:
        token = get_cdse_token()
        half = cfg.scene_size_m / 2.0
        dlat = half / 111320.0
        dlon = half / (111320.0 * math.cos(math.radians(lat)))
        bbox = [lon - dlon, lat - dlat, lon + dlon, lat + dlat]
        
        payload = {
            "input": {
                "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                "data": [{"type": "sentinel-2-l2a", "dataFilter": {"maxCloudCoverage": 20}}]
            },
            "output": {
                "width": 640, "height": 640,
                "responses": [{"identifier":"default","format":{"type":"image/png"}}]
            },
            "evalscript": "//VERSION=3\nfunction setup(){return {input:['B04','B03','B02'],output:{bands:3}}}\nfunction evaluatePixel(s){return [s.B04*2.5, s.B03*2.5, s.B02*2.5]}"
        }
        r = requests.post("https://sh.dataspace.copernicus.eu/api/v1/process", 
                          headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=30)
        if r.status_code == 200:
            with open(img_path, "wb") as f: f.write(r.content)
            return img_path
    except: return None

# ======================= واجهة Streamlit الرئيسية =======================
st.set_page_config(page_title=cfg.page_title, page_icon=cfg.page_icon, layout="wide")
ensure_dirs(cfg.images_dir, cfg.detected_dir, cfg.output_dir, cfg.models_dir)

st.title(cfg.page_title)
uploaded = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded)
    df['Subscription'] = df['Subscription'].apply(clean_meter_id)
    
    st.sidebar.markdown("### 🛠️ خيارات المعالجة")
    if st.sidebar.button("🚀 بدء التحليل المتدرج"):
        # تحميل النماذج
        yolo_model = YOLO(os.path.join(cfg.models_dir, "best.pt"))
        risk_model = RiskModel(os.path.join(cfg.models_dir, "isolation_model.joblib"), 
                               os.path.join(cfg.models_dir, "isolation_scaler.joblib"))

        results = []
        progress = st.progress(0)
        cols = st.columns(3)
        t_start = time.time()

        for i, (_, row) in enumerate(df.iterrows(), 1):
            m_id = row['Subscription']
            try:
                lat, lon = float(row['y']), float(row['x'])
                br, cons = float(row['Breaker']), float(row['consumption'])
                
                img_p = download_image(lat, lon, m_id)
                if not img_p: continue
                
                det = detect_field_progressive(img_p, lat, lon, m_id, yolo_model)
                if not det: continue
                
                # حساب المخاطر بالتمرير الصحيح لكافة الوسائط
                score, pr = risk_model.compute(br, cons, lon, lat, det['area'], det['green'])
                
                results.append([m_id, pr, score, det['edge'], det['center'], det['area'], 
                                cons, br, row.get('Office','-'), lat, lon, det['green']])
                
                # عرض البطاقة في الواجهة
                color = get_risk_color(score)
                with cols[i % 3]:
                    st.markdown(f"""
                    <div style="border:3px solid {color}; border-radius:15px; padding:15px; background:white; margin-bottom:20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1)">
                        <img src="data:image/png;base64,{base64.b64encode(open(det['path'],"rb").read()).decode()}" style="width:100%; border-radius:10px">
                        <h4 style="text-align:center; color:#333; margin-top:10px">عداد {m_id}</h4>
                        <div style="text-align:center; background:{color}; color:white; border-radius:10px; padding:5px; font-weight:bold">{pr} ({score*100:.1f}%)</div>
                        <p style="font-size:12px; margin-top:10px; text-align:right">
                            <b>خضرة:</b> {det['green']*100:.1f}% | <b>مساحة:</b> {det['area']}م²<br>
                            <b>استهلاك:</b> {cons} | <b>قاطع:</b> {br}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                progress.progress(i/len(df))
            except Exception as e:
                st.sidebar.error(f"خطأ في {m_id}: {e}")

        if results:
            # ترتيب النتائج من الأعلى خطورة للأقل
            results.sort(key=lambda x: x[2], reverse=True)
            res_df = pd.DataFrame(results, columns=["Subscription","priority","risk_score","edge_dist","center_dist","area_m2","consumption","breaker","office","lat","lon","green_ratio"])
            
            st.sidebar.success(f"✅ اكتمل التحليل في {time.time()-t_start:.1f} ثانية")
            st.sidebar.download_button("📥 تحميل النتائج (Excel)", data=save_results_excel(res_df), file_name="Results.xlsx")
            st.sidebar.download_button("📥 تحميل تقرير الصور (HTML)", data=save_results_html(results, cfg.detected_dir), file_name="Report.html")

st.markdown("---")
st.caption("مشهور العباس 2026 | نظام الرصد الذكي للفئة الزراعية")
