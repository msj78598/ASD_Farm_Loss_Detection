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

# ======================= الإعدادات المستقرة =======================
@dataclass
class AppConfig:
    map_size: Tuple[int, int] = (640, 640)
    scene_size_m: int = 2500
    calibration_factor: float = 0.6695
    min_confidence_accept: float = 0.45
    min_area_m2: float = 5000.0
    r_start_m: int = 50
    r_step_m: int = 10
    r_max_m: int = 250
    thr_critical: float = 0.80
    thr_high: float = 0.60
    thr_medium: float = 0.35
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    models_dir: str = "models"
    green_ratio_min: float = 0.0
    green_dominance: float = 1.1
    green_min_value: int = 60

cfg = AppConfig()

# ======================= الوظائف الأساسية =======================
def ensure_dirs():
    for p in [cfg.images_dir, cfg.detected_dir, cfg.models_dir]:
        os.makedirs(p, exist_ok=True)

def get_risk_color(score: float) -> str:
    if score >= cfg.thr_critical: return "#8B0000"
    if score >= cfg.thr_high: return "#FF0000"
    if score >= cfg.thr_medium: return "#FF8C00"
    return "#2E8B57"

def save_results_html(rows: List[List], detected_dir: str) -> bytes:
    # تم تحسين بناء الـ HTML ليكون مقاوماً للتلف
    html_start = """<html><head><meta charset='UTF-8'>
    <style>
        body{font-family:tahoma; direction:rtl; background:#f0f2f6; padding:20px;}
        .grid{display:grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap:20px;}
        .card{background:white; border-radius:15px; overflow:hidden; border-top:8px solid; box-shadow:0 4px 6px rgba(0,0,0,0.1); padding:10px; text-align:center;}
    </style></head><body><div class='grid'>"""
    
    cards = []
    for r in rows:
        m_id, priority, score, edge_d, center_d, area, cons, br, office, lat, lon, g_ratio = r
        color = get_risk_color(score)
        img_tag = ""
        pth = os.path.join(detected_dir, f"{m_id}.png")
        if os.path.exists(pth):
            with open(pth, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            img_tag = f"<img src='data:image/png;base64,{b64}' style='width:100%; height:180px; object-fit:cover; border-radius:10px;'>"
        
        cards.append(f"""
        <div class='card' style='border-top-color:{color};'>
            {img_tag}
            <h3>عداد {m_id}</h3>
            <div style='background:{color}; color:white; padding:5px; border-radius:5px;'>{priority} ({score*100:.1f}%)</div>
            <p style='font-size:12px;'>مساحة: {area}م² | خضرة: {g_ratio*100:.1f}%<br>استهلاك: {cons} | مكتب: {office}</p>
            <a href='https://www.google.com/maps?q={lat},{lon}' target='_blank'>📍 الخريطة</a>
        </div>""")
    
    html_end = "</div></body></html>"
    return (html_start + "".join(cards) + html_end).encode("utf-8")

# ======================= محرك الحساب والذكاء =======================
class RiskModel:
    def __init__(self, m_path, s_path):
        self.model = joblib.load(m_path)
        self.scaler = joblib.load(s_path)

    def compute(self, breaker, consumption, lon, lat, area_m2, green_ratio):
        effective_area = area_m2 * green_ratio
        t_cons = effective_area * 0.20
        t_br = effective_area * 0.0013
        
        cons_risk = max(0, min(1, (t_cons - consumption) / t_cons)) if t_cons > 0 else 0
        br_risk = max(0, min(1, (t_br - breaker) / t_br)) if t_br > 0 else 0
        
        X = self.scaler.transform([[breaker, consumption, lon, lat]])
        anomaly = 1.0 if self.model.predict(X)[0] == 1 else 0.0
        
        score = (0.5 * cons_risk) + (0.3 * br_risk) + (0.2 * anomaly)
        
        if score >= cfg.thr_critical: pr = "حرجة جداً"
        elif score >= cfg.thr_high: pr = "قصوى"
        elif score >= cfg.thr_medium: pr = "متوسطة"
        else: pr = "منخفضة"
        return score, pr

# ======================= دوال المعالجة =======================
def estimate_green_ratio(image, box_xyxy):
    crop = image.crop([int(v) for v in box_xyxy])
    arr = np.asarray(crop)
    if arr.size == 0: return 0.0
    G, R, B = arr[...,1].astype(float), arr[...,0].astype(float), arr[...,2].astype(float)
    mask = (G > R * cfg.green_dominance) & (G > B * cfg.green_dominance) & (G > cfg.green_min_value)
    return float(mask.mean())

def detect_field_progressive(img_path, lat, lon, m_id, model_yolo):
    img = Image.open(img_path).convert("RGB")
    res = model_yolo.predict(img, imgsz=640, conf=cfg.min_confidence_accept, verbose=False)[0]
    if not res.boxes: return None

    m_per_px = cfg.scene_size_m / 640.0
    candidates = []
    for box in res.boxes:
        xyxy = box.xyxy.cpu().numpy()[0]
        area = (abs(xyxy[2]-xyxy[0]) * abs(xyxy[3]-xyxy[1]) * (m_per_px**2)) * cfg.calibration_factor
        if area < cfg.min_area_m2: continue
        
        bx, by = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
        dist = geodesic((lat, lon), (lat - ((by-320)*m_per_px/111320), lon + ((bx-320)*m_per_px/(111320*math.cos(math.radians(lat)))))).meters
        
        g_ratio = estimate_green_ratio(img, xyxy)
        edge_dist = max(dist - ((max(abs(xyxy[2]-xyxy[0]), abs(xyxy[3]-xyxy[1]))/2)*m_per_px), 0.0)
        candidates.append({'edge': edge_dist, 'box': xyxy, 'area': int(area), 'green': g_ratio})

    for R in range(cfg.r_start_m, cfg.r_max_m + 1, cfg.r_step_m):
        within = [c for c in candidates if c['edge'] <= R]
        if within:
            best = min(within, key=lambda x: x['edge'])
            draw = ImageDraw.Draw(img)
            draw.rectangle(best['box'].tolist(), outline="green", width=5)
            out_p = os.path.join(cfg.detected_dir, f"{m_id}.png")
            img.save(out_p)
            return best
    return None

# ======================= واجهة المستخدم =======================
st.set_page_config(page_title="نظام الرصد المطور", layout="wide")
ensure_dirs()

# دالة التحميل (يجب أن تكون secrets.toml مضبوطة)
def get_token():
    r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                      data={"grant_type":"client_credentials","client_id":st.secrets["CDSE_CLIENT_ID"],"client_secret":st.secrets["CDSE_CLIENT_SECRET"]})
    return r.json()["access_token"]

def download_img(lat, lon, m_id):
    path = os.path.join(cfg.images_dir, f"{m_id}.png")
    if os.path.exists(path): return path
    try:
        token = get_token()
        d = (cfg.scene_size_m / 2) / 111320.0
        bbox = [lon - d, lat - d, lon + d, lat + d]
        payload = {
            "input": {"bounds": {"bbox": bbox}, "data": [{"type": "sentinel-2-l2a"}]},
            "output": {"width": 640, "height": 640, "responses": [{"format": {"type": "image/png"}}]},
            "evalscript": "//VERSION=3\nfunction setup(){return{input:['B04','B03','B02'],output:{bands:3}}}\nfunction evaluatePixel(s){return[s.B04*2.5,s.B03*2.5,s.B02*2.5]}"
        }
        r = requests.post("https://sh.dataspace.copernicus.eu/api/v1/process", headers={"Authorization": f"Bearer {token}"}, json=payload)
        if r.status_code == 200:
            with open(path, "wb") as f: f.write(r.content)
            return path
    except: return None

# التنفيذ
st.title("🌾 نظام اكتشاف الفاقد الزراعي الذكي")
file = st.file_uploader("رفع ملف Excel", type=["xlsx"])

if file:
    df = pd.read_excel(file)
    df['Subscription'] = df['Subscription'].astype(str).str.replace(".0","", regex=False)
    
    if st.button("🚀 بدء التحليل الآمن"):
        yolo = YOLO(os.path.join(cfg.models_dir, "best.pt"))
        risk_m = RiskModel(os.path.join(cfg.models_dir, "isolation_model.joblib"), 
                           os.path.join(cfg.models_dir, "isolation_scaler.joblib"))
        
        results = []
        prog = st.progress(0)
        grid = st.columns(3)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            m_id = row['Subscription']
            try:
                img_p = download_img(row['y'], row['x'], m_id)
                if not img_p: continue
                
                det = detect_field_progressive(img_p, row['y'], row['x'], m_id, yolo)
                if not det: continue
                
                score, pr = risk_m.compute(row['Breaker'], row['consumption'], row['x'], row['y'], det['area'], det['green'])
                res = [m_id, pr, score, det['edge'], 0, det['area'], row['consumption'], row['Breaker'], row.get('Office','-'), row['y'], row['x'], det['green']]
                results.append(res)
                
                # عرض مباشر خفيف لتجنب انهيار المتصفح
                with grid[i % 3]:
                    color = get_risk_color(score)
                    st.markdown(f"<div style='border:2px solid {color}; padding:5px; border-radius:10px; text-align:center;'><b>عداد {m_id}</b><br>{pr}</div>", unsafe_allow_html=True)
                
                prog.progress((i + 1) / len(df))
            except Exception as e:
                st.write(f"⚠️ خطأ في {m_id}")

        if results:
            results.sort(key=lambda x: x[2], reverse=True)
            res_df = pd.DataFrame(results, columns=["Subscription","priority","risk_score","edge_dist","center_dist","area_m2","consumption","breaker","office","lat","lon","green_ratio"])
            st.success("تم التحليل بنجاح!")
            st.download_button("📥 نتائج Excel", data=res_df.to_csv(index=False).encode('utf-8-sig'), file_name="results.csv")
            st.download_button("📥 تقرير الصور (HTML)", data=save_results_html(results, cfg.detected_dir), file_name="report.html")
