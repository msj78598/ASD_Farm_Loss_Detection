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

# ======================= الإعدادات =======================
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
    risk_low: float = 0.40
    risk_high: float = 0.70
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    models_dir: str = "models"
    green_ratio_min: float = 0.0
    green_dominance: float = 1.1
    green_min_value: int = 60

cfg = AppConfig()

# ======================= الأدوات العامة =======================
def ensure_dirs():
    for p in [cfg.images_dir, cfg.detected_dir, cfg.models_dir]:
        os.makedirs(p, exist_ok=True)

def clean_meter_id(val) -> str:
    if pd.isna(val): return ""
    s = str(val).strip()
    return re.sub(r"\.0+$", "", s)

# ======================= منطق حساب الخضرة "الاحترافي" =======================
def estimate_green_ratio(image: Image.Image, box_xyxy: tuple) -> float:
    """المنطق الذي تفضله: يجمع بين Dominance و ExG و HSV"""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    if x2 <= x1 or y2 <= y1: return 0.0
    crop = image.crop((x1, y1, x2, y2))
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.size == 0: return 0.0

    R = arr[..., 0].astype(np.float32)
    G = arr[..., 1].astype(np.float32)
    B = arr[..., 2].astype(np.float32)

    # 1. اختبار الغلبة (Dominance)
    dom_mask = (G > R * cfg.green_dominance) & (G > B * cfg.green_dominance) & (G > cfg.green_min_value)

    # 2. اختبار مؤشر الخضرة الزائد (ExG)
    Rn, Gn, Bn = R/255.0, G/255.0, B/255.0
    exg = 2.0 * Gn - Rn - Bn
    exg_mask = exg > 0.08

    # 3. اختبار مساحة اللون (HSV) - كشف التدرج الأخضر الزراعي
    hsv = crop.convert("HSV")
    H = np.asarray(hsv.getchannel(0), dtype=np.uint8)
    S = np.asarray(hsv.getchannel(1), dtype=np.uint8)
    V = np.asarray(hsv.getchannel(2), dtype=np.uint8)
    hsv_mask = (H >= 25) & (H <= 67) & (S >= 60) & (V >= 50)

    # دمج كل الفلاتر لضمان عدم ضياع أي بكسل أخضر
    final_mask = dom_mask | exg_mask | hsv_mask
    return float(final_mask.mean())

# ======================= نموذج المخاطر الذكي =======================
class RiskModel:
    def __init__(self, m_path, s_path, low, high):
        self.model = joblib.load(m_path)
        self.scaler = joblib.load(s_path)
        self.low, self.high = low, high

    def compute(self, breaker, consumption, lon, lat, area_m2, green_ratio):
        effective_area = area_m2 * green_ratio
        
        # المعايير المرجعية
        r1 = 1.0 if breaker < (effective_area * 0.0013) else 0.0
        r2 = 1.0 if consumption < (effective_area * 0.20) else 0.0
        
        X = self.scaler.transform([[breaker, consumption, lon, lat]])
        r3 = 1.0 if self.model.predict(X)[0] == 1 else 0.0
        
        score = 0.4 * r1 + 0.4 * r2 + 0.2 * r3
        
        if score >= self.high: pr = "قصوى"
        elif score >= self.low: pr = "متوسطة"
        else: pr = "منخفضة"
        return score, pr

# ======================= الكشف المتدرج =======================
def detect_field_progressive(img_path, lat, lon, m_id, model_yolo):
    image = Image.open(img_path).convert("RGB")
    res = model_yolo.predict(image, imgsz=640, conf=cfg.min_confidence_accept, verbose=False)[0]
    if not res.boxes: return None

    m_per_px = cfg.scene_size_m / 640.0
    cx, cy = 320, 320
    candidates = []

    for box in res.boxes:
        xyxy = box.xyxy.cpu().numpy()[0]
        area = (abs(xyxy[2]-xyxy[0]) * abs(xyxy[3]-xyxy[1]) * (m_per_px**2)) * cfg.calibration_factor
        if area < cfg.min_area_m2: continue

        bx, by = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
        dist = geodesic((lat, lon), (lat - ((by-cy)*m_per_px/111320), lon + ((bx-cx)*m_per_px/(111320*math.cos(math.radians(lat)))))).meters
        
        g_ratio = estimate_green_ratio(image, tuple(xyxy))
        if g_ratio < cfg.green_ratio_min: continue

        radius_m = (max(abs(xyxy[2]-xyxy[0]), abs(xyxy[3]-xyxy[1]))/2) * m_per_px
        edge_dist = max(dist - radius_m, 0.0)
        candidates.append({'edge': edge_dist, 'center': dist, 'box': xyxy, 'area': int(area), 'green': g_ratio})

    if not candidates: return None

    for R in range(cfg.r_start_m, cfg.r_max_m + 1, cfg.r_step_m):
        within = [c for c in candidates if c['edge'] <= R]
        if within:
            best = min(within, key=lambda x: (x['edge'], x['center']))
            draw = ImageDraw.Draw(image)
            draw.rectangle(best['box'].tolist(), outline="green", width=5)
            out_p = os.path.join(cfg.detected_dir, f"{m_id}.png")
            image.save(out_p)
            return best
    return None

# ======================= اتصال الأقمار الصناعية =======================
def get_token():
    try:
        r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                          data={"grant_type":"client_credentials","client_id":st.secrets["CDSE_CLIENT_ID"],"client_secret":st.secrets["CDSE_CLIENT_SECRET"]}, timeout=15)
        return r.json()["access_token"]
    except: return None

def download_img(lat, lon, m_id):
    path = os.path.join(cfg.images_dir, f"{m_id}.png")
    if os.path.exists(path): return path
    token = get_token()
    if not token: return None
    d = (cfg.scene_size_m / 2) / 111320.0
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

# ======================= واجهة المستخدم والتقارير =======================
st.set_page_config(page_title=cfg.page_title, layout="wide")
ensure_dirs()

def get_html_report(results):
    html = ["<html><head><meta charset='UTF-8'><style>body{font-family:tahoma;direction:rtl;background:#f4f7f6;padding:20px;}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:15px;}.card{background:white;border-radius:12px;padding:15px;box-shadow:0 4px 8px rgba(0,0,0,0.1);text-align:center;border-top:6px solid;}</style></head><body><div class='grid'>"]
    colors = {"قصوى": "#ff4d4d", "متوسطة": "#ffa500", "منخفضة": "#4CAF50"}
    for r in results:
        m_id, pr, score, edge_d, cent_d, area, cons, br, off, lat, lon, g_ratio = r
        img_pth = os.path.join(cfg.detected_dir, f"{m_id}.png")
        img_tag = ""
        if os.path.exists(img_pth):
            with open(img_pth, "rb") as f: b64 = base64.b64encode(f.read()).decode()
            img_tag = f"<img src='data:image/png;base64,{b64}' style='width:100%;border-radius:8px;'>"
        html.append(f"<div class='card' style='border-top-color:{colors.get(pr,'#ccc')};'>{img_tag}<h3>عداد {m_id}</h3><p><b>{pr} ({score*100:.1f}%)</b><br>خضرة: {g_ratio*100:.0f}% | مساحة: {area}م²<br>استهلاك: {cons} | قاطع: {br}<br><a href='http://maps.google.com/?q={lat},{lon}'>📍 الموقع</a></p></div>")
    html.append("</div></body></html>")
    return "".join(html).encode("utf-8")

st.title(cfg.page_title)
uploaded = st.file_uploader("رفع ملف Excel", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded)
    df['Subscription'] = df['Subscription'].apply(clean_meter_id)
    
    if st.button("🚀 بدء التحليل المطور"):
        yolo = YOLO(os.path.join(cfg.models_dir, "best.pt"))
        risk_m = RiskModel(os.path.join(cfg.models_dir, "isolation_model.joblib"), 
                           os.path.join(cfg.models_dir, "isolation_scaler.joblib"), cfg.risk_low, cfg.risk_high)
        
        results = []
        prog = st.progress(0)
        grid = st.columns(3)
        
        for i, (_, row) in enumerate(df.iterrows()):
            m_id = row['Subscription']
            try:
                img_p = download_img(row['y'], row['x'], m_id)
                if not img_p: continue
                
                det = detect_field_progressive(img_p, row['y'], row['x'], m_id, yolo)
                if not det: continue
                
                score, pr = risk_m.compute(row['Breaker'], row['consumption'], row['x'], row['y'], det['area'], det['green'])
                results.append([m_id, pr, score, det['edge'], det['center'], det['area'], row['consumption'], row['Breaker'], row.get('Office','-'), row['y'], row['x'], det['green']])
                
                with grid[i % 3]:
                    st.image(det['path'] if 'path' in det else os.path.join(cfg.detected_dir, f"{m_id}.png"), caption=f"عداد {m_id} - {pr}")
                
                prog.progress((i+1)/len(df))
            except Exception as e:
                st.sidebar.error(f"خطأ في {m_id}: {e}")

        if results:
            results.sort(key=lambda x: x[2], reverse=True)
            res_df = pd.DataFrame(results, columns=["Subscription","priority","risk_score","edge_dist","center_dist","area_m2","consumption","breaker","office","lat","lon","green_ratio"])
            st.success("تم الانتهاء!")
            st.download_button("📥 Excel", data=res_df.to_csv(index=False).encode('utf-8-sig'), file_name="results.csv")
            st.download_button("📥 تقرير HTML", data=get_html_report(results), file_name="report.html")
