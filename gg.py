# -*- coding: utf-8 -*-
"""
فاقد زراعي — YOLO + IsolationForest + Copernicus (نسخة نمو مُحسّنة)
- RGB للعرض + NDVI FLOAT32 للحساب
- NDVI مُقنّع بسُحب/ظلال/ماء عبر SCL + فلتر "ضعف الإشارة" (B08+B04<0.20) + قص قمة NDVI<=0.85
- intensity = متوسط أعلى 20% من NDVI الموجب داخل البوكس (يميز الأخضر الداكن)
- لا نستخدم تغطية/مساحة خضراء — فقط قوة النمو + منطق هندسي (مساحة/قاطع/استهلاك) + شذوذ
- تصعيد فوري إذا النمو قوي مع خلل قاطع/استهلاك
"""

import os, io, time, base64, math
from dataclasses import dataclass
from typing import Tuple, List

import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from geopy.distance import geodesic
from ultralytics import YOLO
import joblib

# ======================= إعدادات =======================
@dataclass
class AppConfig:
    map_size: Tuple[int, int] = (640, 640)
    scene_size_m: int = 2500
    calibration_factor: float = 0.6695

    # YOLO
    min_confidence_accept: float = 0.45

    # هندسي
    min_area_m2: float = 5000.0
    max_edge_distance_m: float = 100.0

    # تصنيف الخطر
    risk_low: float = 0.40
    risk_high: float = 0.70

    request_timeout_s: int = 30
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    output_dir: str = "output"
    models_dir: str = "models"
    page_title: str = "🌾 نظام اكتشاف حالات الفاقد للفئة الزراعية"
    page_icon: str = "🌾"

    # نمو (NDVI)
    topk_ratio: float = 0.20              # نأخذ متوسط أعلى 20% من NDVI الموجب
    intensity_escalate_thr: float = 0.55  # نمو قوي ⇒ تصعيد عند وجود خلل
    ndvi_min_valid: float = 0.10          # أقل NDVI نعتبره نباتًا (بعد الفلاتر)
    ndvi_max_clip: float = 0.85           # قص القيم الشاذة العليا

    # أوزان الخطر
    w_breaker: float = 0.35
    w_consumption: float = 0.35
    w_anomaly: float = 0.15
    w_green: float = 0.15                 # خطر نباتي عكسي = 1 - intensity

    escalation_score: float = 0.95

cfg = AppConfig()

# ======================= أدوات عامة =======================
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
    # rows: [meter, pr, score, edge, area, cons, br, off, lat, lon, intensity]
    html = ["<html><head><meta charset='UTF-8'></head><body><div style='display:flex;flex-wrap:wrap;'>"]
    for r in rows:
        meter_id, priority, risk, distance, area, consumption, breaker, office, lat, lon, intensity = r
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
  خطر: {risk*100:.1f}% | 🌱 نمو (NDVI):{intensity*100:.0f}% | مسافة:{distance:.1f}م | مساحة:{area}م²<br>
  استهلاك:{consumption} | قاطع:{breaker} | مكتب:{office}<br>
  <a href='https://maps.google.com?q={lat},{lon}'>📍 الموقع</a>
</div>""")
    html.append("</div></body></html>")
    return "\n".join(html).encode("utf-8")

# ======================= النماذج =======================
@st.cache_resource
def load_yolo(model_path: str):
    return YOLO(model_path)

class RiskModel:
    def __init__(self, model_path, scaler_path, low_thr, high_thr):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.low_thr, self.high_thr = low_thr, high_thr

    @staticmethod
    def r_green(intensity: float) -> float:
        # خطر نباتي عكسي لقوة النمو [0..1]
        return float(1.0 - float(np.clip(intensity, 0.0, 1.0)))

    def compute(self, breaker, consumption, lon, lat, area_m2, intensity):
        # r1/r2 قواعد هندسية
        r1_base = 1.0 if breaker < area_m2 * 0.006 else 0.0
        r2_base = 1.0 if consumption < area_m2 * 0.4 else 0.0

        # شذوذ
        X = np.array([[breaker, consumption, lon, lat]], dtype=float)
        Xs = self.scaler.transform(X)
        anomaly = self.model.predict(Xs)[0]
        r3 = 1.0 if anomaly == 1 else 0.0

        # تضخيم أثر الخلل مع نمو أقوى
        r1 = min(1.0, r1_base * (0.6 + 1.0 * intensity))
        r2 = min(1.0, r2_base * (0.6 + 1.2 * intensity))
        r4 = self.r_green(intensity)

        score = cfg.w_breaker*r1 + cfg.w_consumption*r2 + cfg.w_anomaly*r3 + cfg.w_green*r4

        # تصعيد فوري: نمو قوي + خلل قاطع/استهلاك
        if intensity >= cfg.intensity_escalate_thr and (r1_base == 1.0 or r2_base == 1.0):
            score = max(score, cfg.escalation_score)

        pr = "قصوى" if score >= self.high_thr else ("متوسطة" if score >= self.low_thr else "منخفضة")
        return score, pr

# ======================= تنزيل RGB + NDVI =======================
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
        raise RuntimeError(f"CDSE token error {r.status_code}: {r.text[:200]}")
    js = r.json()
    st.session_state["_cdse_token"] = js["access_token"]
    st.session_state["_cdse_token_exp"] = time.time() + int(js.get("expires_in", 3600))
    return st.session_state["_cdse_token"]

def bbox_from_meters(lat: float, lon: float, size_m: float):
    half = size_m/2.0
    dlat = half/111320.0
    dlon = half/(111320.0*math.cos(math.radians(lat)))
    return [lon-dlon, lat-dlat, lon+dlon, lat+dlat]

def _process(bounds_bbox, responses, evalscript, token, timeout):
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    payload = {
        "input": {
            "bounds": {"bbox": bounds_bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {"maxCloudCoverage": 50, "mosaickingOrder": "mostRecent"},
                "processing": {"upsampling": "NEAREST", "downsampling": "NEAREST"}
            }]
        },
        "output": {"width": cfg.map_size[0], "height": cfg.map_size[1], "responses": responses},
        "evalscript": evalscript
    }
    headers = {"Authorization": f"Bearer {token}"}
    return requests.post(url, headers=headers, json=payload, timeout=timeout)

@st.cache_data(show_spinner=False, ttl=24*3600)
def download_rgb_and_ndvi(lat, lon, meter_id, timeout=30):
    """
    ينـزّل:
      - RGB لعرض الصورة
      - NDVI (FLOAT32) مع فلاتر: SCL + فلتر إشارة + قص قمة
    """
    rgb_path = os.path.join(cfg.images_dir, f"{meter_id}.png")
    ndvi_path = os.path.join(cfg.images_dir, f"{meter_id}_ndvi.tif")
    if os.path.exists(rgb_path) and os.path.exists(ndvi_path):
        return rgb_path, ndvi_path

    bbox = bbox_from_meters(lat, lon, cfg.scene_size_m)
    token = get_cdse_token()

    # 1) RGB
    eval_rgb = """
//VERSION=3
function setup(){return {input:["B04","B03","B02"],output:{bands:3}}}
function evaluatePixel(s){return [s.B04*1.8, s.B03*1.8, s.B02*1.8]}
"""
    r1 = _process(bbox, [{"identifier":"default","format":{"type":"image/png"}}], eval_rgb, token, timeout)
    if r1.status_code != 200: return None, None
    with open(rgb_path, "wb") as f: f.write(r1.content)

    # 2) NDVI مع فلاتر (SCL + low-signal + upper clip)
    eval_ndvi = """
//VERSION=3
function setup(){return {input:["B08","B04","SCL"],output:{bands:1,sampleType:"FLOAT32"}}}
function bad(c){return (c==3)||(c==6)||(c==8)||(c==9)||(c==10)||(c==11);} // ظل/ماء/سحب/ثلج
function evaluatePixel(s){
  var sum = s.B08 + s.B04;                 // قوة الإشارة
  var ndvi = (s.B08 - s.B04) / (sum + 1e-6);
  // استبعاد بكسلات غير موثوقة
  if (bad(s.SCL) || sum < 0.20) { ndvi = -1.0; }
  // قص الطرف العلوي لمنع التضخم
  if (ndvi > 0.85) ndvi = 0.85;
  return [ndvi];
}
"""
    r2 = _process(bbox, [{"identifier":"default","format":{"type":"image/tiff"}}], eval_ndvi, token, timeout)
    if r2.status_code != 200: return None, None
    with open(ndvi_path, "wb") as f: f.write(r2.content)

    return rgb_path, ndvi_path

# ======================= NDVI تحليل (intensity فقط) =======================
def ndvi_from_tiff(path: str) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img).astype(np.float32)
    return np.clip(arr, -1.0, 1.0)

def vegetation_intensity(ndvi: np.ndarray, box: Tuple[float, float, float, float]) -> float:
    x1,y1,x2,y2 = [int(v) for v in box]
    x1 = max(0, min(ndvi.shape[1]-1, x1)); x2 = max(0, min(ndvi.shape[1], x2))
    y1 = max(0, min(ndvi.shape[0]-1, y1)); y2 = max(0, min(ndvi.shape[0], y2))
    if x2<=x1 or y2<=y1: return 0.0
    crop = ndvi[y1:y2, x1:x2]
    if crop.size == 0: return 0.0

    # قيم نباتية موثوقة فقط: [0.10 .. 0.85]
    valid = crop[(crop >= cfg.ndvi_min_valid) & (crop <= cfg.ndvi_max_clip)]
    if valid.size == 0: return 0.0

    # متوسط أعلى 20% (الأغمق/الأقوى)
    k = max(1, int(round(cfg.topk_ratio * valid.size)))
    topk = np.partition(valid, -k)[-k:]
    intensity = float(np.clip(topk.mean(), 0.0, 1.0))
    return intensity

# ======================= الكشف =======================
@dataclass
class FieldDetection:
    bbox_xyxy: Tuple[float, float, float, float]
    conf: float
    area_m2: int
    center_latlon: Tuple[float, float]
    edge_distance_m: float
    out_img_path: str
    intensity: float

def detect_boxes(image: Image.Image, model: YOLO, min_conf=0.5):
    res = model.predict(source=image, imgsz=640, conf=min_conf, verbose=False)[0]
    if (not res) or (res.boxes is None) or (len(res.boxes)==0):
        return []
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    idxs = np.argsort(-confs)
    return [(boxes[i], float(confs[i])) for i in idxs]

def detect_field(rgb_path, ndvi_path, lat, lon, meter_id, model_yolo,
                 calibration_factor, min_conf_accept,
                 min_area_m2, max_edge_distance_m, detected_dir):
    image = Image.open(rgb_path).convert("RGB")
    ndvi = ndvi_from_tiff(ndvi_path)
    candidates = detect_boxes(image, model_yolo, min_conf=min_conf_accept)
    if not candidates: return None

    m_per_px = cfg.scene_size_m / float(cfg.map_size[0])

    for box, conf in candidates:
        w_px, h_px = abs(box[2]-box[0]), abs(box[3]-box[1])
        area = w_px * h_px * (m_per_px**2)
        corrected = area * calibration_factor
        if corrected < min_area_m2: continue

        cx, cy = image.width/2, image.height/2
        bx, by = (box[0]+box[2])/2, (box[1]+box[3])/2
        dx_m, dy_m = (bx-cx)*m_per_px, (by-cy)*m_per_px
        dlat = -(dy_m / 111320.0)
        dlon = dx_m / (40075000.0 * math.cos(math.radians(lat)) / 360.0)
        flat, flon = lat+dlat, lon+dlon

        radius_px = max(w_px, h_px)/2
        radius_m = radius_px * m_per_px
        dist = geodesic((lat, lon), (flat, flon)).meters
        edge = max(dist - radius_m, 0)
        if edge > max_edge_distance_m: continue

        intensity = vegetation_intensity(ndvi, tuple(box.tolist()))

        # رسم/حفظ
        draw = ImageDraw.Draw(image)
        draw.rectangle(box.tolist(), outline="green", width=3)
        draw.line([(cx, cy), (bx, by)], fill="yellow", width=2)
        draw.text((int(box[0])+4, int(box[1])+4), f"NDVI {intensity*100:.0f}%", fill="white")

        os.makedirs(detected_dir, exist_ok=True)
        out_path = os.path.join(detected_dir, f"{meter_id}.png")
        image.save(out_path)

        return FieldDetection(tuple(box.tolist()), conf, int(corrected), (flat, flon), round(edge,2),
                              out_path, intensity)

    return None

# ======================= الواجهة + التشغيل =======================
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

    breaker_filter = st.sidebar.selectbox("سعة القاطع", ["الكل"] + sorted(df["Breaker"].unique().tolist()))
    sort_order = st.sidebar.radio("ترتيب حسب الاستهلاك", ["بدون ترتيب", "تصاعدي", "تنازلي"])
    if breaker_filter != "الكل": df = df[df["Breaker"] == breaker_filter]
    if sort_order == "تصاعدي": df = df.sort_values(by="consumption", ascending=True)
    elif sort_order == "تنازلي": df = df.sort_values(by="consumption", ascending=False)

    # معاينة صور
    if st.sidebar.button("📥 تنزيل/عرض الصور"):
        progress = st.sidebar.progress(0)
        cols = st.columns(4)
        shown, n = 0, len(df)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            meter = str(row["Subscription"]); lat, lon = float(row["y"]), float(row["x"])
            rgb_path, _ = download_rgb_and_ndvi(lat, lon, meter)
            if rgb_path:
                with open(rgb_path, "rb") as f: b64 = base64.b64encode(f.read()).decode()
                cols[shown % 4].markdown(
                    f'<div style="border:1px solid #ddd;border-radius:8px;padding:6px;margin:6px;text-align:center">'
                    f'<img src="data:image/png;base64,{b64}" width="230" style="border-radius:6px"><br>'
                    f'<small>عداد {meter}<br>Lat {lat:.6f}, Lon {lon:.6f}</small></div>',
                    unsafe_allow_html=True
                )
                shown += 1
            progress.progress(i / max(n,1))
        st.sidebar.success(f"✅ تم عرض {shown} صورة")
        st.stop()

    # تشغيل التحليل
    if st.sidebar.button("🚀 بدء التحليل"):
        model_yolo = load_yolo(MODEL_PATH)
        risk_model = RiskModel(ML_MODEL_PATH, SCALER_PATH, cfg.risk_low, cfg.risk_high)

        progress = st.sidebar.progress(0)
        results, cols, col_i = [], st.columns(2), 0
        n = len(df); t0 = time.time()

        for i, (_, row) in enumerate(df.iterrows(), 1):
            try:
                meter = str(row["Subscription"])
                lat, lon = float(row["y"]), float(row["x"])
                br, cons, off = float(row["Breaker"]), float(row["consumption"]), str(row["Office"])

                rgb_path, ndvi_path = download_rgb_and_ndvi(lat, lon, meter)
                if not (rgb_path and ndvi_path): progress.progress(i/n); continue

                det = detect_field(rgb_path, ndvi_path, lat, lon, meter, model_yolo,
                                   cfg.calibration_factor, cfg.min_confidence_accept,
                                   cfg.min_area_m2, cfg.max_edge_distance_m, cfg.detected_dir)
                if det is None: progress.progress(i/n); continue

                score, pr = risk_model.compute(br, cons, lon, lat, det.area_m2, det.intensity)

                with open(det.out_img_path, "rb") as f: img64 = base64.b64encode(f.read()).decode()
                cols[col_i % 2].markdown(f"""
<div style="border:4px solid {colors.get(pr,'#ccc')};padding:10px;border-radius:12px;margin:6px;text-align:center;">
  <img src="data:image/png;base64,{img64}" width="360"><br>
  <strong>عداد {meter} ({pr})</strong><br>
  خطر:{score*100:.1f}% | 🌱 نمو (NDVI):{det.intensity*100:.0f}% | مسافة:{det.edge_distance_m:.1f}م | مساحة:{det.area_m2}م²<br>
  استهلاك:{cons} | قاطع:{br} | مكتب:{off}<br>
  <a href="https://maps.google.com?q={lat},{lon}">📍 الموقع</a>
</div>""", unsafe_allow_html=True)

                results.append([meter, pr, score, det.edge_distance_m, det.area_m2,
                                cons, br, off, lat, lon, det.intensity])
                col_i += 1

            except Exception as e:
                st.warning(f"⚠️ خطأ في العداد {row.get('Subscription','?')}: {e}")
            finally:
                progress.progress(i/n)

        if results:
            res_df = pd.DataFrame(results, columns=[
                "Subscription","priority","risk_score","edge_distance_m","area_m2",
                "consumption","breaker","office","lat","lon","intensity"
            ])
            st.sidebar.download_button("📥 نتائج Excel", data=save_results_excel(res_df), file_name="results.xlsx")
            st.sidebar.download_button("📥 تقرير HTML", data=save_results_html(results, colors, cfg.detected_dir),
                                       file_name="report.html", mime="text/html")
        st.sidebar.success(f"⏱️ اكتمل التحليل في {round(time.time()-t0,1)} ثانية")

st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس | 00966553339838 | ")
