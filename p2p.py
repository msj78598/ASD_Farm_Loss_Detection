# -*- coding: utf-8 -*-
"""
"""

import os, io, time, base64, math, re, calendar
from dataclasses import dataclass
from typing import Tuple, List, Optional
from collections import defaultdict
from datetime import date

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
    max_edge_distance_m: float = 50.0
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
    green_ratio_min: float = 0.30   # 30% الحد الأدنى للخضرة
    green_dominance: float = 1.1    # G أعلى من R و B بهذه النسبة
    green_min_value: int = 60       # حد أدنى لقيمة G

cfg = AppConfig()

CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
TOKEN_URL   = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

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

# ======================= تحميل النماذج =======================
@st.cache_resource
def load_yolo(model_path: str):
    return YOLO(model_path)

class RiskModel:
    def __init__(self, model_path, scaler_path, low_thr, high_thr):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.low_thr, self.high_thr = low_thr, high_thr

    def compute(self, breaker, consumption, lon, lat, area_m2):
        X = np.array([[breaker, consumption, lon, lat]], dtype=float)
        Xs = self.scaler.transform(X)
        anomaly = self.model.predict(Xs)[0]
        r1 = 1.0 if breaker < area_m2 * 0.006 else 0.0
        r2 = 1.0 if consumption < area_m2 * 0.4 else 0.0
        r3 = 1.0 if anomaly == 1 else 0.0
        score = 0.4*r1 + 0.4*r2 + 0.2*r3
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
    out_img_path: str
    green_ratio: float
    acq_date: Optional[str] = None

def detect_boxes(image: Image.Image, model: YOLO, min_conf=0.5):
    res = model.predict(source=image, imgsz=640, conf=min_conf, verbose=False)[0]
    if not res or not res.boxes or len(res.boxes) == 0:
        return []
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    idxs = np.argsort(-confs)
    return [(boxes[i], float(confs[i])) for i in idxs]

def detect_field(img_path, lat, lon, meter_id, model_yolo,
                 calibration_factor, min_conf_accept,
                 min_area_m2, max_edge_distance_m, detected_dir,
                 acq_date: Optional[str] = None):
    image = Image.open(img_path).convert("RGB")
    candidates = detect_boxes(image, model_yolo, min_conf=min_conf_accept)
    if not candidates:
        return None

    m_per_px = cfg.scene_size_m / float(cfg.map_size[0])

    for box, conf in candidates:
        w_px, h_px = abs(box[2]-box[0]), abs(box[3]-box[1])
        area = w_px * h_px * (m_per_px**2)
        corrected = area * calibration_factor
        if corrected < min_area_m2:
            continue

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
        if edge > max_edge_distance_m:
            continue

        green_ratio = estimate_green_ratio(image, tuple(box.tolist()))
        if green_ratio < cfg.green_ratio_min:
            continue

        draw = ImageDraw.Draw(image)
        draw.rectangle(box.tolist(), outline="green", width=3)
        draw.line([(cx, cy), (bx, by)], fill="yellow", width=2)
        label = f"Green {green_ratio*100:.0f}%"
        draw.text((int(box[0])+4, int(box[1])+4), label, fill="white")

        os.makedirs(detected_dir, exist_ok=True)
        suffix = f"_{acq_date}" if acq_date else ""
        out_name = f"{meter_id}{suffix}.png"
        out_path = os.path.join(detected_dir, out_name)
        image.save(out_path)

        return FieldDetection(tuple(box.tolist()), conf, int(corrected),
                              (flat, flon), round(edge,2), out_path,
                              green_ratio, acq_date)

    return None

# ======================= CDSE Token & Download =======================
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
def get_month_s2_dates(lat: float, lon: float, year: int, month: int, max_items: int = 20) -> List[str]:
    """
    ترجع قائمة بتواريخ (YYYY-MM-DD) لكل مشهد Sentinel-2 متاح
    فوق هذا الموقع خلال الشهر المحدد.
    """
    token = get_cdse_token()
    bbox = bbox_from_meters(lat, lon, cfg.scene_size_m)
    last_day = calendar.monthrange(year, month)[1]
    dt_range = f"{year}-{month:02d}-01T00:00:00Z/{year}-{month:02d}-{last_day:02d}T23:59:59Z"

    payload = {
        "bbox": bbox,
        "collections": ["sentinel-2-l2a"],
        "datetime": dt_range,
        "limit": max_items,
        "sortby": [{"field": "properties.datetime", "direction": "asc"}]
    }
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(CATALOG_URL, headers=headers, json=payload, timeout=30)
    if r.status_code != 200:
        st.warning(f"Catalog status {r.status_code}: {r.text[:200]}")
        return []

    js = r.json()
    feats = js.get("features", [])
    dates = set()
    for f in feats:
        props = f.get("properties", {})
        dt_str = props.get("datetime") or props.get("date") or ""
        if "T" in dt_str:
            dt_str = dt_str.split("T")[0]
        if dt_str:
            dates.add(dt_str)
    return sorted(list(dates))

@st.cache_data(show_spinner=False, ttl=24*3600)
def download_image(lat: float, lon: float, meter_id: str,
                   acq_date: Optional[str] = None,
                   timeout: int = 30):
    """
    تنزيل مشهد Sentinel-2 True Color بحجم ثابت 640px
    إذا تم تمرير acq_date (نص YYYY-MM-DD) نحصر الصورة في نفس اليوم.
    """
    suffix = f"_{acq_date}" if acq_date else ""
    img_path = os.path.join(cfg.images_dir, f"{meter_id}{suffix}.png")
    if os.path.exists(img_path):
        return img_path

    def _request(token):
        bbox = bbox_from_meters(lat, lon, cfg.scene_size_m)
        data_filter = {
            "maxCloudCoverage": 60,
            "mosaickingOrder": "mostRecent"
        }
        if acq_date:
            data_filter["timeRange"] = {
                "from": f"{acq_date}T00:00:00Z",
                "to": f"{acq_date}T23:59:59Z"
            }

        url = "https://sh.dataspace.copernicus.eu/api/v1/process"
        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": data_filter,
                    "processing": {"upsampling": "NEAREST", "downsampling": "NEAREST"}
                }]
            },
            "output": {
                "width": cfg.map_size[0],
                "height": cfg.map_size[1],
                "responses": [{"identifier":"default","format":{"type":"image/png"}}]
            },
            "evalscript": """
//VERSION=3
function setup(){return {input:["B04","B03","B02"],output:{bands:3}}}
function evaluatePixel(s){
  return [s.B04*1.8, s.B03*1.8, s.B02*1.8]
}
"""
        }
        headers = {"Authorization": f"Bearer {token}"}
        return requests.post(url, headers=headers, json=payload, timeout=timeout)

    token = get_cdse_token()
    r = _request(token)
    if r.status_code == 401:
        token = get_cdse_token()
        r = _request(token)

    if r.status_code == 200:
        with open(img_path, "wb") as f:
            f.write(r.content)
        return img_path
    else:
        st.warning(f"Copernicus status {r.status_code} للعداد {meter_id}: {r.text[:200]}")
        return None

# ======================= واجهة Streamlit =======================
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

    edge_limit = st.sidebar.slider("أقصى انزياح بين مركز العداد/الحقل (متر)", 10, 50, 50, step=5)

    # ===== اختيار شهر وسنة الاستهلاك =====
    today = date.today()
    sel_year = st.sidebar.number_input("سنة الاستهلاك", min_value=2016, max_value=today.year,
                                       value=today.year, step=1)
    month_names = {
        1: "01 - يناير", 2: "02 - فبراير", 3: "03 - مارس", 4: "04 - أبريل",
        5: "05 - مايو", 6: "06 - يونيو", 7: "07 - يوليو", 8: "08 - أغسطس",
        9: "09 - سبتمبر", 10: "10 - أكتوبر", 11: "11 - نوفمبر", 12: "12 - ديسمبر"
    }
    sel_month = st.sidebar.selectbox("شهر الاستهلاك", list(month_names.keys()),
                                     format_func=lambda m: month_names[m])

    if breaker_filter != "الكل":
        df = df[df["Breaker"] == breaker_filter]
    if sort_order == "تصاعدي":
        df = df.sort_values(by="consumption", ascending=True)
    elif sort_order == "تنازلي":
        df = df.sort_values(by="consumption", ascending=False)

    preview_only = st.sidebar.checkbox("🖼️ عرض صورة أحدث مشهد فقط")

    if st.sidebar.button("📥 تنزيل/عرض الصور (مشهد واحد لكل عداد)"):
        progress = st.sidebar.progress(0)
        cols = st.columns(4)
        shown, n, t0 = 0, len(df), time.time()
        for i, (_, row) in enumerate(df.iterrows(), 1):
            meter = clean_meter_id(row["Subscription"])
            lat, lon = float(row["y"]), float(row["x"])
            p = download_image(lat, lon, meter)  # أحدث صورة من المنصة (بدون فلتر شهر)
            if p:
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                cols[shown % 4].markdown(f"""
<div style="border:1px solid #ddd;border-radius:8px;padding:6px;margin:6px;text-align:center">
  <img src="data:image/png;base64,{b64}" width="230" style="border-radius:6px"><br>
  <small>عداد {meter}<br>Lat {lat:.6f}, Lon {lon:.6f}</small>
</div>""", unsafe_allow_html=True)
                shown += 1
            progress.progress(i / max(n,1))
        st.sidebar.success(f"✅ تم عرض {shown} صورة في {time.time()-t0:.1f} ثانية")
        st.stop()

    # تشغيل التحليل على كل صور الشهر المختار
    if st.sidebar.button("🚀 بدء التحليل لشهر الاستهلاك المختار"):
        model_yolo = load_yolo(MODEL_PATH)
        risk_model = RiskModel(ML_MODEL_PATH, SCALER_PATH, cfg.risk_low, cfg.risk_high)

        progress = st.sidebar.progress(0)
        results = []
        ts_rows = []
        gallery = defaultdict(list)   # meter_id -> قائمة صور/تواريخ
        cols = st.columns(3)
        col_i = 0
        t0 = time.time()
        n = len(df)

        for i, (_, row) in enumerate(df.iterrows(), 1):
            try:
                meter = clean_meter_id(row["Subscription"])
                lat, lon = float(row["y"]), float(row["x"])
                br, cons, off = float(row["Breaker"]), float(row["consumption"]), str(row["Office"])

                # جميع تواريخ مشاهد Sentinel-2 في هذا الشهر فوق العداد
                dates_for_meter = get_month_s2_dates(lat, lon, int(sel_year), int(sel_month))
                if not dates_for_meter:
                    # احتياط: لو ما وجدنا أي مشهد في الكاتالوج لهذا الشهر، لا نوقف البرنامج
                    progress.progress(i / n)
                    continue

                dets_for_meter: List[FieldDetection] = []
                for d in dates_for_meter:
                    img_path = download_image(lat, lon, meter, acq_date=d)
                    if not img_path:
                        continue

                    det = detect_field(
                        img_path, lat, lon, meter, model_yolo,
                        cfg.calibration_factor, cfg.min_confidence_accept,
                        cfg.min_area_m2, edge_limit, cfg.detected_dir,
                        acq_date=d
                    )
                    if det is None:
                        # نحفظ صف في السلسلة الزمنية مع خضرة صفرية لو حاب
                        ts_rows.append([meter, d, 0.0, 0, cons, br, off, lat, lon])
                        continue

                    dets_for_meter.append(det)
                    ts_rows.append([
                        meter, d, det.green_ratio, det.area_m2,
                        cons, br, off, lat, lon
                    ])
                    gallery[meter].append({
                        "date": d,
                        "green": det.green_ratio,
                        "area_m2": det.area_m2,
                        "img_path": det.out_img_path
                    })

                if not dets_for_meter:
                    progress.progress(i / n)
                    continue

                # نختار أفضل صورة حسب أعلى خضرة
                best_det = max(dets_for_meter, key=lambda d: d.green_ratio)
                green_vals = [d.green_ratio for d in dets_for_meter]
                green_mean = float(np.mean(green_vals))
                score, pr = risk_model.compute(br, cons, lon, lat, best_det.area_m2)

                results.append([
                    meter, pr, score, best_det.edge_distance_m,
                    best_det.area_m2, cons, br, off, lat, lon,
                    green_mean, len(dets_for_meter)
                ])

                # عرض الكرت + مجلد الصور
                with cols[col_i % 3]:
                    with open(best_det.out_img_path, "rb") as f:
                        img64 = base64.b64encode(f.read()).decode()
                    st.markdown(f"""
<div style="border:4px solid {colors.get(pr,'#ccc')};padding:10px;border-radius:12px;margin:6px;text-align:center;">
  <img src="data:image/png;base64,{img64}" width="260"><br>
  <strong>عداد {meter} ({pr})</strong><br>
  خطر:{score*100:.1f}% | مسافة:{best_det.edge_distance_m:.1f}م | مساحة:{best_det.area_m2}م²<br>
  خضرة (أفضل صورة):{best_det.green_ratio*100:.0f}% | متوسط خضرة الشهر:{green_mean*100:.0f}% ({len(dets_for_meter)} صورة)<br>
  استهلاك:{cons} | قاطع:{br} | مكتب:{off}<br>
  <a href="https://maps.google.com?q={lat},{lon}">📍 الموقع</a>
</div>""", unsafe_allow_html=True)

                    with st.expander("📂 عرض جميع صور هذا العداد خلال الشهر"):
                        for info in gallery[meter]:
                            with open(info["img_path"], "rb") as f:
                                g64 = base64.b64encode(f.read()).decode()
                            st.markdown(f"""
<div style="border:1px solid #ddd;border-radius:8px;padding:6px;margin:6px;text-align:center;">
  <img src="data:image/png;base64,{g64}" width="230" style="border-radius:6px"><br>
  <small>تاريخ الصورة: {info['date']} | خضرة: {info['green']*100:.0f}% | مساحة: {info['area_m2']} م²</small>
</div>""", unsafe_allow_html=True)

                col_i += 1
                progress.progress(i / n)

            except Exception as e:
                st.warning(f"⚠️ خطأ في العداد {row.get('Subscription','?')}: {e}")
                progress.progress(i / n)
                continue

        if results:
            res_df = pd.DataFrame(results, columns=[
                "Subscription","priority","risk_score","edge_distance_m","area_m2",
                "consumption","breaker","office","lat","lon",
                "green_ratio_mean_month","num_images_in_month"
            ])
            st.sidebar.download_button(
                "📥 نتائج ملخصة لكل عداد (Excel)",
                data=save_results_excel(res_df),
                file_name="results_summary.xlsx"
            )

        if ts_rows:
            ts_df = pd.DataFrame(ts_rows, columns=[
                "Subscription","date","green_ratio","area_m2",
                "consumption","breaker","office","lat","lon"
            ])
            st.sidebar.download_button(
                "📥 السلسلة الزمنية (كل عداد + كل تاريخ) (Excel)",
                data=save_results_excel(ts_df),
                file_name="results_timeseries.xlsx"
            )

        st.sidebar.success(f"⏱️ اكتمل التحليل في {round(time.time()-t0,1)} ثانية")

st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس | 00966553339838 | ")
