# -*- coding: utf-8 -*-
"""
نظام اكتشاف حالات الفاقد للفئة الزراعية (Streamlit + YOLO + Isolation Forest + Copernicus)
- فلتر "كثافة المحصول (الخضرة)" قبل التشغيل (منخفض/متوسط/عالي)
- قياسان نباتيان:
   1) green_coverage: نسبة تغطية الخضرة داخل البوكس
   2) growth_strength: قوة النمو (يزيد مع الأخضر الداكن المشبع)
- ربط الخطر بالهندسة (مساحة/قاطع/استهلاك) + الشذوذ + النمو
- تصعيد فوري للحالات ذات نمو مرتفع مع استهلاك/قاطع غير ملائمين
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

# ======================= إعدادات ثابتة =======================
@dataclass
class AppConfig:
    map_size: Tuple[int, int] = (640, 640)
    scene_size_m: int = 2500
    calibration_factor: float = 0.6695

    # كشف YOLO
    min_confidence_accept: float = 0.45

    # شروط هندسية
    min_area_m2: float = 5000.0
    max_edge_distance_m: float = 100.0

    # عتبات تصنيف الخطر
    risk_low: float = 0.40
    risk_high: float = 0.70

    request_timeout_s: int = 30
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    output_dir: str = "output"
    models_dir: str = "models"
    page_title: str = "🌾 نظام اكتشاف حالات الفاقد للفئة الزراعية"
    page_icon: str = "🌾"

    # ====== إعدادات "كثافة المحصول" ======
    # (1) فلتر تغطية الخضرة (يتغير من الواجهة)
    green_ratio_min: float = 0.40     # (منخفض 0.25 / متوسط 0.40 / عالي 0.55)
    # (2) حساسية الأقنعة
    green_dominance: float = 1.1      # G أعلى من R و B بهذه النسبة
    green_min_value: int = 60         # حد أدنى لقيمة G
    # (3) بارامترات وزن "الأخضر الداكن" في HSV
    hue_center: int = 50              # ~درجة الأخضر على 0..255
    hue_halfwidth: int = 40           # عرض نافذة الأخضر
    v_dark_clip: float = 0.15         # تجاهل الظلال الأشد ظلمة
    v_bright_clip: float = 0.85       # تقليل أثر السطوع الشديد
    dark_gamma: float = 1.0           # قوة ترجيح الظلام (كلما زادت زاد تفضيل الداكن)

    # أوزان معادلة الخطر
    w_breaker: float = 0.30
    w_consumption: float = 0.40
    w_anomaly: float = 0.15
    w_green: float = 0.15             # خطر نباتي عكسي (يقل مع قوة النمو)

    # تصعيد فوري إذا نمو مرتفع واستهلاك/قاطع غير مناسبين
    growth_escalation_thr: float = 0.70
    escalation_score: float = 0.85

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
    # rows: [meter, pr, score, edge, area, cons, br, off, lat, lon, green_cov, growth_str]
    html = ["<html><head><meta charset='UTF-8'></head><body><div style='display:flex;flex-wrap:wrap;'>"]
    for r in rows:
        meter_id, priority, risk, distance, area, consumption, breaker, office, lat, lon, green_cov, growth_str = r
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
  خطر: {risk*100:.1f}% | 🌿 تغطية:{green_cov*100:.0f}% | 🌱 نمو:{growth_str*100:.0f}% | مسافة:{distance:.1f}م | مساحة:{area}م²<br>
  استهلاك:{consumption} | قاطع:{breaker} | مكتب:{office}<br>
  <a href='https://maps.google.com?q={lat},{lon}'>📍 الموقع</a>
</div>""")
    html.append("</div></body></html>")
    return "\n".join(html).encode("utf-8")

# ======================= تحميل النماذج =======================
@st.cache_resource
def load_yolo(model_path: str):
    return YOLO(model_path)

class RiskModel:
    def __init__(self, model_path, scaler_path, low_thr, high_thr):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.low_thr, self.high_thr = low_thr, high_thr

    @staticmethod
    def vegetation_risk(growth_strength: float) -> float:
        # r_green: خطر نباتي عكسي لقوة النمو
        # 1 عند نمو=0 ، يقترب من 0 عند نمو>=0.7
        return float(max(0.0, 1.0 - growth_strength / 0.7))

    def compute(self, breaker, consumption, lon, lat, area_m2, green_coverage, growth_strength):
        # r1/r2 الأساسيان (Binary)
        r1_base = 1.0 if breaker < area_m2 * 0.006 else 0.0
        r2_base = 1.0 if consumption < area_m2 * 0.4 else 0.0

        # شذوذ
        X = np.array([[breaker, consumption, lon, lat]], dtype=float)
        Xs = self.scaler.transform(X)
        anomaly = self.model.predict(Xs)[0]
        r3 = 1.0 if anomaly == 1 else 0.0

        # تضخيم أثر r1/r2 عند نمو مرتفع
        r1 = min(1.0, r1_base * (0.6 + 0.6 * growth_strength))
        r2 = min(1.0, r2_base * (0.7 + 0.8 * growth_strength))

        # خطر نباتي عكسي
        r4 = self.vegetation_risk(growth_strength)

        # مجموع مرجّح
        score = cfg.w_breaker*r1 + cfg.w_consumption*r2 + cfg.w_anomaly*r3 + cfg.w_green*r4

        # تصعيد فوري: نمو قوي + خلل قاطع/استهلاك
        if growth_strength >= cfg.growth_escalation_thr and (r1_base == 1.0 or r2_base == 1.0):
            score = max(score, cfg.escalation_score)

        # تصنيف
        if score >= self.high_thr:
            pr = "قصوى"
        elif score >= self.low_thr:
            pr = "متوسطة"
        else:
            pr = "منخفضة"
        return score, pr

# ======================= قياس الخضرة والنمو =======================
def estimate_vegetation(image: Image.Image, box_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    يعيد:
      green_coverage ∈ [0..1]  : نسبة البوكس المصنفة خضراء
      growth_strength ∈ [0..1] : قوة النمو (يزيد للأخضر الداكن المشبع)
    """
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    crop = image.crop((x1, y1, x2, y2))

    # 1) أقنعة خضراء بسيطة
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.size == 0:
        return 0.0, 0.0
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

    hsv_mask = (H >= cfg.hue_center - cfg.hue_halfwidth) & (H <= cfg.hue_center + cfg.hue_halfwidth) & (S >= 40)

    green_mask = dominance_mask | exg_mask | hsv_mask
    if green_mask.size == 0:
        return 0.0, 0.0

    green_coverage = float(green_mask.mean())

    # 2) قوة النمو (تفضيل الأخضر الداكن المشبع)
    # وزن قرب اللون من الأخضر
    hue_weight = np.clip(1.0 - np.abs(H.astype(np.int16) - cfg.hue_center) / float(cfg.hue_halfwidth), 0.0, 1.0)
    S_norm = S / 255.0
    V_norm = V / 255.0

    # وزن الظلام: نريد مكافأة البكسلات الغامقة لكن نتجنب الظلال الشديدة والسطوع العالي
    v_adj = np.clip(1.0 - V_norm, 0.0, 1.0)
    # قصّ الأطراف: أقل من v_dark_clip = تجاهل، أكثر من v_bright_clip = تقليل
    v_adj = np.clip((v_adj - cfg.v_dark_clip) / max(1e-6, (cfg.v_bright_clip - cfg.v_dark_clip)), 0.0, 1.0)
    dark_weight = v_adj ** cfg.dark_gamma

    # درجة النمو لكل بكسل
    growth_pix = hue_weight * S_norm * dark_weight

    # متوسط القوة داخل البكسلات الخضراء فقط
    if green_mask.any():
        growth_strength = float(growth_pix[green_mask].mean())
    else:
        growth_strength = 0.0

    return green_coverage, growth_strength

# ======================= الكشف =======================
@dataclass
class FieldDetection:
    bbox_xyxy: Tuple[float, float, float, float]
    conf: float
    area_m2: int
    center_latlon: Tuple[float, float]
    edge_distance_m: float
    out_img_path: str
    green_coverage: float
    growth_strength: float

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
                 min_area_m2, max_edge_distance_m, detected_dir):
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

        # حساب الخضرة وقوة النمو
        green_cov, growth_str = estimate_vegetation(image, tuple(box.tolist()))
        # فلتر تغطية الخضرة
        if green_cov < cfg.green_ratio_min:
            continue

        # رسم/حفظ أول صندوق ينجح الشروط
        draw = ImageDraw.Draw(image)
        draw.rectangle(box.tolist(), outline="green", width=3)
        draw.line([(cx, cy), (bx, by)], fill="yellow", width=2)
        draw.text((int(box[0])+4, int(box[1])+4), f"cov {green_cov*100:.0f}% | grow {growth_str*100:.0f}%", fill="white")

        os.makedirs(detected_dir, exist_ok=True)
        out_path = os.path.join(detected_dir, f"{meter_id}.png")
        image.save(out_path)

        return FieldDetection(tuple(box.tolist()), conf, int(corrected), (flat, flon), round(edge,2),
                              out_path, green_cov, growth_str)

    return None

# ======================= CDSE Token & Download =======================
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
def download_image(lat, lon, meter_id, timeout=30):
    """
    ينـزّل مشهد Sentinel-2 True Color بحجم ثابت على 640px
    """
    img_path = os.path.join(cfg.images_dir, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path

    def _request(token):
        bbox = bbox_from_meters(lat, lon, cfg.scene_size_m)
        url = "https://sh.dataspace.copernicus.eu/api/v1/process"
        payload = {
            "input": {
                "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {"maxCloudCoverage": 60, "mosaickingOrder": "mostRecent"},
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
  // تضخيم معتدل لتفادي قص الألوان
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

# ===== واجهة اختيار فلتر الخضرة =====
st.sidebar.markdown("### 🌿 كثافة المحصول (فلتر الخضرة)")
green_level = st.sidebar.selectbox(
    "اختر مستوى الصرامة:",
    ["منخفض (أقل استبعاد)", "متوسط (افتراضي)", "عالي (أكثر استبعاد)"],
    index=1
)
if green_level.startswith("منخفض"):
    cfg.green_ratio_min = 0.25
elif green_level.startswith("عالي"):
    cfg.green_ratio_min = 0.55
else:
    cfg.green_ratio_min = 0.40
st.sidebar.caption(f"الحد الأدنى لتغطية الخضرة: {int(cfg.green_ratio_min*100)}%")

if uploaded:
    df = read_excel(uploaded)
    st.sidebar.info(f"🔢 عدد الحالات: {len(df)}")

    breaker_filter = st.sidebar.selectbox("سعة القاطع", ["الكل"] + sorted(df["Breaker"].unique().tolist()))
    sort_order = st.sidebar.radio("ترتيب حسب الاستهلاك", ["بدون ترتيب", "تصاعدي", "تنازلي"])
    if breaker_filter != "الكل": df = df[df["Breaker"] == breaker_filter]
    if sort_order == "تصاعدي": df = df.sort_values(by="consumption", ascending=True)
    elif sort_order == "تنازلي": df = df.sort_values(by="consumption", ascending=False)

    # معاينة صور فقط
    preview_only = st.sidebar.checkbox("🖼️ عرض الصور فقط (بدون تشغيل النموذج)")
    if st.sidebar.button("📥 تنزيل/عرض الصور"):
        progress = st.sidebar.progress(0)
        cols = st.columns(4)
        shown, n, t0 = 0, len(df), time.time()
        for i, (_, row) in enumerate(df.iterrows(), 1):
            meter = str(row["Subscription"]); lat, lon = float(row["y"]), float(row["x"])
            p = download_image(lat, lon, meter)
            if p:
                with open(p, "rb") as f: b64 = base64.b64encode(f.read()).decode()
                cols[shown % 4].markdown(f"""
<div style="border:1px solid #ddd;border-radius:8px;padding:6px;margin:6px;text-align:center">
  <img src="data:image/png;base64,{b64}" width="230" style="border-radius:6px"><br>
  <small>عداد {meter}<br>Lat {lat:.6f}, Lon {lon:.6f}</small>
</div>""", unsafe_allow_html=True)
                shown += 1
            progress.progress(i / max(n,1))
        st.sidebar.success(f"✅ تم عرض {shown} صورة في {time.time()-t0:.1f} ثانية")
        st.stop()

    # تشغيل التحليل
    if st.sidebar.button("🚀 بدء التحليل"):
        model_yolo = load_yolo(MODEL_PATH)
        risk_model = RiskModel(ML_MODEL_PATH, SCALER_PATH, cfg.risk_low, cfg.risk_high)

        progress = st.sidebar.progress(0)
        results, cols, col_i = [], st.columns(3), 0
        t0 = time.time()
        n = len(df)

        for i, (_, row) in enumerate(df.iterrows(), 1):
            try:
                meter = str(row["Subscription"])
                lat, lon = float(row["y"]), float(row["x"])
                br, cons, off = float(row["Breaker"]), float(row["consumption"]), str(row["Office"])

                img_path = download_image(lat, lon, meter)
                if not img_path:
                    progress.progress(i / n); continue

                det = detect_field(
                    img_path, lat, lon, meter, model_yolo,
                    cfg.calibration_factor, cfg.min_confidence_accept,
                    cfg.min_area_m2, cfg.max_edge_distance_m, cfg.detected_dir
                )
                if det is None:
                    progress.progress(i / n); continue

                score, pr = risk_model.compute(br, cons, lon, lat, det.area_m2,
                                               det.green_coverage, det.growth_strength)

                results.append([meter, pr, score, det.edge_distance_m, det.area_m2, cons, br, off, lat, lon,
                                det.green_coverage, det.growth_strength])

                with open(det.out_img_path, "rb") as f: img64 = base64.b64encode(f.read()).decode()
                cols[col_i % 3].markdown(f"""
<div style="border:4px solid {colors.get(pr,'#ccc')};padding:10px;border-radius:12px;margin:6px;text-align:center;">
  <img src="data:image/png;base64,{img64}" width="260"><br>
  <strong>عداد {meter} ({pr})</strong><br>
  خطر:{score*100:.1f}% | 🌿 تغطية:{det.green_coverage*100:.0f}% | 🌱 نمو:{det.growth_strength*100:.0f}% | مسافة:{det.edge_distance_m:.1f}م | مساحة:{det.area_m2}م²<br>
  استهلاك:{cons} | قاطع:{br} | مكتب:{off}<br>
  <a href="https://maps.google.com?q={lat},{lon}">📍 الموقع</a>
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
                "consumption","breaker","office","lat","lon","green_coverage","growth_strength"
            ])
            st.sidebar.download_button("📥 نتائج Excel", data=save_results_excel(res_df), file_name="results.xlsx")
            st.sidebar.download_button("📥 تقرير HTML", data=save_results_html(results, colors, cfg.detected_dir),
                                       file_name="report.html", mime="text/html")
        st.sidebar.success(f"⏱️ اكتمل التحليل في {round(time.time()-t0,1)} ثانية")

st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس | 00966553339838 | ")
