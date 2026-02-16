# -*- coding: utf-8 -*-
"""
Streamlit app for detecting agricultural water/electricity loss cases
with Sentinel-2 images + YOLO detection + Isolation Forest risk model.
"""

import os
import io
import time
import base64
import math
import re
import logging
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

# ======================= إعداد اللوجر =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ======================= إعدادات ثابتة =======================
@dataclass
class AppConfig:
    # أبعاد الصورة الناتجة من Sentinel
    map_size: Tuple[int, int] = (640, 640)
    # عرض/ارتفاع المشهد بالأمتار (ثابت)
    scene_size_m: int = 2500
    # معامل معايرة المساحة
    calibration_factor: float = 0.6695
    # أدنى ثقة لقبول صندوق YOLO
    min_confidence_accept: float = 0.45
    # أدنى مساحة (م²) لقبول الحقل
    min_area_m2: float = 5000.0

    # ✅ Progressive search by EDGE distance
    r_start_m: int = 50
    r_step_m: int = 10
    r_max_m: int = 200

    # عتبات تصنيف المخاطر
    risk_low: float = 0.40
    risk_high: float = 0.70

    # إعدادات الاتصال بـ Copernicus
    request_timeout_s: int = 30

    # مجلدات العمل
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    output_dir: str = "output"
    models_dir: str = "models"

    # إعدادات صفحة Streamlit
    page_title: str = "🌾 نظام اكتشاف حالات الفاقد للفئة الزراعية"
    page_icon: str = "🌾"

    # ====== إعدادات فلترة “الخضرة” ======
    green_ratio_min: float = 0.0
    green_dominance: float = 1.1
    green_min_value: int = 60

    # ====== إعدادات نموذج المخاطر (بدلاً من أرقام ثابتة داخل الكود) ======
    breaker_area_coef: float = 0.0013  # معامل علاقة سعة القاطع بالمساحة الخضراء
    min_cons_coef: float = 0.20        # الحد الأدنى للاستهلاك بالنسبة للمساحة الخضراء
    w_r1: float = 0.4                  # وزن معيار القاطع
    w_r2: float = 0.4                  # وزن معيار الاستهلاك
    w_r3: float = 0.2                  # وزن نموذج العزلة (Isolation Forest)


cfg = AppConfig()

# ======================= أدوات عامة =======================
def ensure_dirs(*paths: str) -> None:
    """إنشاء المجلدات المطلوبة إن لم تكن موجودة."""
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
            # التعامل مع الصيغ العلمية مثل 1.23E+08
            if re.fullmatch(r"[0-9]+(\.[0-9]+)?[eE][\+\-]?\d+", s):
                return str(int(float(s)))
        except Exception:
            pass
        s = re.sub(r"\.0+$", "", s)
        return s


def read_excel(file_obj) -> pd.DataFrame:
    """قراءة ملف الإكسل وتحضير البيانات الأساسية."""
    df = pd.read_excel(file_obj, dtype={"Subscription": str})
    df["Subscription"] = df["Subscription"].apply(clean_meter_id)
    # حذف الصفوف التي لا تحتوي على بيانات أساسية مطلوبة
    return df.dropna(subset=["Subscription", "Office", "Breaker", "consumption", "x", "y"])


def save_results_excel(df: pd.DataFrame) -> bytes:
    """تصدير النتائج إلى ملف Excel في الذاكرة."""
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf.read()


def save_results_html(rows: List[List], colors: dict, detected_dir: str) -> bytes:
    """
    توليد تقرير HTML بسيط يحتوي على صور الحقول وبياناتها.
    rows: [meter, pr, score, edge_d, center_d, area, cons, br, off, lat, lon]
    """
    from html import escape

    html = [
        "<html><head><meta charset='UTF-8'><style>"
        "body{font-family:Arial, sans-serif;}"
        ".card{border-radius:10px;margin:6px;text-align:center;padding:10px;}"
        ".img{border-radius:8px;}"
        "</style></head><body><div style='display:flex;flex-wrap:wrap;'>"
    ]
    for r in rows:
        meter_id, priority, risk, edge_d, center_d, area, consumption, breaker, office, lat, lon = r
        border = colors.get(priority, "#ccc")
        pth = os.path.join(detected_dir, f"{meter_id}.png")
        img_tag = ""
        if os.path.exists(pth):
            with open(pth, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            img_tag = (
                f"<img src='data:image/png;base64,{img_b64}' "
                f"width='250' class='img'>"
            )

        html.append(f"""
<div class='card' style='border:4px solid {border};'>
  {img_tag}<br>
  <strong>عداد {escape(str(meter_id))} ({escape(str(priority))})</strong><br>
  خطر: {risk*100:.1f}% | حافة: {edge_d:.1f}م | مركز: {center_d:.1f}م | مساحة: {area}م²<br>
  الاستهلاك: {consumption} | القاطع: {breaker} | المكتب: {escape(str(office))}<br>
  <a href='https://maps.google.com?q={lat},{lon}'>📍 الموقع</a>
  &nbsp;|&nbsp;
  <a href='https://wa.me/?text=عداد:{meter_id}%20الموقع:{lat},{lon}'>📲 واتساب</a>
</div>""")
    html.append("</div></body></html>")
    return "\n".join(html).encode("utf-8")


def make_result_card_html(
    meter: str,
    pr: str,
    score: float,
    det,
    cons: float,
    br: float,
    off: str,
    lat: float,
    lon: float,
    border_color: str,
) -> str:
    """توليد HTML لعرض بطاقة نتيجة واحدة في Streamlit."""
    with open(det.out_img_path, "rb") as f:
        img64 = base64.b64encode(f.read()).decode()

    card_html = f"""
<div style="border:4px solid {border_color};padding:10px;border-radius:12px;margin:6px;text-align:center;">
  <img src="data:image/png;base64,{img64}" width="260" style="border-radius:8px;"><br>
  <strong>عداد {meter} ({pr})</strong><br>
  خطر:{score*100:.1f}% | حافة:{det.edge_distance_m:.1f}م | مركز:{det.center_distance_m:.1f}م<br>
  مساحة:{det.area_m2}م² | خضرة:{det.green_ratio*100:.0f}%<br>
  استهلاك:{cons} | قاطع:{br} | مكتب:{off}<br>
  <a href="https://maps.google.com?q={lat},{lon}">📍 الموقع</a>
</div>
"""
    return card_html


# ======================= تحميل النماذج =======================
@st.cache_resource
def load_yolo(model_path: str):
    """تحميل نموذج YOLO مرة واحدة لكل جلسة."""
    logger.info(f"Loading YOLO model from {model_path}")
    return YOLO(model_path)


class RiskModel:
    """
    نموذج حساب مستوى المخاطر:
    - يستخدم Isolation Forest + قواعد منطقية تعتمد على:
      * سعة القاطع
      * الاستهلاك
      * المساحة الخضراء الفعلية
    """

    def __init__(self, model_path: str, scaler_path: str, config: AppConfig):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.low_thr = config.risk_low
        self.high_thr = config.risk_high
        self.breaker_area_coef = config.breaker_area_coef
        self.min_cons_coef = config.min_cons_coef
        self.w_r1 = config.w_r1
        self.w_r2 = config.w_r2
        self.w_r3 = config.w_r3

    def compute(
        self,
        breaker: float,
        consumption: float,
        lon: float,
        lat: float,
        area_m2: float,
        green_ratio: float,
    ):
        """
        breaker: سعة القاطع (أمبير)
        consumption: الاستهلاك (الوحدة حسب بياناتك - مثلاً ك.و.س)
        lon, lat: إحداثيات العداد
        area_m2: المساحة المكتشفة (م²)
        green_ratio: نسبة الخضرة داخل الصندوق [0..1]
        """
        # 1. المساحة الخضراء الفعلية
        effective_area = area_m2 * green_ratio

        # 2. تحضير البيانات لنموذج العزلة
        X = np.array([[breaker, consumption, lon, lat]], dtype=float)
        Xs = self.scaler.transform(X)
        anomaly = self.model.predict(Xs)[0]
        # ملاحظة: تأكد من أن 1 تعني حالة شاذة في نموذجك أو عدّل الشرط حسب تدريبك

        # 3. معايير المخاطر
        # (r1) هل سعة القاطع أقل من المطلوب لمساحة خضراء معينة؟
        r1 = 1.0 if breaker < (effective_area * self.breaker_area_coef) else 0.0

        # (r2) هل الاستهلاك أقل من الحد الأدنى المتوقع لهذه المساحة؟
        r2 = 1.0 if consumption < (effective_area * self.min_cons_coef) else 0.0

        # (r3) ناتج نموذج العزلة (هنا نفترض أن 1 تعني حالة شاذة / خطر أعلى)
        r3 = 1.0 if anomaly == 1 else 0.0

        # 4. حساب النتيجة النهائية
        score = self.w_r1 * r1 + self.w_r2 * r2 + self.w_r3 * r3

        # 5. تصنيف الأولوية
        if score >= self.high_thr:
            pr = "قصوى"
        elif score >= self.low_thr:
            pr = "متوسطة"
        else:
            pr = "منخفضة"

        return score, pr


# ======================= دالة تقدير الخضرة =======================
def estimate_green_ratio(
    rgb_arr: np.ndarray,
    hsv_arr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    box_xyxy: Tuple[float, float, float, float],
) -> float:
    """
    تقدير نسبة الخضرة داخل الصندوق المحدد من صورة كاملة تم حساب RGB/HSV لها مسبقاً.
    - rgb_arr: مصفوفة RGB للصورة الأصلية (H, W, 3)
    - hsv_arr: (H_arr, S_arr, V_arr) للصورة كاملة أيضاً
    - box_xyxy: إحداثيات الصندوق داخل الصورة
    """
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # تقطيع من الـ numpy مباشرة لتوفير الوقت (بدل crop لكل صندوق)
    crop_rgb = rgb_arr[y1:y2, x1:x2, :]
    if crop_rgb.size == 0:
        return 0.0

    R = crop_rgb[..., 0].astype(np.float32)
    G = crop_rgb[..., 1].astype(np.float32)
    B = crop_rgb[..., 2].astype(np.float32)

    # 1) هيمنة الأخضر
    dominance_mask = (
        (G > R * cfg.green_dominance)
        & (G > B * cfg.green_dominance)
        & (G > cfg.green_min_value)
    )

    # 2) ExG
    Rn = R / 255.0
    Gn = G / 255.0
    Bn = B / 255.0
    exg = 2.0 * Gn - Rn - Bn
    exg_mask = exg > 0.08

    # 3) HSV
    H_full, S_full, V_full = hsv_arr
    H = H_full[y1:y2, x1:x2]
    S = S_full[y1:y2, x1:x2]
    V = V_full[y1:y2, x1:x2]
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
    center_distance_m: float
    out_img_path: str
    green_ratio: float


def detect_boxes(image: Image.Image, model: YOLO, min_conf: float = 0.5):
    """تشغيل YOLO على صورة واحدة وإرجاع الصناديق المرتبة من الأعلى ثقةً."""
    res = model.predict(source=image, imgsz=640, conf=min_conf, verbose=False)[0]
    if not res or not res.boxes or len(res.boxes) == 0:
        return []
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    idxs = np.argsort(-confs)
    return [(boxes[i], float(confs[i])) for i in idxs]


def detect_field_progressive(
    img_path: str,
    lat: float,
    lon: float,
    meter_id: str,
    model_yolo: YOLO,
    calibration_factor: float,
    min_conf_accept: float,
    min_area_m2: float,
    detected_dir: str,
    r_start: int = 50,
    r_step: int = 10,
    r_max: int = 200,
) -> Optional[FieldDetection]:
    """
    ✅ Progressive search by EDGE distance:
    - البحث عن الحقل الأقرب باستخدام مسافة الحافة EDGE.
    - R = 50, 60, 70, ..., 200
    - أول R يوجد فيه صندوق بـ edge_distance <= R يتم اختياره (الأقرب بالحافة ثم بالمركز).
    """
    image = Image.open(img_path).convert("RGB")
    boxes = detect_boxes(image, model_yolo, min_conf=min_conf_accept)
    if not boxes:
        return None

    m_per_px = cfg.scene_size_m / float(cfg.map_size[0])
    cx, cy = image.width / 2, image.height / 2

    # تجهيز مصفوفات RGB/HSV مرة واحدة للصورة كاملة
    rgb_arr = np.asarray(image, dtype=np.uint8)
    hsv_img = image.convert("HSV")
    H_full = np.asarray(hsv_img.getchannel(0), dtype=np.uint8)
    S_full = np.asarray(hsv_img.getchannel(1), dtype=np.uint8)
    V_full = np.asarray(hsv_img.getchannel(2), dtype=np.uint8)
    hsv_arr = (H_full, S_full, V_full)

    candidates = []
    for box, conf in boxes:
        w_px = abs(box[2] - box[0])
        h_px = abs(box[3] - box[1])
        area = w_px * h_px * (m_per_px**2)
        corrected = area * calibration_factor
        if corrected < min_area_m2:
            continue

        # حساب إحداثيات مركز الصندوق بالنسبة للصورة
        bx = (box[0] + box[2]) / 2
        by = (box[1] + box[3]) / 2
        dx_m = (bx - cx) * m_per_px
        dy_m = (by - cy) * m_per_px

        # تحويل فرق الإحداثيات إلى lat/lon
        dlat = -(dy_m / 111320.0)
        dlon = dx_m / (40075000.0 * math.cos(math.radians(lat)) / 360.0)
        flat = lat + dlat
        flon = lon + dlon

        center_dist = geodesic((lat, lon), (flat, flon)).meters

        # تقدير نسبة الخضرة
        green_ratio = estimate_green_ratio(rgb_arr, hsv_arr, tuple(box.tolist()))
        if green_ratio < cfg.green_ratio_min:
            continue

        radius_px = max(w_px, h_px) / 2
        radius_m = radius_px * m_per_px
        edge_dist = max(center_dist - radius_m, 0.0)

        candidates.append(
            (
                edge_dist,
                center_dist,
                box,
                conf,
                int(corrected),
                (flat, flon),
                green_ratio,
            )
        )

    if not candidates:
        return None

    chosen = None
    chosen_R = None
    for R in range(r_start, r_max + 1, r_step):
        within = [c for c in candidates if c[0] <= R]  # c[0] = edge_dist
        if within:
            chosen = min(within, key=lambda x: (x[0], x[1]))  # (edge, center)
            chosen_R = R
            break

    if chosen is None:
        return None

    edge_dist, center_dist, box, conf, area_m2, (flat, flon), green_ratio = chosen

    # رسم الصندوق وخط من مركز الصورة إلى مركز الصندوق
    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    draw.line(
        [(cx, cy), ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)],
        fill="yellow",
        width=2,
    )
    draw.text(
        (int(box[0]) + 4, int(box[1]) + 4),
        f"R<= {chosen_R}m | Edge:{edge_dist:.1f}m | Center:{center_dist:.1f}m | Green {green_ratio*100:.0f}%",
        fill="white",
    )

    os.makedirs(detected_dir, exist_ok=True)
    out_path = os.path.join(detected_dir, f"{meter_id}.png")
    image.save(out_path)

    return FieldDetection(
        bbox_xyxy=tuple(box.tolist()),
        conf=float(conf),
        area_m2=int(area_m2),
        center_latlon=(flat, flon),
        edge_distance_m=float(edge_dist),
        center_distance_m=float(center_dist),
        out_img_path=out_path,
        green_ratio=float(green_ratio),
    )


# ======================= CDSE Token & Download =======================
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)


def get_cdse_token() -> str:
    """الحصول على توكن CDSE مع استخدام session_state كـ cache."""
    tok = st.session_state.get("_cdse_token")
    exp = st.session_state.get("_cdse_token_exp", 0)
    if tok and time.time() < exp - 60:
        return tok

    cid = st.secrets.get("CDSE_CLIENT_ID")
    csec = st.secrets.get("CDSE_CLIENT_SECRET")
    if not cid or not csec:
        raise RuntimeError(
            "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET غير موجودة في secrets.toml"
        )

    data = {
        "grant_type": "client_credentials",
        "client_id": cid,
        "client_secret": csec,
    }
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
    """حساب صندوق (BBox) بالإحداثيات الجغرافية لمساحة مربعة حول نقطة معينة."""
    half = size_m / 2.0
    dlat = half / 111320.0
    dlon = half / (111320.0 * math.cos(math.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def download_image(lat: float, lon: float, meter_id: str, timeout: int = 30):
    """
    تحميل صورة Sentinel-2 من Copernicus لموقع عداد معين.
    تستخدم Cache على مستوى الملف (images/meter_id.png).
    """
    img_path = os.path.join(cfg.images_dir, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path

    def _request(token: str):
        bbox = bbox_from_meters(lat, lon, cfg.scene_size_m)
        url = "https://sh.dataspace.copernicus.eu/api/v1/process"
        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {
                        "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                    },
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "maxCloudCoverage": 60,
                            "mosaickingOrder": "mostRecent",
                        },
                        "processing": {
                            "upsampling": "NEAREST",
                            "downsampling": "NEAREST",
                        },
                    }
                ],
            },
            "output": {
                "width": cfg.map_size[0],
                "height": cfg.map_size[1],
                "responses": [
                    {"identifier": "default", "format": {"type": "image/png"}}
                ],
            },
            "evalscript": """//VERSION=3
function setup(){return {input:["B04","B03","B02"],output:{bands:3}}}
function evaluatePixel(s){
  return [s.B04*1.8, s.B03*1.8, s.B02*1.8]
}
""",
        }
        headers = {"Authorization": f"Bearer {token}"}
        return requests.post(url, headers=headers, json=payload, timeout=timeout)

    token = get_cdse_token()
    r = _request(token)
    if r.status_code == 401:
        # إعادة المحاولة بتوكن جديد
        token = get_cdse_token()
        r = _request(token)

    if r.status_code == 200:
        with open(img_path, "wb") as f:
            f.write(r.content)
        return img_path
    else:
        logger.warning(
            "Copernicus status %s for meter %s: %s",
            r.status_code,
            meter_id,
            r.text[:200],
        )
        st.warning(
            f"Copernicus status {r.status_code} للعداد {meter_id}: {r.text[:200]}"
        )
        return None


# ======================= واجهة Streamlit =======================
st.set_page_config(
    page_title=cfg.page_title, page_icon=cfg.page_icon, layout="wide"
)
ensure_dirs(cfg.images_dir, cfg.detected_dir, cfg.output_dir, cfg.models_dir)

MODEL_PATH = os.path.join(cfg.models_dir, "best.pt")
ML_MODEL_PATH = os.path.join(cfg.models_dir, "isolation_model.joblib")
SCALER_PATH = os.path.join(cfg.models_dir, "isolation_scaler.joblib")

st.title(cfg.page_title)

uploaded = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])
colors = {"قصوى": "#ff4d4d", "متوسطة": "#ffa500", "منخفضة": "#4CAF50"}

if uploaded:
    df = read_excel(uploaded)
    st.sidebar.info(f"🔢 عدد الحالات في الملف: {len(df)}")

    # ====== إعدادات إضافية من واجهة المستخدم ======
    st.sidebar.markdown("### ⚙️ إعدادات التحليل")

    # فلترة حسب القاطع
    breaker_filter = st.sidebar.selectbox(
        "سعة القاطع", ["الكل"] + sorted(df["Breaker"].unique().tolist())
    )

    # ترتيب حسب الاستهلاك
    sort_order = st.sidebar.radio(
        "ترتيب حسب الاستهلاك", ["بدون ترتيب", "تصاعدي", "تنازلي"]
    )

    # إمكانية تعديل أدنى مساحة وأدنى ثقة من الواجهة
    cfg.min_area_m2 = st.sidebar.number_input(
        "الحد الأدنى للمساحة (م²)",
        min_value=1000.0,
        max_value=50000.0,
        value=cfg.min_area_m2,
        step=500.0,
    )
    cfg.min_confidence_accept = st.sidebar.slider(
        "أدنى ثقة لنموذج YOLO",
        min_value=0.1,
        max_value=0.9,
        value=cfg.min_confidence_accept,
        step=0.05,
    )
    cfg.green_ratio_min = st.sidebar.slider(
        "الحد الأدنى لنسبة الخضرة داخل الحقل (%)",
        min_value=0.0,
        max_value=100.0,
        value=cfg.green_ratio_min * 100.0,
        step=5.0,
    ) / 100.0

    # فلترة حسب القاطع والاستهلاك
    if breaker_filter != "الكل":
        df = df[df["Breaker"] == breaker_filter]

    if sort_order == "تصاعدي":
        df = df.sort_values(by="consumption", ascending=True)
    elif sort_order == "تنازلي":
        df = df.sort_values(by="consumption", ascending=False)

    preview_only = st.sidebar.checkbox("🖼️ عرض الصور فقط (بدون تشغيل نموذج المخاطر)")
    if st.sidebar.button("📥 تنزيل/عرض الصور"):
        progress = st.sidebar.progress(0)
        cols = st.columns(4)
        shown = 0
        n = len(df)
        t0 = time.time()

        for i, (_, row) in enumerate(df.iterrows(), 1):
            meter = clean_meter_id(row["Subscription"])
            lat = float(row["y"])
            lon = float(row["x"])
            p = download_image(lat, lon, meter)
            if p:
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                cols[shown % 4].markdown(
                    f"""
<div style="border:1px solid #ddd;border-radius:8px;padding:6px;margin:6px;text-align:center">
  <img src="data:image/png;base64,{b64}" width="230" style="border-radius:6px"><br>
  <small>عداد {meter}<br>Lat {lat:.6f}, Lon {lon:.6f}</small>
</div>""",
                    unsafe_allow_html=True,
                )
                shown += 1
            progress.progress(i / max(n, 1))

        st.sidebar.success(
            f"✅ تم عرض {shown} صورة في {time.time()-t0:.1f} ثانية"
        )
        st.stop()

    if st.sidebar.button("🚀 بدء التحليل"):
        model_yolo = load_yolo(MODEL_PATH)
        risk_model = RiskModel(ML_MODEL_PATH, SCALER_PATH, cfg)

        progress = st.sidebar.progress(0)
        results = []
        cols = st.columns(3)
        col_i = 0
        t0 = time.time()
        n = len(df)

        for i, (_, row) in enumerate(df.iterrows(), 1):
            try:
                meter = clean_meter_id(row["Subscription"])
                lat = float(row["y"])
                lon = float(row["x"])
                br = float(row["Breaker"])
                cons = float(row["consumption"])
                off = str(row["Office"])

                img_path = download_image(lat, lon, meter)
                if not img_path:
                    progress.progress(i / max(n, 1))
                    continue

                det = detect_field_progressive(
                    img_path,
                    lat,
                    lon,
                    meter,
                    model_yolo,
                    cfg.calibration_factor,
                    cfg.min_confidence_accept,
                    cfg.min_area_m2,
                    cfg.detected_dir,
                    r_start=cfg.r_start_m,
                    r_step=cfg.r_step_m,
                    r_max=cfg.r_max_m,
                )

                if det is None:
                    progress.progress(i / max(n, 1))
                    continue

                # إذا تم اختيار "عرض الصور فقط" لا نحسب المخاطر
                if preview_only:
                    score, pr = 0.0, "غير محسوب"
                else:
                    score, pr = risk_model.compute(
                        br, cons, lon, lat, det.area_m2, det.green_ratio
                    )

                # حفظ النتيجة للـ Excel/HTML
                results.append(
                    [
                        meter,
                        pr,
                        score,
                        det.edge_distance_m,
                        det.center_distance_m,
                        det.area_m2,
                        cons,
                        br,
                        off,
                        lat,
                        lon,
                    ]
                )

                border_color = colors.get(pr, "#ccc")
                card_html = make_result_card_html(
                    meter,
                    pr,
                    score,
                    det,
                    cons,
                    br,
                    off,
                    lat,
                    lon,
                    border_color,
                )
                cols[col_i % 3].markdown(card_html, unsafe_allow_html=True)
                col_i += 1

                progress.progress(i / max(n, 1))

            except Exception as e:
                logger.exception(
                    "Error processing meter %s", row.get("Subscription", "?")
                )
                st.warning(
                    f"⚠️ خطأ في العداد {row.get('Subscription','?')}: {e}"
                )
                progress.progress(i / max(n, 1))
                continue

        if results:
            res_df = pd.DataFrame(
                results,
                columns=[
                    "Subscription",
                    "priority",
                    "risk_score",
                    "edge_distance_m",
                    "center_distance_m",
                    "area_m2",
                    "consumption",
                    "breaker",
                    "office",
                    "lat",
                    "lon",
                ],
            )

            st.sidebar.markdown("### 📊 ملخص سريع")
            st.sidebar.write(res_df["priority"].value_counts())

            st.sidebar.download_button(
                "📥 نتائج Excel",
                data=save_results_excel(res_df),
                file_name="results.xlsx",
            )
            st.sidebar.download_button(
                "📥 تقرير HTML",
                data=save_results_html(results, colors, cfg.detected_dir),
                file_name="report.html",
                mime="text/html",
            )

        st.sidebar.success(f"⏱️ اكتمل التحليل في {round(time.time()-t0,1)} ثانية")

st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس 2026 | 00966553339838 | ")
