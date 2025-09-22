# -*- coding: utf-8 -*-
"""
ملف واحد متكامل — نظام اكتشاف حالات الفاقد للفئة الزراعية (Streamlit + YOLO + Isolation Forest)

الميزات:
- تنزيل صور Google Static Maps ثم كشف الحقل الزراعي عبر YOLO بأعلى صندوق ثقة.
- حساب المساحة (م²) بدقة صحيحة (meters-per-pixel بحسب خط العرض والـ zoom) + مسافة الحافة إلى العداد.
- درجة مخاطر (0..1) = قواعد بسيطة + Isolation Forest → أولوية (منخفضة/متوسطة/قصوى).
- خيارات إضافية: فرز حسب الأقرب لنقطة مرجعية، حد أدنى لدرجة الخطر قبل العرض/التصدير.
- موثوقية عالية: Retries + Backoff للشبكة، Cache، Checkpointing للاستئناف عند التوقف.
- تنزيلات مستقرة (Excel/HTML) عبر st.session_state بدون إعادة الحساب.

تشغيل:
    streamlit run app_single.py

هام:
- لا تضع مفتاح Google داخل الكود. استخدم Streamlit Secrets:
  GOOGLE_MAPS_API_KEY = "YOUR-KEY"
- ضع النماذج في مجلد models/ (best.pt, isolation_model.joblib, isolation_scaler.joblib)
- أضف إسناد "Map data © Google".
"""

import os, time, base64, math, traceback, io
from dataclasses import dataclass
from typing import Optional, Tuple, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from geopy.distance import geodesic
import joblib


# =========================
# 1) الإعدادات العامة
# =========================
@dataclass
class AppConfig:
    # الخرائط/الصور
    zoom: int = 15                           # يجب أن يطابق Zoom لتحميل الخرائط والحساب
    map_size: Tuple[int, int] = (640, 640)
    map_type: str = "satellite"
    calibration_factor: float = 0.6695       # معامل معايرة للمساحة (حسب تجاربكم)

    # عتبات الكشف
    yolo_conf_threshold: float = 0.5         # حد أولي لـ YOLO (فلترة مقترحات)
    min_confidence_accept: float = 0.9       # حد قبول نهائي للصندوق الأفضل
    min_area_m2: float = 5000.0              # أدنى مساحة مقبولة (بعد المعايرة)
    max_edge_distance_m: float = 100.0       # أقصى مسافة حافة-إلى-عداد

    # عتبات المخاطر
    risk_low: float = 0.40                   # 0.40–<0.70 متوسطة
    risk_high: float = 0.70                  # ≥0.70 قصوى

    # اعتمادية/أداء
    request_timeout_s: int = 20
    max_retries: int = 3
    retry_backoff_s: float = 1.0
    save_checkpoint_every: int = 20          # كل كم حالة نحفظ نقطة الاستئناف

    # مسارات
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    output_dir: str = "output"
    models_dir: str = "models"
    form_template_path: str = "TEMPLATE.xlsx"

    # واجهة
    page_title: str = "🌾 نظام اكتشاف حالات الفاقد للفئة الزراعية"
    page_icon: str = "🌾"


# =========================
# 2) أدوات مساعدة Utilities
# =========================
def meters_per_pixel(lat: float, zoom: int) -> float:
    """دقة الأرض (م/بكسل) عند خط عرض lat ومستوى تكبير zoom."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)

def build_session(total_retries: int = 3, backoff_factor: float = 0.5,
                  status_forcelist=(429, 500, 502, 503, 504)):
    """جلسة HTTP مع سياسة إعادة المحاولة Backoff واحترام Retry-After."""
    session = requests.Session()
    retries = Retry(
        total=total_retries, read=total_retries, connect=total_retries,
        backoff_factor=backoff_factor, status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False, respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


# =========================
# 3) إدخال/إخراج البيانات
# =========================
def read_excel(file_obj) -> pd.DataFrame:
    """قراءة ملف الإدخال وإسقاط الصفوف الناقصة للأعمدة الأساسية."""
    df = pd.read_excel(file_obj)
    df = df.dropna(subset=["Subscription", "Office", "Breaker", "consumption", "x", "y"])
    return df

def save_results_excel(df: pd.DataFrame) -> bytes:
    """تحويل النتائج إلى Excel (bytes) للتنزيل."""
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf.read()

def save_results_html(rows: List[List], colors: dict, detected_dir: str) -> bytes:
    """إنشاء تقرير HTML مرئي لكل نتيجة مع الصورة وروابط الموقع/واتساب."""
    html = ["<html><head><meta charset='UTF-8'></head><body><div style='display:flex;flex-wrap:wrap;'>"]
    for r in rows:
        meter_id, priority, risk, distance, area, consumption, breaker, office, lat, lon = r
        border = colors.get(priority, "#ccc")
        img_path = os.path.join(detected_dir, f"{meter_id}.png")
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            img_tag = f"<img src='data:image/png;base64,{img_b64}' width='250' style='border-radius:8px;'>"
        else:
            img_tag = ""
        html.append(f"""
<div style='border:4px solid {border};padding:10px;border-radius:10px;margin:6px;text-align:center;'>
  {img_tag}<br>
  <strong>عداد {meter_id} ({priority})</strong><br>
  درجة الخطر: {risk*100:.1f}% | المسافة: {distance:.1f}م | المساحة: {area}م²<br>
  الاستهلاك: {consumption} | القاطع: {breaker} | المكتب: {office}<br>
  <a href='https://maps.google.com?q={lat},{lon}'>📍 الموقع</a>
  <a href='https://wa.me/?text=عداد:{meter_id}%20الموقع:{lat},{lon}'>📲 واتساب</a>
</div>""")
    html.append("</div><div style='margin-top:12px;font-size:12px;color:#666'>Map data © Google</div></body></html>")
    return "\n".join(html).encode("utf-8")


# =========================
# 4) رؤية حاسوبية (YOLO)
# =========================
@dataclass
class FieldDetection:
    bbox_xyxy: Tuple[float, float, float, float]
    conf: float
    area_m2: float
    center_latlon: Tuple[float, float]
    edge_distance_m: float
    out_img_path: str

@st.cache_resource
def load_yolo(model_path: str) -> YOLO:
    """تحميل YOLO مرة واحدة (Cache)."""
    return YOLO(model_path)

def detect_best_box(image: Image.Image, model: YOLO, min_conf=0.5):
    """تشغيل YOLO وإرجاع الصندوق الأعلى ثقة."""
    results = model.predict(source=image, imgsz=640, conf=min_conf, verbose=False)[0]
    if results is None or results.boxes is None or len(results.boxes) == 0:
        return None, None
    confs = results.boxes.conf.cpu().numpy()
    idx = int(confs.argmax())
    return results.boxes.xyxy[idx].cpu().numpy(), float(confs[idx])

def detect_field(img_path: str, lat: float, lon: float, meter_id: str, model_yolo: YOLO,
                 zoom: int, calibration_factor: float, min_conf_accept: float,
                 min_area_m2: float, max_edge_distance_m: float, detected_dir: str) -> Optional[FieldDetection]:
    """كشف الحقل وحساب المساحة/المسافة وحفظ صورة مع الصندوق."""
    image = Image.open(img_path).convert("RGB")
    box, conf = detect_best_box(image, model_yolo, min_conf=min_conf_accept)
    if box is None or conf < min_conf_accept:
        return None

    # حساب المساحة (م²)
    res = meters_per_pixel(lat, zoom)
    width_px = abs(box[2] - box[0])
    height_px = abs(box[3] - box[1])
    area = width_px * height_px * (res ** 2)
    corrected_area = area * calibration_factor
    if corrected_area < min_area_m2:
        return None

    # تحويل إزاحة المركز إلى إحداثيات جغرافية تقريبية
    img_cx, img_cy = (image.width / 2.0, image.height / 2.0)
    bx_cx, bx_cy = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    dx_m = (bx_cx - img_cx) * res
    dy_m = (bx_cy - img_cy) * res

    dlat = -(dy_m / 111320.0)
    dlon = dx_m / (40075000.0 * math.cos(math.radians(lat)) / 360.0)
    field_lat = lat + dlat
    field_lon = lon + dlon

    # مسافة الحافة
    radius_px = max(width_px, height_px) / 2.0
    radius_m = radius_px * res
    center_distance = geodesic((lat, lon), (field_lat, field_lon)).meters
    edge_distance = max(center_distance - radius_m, 0.0)
    if edge_distance > max_edge_distance_m:
        return None

    # رسم/حفظ
    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    draw.line([(img_cx, img_cy), (bx_cx, bx_cy)], fill="yellow", width=2)
    os.makedirs(detected_dir, exist_ok=True)
    out_path = os.path.join(detected_dir, f"{meter_id}.png")
    image.save(out_path)

    return FieldDetection(tuple(box.tolist()), conf, int(corrected_area),
                          (field_lat, field_lon), round(edge_distance, 2), out_path)


# =========================
# 5) نموذج المخاطر
# =========================
class RiskOutput:
    def __init__(self, score: float, priority: str):
        self.score = score
        self.priority = priority

class RiskModel:
    def __init__(self, model_path: str, scaler_path: str, low_thr: float, high_thr: float):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.low_thr = low_thr
        self.high_thr = high_thr

    def compute(self, breaker: float, consumption: float, lon: float, lat: float, area_m2: float) -> RiskOutput:
        X = np.array([[breaker, consumption, lon, lat]], dtype=float)
        Xs = self.scaler.transform(X)
        anomaly = self.model.predict(Xs)[0]  # 1 == anomaly

        r1 = 1.0 if breaker < area_m2 * 0.006 else 0.0
        r2 = 1.0 if consumption < area_m2 * 0.4 else 0.0
        r3 = 1.0 if anomaly == 1 else 0.0
        score = 0.4 * r1 + 0.4 * r2 + 0.2 * r3

        if score >= self.high_thr:
            priority = "قصوى"
        elif score >= self.low_thr:
            priority = "متوسطة"
        else:
            priority = "منخفضة"
        return RiskOutput(score=score, priority=priority)


# =========================
# 6) واجهة Streamlit
# =========================
cfg = AppConfig()
st.set_page_config(page_title=cfg.page_title, page_icon=cfg.page_icon, layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, cfg.images_dir)
DETECTED_DIR = os.path.join(BASE_DIR, cfg.detected_dir)
OUTPUT_DIR = os.path.join(BASE_DIR, cfg.output_dir)
MODELS_DIR = os.path.join(BASE_DIR, cfg.models_dir)
FORM_PATH = os.path.join(BASE_DIR, cfg.form_template_path)
MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
ML_MODEL_PATH = os.path.join(MODELS_DIR, "isolation_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "isolation_scaler.joblib")

ensure_dirs(IMG_DIR, DETECTED_DIR, OUTPUT_DIR, MODELS_DIR)

# جلسة شبكة مع Retries
_session = build_session(total_retries=cfg.max_retries, backoff_factor=cfg.retry_backoff_s)

@st.cache_resource
def _load_models_cached():
    yolo = load_yolo(MODEL_PATH)
    risk = RiskModel(ML_MODEL_PATH, SCALER_PATH, cfg.risk_low, cfg.risk_high)
    return yolo, risk

@st.cache_data(show_spinner=False, ttl=24*3600)
def download_image(lat: float, lon: float, meter_id: str, zoom: int, size: Tuple[int,int], map_type: str, timeout: int) -> Optional[str]:
    """تنزيل صورة Static Maps وتخزينها على القرص (مع Cache)."""
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
    params = {
        "center": f"{lat},{lon}",
        "zoom": str(zoom),
        "size": f"{size[0]}x{size[1]}",
        "maptype": map_type,
        "markers": f"color:red|label:X|{lat},{lon}",
        "key": api_key
    }
    try:
        r = _session.get(base_url, params=params, timeout=timeout)
        if r.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(r.content)
            return img_path
        else:
            st.warning(f"Static Maps status {r.status_code} للعداد {meter_id}")
            return None
    except Exception as e:
        st.error(f"خطأ شبكة أثناء تنزيل صورة {meter_id}: {e}")
        return None


# عنوان الصفحة + تنزيل القالب
st.title(cfg.page_title)
if os.path.exists(FORM_PATH):
    st.download_button("📥 تحميل نموذج البيانات (TEMPLATE.xlsx)", open(FORM_PATH, "rb"), file_name="TEMPLATE.xlsx")

uploaded_file = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

# إعدادات أساسية
st.sidebar.header("إعدادات التحليل")
ui_zoom   = st.sidebar.slider("مستوى التكبير (Zoom)", 10, 20, cfg.zoom, help="يجب أن يطابق Zoom المستخدم لتنزيل الصور.")
ui_min_cf = st.sidebar.slider("حد قبول ثقة YOLO", 0.50, 0.99, cfg.min_confidence_accept, 0.01)
ui_min_ar = st.sidebar.number_input("أدنى مساحة مقبولة (م²)", value=float(cfg.min_area_m2), step=1000.0)
ui_max_ed = st.sidebar.number_input("أقصى مسافة حافة-إلى-عداد (م)", value=float(cfg.max_edge_distance_m), step=10.0)
ui_calib  = st.sidebar.number_input("معامل المعايرة (Calibration)", value=float(cfg.calibration_factor), step=0.01, format="%.4f")
st.sidebar.caption("Map data © Google")

# خيارات إضافية: فرز بالأقرب + حد أدنى للمخاطر
st.sidebar.subheader("خيارات الفرز/التصفية الإضافية")
enable_nearest = st.sidebar.checkbox("فرز حسب الأقرب لنقطة مرجعية", value=False)
ref_lat = st.sidebar.number_input("خط العرض للنقطة المرجعية", value=0.0, format="%.6f", disabled=not enable_nearest)
ref_lon = st.sidebar.number_input("خط الطول للنقطة المرجعية", value=0.0, format="%.6f", disabled=not enable_nearest)
risk_min = st.sidebar.slider("أدنى درجة خطر لإظهار/تصدير الحالة", 0.0, 1.0, 0.0, 0.01)

# ألوان الأولويات
colors = {"قصوى": "#ff4d4d", "متوسطة": "#ffa500", "منخفضة": "#4CAF50"}

# حالة التنزيلات لتبقى بعد أي rerun
if "downloads_ready" not in st.session_state:
    st.session_state["downloads_ready"] = False
    st.session_state["excel_bytes"] = None
    st.session_state["html_bytes"] = None
    st.session_state["results_df"] = None

# عند وجود ملف
if uploaded_file:
    df = read_excel(uploaded_file)

    # فلاتر/ترتيب قبل التحليل (حسب اختيارك)
    breaker_filter = st.sidebar.selectbox("سعة القاطع", ["الكل"] + sorted(df["Breaker"].unique().tolist()))
    sort_order = st.sidebar.radio("ترتيب حسب الاستهلاك", ["بدون ترتيب", "تصاعدي", "تنازلي"])
    if breaker_filter != "الكل":
        df = df[df["Breaker"] == breaker_filter]
    if sort_order == "تصاعدي":
        df = df.sort_values(by="consumption", ascending=True)
    elif sort_order == "تنازلي":
        df = df.sort_values(by="consumption", ascending=False)

    st.sidebar.info(f"🔢 عدد الحالات في الملف: {len(df)}")

    # أزرار التشغيل/الاستئناف
    col_run, col_resume = st.sidebar.columns(2)
    start_new = col_run.button("🚀 بدء التحليل")
    resume    = col_resume.button("⏯️ استئناف من آخر نقطة")

    if start_new or resume:
        model_yolo, risk_model = _load_models_cached()
        progress_bar = st.sidebar.progress(0)
        status_area  = st.sidebar.empty()
        t0 = time.time()

        results_rows: List[List] = []
        checkpoint_csv = os.path.join(OUTPUT_DIR, "results_checkpoint.csv")
        processed_ids = set()

        # تحميل نقطة استئناف
        if resume and os.path.exists(checkpoint_csv):
            try:
                cdf = pd.read_csv(checkpoint_csv)
                results_rows = cdf.values.tolist()
                processed_ids = set(cdf["Subscription"].astype(str).tolist())
                st.sidebar.success(f"تم تحميل {len(processed_ids)} حالة من المحفوظ.")
            except Exception:
                st.sidebar.warning("تعذر قراءة نقطة الاستئناف — سيبدأ تشغيل جديد.")

        cols = st.columns(3)
        col_index = 0
        n = len(df)

        for i, (_, row) in enumerate(df.iterrows(), 1):
            try:
                meter_id = str(row["Subscription"])
                if meter_id in processed_ids:
                    progress_bar.progress(i / max(n,1))
                    continue

                lat, lon = float(row["y"]), float(row["x"])
                breaker = float(row["Breaker"])
                consumption = float(row["consumption"])
                office = str(row["Office"])

                img_path = download_image(lat, lon, meter_id, ui_zoom, cfg.map_size, cfg.map_type, cfg.request_timeout_s)
                if not img_path:
                    status_area.warning(f"تعذر تنزيل صورة للعداد {meter_id}")
                    progress_bar.progress(i / max(n,1))
                    continue

                det = detect_field(img_path, lat, lon, meter_id, model_yolo,
                                   ui_zoom, ui_calib, ui_min_cf, ui_min_ar, ui_max_ed, DETECTED_DIR)
                if det is None:
                    progress_bar.progress(i / max(n,1))
                    continue

                rk = risk_model.compute(breaker, consumption, lon, lat, det.area_m2)
                results_rows.append([meter_id, rk.priority, rk.score, det.edge_distance_m,
                                     det.area_m2, consumption, breaker, office, lat, lon])

                # بطاقة مرئية
                with open(det.out_img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                cols[col_index % 3].markdown(f'''
<div style="border:4px solid {colors.get(rk.priority, '#ccc')};padding:10px;border-radius:12px;margin:6px;text-align:center;">
  <img src="data:image/png;base64,{img_b64}" width="260" style="border-radius:8px;"><br>
  <strong>عداد {meter_id} ({rk.priority})</strong><br>
  درجة الخطر:{rk.score*100:.1f}% | المسافة:{det.edge_distance_m:.1f}م | المساحة:{det.area_m2}م²<br>
  الاستهلاك:{consumption} | القاطع:{breaker} | المكتب:{office}<br>
  <a href="https://maps.google.com?q={lat},{lon}">📍 الموقع</a>
  <a href="https://wa.me/?text=عداد:{meter_id}%20الموقع:{lat},{lon}">📲 واتساب</a>
</div>
''', unsafe_allow_html=True)
                col_index += 1

                # حفظ نقطة استئناف كل N صفوف
                if len(results_rows) % cfg.save_checkpoint_every == 0:
                    pd.DataFrame(results_rows, columns=[
                        "Subscription","priority","risk_score","edge_distance_m","area_m2",
                        "consumption","breaker","office","lat","lon"
                    ]).to_csv(checkpoint_csv, index=False)

                progress_bar.progress(i / max(n,1))

            except Exception as e:
                status_area.error(f"خطأ في معالجة العداد {row.get('Subscription','?')}: {e}")
                continue

        # === المخرجات النهائية + التصفية/الفرز ===
        if results_rows:
            results_df = pd.DataFrame(results_rows, columns=[
                "Subscription","priority","risk_score","edge_distance_m","area_m2",
                "consumption","breaker","office","lat","lon"
            ])

            # تصفية حسب الحد الأدنى لدرجة الخطر
            if risk_min > 0:
                results_df = results_df[results_df["risk_score"] >= risk_min].copy()

            # فرز حسب الأقرب (إذا مفعّل)
            if enable_nearest and not (ref_lat == 0.0 and ref_lon == 0.0):
                results_df["dist_to_ref_m"] = results_df.apply(
                    lambda r: geodesic((r["lat"], r["lon"]), (ref_lat, ref_lon)).meters, axis=1
                )
                results_df = results_df.sort_values("dist_to_ref_m", ascending=True)

            # ملفات التنزيل
            excel_bytes = save_results_excel(results_df)
            html_bytes  = save_results_html(
                results_df[["Subscription","priority","risk_score","edge_distance_m","area_m2",
                            "consumption","breaker","office","lat","lon"]].values.tolist(),
                colors,
                DETECTED_DIR
            )

            # تخزين بالجلسة حتى تبقى الأزرار فعّالة بعد أي rerun
            st.session_state["downloads_ready"] = True
            st.session_state["excel_bytes"] = excel_bytes
            st.session_state["html_bytes"]  = html_bytes
            st.session_state["results_df"]  = results_df

            # حفظ CSV نهائي وإزالة نقطة الاستئناف
            results_df.to_csv(os.path.join(OUTPUT_DIR, "results_final.csv"), index=False)
            if os.path.exists(checkpoint_csv):
                os.remove(checkpoint_csv)

        dt = time.time() - t0
        st.sidebar.success(f"⏱️ اكتمال التحليل في {dt:.1f} ثانية")
        st.toast("انتهى التحليل ✔️", icon="✅")

    # ===== قسم التنزيلات الدائم =====
    if st.session_state.get("downloads_ready"):
        st.sidebar.markdown("### تنزيل المخرجات")
        st.sidebar.download_button(
            "📥 تحميل النتائج Excel",
            data=st.session_state["excel_bytes"],
            file_name="results.xlsx",
            use_container_width=True
        )
        st.sidebar.download_button(
            "🌐 تحميل التقرير الكامل HTML",
            data=st.session_state["html_bytes"],
            file_name="report.html",
            mime="text/html",
            use_container_width=True
        )

# فاصل وتعريف المطوّر
st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس | 00966553339838 | ")
