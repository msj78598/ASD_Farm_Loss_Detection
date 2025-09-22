# -*- coding: utf-8 -*-
"""
ملف واحد متكامل — نظام اكتشاف حالات الفاقد للفئة الزراعية (Streamlit + YOLO + Isolation Forest)

الغرض:
- تنزيل صور أقمار صناعية ثابتة (Google Static Maps) لإحداثيات العدادات.
- كشف الحقول الزراعية عبر YOLO واستخراج الصندوق الأفضل (أعلى ثقة).
- حساب مساحة الحقل بالمتر المربع ومسافة حافة الحقل إلى موقع العداد.
- تقييم "درجة المخاطر" Risk Score (قواعد + Isolation Forest) وتحديد الأولوية (منخفضة/متوسطة/قصوى).
- عرض بطاقات النتائج مع الصور وروابط الموقع/واتساب، وتصدير تقارير Excel وHTML.
- التعامل مع أحجام كبيرة من الحالات عبر:
  (1) إعادة المحاولة للشبكة Retries + Backoff
  (2) Cache للصور والنماذج
  (3) حفظ تقدّم التشغيل (Checkpointing) واستئنافه عند التوقف المفاجئ.

ملخص البنية داخل هذا الملف الواحد:
- AppConfig: إعدادات قابلة للتعديل (Zoom، عتبات، مسارات، إلخ).
- أدوات Utilities: الدقة المكانية، جلسة HTTP مع Retries، إنشاء المجلدات.
- Data IO: قراءة Excel، إنشاء تقارير Excel/HTML.
- Vision: تحميل YOLO، انتقاء أفضل صندوق، حسابات المساحة/الإزاحة/المسافة.
- Risk: تحميل Isolation Forest + Scaler وحساب درجة المخاطر.
- واجهة Streamlit: عناصر التحكم، تشغيل الدُفعات مع تقدّم واقعي، حفظ واستئناف التقدّم.

تشغيل سريع:
    pip install -r requirements.txt   # أو اعتمادات الحزمة المذكورة أعلاه
    streamlit run app_single.py

أمن وامتثال:
- لا تضع مفتاح Google API داخل الكود؛ استخدم Streamlit Secrets (secrets.toml).
- أظهر إسناد: "Map data © Google" في الواجهة والتقارير.
- تفحّص سياسة الخصوصية لملفات Excel.

"""

# -----------------------------
# الواردات
# -----------------------------
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

# -----------------------------
# 1) الإعدادات العامة AppConfig
# -----------------------------
@dataclass
class AppConfig:
    # إعدادات الخرائط/الصور
    zoom: int = 15                     # يجب أن يطابق Zoom المستخدم لتنزيل صورة Static Maps
    map_size: Tuple[int, int] = (640, 640)
    map_type: str = "satellite"
    calibration_factor: float = 0.6695 # معامل معايرة للمساحة (تجريبي/ميداني)

    # عتبات الكشف
    yolo_conf_threshold: float = 0.5    # حد أولي لاستبعاد المقترحات السيئة (YOLO)
    min_confidence_accept: float = 0.9  # حد القبول النهائي للصندوق الأفضل
    min_area_m2: float = 5000.0         # أصغر مساحة حقل مقبولة بعد المعايرة
    max_edge_distance_m: float = 100.0  # أقصى مسافة من حافة الحقل إلى العداد

    # عتبات المخاطر
    risk_low: float = 0.40              # 0.40–<0.70 متوسطة
    risk_high: float = 0.70             # ≥0.70 قصوى

    # اعتمادية وأداء
    request_timeout_s: int = 20
    max_retries: int = 3
    retry_backoff_s: float = 1.0
    save_checkpoint_every: int = 20     # كل كم حالة نحفظ نقطة استئناف

    # مسارات (نسبية لمجلد التطبيق)
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    output_dir: str = "output"
    models_dir: str = "models"
    form_template_path: str = "TEMPLATE.xlsx"

    # واجهة
    page_title: str = "🌾 نظام اكتشاف حالات الفاقد للفئة الزراعية"
    page_icon: str = "🌾"

# -----------------------------
# 2) أدوات Utilities
# -----------------------------
def meters_per_pixel(lat: float, zoom: int) -> float:
    """احسب دقة الأرض بالمتر لكل بكسل عند خط عرض وزوم معين.
    الصيغة القياسية: 156543.03392 * cos(phi) / 2**zoom
    """
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)

def build_session(total_retries: int = 3, backoff_factor: float = 0.5,
                  status_forcelist=(429, 500, 502, 503, 504)):
    """جلسة HTTP مع سياسة إعادة المحاولة وBackoff للموثوقية."""
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
    """أنشئ المجلدات إذا لم تكن موجودة."""
    for p in paths:
        os.makedirs(p, exist_ok=True)

# -----------------------------
# 3) إدخال/إخراج البيانات Data IO
# -----------------------------
def read_excel(file_obj) -> pd.DataFrame:
    """اقرأ ملف الإدخال Excel مع إسقاط الصفوف الناقصة للأعمدة الأساسية."""
    df = pd.read_excel(file_obj)
    df = df.dropna(subset=["Subscription", "Office", "Breaker", "consumption", "x", "y"])
    return df

def save_results_excel(df: pd.DataFrame) -> bytes:
    """حوّل نتائج التحليل إلى ملف Excel (بايتات) للتنزيل."""
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf.read()

def save_results_html(rows: List[List], colors: dict, detected_dir: str) -> bytes:
    """كوّن تقرير HTML مرئي يحوي بطاقات لكل نتيجة وصورة الكشف وروابط الموقع/واتساب."""
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

# -----------------------------
# 4) رؤية حاسوبية Vision (YOLO)
# -----------------------------
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
    """تحميل نموذج YOLO مرة واحدة (Cache)."""
    return YOLO(model_path)

def detect_best_box(image: Image.Image, model: YOLO, min_conf=0.5):
    """نفّذ التنبؤ على الصورة وأرجع الصندوق الأعلى ثقة."""
    results = model.predict(source=image, imgsz=640, conf=min_conf, verbose=False)[0]
    if results is None or results.boxes is None or len(results.boxes) == 0:
        return None, None
    confs = results.boxes.conf.cpu().numpy()
    idx = int(confs.argmax())
    return results.boxes.xyxy[idx].cpu().numpy(), float(confs[idx])

def detect_field(img_path: str, lat: float, lon: float, meter_id: str, model_yolo: YOLO,
                 zoom: int, calibration_factor: float, min_conf_accept: float,
                 min_area_m2: float, max_edge_distance_m: float, detected_dir: str) -> Optional[FieldDetection]:
    """كشف الحقل من صورة القمر الصناعي وحساب المساحة/المسافة ورسم الصندوق وحفظ صورة مخرجة."""
    image = Image.open(img_path).convert("RGB")
    box, conf = detect_best_box(image, model_yolo, min_conf=min_conf_accept)
    if box is None or conf < min_conf_accept:
        return None

    # حساب مساحة الحقل (م²) باستخدام دقة متر/بكسل الصحيحة عند خط العرض والـZoom المختار
    res = meters_per_pixel(lat, zoom)
    width_px = abs(box[2] - box[0])
    height_px = abs(box[3] - box[1])
    area = width_px * height_px * (res ** 2)
    corrected_area = area * calibration_factor
    if corrected_area < min_area_m2:
        return None

    # تحويل انزياح مركز الصندوق إلى إحداثيات جغرافية تقريبية
    img_cx, img_cy = (image.width / 2.0, image.height / 2.0)
    bx_cx, bx_cy = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    dx_m = (bx_cx - img_cx) * res
    dy_m = (bx_cy - img_cy) * res

    dlat = -(dy_m / 111320.0)
    dlon = dx_m / (40075000.0 * math.cos(math.radians(lat)) / 360.0)
    field_lat = lat + dlat
    field_lon = lon + dlon

    # مسافة الحافة للعداد (مسافة مركز-إلى-مركز ناقص نصف القطر بالمتر)
    radius_px = max(width_px, height_px) / 2.0
    radius_m = radius_px * res
    center_distance = geodesic((lat, lon), (field_lat, field_lon)).meters
    edge_distance = max(center_distance - radius_m, 0.0)
    if edge_distance > max_edge_distance_m:
        return None

    # رسم الصندوق والخط وحفظ الصورة
    draw = ImageDraw.Draw(image)
    draw.rectangle(box.tolist(), outline="green", width=3)
    draw.line([(img_cx, img_cy), (bx_cx, bx_cy)], fill="yellow", width=2)
    os.makedirs(detected_dir, exist_ok=True)
    out_path = os.path.join(detected_dir, f"{meter_id}.png")
    image.save(out_path)

    return FieldDetection(tuple(box.tolist()), conf, int(corrected_area),
                          (field_lat, field_lon), round(edge_distance, 2), out_path)

# -----------------------------
# 5) نموذج المخاطر Risk (Isolation Forest + قواعد)
# -----------------------------
class RiskOutput:
    """حاوية مبسطة لنتيجة المخاطر."""
    def __init__(self, score: float, priority: str):
        self.score = score
        self.priority = priority

class RiskModel:
    """تحميل Isolation Forest وScaler + حساب درجة المخاطر والرتبة."""
    def __init__(self, model_path: str, scaler_path: str, low_thr: float, high_thr: float):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.low_thr = low_thr
        self.high_thr = high_thr

    def compute(self, breaker: float, consumption: float, lon: float, lat: float, area_m2: float) -> RiskOutput:
        # تنبؤ الشذوذ
        X = np.array([[breaker, consumption, lon, lat]], dtype=float)
        Xs = self.scaler.transform(X)
        anomaly = self.model.predict(Xs)[0]  # 1 == anomaly

        # قواعد بسيطة (يمكن ضبطها حسب بياناتكم)
        r1 = 1.0 if breaker < area_m2 * 0.006 else 0.0
        r2 = 1.0 if consumption < area_m2 * 0.4 else 0.0
        r3 = 1.0 if anomaly == 1 else 0.0
        score = 0.4 * r1 + 0.4 * r2 + 0.2 * r3  # 0..1

        if score >= self.high_thr:
            priority = "قصوى"
        elif score >= self.low_thr:
            priority = "متوسطة"
        else:
            priority = "منخفضة"
        return RiskOutput(score=score, priority=priority)

# -----------------------------
# 6) واجهة Streamlit وتدفق التشغيل
# -----------------------------
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

# لطلبات الشبكة (خرائط Google) مع Retries
_session = build_session(total_retries=cfg.max_retries, backoff_factor=cfg.retry_backoff_s)

@st.cache_resource
def _load_models_cached():
    """تحميل YOLO + RiskModel مرة واحدة."""
    yolo = load_yolo(MODEL_PATH)
    risk = RiskModel(ML_MODEL_PATH, SCALER_PATH, cfg.risk_low, cfg.risk_high)
    return yolo, risk

@st.cache_data(show_spinner=False, ttl=24*3600)
def download_image(lat: float, lon: float, meter_id: str, zoom: int, size: Tuple[int,int], map_type: str, timeout: int) -> Optional[str]:
    """تنزيل صورة Google Static Map (مع Cache) ثم حفظها كملف PNG وإرجاع مسارها."""
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

# عنوان الصفحة + تنزيل قالب (إن وجد)
st.title(cfg.page_title)
if os.path.exists(FORM_PATH):
    st.download_button("📥 تحميل نموذج البيانات (TEMPLATE.xlsx)", open(FORM_PATH, "rb"), file_name="TEMPLATE.xlsx")

# رفع ملف الإدخال
uploaded_file = st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])

# عناصر ضبط من الشريط الجانبي
st.sidebar.header("إعدادات التحليل")
ui_zoom = st.sidebar.slider("مستوى التكبير (Zoom)", 10, 20, cfg.zoom, help="يجب أن يطابق مستوى التكبير المستخدم لتنزيل الصور.")
ui_min_conf = st.sidebar.slider("حد قبول ثقة YOLO", 0.50, 0.99, cfg.min_confidence_accept, 0.01)
ui_min_area = st.sidebar.number_input("أدنى مساحة مقبولة (م²)", value=float(cfg.min_area_m2), step=1000.0)
ui_max_edge = st.sidebar.number_input("أقصى مسافة حافة-إلى-عداد (م)", value=float(cfg.max_edge_distance_m), step=10.0)
ui_calib = st.sidebar.number_input("معامل المعايرة (Calibration)", value=float(cfg.calibration_factor), step=0.01, format="%.4f")
st.sidebar.caption("Map data © Google")

# ألوان الأولويات
colors = {"قصوى": "#ff4d4d", "متوسطة": "#ffa500", "منخفضة": "#4CAF50"}

# عند رفع ملف
if uploaded_file:
    df = read_excel(uploaded_file)

    # فلاتر/ترتيب
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
    resume = col_resume.button("⏯️ استئناف من آخر نقطة")

    if start_new or resume:
        model_yolo, risk_model = _load_models_cached()
        progress_bar = st.sidebar.progress(0)
        status_area = st.sidebar.empty()
        t0 = time.time()

        results_rows: List[List] = []
        checkpoint_csv = os.path.join(OUTPUT_DIR, "results_checkpoint.csv")
        processed_ids = set()

        # تحميل نقطة الاستئناف إن طُلِب
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

        # المعالجة الدُفعية مع تجاوز أخطاء الصفوف وحفظ تقدّم دوري
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
                                   ui_zoom, ui_calib, ui_min_conf, ui_min_area, ui_max_edge, DETECTED_DIR)
                if det is None:
                    progress_bar.progress(i / max(n,1))
                    continue

                rk = risk_model.compute(breaker, consumption, lon, lat, det.area_m2)
                results_rows.append([meter_id, rk.priority, rk.score, det.edge_distance_m, det.area_m2,
                                     consumption, breaker, office, lat, lon])

                # بطاقة مرئية في الواجهة
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

                # حفظ نقطة الاستئناف كل N حالات
                if len(results_rows) % cfg.save_checkpoint_every == 0:
                    pd.DataFrame(results_rows, columns=[
                        "Subscription","priority","risk_score","edge_distance_m","area_m2","consumption","breaker","office","lat","lon"
                    ]).to_csv(checkpoint_csv, index=False)

                progress_bar.progress(i / max(n,1))

            except Exception as e:
                status_area.error(f"خطأ في معالجة العداد {row.get('Subscription','?')}: {e}")
                # لا نوقف التشغيل—نواصل للحالة التالية
                continue

        # توليد المخرجات النهائية وتنزيلها
        if results_rows:
            results_df = pd.DataFrame(results_rows, columns=[
                "Subscription","priority","risk_score","edge_distance_m","area_m2","consumption","breaker","office","lat","lon"
            ])
            excel_bytes = save_results_excel(results_df)
            html_bytes = save_results_html(results_rows, colors, DETECTED_DIR)

            st.sidebar.download_button("📥 تحميل النتائج Excel", data=excel_bytes, file_name="results.xlsx")
            st.sidebar.download_button("📥 تحميل التقرير الكامل HTML", data=html_bytes, file_name="report.html", mime="text/html")

            # حفظ CSV نهائي وإزالة نقطة الاستئناف
            results_df.to_csv(os.path.join(OUTPUT_DIR, "results_final.csv"), index=False)
            if os.path.exists(checkpoint_csv):
                os.remove(checkpoint_csv)

        dt = time.time() - t0
        st.sidebar.success(f"⏱️ اكتمل التحليل في {dt:.1f} ثانية")
        st.toast("انتهى التحليل ✔️", icon="✅")

# فاصل وتعريف المطوّر
st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس | 00966553339838 | ")
