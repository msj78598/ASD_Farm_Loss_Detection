سأعطيك نسخة محسّنة كاملة في ملف واحد (كما طلبت)، لكنها تحتوي على معظم التعديلات المقترحة، مع الحفاظ على نفس منطق العمل الأصلي.

التعديلات المضافة باختصار داخل نفس الملف:
• تحسين الـ imports وتنظيمها.
• جعل معاملات RiskModel في AppConfig بدل أرقام ثابتة.
• تحسين estimategreenratio بتقليل التكرار (تحويل HSV للصورة مرة واحدة).
• إضافة إمكانية تعديل بعض الإعدادات من واجهة Streamlit (thresholds رئيسية).
• تحسين HTML وواجهة العرض، مع دالة مساعدة لعرض بطاقة النتيجة.
• إضافة تسجيل (logging) أساسي للأخطاء.

> ملاحظة: هذا الكود يفترض نفس الملفات والنماذج (best.pt, isolationmodel.joblib, isolationscaler.joblib) موجودة تحت models/ كما في نسختك الأصلية.

``python
-- coding: utf-8 --
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

======================= إعداد اللوجر =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(name)

======================= إعدادات ثابتة =======================
@dataclass
class AppConfig:
    # أبعاد الصورة الناتجة من Sentinel
    mapsize: Tuple[int, int] = (640, 640)
    # عرض/ارتفاع المشهد بالأمتار (ثابت)
    scenesizem: int = 2500
    # معامل معايرة المساحة
    calibrationfactor: float = 0.6695
    # أدنى ثقة لقبول صندوق YOLO
    minconfidenceaccept: float = 0.45
    # أدنى مساحة (م²) لقبول الحقل
    minaream2: float = 5000.0

    # ✅ Progressive search by EDGE distance
    rstartm: int = 50
    rstepm: int = 10
    rmaxm: int = 200

    # عتبات تصنيف المخاطر
    risklow: float = 0.40
    riskhigh: float = 0.70

    # إعدادات الاتصال بـ Copernicus
    requesttimeouts: int = 30

    # مجلدات العمل
    imagesdir: str = "images"
    detecteddir: str = "DETECTEDFIELDS"
    outputdir: str = "output"
    modelsdir: str = "models"

    # إعدادات صفحة Streamlit
    pagetitle: str = "🌾 نظام اكتشاف حالات الفاقد للفئة الزراعية"
    pageicon: str = "🌾"

    # ====== إعدادات فلترة “الخضرة” ======
    greenratiomin: float = 0.0
    greendominance: float = 1.1
    greenminvalue: int = 60

    # ====== إعدادات نموذج المخاطر (بدلاً من أرقام ثابتة داخل الكود) ======
    breakerareacoef: float = 0.0013  # معامل علاقة سعة القاطع بالمساحة الخضراء
    minconscoef: float = 0.20        # الحد الأدنى للاستهلاك بالنسبة للمساحة الخضراء
    wr1: float = 0.4                  # وزن معيار القاطع
    wr2: float = 0.4                  # وزن معيار الاستهلاك
    wr3: float = 0.2                  # وزن نموذج العزلة (Isolation Forest)

cfg = AppConfig()

======================= أدوات عامة =======================
def ensuredirs(paths: str) -> None:
    """إنشاء المجلدات المطلوبة إن لم تكن موجودة."""
    for p in paths:
        os.makedirs(p, existok=True)

def cleanmeterid(val) -> str:
    """إرجاع رقم العداد كنص نظيف بدون .0 أو صيغة علمية أو مسافات."""
    if pd.isna(val):
        return ""
    try:
        f = float(val)
        if f.isinteger():
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

def readexcel(fileobj) -> pd.DataFrame:
    """قراءة ملف الإكسل وتحضير البيانات الأساسية."""
    df = pd.readexcel(fileobj, dtype={"Subscription": str})
    df["Subscription"] = df["Subscription"].apply(cleanmeterid)
    # حذف الصفوف التي لا تحتوي على بيانات أساسية مطلوبة
    return df.dropna(subset=["Subscription", "Office", "Breaker", "consumption", "x", "y"])

def saveresultsexcel(df: pd.DataFrame) -> bytes:
    """تصدير النتائج إلى ملف Excel في الذاكرة."""
    buf = io.BytesIO()
    df.toexcel(buf, index=False)
    buf.seek(0)
    return buf.read()

def saveresultshtml(rows: List[List], colors: dict, detecteddir: str) -> bytes:
    """
    توليد تقرير HTML بسيط يحتوي على صور الحقول وبياناتها.
    rows: [meter, pr, score, edged, centerd, area, cons, br, off, lat, lon]
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
        meterid, priority, risk, edged, centerd, area, consumption, breaker, office, lat, lon = r
        border = colors.get(priority, "#ccc")
        pth = os.path.join(detecteddir, f"{meterid}.png")
        imgtag = ""
        if os.path.exists(pth):
            with open(pth, "rb") as f:
                imgb64 = base64.b64encode(f.read()).decode()
            imgtag = (
                f"<img src='data:image/png;base64,{imgb64}' "
                f"width='250' class='img'>"
            )

        html.append(f"""
<div class='card' style='border:4px solid {border};'>
  {imgtag}<br>
  <strong>عداد {escape(str(meterid))} ({escape(str(priority))})</strong><br>
  خطر: {risk100:.1f}% | حافة: {edged:.1f}م | مركز: {centerd:.1f}م | مساحة: {area}م²<br>
  الاستهلاك: {consumption} | القاطع: {breaker} | المكتب: {escape(str(office))}<br>
  <a href='https://maps.google.com?q={lat},{lon}'>📍 الموقع</a>
  &nbsp;|&nbsp;
  <a href='https://wa.me/?text=عداد:{meterid}%20الموقع:{lat},{lon}'>📲 واتساب</a>
</div>""")
    html.append("</div></body></html>")
    return "\n".join(html).encode("utf-8")

def makeresultcardhtml(
    meter: str,
    pr: str,
    score: float,
    det,
    cons: float,
    br: float,
    off: str,
    lat: float,
    lon: float,
    bordercolor: str,
) -> str:
    """توليد HTML لعرض بطاقة نتيجة واحدة في Streamlit."""
    with open(det.outimgpath, "rb") as f:
        img64 = base64.b64encode(f.read()).decode()

    cardhtml = f"""
<div style="border:4px solid {bordercolor};padding:10px;border-radius:12px;margin:6px;text-align:center;">
  <img src="data:image/png;base64,{img64}" width="260" style="border-radius:8px;"><br>
  <strong>عداد {meter} ({pr})</strong><br>
  خطر:{score100:.1f}% | حافة:{det.edgedistancem:.1f}م | مركز:{det.centerdistancem:.1f}م<br>
  مساحة:{det.aream2}م² | خضرة:{det.greenratio100:.0f}%<br>
  استهلاك:{cons} | قاطع:{br} | مكتب:{off}<br>
  <a href="https://maps.google.com?q={lat},{lon}">📍 الموقع</a>
</div>
"""
    return cardhtml

======================= تحميل النماذج =======================
@st.cacheresource
def loadyolo(modelpath: str):
    """تحميل نموذج YOLO مرة واحدة لكل جلسة."""
    logger.info(f"Loading YOLO model from {modelpath}")
    return YOLO(modelpath)

class RiskModel:
    """
    نموذج حساب مستوى المخاطر:
    - يستخدم Isolation Forest + قواعد منطقية تعتمد على:
       سعة القاطع
       الاستهلاك
       المساحة الخضراء الفعلية
    """

    def init(self, modelpath: str, scalerpath: str, config: AppConfig):
        self.model = joblib.load(modelpath)
        self.scaler = joblib.load(scalerpath)
        self.lowthr = config.risklow
        self.highthr = config.riskhigh
        self.breakerareacoef = config.breakerareacoef
        self.minconscoef = config.minconscoef
        self.wr1 = config.wr1
        self.wr2 = config.wr2
        self.wr3 = config.wr3

    def compute(
        self,
        breaker: float,
        consumption: float,
        lon: float,
        lat: float,
        aream2: float,
        greenratio: float,
    ):
        """
        breaker: سعة القاطع (أمبير)
        consumption: الاستهلاك (الوحدة حسب بياناتك - مثلاً ك.و.س)
        lon, lat: إحداثيات العداد
        aream2: المساحة المكتشفة (م²)
        greenratio: نسبة الخضرة داخل الصندوق [0..1]
        """
        # 1. المساحة الخضراء الفعلية
        effectivearea = aream2  greenratio

        # 2. تحضير البيانات لنموذج العزلة
        X = np.array([[breaker, consumption, lon, lat]], dtype=float)
        Xs = self.scaler.transform(X)
        anomaly = self.model.predict(Xs)[0]
        # ملاحظة: تأكد من أن 1 تعني حالة شاذة في نموذجك أو عدّل الشرط حسب تدريبك

        # 3. معايير المخاطر
        # (r1) هل سعة القاطع أقل من المطلوب لمساحة خضراء معينة؟
        r1 = 1.0 if breaker < (effectivearea  self.breakerareacoef) else 0.0

        # (r2) هل الاستهلاك أقل من الحد الأدنى المتوقع لهذه المساحة؟
        r2 = 1.0 if consumption < (effectivearea  self.minconscoef) else 0.0

        # (r3) ناتج نموذج العزلة (هنا نفترض أن 1 تعني حالة شاذة / خطر أعلى)
        r3 = 1.0 if anomaly == 1 else 0.0

        # 4. حساب النتيجة النهائية
        score = self.wr1  r1 + self.wr2  r2 + self.wr3  r3

        # 5. تصنيف الأولوية
        if score >= self.highthr:
            pr = "قصوى"
        elif score >= self.lowthr:
            pr = "متوسطة"
        else:
            pr = "منخفضة"

        return score, pr

======================= دالة تقدير الخضرة =======================
def estimategreenratio(
    rgbarr: np.ndarray,
    hsvarr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    boxxyxy: Tuple[float, float, float, float],
) -> float:
    """
    تقدير نسبة الخضرة داخل الصندوق المحدد من صورة كاملة تم حساب RGB/HSV لها مسبقاً.
    - rgbarr: مصفوفة RGB للصورة الأصلية (H, W, 3)
    - hsvarr: (Harr, Sarr, Varr) للصورة كاملة أيضاً
    - boxxyxy: إحداثيات الصندوق داخل الصورة
    """
    x1, y1, x2, y2 = [int(v) for v in boxxyxy]
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # تقطيع من الـ numpy مباشرة لتوفير الوقت (بدل crop لكل صندوق)
    croprgb = rgbarr[y1:y2, x1:x2, :]
    if croprgb.size == 0:
        return 0.0

    R = croprgb[..., 0].astype(np.float32)
    G = croprgb[..., 1].astype(np.float32)
    B = croprgb[..., 2].astype(np.float32)

    # 1) هيمنة الأخضر
    dominancemask = (
        (G > R  cfg.greendominance)
        & (G > B  cfg.greendominance)
        & (G > cfg.greenminvalue)
    )

    # 2) ExG
    Rn = R / 255.0
    Gn = G / 255.0
    Bn = B / 255.0
    exg = 2.0  Gn - Rn - Bn
    exgmask = exg > 0.08

    # 3) HSV
    Hfull, Sfull, Vfull = hsvarr
    H = Hfull[y1:y2, x1:x2]
    S = Sfull[y1:y2, x1:x2]
    V = Vfull[y1:y2, x1:x2]
    hsvmask = (H >= 25) & (H <= 67) & (S >= 60) & (V >= 50)

    greenmask = dominancemask | exgmask | hsvmask
    return float(greenmask.mean())

======================= الكشف =======================
@dataclass
class FieldDetection:
    bboxxyxy: Tuple[float, float, float, float]
    conf: float
    aream2: int
    centerlatlon: Tuple[float, float]
    edgedistancem: float
    centerdistancem: float
    outimgpath: str
    greenratio: float

def detectboxes(image: Image.Image, model: YOLO, minconf: float = 0.5):
    """تشغيل YOLO على صورة واحدة وإرجاع الصناديق المرتبة من الأعلى ثقةً."""
    res = model.predict(source=image, imgsz=640, conf=minconf, verbose=False)[0]
    if not res or not res.boxes or len(res.boxes) == 0:
        return []
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    idxs = np.argsort(-confs)
    return [(boxes[i], float(confs[i])) for i in idxs]

def detectfieldprogressive(
    imgpath: str,
    lat: float,
    lon: float,
    meterid: str,
    modelyolo: YOLO,
    calibrationfactor: float,
    minconfaccept: float,
    minaream2: float,
    detecteddir: str,
    rstart: int = 50,
    rstep: int = 10,
    rmax: int = 200,
) -> Optional[FieldDetection]:
    """
    ✅ Progressive search by EDGE distance:
    - البحث عن الحقل الأقرب باستخدام مسافة الحافة EDGE.
    - R = 50, 60, 70, ..., 200
    - أول R يوجد فيه صندوق بـ edgedistance <= R يتم اختياره (الأقرب بالحافة ثم بالمركز).
    """
    image = Image.open(imgpath).convert("RGB")
    boxes = detectboxes(image, modelyolo, minconf=minconfaccept)
    if not boxes:
        return None

    mperpx = cfg.scenesizem / float(cfg.mapsize[0])
    cx, cy = image.width / 2, image.height / 2

    # تجهيز مصفوفات RGB/HSV مرة واحدة للصورة كاملة
    rgbarr = np.asarray(image, dtype=np.uint8)
    hsvimg = image.convert("HSV")
    Hfull = np.asarray(hsvimg.getchannel(0), dtype=np.uint8)
    Sfull = np.asarray(hsvimg.getchannel(1), dtype=np.uint8)
    Vfull = np.asarray(hsvimg.getchannel(2), dtype=np.uint8)
    hsvarr = (Hfull, Sfull, Vfull)

    candidates = []
    for box, conf in boxes:
        wpx = abs(box[2] - box[0])
        hpx = abs(box[3] - box[1])
        area = wpx  hpx  (mperpx*2)
        corrected = area  calibrationfactor
        if corrected < minaream2:
            continue

        # حساب إحداثيات مركز الصندوق بالنسبة للصورة
        bx = (box[0] + box[2]) / 2
        by = (box[1] + box[3]) / 2
        dxm = (bx - cx)  mperpx
        dym = (by - cy)  mperpx

        # تحويل فرق الإحداثيات إلى lat/lon
        dlat = -(dym / 111320.0)
        dlon = dxm / (40075000.0  math.cos(math.radians(lat)) / 360.0)
        flat = lat + dlat
        flon = lon + dlon

        centerdist = geodesic((lat, lon), (flat, flon)).meters

        # تقدير نسبة الخضرة
        greenratio = estimategreenratio(rgbarr, hsvarr, tuple(box.tolist()))
        if greenratio < cfg.greenratiomin:
            continue

        radiuspx = max(wpx, hpx) / 2
        radiusm = radiuspx  mperpx
        edgedist = max(centerdist - radiusm, 0.0)

        candidates.append(
            (
                edgedist,
                centerdist,
                box,
                conf,
                int(corrected),
                (flat, flon),
                greenratio,
            )
        )

    if not candidates:
        return None

    chosen = None
    chosenR = None
    for R in range(rstart, rmax + 1, rstep):
        within = [c for c in candidates if c[0] <= R]  # c[0] = edgedist
        if within:
            chosen = min(within, key=lambda x: (x[0], x[1]))  # (edge, center)
            chosenR = R
            break

    if chosen is None:
        return None

    edgedist, centerdist, box, conf, aream2, (flat, flon), greenratio = chosen

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
        f"R<= {chosenR}m | Edge:{edgedist:.1f}m | Center:{centerdist:.1f}m | Green {greenratio100:.0f}%",
        fill="white",
    )

    os.makedirs(detecteddir, existok=True)
    outpath = os.path.join(detecteddir, f"{meterid}.png")
    image.save(outpath)

    return FieldDetection(
        bboxxyxy=tuple(box.tolist()),
        conf=float(conf),
        aream2=int(aream2),
        centerlatlon=(flat, flon),
        edgedistancem=float(edgedist),
        centerdistancem=float(centerdist),
        outimgpath=outpath,
        greenratio=float(greenratio),
    )

======================= CDSE Token & Download =======================
TOKENURL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

def getcdsetoken() -> str:
    """الحصول على توكن CDSE مع استخدام sessionstate كـ cache."""
    tok = st.sessionstate.get("cdsetoken")
    exp = st.sessionstate.get("cdsetokenexp", 0)
    if tok and time.time() < exp - 60:
        return tok

    cid = st.secrets.get("CDSECLIENTID")
    csec = st.secrets.get("CDSECLIENTSECRET")
    if not cid or not csec:
        raise RuntimeError(
            "CDSECLIENTID / CDSECLIENTSECRET غير موجودة في secrets.toml"
        )

    data = {
        "granttype": "clientcredentials",
        "clientid": cid,
        "clientsecret": csec,
    }
    r = requests.post(TOKENURL, data=data, timeout=20)
    if r.statuscode != 200:
        raise RuntimeError(f"CDSE token error {r.statuscode}: {r.text[:200]}")
    js = r.json()
    access = js["accesstoken"]
    expires = int(js.get("expiresin", 3600))

    st.sessionstate["cdsetoken"] = access
    st.sessionstate["cdsetokenexp"] = time.time() + expires
    return access

def bboxfrommeters(lat: float, lon: float, sizem: float):
    """حساب صندوق (BBox) بالإحداثيات الجغرافية لمساحة مربعة حول نقطة معينة."""
    half = sizem / 2.0
    dlat = half / 111320.0
    dlon = half / (111320.0  math.cos(math.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]

@st.cachedata(showspinner=False, ttl=24  3600)
def downloadimage(lat: float, lon: float, meterid: str, timeout: int = 30):
    """
    تحميل صورة Sentinel-2 من Copernicus لموقع عداد معين.
    تستخدم Cache على مستوى الملف (images/meterid.png).
    """
    imgpath = os.path.join(cfg.imagesdir, f"{meterid}.png")
    if os.path.exists(imgpath):
        return imgpath

    def request(token: str):
        bbox = bboxfrommeters(lat, lon, cfg.scenesizem)
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
                "width": cfg.mapsize[0],
                "height": cfg.mapsize[1],
                "responses": [
                    {"identifier": "default", "format": {"type": "image/png"}}
                ],
            },
            "evalscript": """//VERSION=3
function setup(){return {input:["B04","B03","B02"],output:{bands:3}}}
function evaluatePixel(s){
  return [s.B041.8, s.B031.8, s.B021.8]
}
""",
        }
        headers = {"Authorization": f"Bearer {token}"}
        return requests.post(url, headers=headers, json=payload, timeout=timeout)

    token = getcdsetoken()
    r = request(token)
    if r.statuscode == 401:
        # إعادة المحاولة بتوكن جديد
        token = getcdsetoken()
        r = request(token)

    if r.statuscode == 200:
        with open(imgpath, "wb") as f:
            f.write(r.content)
        return imgpath
    else:
        logger.warning(
            "Copernicus status %s for meter %s: %s",
            r.statuscode,
            meterid,
            r.text[:200],
        )
        st.warning(
            f"Copernicus status {r.statuscode} للعداد {meterid}: {r.text[:200]}"
        )
        return None

======================= واجهة Streamlit =======================
st.setpageconfig(
    pagetitle=cfg.pagetitle, pageicon=cfg.pageicon, layout="wide"
)
ensuredirs(cfg.imagesdir, cfg.detecteddir, cfg.outputdir, cfg.modelsdir)

MODELPATH = os.path.join(cfg.modelsdir, "best.pt")
MLMODELPATH = os.path.join(cfg.modelsdir, "isolationmodel.joblib")
SCALERPATH = os.path.join(cfg.modelsdir, "isolationscaler.joblib")

st.title(cfg.pagetitle)

uploaded = st.fileuploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])
colors = {"قصوى": "#ff4d4d", "متوسطة": "#ffa500", "منخفضة": "#4CAF50"}

if uploaded:
    df = readexcel(uploaded)
    st.sidebar.info(f"🔢 عدد الحالات في الملف: {len(df)}")

    # ====== إعدادات إضافية من واجهة المستخدم ======
    st.sidebar.markdown("### ⚙️ إعدادات التحليل")

    # فلترة حسب القاطع
    breakerfilter = st.sidebar.selectbox(
        "سعة القاطع", ["الكل"] + sorted(df["Breaker"].unique().tolist())
    )

    # ترتيب حسب الاستهلاك
    sortorder = st.sidebar.radio(
        "ترتيب حسب الاستهلاك", ["بدون ترتيب", "تصاعدي", "تنازلي"]
    )

    # إمكانية تعديل أدنى مساحة وأدنى ثقة من الواجهة
    cfg.minaream2 = st.sidebar.numberinput(
        "الحد الأدنى للمساحة (م²)",
        minvalue=1000.0,
        maxvalue=50000.0,
        value=cfg.minaream2,
        step=500.0,
    )
    cfg.minconfidenceaccept = st.sidebar.slider(
        "أدنى ثقة لنموذج YOLO",
        minvalue=0.1,
        maxvalue=0.9,
        value=cfg.minconfidenceaccept,
        step=0.05,
    )
    cfg.greenratiomin = st.sidebar.slider(
        "الحد الأدنى لنسبة الخضرة داخل الحقل (%)",
        minvalue=0.0,
        maxvalue=100.0,
        value=cfg.greenratiomin * 100.0,
        step=5.0,
    ) / 100.0

    # فلترة حسب القاطع والاستهلاك
    if breakerfilter != "الكل":
        df = df[df["Breaker"] == breakerfilter]

    if sortorder == "تصاعدي":
        df = df.sortvalues(by="consumption", ascending=True)
    elif sortorder == "تنازلي":
        df = df.sortvalues(by="consumption", ascending=False)

    previewonly = st.sidebar.checkbox("🖼️ عرض الصور فقط (بدون تشغيل نموذج المخاطر)")
    if st.sidebar.button("📥 تنزيل/عرض الصور"):
        progress = st.sidebar.progress(0)
        cols = st.columns(4)
        shown = 0
        n = len(df)
        t0 = time.time()

        for i, (, row) in enumerate(df.iterrows(), 1):
            meter = cleanmeterid(row["Subscription"])
            lat = float(row["y"])
            lon = float(row["x"])
            p = downloadimage(lat, lon, meter)
            if p:
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                cols[shown % 4].markdown(
                    f"""
<div style="border:1px solid #ddd;border-radius:8px;padding:6px;margin:6px;text-align:center">
  <img src="data:image/png;base64,{b64}" width="230" style="border-radius:6px"><br>
  <small>عداد {meter}<br>Lat {lat:.6f}, Lon {lon:.6f}</small>
</div>""",
                    unsafeallowhtml=True,
                )
                shown += 1
            progress.progress(i / max(n, 1))

        st.sidebar.success(
            f"✅ تم عرض {shown} صورة في {time.time()-t0:.1f} ثانية"
        )
        st.stop()

    if st.sidebar.button("🚀 بدء التحليل"):
        modelyolo = loadyolo(MODELPATH)
        riskmodel = RiskModel(MLMODELPATH, SCALERPATH, cfg)

        progress = st.sidebar.progress(0)
        results = []
        cols = st.columns(3)
        coli = 0
        t0 = time.time()
        n = len(df)

        for i, (, row) in enumerate(df.iterrows(), 1):
            try:
                meter = cleanmeterid(row["Subscription"])
                lat = float(row["y"])
                lon = float(row["x"])
                br = float(row["Breaker"])
                cons = float(row["consumption"])
                off = str(row["Office"])

                imgpath = downloadimage(lat, lon, meter)
                if not imgpath:
                    progress.progress(i / max(n, 1))
                    continue

                det = detectfieldprogressive(
                    imgpath,
                    lat,
                    lon,
                    meter,
                    modelyolo,
                    cfg.calibrationfactor,
                    cfg.minconfidenceaccept,
                    cfg.minaream2,
                    cfg.detecteddir,
                    rstart=cfg.rstartm,
                    rstep=cfg.rstepm,
                    rmax=cfg.rmaxm,
                )

                if det is None:
                    progress.progress(i / max(n, 1))
                    continue

                # إذا تم اختيار "عرض الصور فقط" لا نحسب المخاطر
                if previewonly:
                    score, pr = 0.0, "غير محسوب"
                else:
                    score, pr = riskmodel.compute(
                        br, cons, lon, lat, det.aream2, det.greenratio
                    )

                # حفظ النتيجة للـ Excel/HTML
                results.append(
                    [
                        meter,
                        pr,
                        score,
                        det.edgedistancem,
                        det.centerdistancem,
                        det.aream2,
                        cons,
                        br,
                        off,
                        lat,
                        lon,
                    ]
                )

                bordercolor = colors.get(pr, "#ccc")
                cardhtml = makeresultcardhtml(
                    meter,
                    pr,
                    score,
                    det,
                    cons,
                    br,
                    off,
                    lat,
                    lon,
                    bordercolor,
                )
                cols[coli % 3].markdown(cardhtml, unsafeallowhtml=True)
                coli += 1

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
            resdf = pd.DataFrame(
                results,
                columns=[
                    "Subscription",
                    "priority",
                    "riskscore",
                    "edgedistancem",
                    "centerdistancem",
                    "aream2",
                    "consumption",
                    "breaker",
                    "office",
                    "lat",
                    "lon",
                ],
            )

            st.sidebar.markdown("### 📊 ملخص سريع")
            st.sidebar.write(resdf["priority"].valuecounts())

            st.sidebar.downloadbutton(
                "📥 نتائج Excel",
                data=saveresultsexcel(resdf),
                filename="results.xlsx",
            )
            st.sidebar.downloadbutton(
                "📥 تقرير HTML",
                data=saveresultshtml(results, colors, cfg.detecteddir),
                file_name="report.html",
                mime="text/html",
            )

        st.sidebar.success(f"⏱️ اكتمل التحليل في {round(time.time()-t0,1)} ثانية")

st.markdown("---")
st.markdown("👨‍💻 تطوير : مشهور العباس 2026 | 00966553339838 | ")
``

إذا ترغب:
• أستطيع بعد ذلك تقسيم هذا الملف نفسه إلى هيكل مشروع كامل (عدة ملفات ومجلدات)، أو
• تخصيص جزء معيّن لتحسين إضافي (مثلاً فقط منطق RiskModel أو فقط واجهة Streamlit).
