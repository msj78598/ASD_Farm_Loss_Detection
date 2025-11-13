# -*- coding: utf-8 -*-
"""
نظام انتقاء حالات الفاقد الزراعي بمحرك تفكير (GPT-like):
- تنزيل RGB + NDVI (FLOAT32) مع فلاتر SCL + فلتر ضعف الإشارة + قص القمم
- كشف YOLO ثم حساب مؤشرات النمو: intensity / cohesion / peakiness
- RuleEngine يجمع "أدلة" مرجّحة ويعطي قرارًا مفسّرًا + نسبة خطر نهائية
- فلاتر واجهة: سعة القاطع، حد أدنى للخطر، تفعيل الفلتر الزراعي، ثقة YOLO
"""

import os, io, time, base64, math, collections
from dataclasses import dataclass
from typing import Tuple, List, Dict

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
    default_yolo_conf: float = 0.35

    # هندسي
    min_area_m2: float = 5000.0
    max_edge_distance_m: float = 120.0

    # ملفات
    images_dir: str = "images"
    detected_dir: str = "DETECTED_FIELDS"
    output_dir: str = "output"
    models_dir: str = "models"
    page_title: str = "🌾 نظام اكتشاف حالات الفاقد للفئة الزراعية"
    page_icon: str = "🌾"

    # NDVI
    topk_ratio: float = 0.20
    ndvi_min_valid: float = 0.10
    ndvi_green_mask: float = 0.35
    ndvi_max_clip: float = 0.85
    low_signal_sum: float = 0.20

    # فلتر زراعي مبدئي (يمكن تغييره من الواجهة)
    pass_intensity: float = 0.55
    pass_cohesion: float = 0.45
    pass_peakiness: float = 0.40

cfg = AppConfig()

# ======================= أدوات عامة =======================
def ensure_dirs(*paths): [os.makedirs(p, exist_ok=True) for p in paths]

def read_excel(file_obj) -> pd.DataFrame:
    df = pd.read_excel(file_obj)
    return df.dropna(subset=["Subscription", "Office", "Breaker", "consumption", "x", "y"])

def save_results_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO(); df.to_excel(buf, index=False); buf.seek(0); return buf.read()

# ======================= نماذج =======================
@st.cache_resource
def load_yolo(model_path: str): return YOLO(model_path)

@st.cache_resource
def load_iso(model_path: str, scaler_path: str):
    return joblib.load(model_path), joblib.load(scaler_path)

# ======================= Copernicus =======================
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

def get_cdse_token():
    tok = st.session_state.get("_cdse_token"); exp = st.session_state.get("_cdse_token_exp", 0)
    if tok and time.time() < exp - 60: return tok
    cid = st.secrets.get("CDSE_CLIENT_ID"); csec = st.secrets.get("CDSE_CLIENT_SECRET")
    if not cid or not csec: raise RuntimeError("CDSE_CLIENT_ID / CDSE_CLIENT_SECRET غير موجودة في secrets.toml")
    r = requests.post(TOKEN_URL, data={"grant_type":"client_credentials","client_id":cid,"client_secret":csec}, timeout=20)
    r.raise_for_status(); js = r.json()
    st.session_state["_cdse_token"] = js["access_token"]; st.session_state["_cdse_token_exp"] = time.time()+int(js.get("expires_in",3600))
    return st.session_state["_cdse_token"]

def bbox_from_meters(lat: float, lon: float, size_m: float):
    half=size_m/2; dlat=half/111320.0; dlon=half/(111320.0*math.cos(math.radians(lat)))
    return [lon-dlon, lat-dlat, lon+dlon, lat+dlat]

def _process(bounds_bbox, responses, evalscript, token, timeout=30):
    url="https://sh.dataspace.copernicus.eu/api/v1/process"
    payload={
        "input":{"bounds":{"bbox":bounds_bbox,"properties":{"crs":"http://www.opengis.net/def/crs/EPSG/0/4326"}},
                 "data":[{"type":"sentinel-2-l2a","dataFilter":{"maxCloudCoverage":50,"mosaickingOrder":"mostRecent"},
                          "processing":{"upsampling":"NEAREST","downsampling":"NEAREST"}}]},
        "output":{"width":cfg.map_size[0],"height":cfg.map_size[1],"responses":responses},
        "evalscript":evalscript}
    return requests.post(url, headers={"Authorization":f"Bearer {token}"}, json=payload, timeout=timeout)

@st.cache_data(show_spinner=False, ttl=24*3600)
def download_rgb_and_ndvi(lat, lon, meter_id):
    rgb_path=os.path.join(cfg.images_dir,f"{meter_id}.png")
    ndvi_path=os.path.join(cfg.images_dir,f"{meter_id}_ndvi.tif")
    if os.path.exists(rgb_path) and os.path.exists(ndvi_path): return rgb_path, ndvi_path
    bbox=bbox_from_meters(lat, lon, cfg.scene_size_m); token=get_cdse_token()

    eval_rgb = """
//VERSION=3
function setup(){return {input:["B04","B03","B02"],output:{bands:3}}}
function evaluatePixel(s){return [s.B04*1.8, s.B03*1.8, s.B02*1.8]}
"""
    r1=_process(bbox,[{"identifier":"default","format":{"type":"image/png"}}],eval_rgb,token)
    if r1.status_code!=200: return None,None
    with open(rgb_path,"wb") as f: f.write(r1.content)

    eval_ndvi=f"""
//VERSION=3
function setup(){{return {{input:["B08","B04","SCL"],output:{{bands:1,sampleType:"FLOAT32"}}}}}}
function bad(c){{return (c==3)||(c==6)||(c==8)||(c==9)||(c==10)||(c==11);}}
function evaluatePixel(s){{
  var sum=s.B08+s.B04; var ndvi=(s.B08-s.B04)/(sum+1e-6);
  if(bad(s.SCL) || sum<{cfg.low_signal_sum:.2f}) ndvi=-1.0;
  if(ndvi>{cfg.ndvi_max_clip:.2f}) ndvi={cfg.ndvi_max_clip:.2f};
  return [ndvi];
}}"""
    r2=_process(bbox,[{"identifier":"default","format":{"type":"image/tiff"}}],eval_ndvi,token)
    if r2.status_code!=200: return None,None
    with open(ndvi_path,"wb") as f: f.write(r2.content)
    return rgb_path, ndvi_path

# ======================= NDVI مؤشرات =======================
def ndvi_from_tiff(path: str)->np.ndarray:
    arr=np.array(Image.open(path)).astype(np.float32)
    return np.clip(arr,-1.0,1.0)

def _largest_connected_ratio(mask: np.ndarray)->float:
    if mask.size==0 or mask.max()==0: return 0.0
    H,W=mask.shape; seen=np.zeros_like(mask,dtype=bool); best=0
    for y in range(H):
        if mask[y].max()==0: continue
        for x in range(W):
            if mask[y,x] and not seen[y,x]:
                q=collections.deque([(y,x)]); seen[y,x]=True; cnt=0
                while q:
                    cy,cx=q.popleft(); cnt+=1
                    for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                        if 0<=ny<H and 0<=nx<W and mask[ny,nx] and not seen[ny,nx]:
                            seen[ny,nx]=True; q.append((ny,nx))
                best=max(best,cnt)
    return float(best)/float(mask.size)

def _peakiness(values: np.ndarray, bins: int=32)->float:
    if values.size==0: return 0.0
    hist,_=np.histogram(values,bins=bins,range=(0.0,1.0),density=False)
    p=hist.astype(np.float32); p=p/(p.sum()+1e-8)
    ent=-(p*np.log(p+1e-12)).sum(); ent_norm=ent/(np.log(bins)+1e-12)
    return float(np.clip(1.0-ent_norm,0.0,1.0))

def vegetation_signature(ndvi: np.ndarray, box: Tuple[float,float,float,float]):
    x1,y1,x2,y2=[int(v) for v in box]
    x1=max(0,min(ndvi.shape[1]-1,x1)); x2=max(0,min(ndvi.shape[1],x2))
    y1=max(0,min(ndvi.shape[0]-1,y1)); y2=max(0,min(ndvi.shape[0],y2))
    if x2<=x1 or y2<=y1: return 0.0,0.0,0.0
    crop=ndvi[y1:y2, x1:x2]; 
    if crop.size==0: return 0.0,0.0,0.0

    valid=crop[(crop>=cfg.ndvi_min_valid)&(crop<=cfg.ndvi_max_clip)]
    if valid.size==0: return 0.0,0.0,0.0

    # 1) شدة الأخضر الداكن
    k=max(1,int(round(cfg.topk_ratio*valid.size)))
    topk=np.partition(valid,-k)[-k:]
    base_intensity=float(np.clip(topk.mean(),0.0,1.0))

    # 2) تماسك مكاني
    green_mask=(crop>=cfg.ndvi_green_mask)&(crop<=cfg.ndvi_max_clip)
    cohesion=_largest_connected_ratio(green_mask.astype(np.uint8))

    # 3) ذروة توزيع
    peakiness=_peakiness(valid,bins=32)

    # دمج (يحاكي ترجيح GPT: يعطي وزن أعلى للتماسك)
    intensity_final=base_intensity*float(np.clip(0.5+0.50*cohesion+0.15*peakiness,0.5,1.0))
    if cohesion<0.25 and base_intensity<0.60: intensity_final*=0.4

    return float(np.clip(intensity_final,0.0,1.0)), float(np.clip(cohesion,0.0,1.0)), float(np.clip(peakiness,0.0,1.0))

# ======================= YOLO والكشف =======================
def detect_boxes(image: Image.Image, model: YOLO, min_conf=0.35):
    res=model.predict(source=image, imgsz=640, conf=min_conf, verbose=False)[0]
    if (not res) or (res.boxes is None) or (len(res.boxes)==0): return []
    boxes=res.boxes.xyxy.cpu().numpy(); confs=res.boxes.conf.cpu().numpy()
    idxs=np.argsort(-confs); return [(boxes[i], float(confs[i])) for i in idxs]

@dataclass
class FieldDetection:
    bbox_xyxy: Tuple[float,float,float,float]
    conf: float
    area_m2: int
    center_latlon: Tuple[float,float]
    edge_distance_m: float
    out_img_path: str
    intensity: float
    cohesion: float
    peakiness: float

def detect_field(rgb_path, ndvi_path, lat, lon, meter_id, model_yolo,
                 yolo_conf, min_area_m2, max_edge_distance_m, detected_dir,
                 apply_strict, thr_i, thr_c, thr_p):
    image=Image.open(rgb_path).convert("RGB"); ndvi=ndvi_from_tiff(ndvi_path)
    candidates=detect_boxes(image, model_yolo, min_conf=yolo_conf)
    if not candidates: return None

    m_per_px=cfg.scene_size_m/float(cfg.map_size[0])
    for box, conf in candidates:
        w_px,h_px=abs(box[2]-box[0]), abs(box[3]-box[1])
        area=w_px*h_px*(m_per_px**2); corrected=area*cfg.calibration_factor
        if corrected<min_area_m2: continue

        cx,cy=image.width/2, image.height/2
        bx,by=(box[0]+box[2])/2, (box[1]+box[3])/2
        dx_m,dy_m=(bx-cx)*m_per_px, (by-cy)*m_per_px
        dlat=-(dy_m/111320.0); dlon=dx_m/(40075000.0*math.cos(math.radians(lat))/360.0)
        flat,flon=lat+dlat, lon+dlon

        radius_px=max(w_px,h_px)/2; radius_m=radius_px*m_per_px
        dist=geodesic((lat,lon),(flat,flon)).meters
        edge=max(dist-radius_m,0); 
        if edge>max_edge_distance_m: continue

        intensity, cohesion, peakiness=vegetation_signature(ndvi, tuple(box.tolist()))
        if apply_strict and not (intensity>=thr_i and cohesion>=thr_c and peakiness>=thr_p):
            continue

        draw=ImageDraw.Draw(image)
        draw.rectangle(box.tolist(), outline="green", width=3)
        draw.line([(cx,cy),(bx,by)], fill="yellow", width=2)
        draw.text((int(box[0])+4, int(box[1])+4), f"NDVI {intensity*100:.0f}%", fill="white")
        os.makedirs(detected_dir, exist_ok=True)
        out_path=os.path.join(detected_dir,f"{meter_id}.png"); image.save(out_path)

        return FieldDetection(tuple(box.tolist()), conf, int(corrected), (flat,flon), round(edge,2),
                              out_path, intensity, cohesion, peakiness)
    return None

# ======================= RuleEngine (تفكير GPTي مبسّط) =======================
class RuleEngine:
    """
    يجمع أدلة موزونة ويشرح القرار:
    - دليل نمو: intensity/cohesion/peakiness
    - دليل قاطع: عجز القاطع وفق مساحة
    - دليل استهلاك: عجز الاستهلاك وفق مساحة × نمو
    - دليل شذوذ: IsolationForest
    - دليل نباتي عكسي: 1 - intensity
    ثم يصدر: risk_score, label, rationale[]
    """

    def __init__(self, iso_model, scaler):
        self.iso = iso_model
        self.scaler = scaler
        self.w = dict(growth=0.30, breaker=0.28, consumption=0.28, anomaly=0.08, green_inv=0.06)

    @staticmethod
    def breaker_gap(area_m2, breaker_a):
        req = area_m2 * 0.006
        if req <= 0: return 0.0, req
        if breaker_a >= req: return 0.0, req
        return float(np.clip((req - breaker_a)/req,0,1)), req

    @staticmethod
    def consumption_gap(area_m2, intensity, consumption):
        exp = area_m2 * 0.004 * (0.6 + 0.8*float(np.clip(intensity,0,1)))
        if exp <= 0: return 0.0, exp
        if consumption >= exp: return 0.0, exp
        return float(np.clip((exp-consumption)/exp,0,1)), exp

    def decide(self, *, area_m2, breaker, consumption, lon, lat,
               intensity, cohesion, peakiness) -> Dict:
        # 1) نمو
        growth = float(np.clip(0.6*intensity + 0.3*cohesion + 0.1*peakiness, 0, 1))

        # 2) فجوات
        r_breaker, reqA = self.breaker_gap(area_m2, breaker)
        r_cons, expC = self.consumption_gap(area_m2, intensity, consumption)

        # 3) شذوذ
        X = self.scaler.transform(np.array([[breaker, consumption, lon, lat]], dtype=float))
        r_anom = 1.0 if self.iso.predict(X)[0]==1 else 0.0

        # 4) نباتي عكسي
        r_green_inv = float(1.0 - intensity)

        # 5) تجميع
        score = (self.w["growth"]*growth + self.w["breaker"]*r_breaker +
                 self.w["consumption"]*r_cons + self.w["anomaly"]*r_anom +
                 self.w["green_inv"]*r_green_inv)

        # 6) تصعيد فوري (نمو قوي + فجوة كبرى)
        if intensity >= 0.55 and (r_breaker >= 0.5 or r_cons >= 0.5):
            score = max(score, 0.95)

        label = "قصوى" if score >= 0.70 else ("متوسطة" if score >= 0.40 else "منخفضة")

        # 7) تفسير قرار (نص موجز)
        why = []
        if intensity >= 0.65:           why.append("نمو قوي (أخضر داكن متماسك)")
        elif intensity >= 0.50:         why.append("نمو متوسط")
        else:                           why.append("نمو ضعيف/متفرق")

        if r_breaker >= 0.5:            why.append(f"قاطع أقل من المطلوب ({breaker:.0f}A < {reqA:.0f}A)")
        if r_cons >= 0.5:               why.append(f"استهلاك أدنى من المتوقع ({consumption:.0f} < {expC:.0f})")
        if r_anom == 1.0:               why.append("شذوذ بالبيانات (عالي المخاطر)")
        if intensity < 0.35:            why.append("خضرة ضعيفة (دليل مخالف)")

        return dict(
            risk_score=float(np.clip(score,0,1)),
            label=label,
            growth=growth,
            parts=dict(r_breaker=r_breaker, r_consumption=r_cons, r_anomaly=r_anom, r_green_inv=r_green_inv),
            expect=dict(reqA=reqA, expC=expC),
            why=why
        )

# ======================= واجهة =======================
st.set_page_config(page_title=cfg.page_title, page_icon=cfg.page_icon, layout="wide")
ensure_dirs(cfg.images_dir, cfg.detected_dir, cfg.output_dir, cfg.models_dir)
MODEL_PATH=os.path.join(cfg.models_dir,"best.pt")
ISO_PATH=os.path.join(cfg.models_dir,"isolation_model.joblib")
SCALER_PATH=os.path.join(cfg.models_dir,"isolation_scaler.joblib")

st.title(cfg.page_title)
uploaded=st.file_uploader("📁 رفع ملف البيانات (Excel)", type=["xlsx"])
colors={"قصوى":"#ff4d4d","متوسطة":"#ffa500","منخفضة":"#4CAF50"}

if uploaded:
    df=read_excel(uploaded)
    st.sidebar.info(f"🔢 عدد الحالات: {len(df)}")

    # تحكمات
    breaker_filter=st.sidebar.selectbox("سعة القاطع", ["الكل"]+sorted(df["Breaker"].unique().tolist()))
    risk_threshold=st.sidebar.slider("≥ عرض فقط الحالات ذات الخطر", 0, 100, 50, step=5)
    yolo_conf=st.sidebar.slider("YOLO ثقة الكشف", 0.10, 0.80, cfg.default_yolo_conf, 0.05)
    apply_strict=st.sidebar.checkbox("تفعيل فلتر الزراعة الصارم", value=False)
    thr_i=st.sidebar.slider("حد النمو (Intensity)", 0.0,1.0,cfg.pass_intensity,0.05)
    thr_c=st.sidebar.slider("حد التماسك (Cohesion)", 0.0,1.0,cfg.pass_cohesion,0.05)
    thr_p=st.sidebar.slider("حد الذروة (Peakiness)", 0.0,1.0,cfg.pass_peakiness,0.05)

    sort_order=st.sidebar.radio("ترتيب حسب الاستهلاك", ["بدون ترتيب","تصاعدي","تنازلي"])
    if breaker_filter!="الكل": df=df[df["Breaker"]==breaker_filter]
    if sort_order=="تصاعدي": df=df.sort_values(by="consumption", ascending=True)
    elif sort_order=="تنازلي": df=df.sort_values(by="consumption", ascending=False)

    # أزرار مساعدة
    if st.sidebar.button("📥 تنزيل/عرض الصور"):
        cols=st.columns(4); prog=st.sidebar.progress(0); n=len(df); shown=0
        for i,(_,row) in enumerate(df.iterrows(),1):
            meter=str(row["Subscription"]); lat,lon=float(row["y"]),float(row["x"])
            rgb_path,_=download_rgb_and_ndvi(lat,lon,meter)
            if rgb_path:
                with open(rgb_path,"rb") as f: b64=base64.b64encode(f.read()).decode()
                cols[shown%4].markdown(f'<div style="border:1px solid #ddd;border-radius:8px;padding:6px;margin:6px;text-align:center"><img src="data:image/png;base64,{b64}" width="230"><br><small>عداد {meter}<br>Lat {lat:.6f}, Lon {lon:.6f}</small></div>', unsafe_allow_html=True)
                shown+=1
            prog.progress(i/max(n,1))
        st.sidebar.success(f"✅ تم عرض {shown} صورة")
        st.stop()

    # بدء التحليل
    if st.sidebar.button("🚀 بدء التحليل"):
        model_yolo=load_yolo(MODEL_PATH)
        iso, scaler = load_iso(ISO_PATH, SCALER_PATH)
        judge = RuleEngine(iso, scaler)

        cols=st.columns(2); prog=st.sidebar.progress(0); n=len(df); col_i=0
        results=[]; dropped_strict=dropped_area=dropped_edge=dropped_yolo=0

        for i,(_,row) in enumerate(df.iterrows(),1):
            try:
                meter=str(row["Subscription"]); lat,lon=float(row["y"]),float(row["x"])
                br,cons,off=float(row["Breaker"]),float(row["consumption"]),str(row["Office"])

                rgb_path, ndvi_path = download_rgb_and_ndvi(lat,lon,meter)
                if not (rgb_path and ndvi_path): prog.progress(i/n); continue

                det = detect_field(rgb_path, ndvi_path, lat, lon, meter, model_yolo,
                                   yolo_conf, cfg.min_area_m2, cfg.max_edge_distance_m, cfg.detected_dir,
                                   apply_strict, thr_i, thr_c, thr_p)
                if det is None:
                    prog.progress(i/n); continue

                verdict = judge.decide(
                    area_m2=det.area_m2, breaker=br, consumption=cons, lon=lon, lat=lat,
                    intensity=det.intensity, cohesion=det.cohesion, peakiness=det.peakiness
                )

                score=verdict["risk_score"]; pr=verdict["label"]
                if score*100 < risk_threshold: prog.progress(i/n); continue

                with open(det.out_img_path,"rb") as f: img64=base64.b64encode(f.read()).decode()
                why_html = "؛ ".join(verdict["why"]) if verdict["why"] else "—"
                parts = verdict["parts"]; expect = verdict["expect"]

                cols[col_i%2].markdown(f"""
<div style="border:4px solid {colors.get(pr,'#ccc')};padding:12px;border-radius:12px;margin:6px;text-align:center;">
  <img src="data:image/png;base64,{img64}" width="360"><br>
  <strong>عداد {meter} — {pr} ({score*100:.1f}%)</strong><br>
  🌱 نمو:{det.intensity*100:.0f}% | 🧩 تماسك:{det.cohesion*100:.0f}% | 📈 ذروة:{det.peakiness*100:.0f}% | ↔ مسافة:{det.edge_distance_m:.1f}م | مساحة:{det.area_m2}م²<br>
  ⚡ قاطع:{br}A (مطلوب≈{expect['reqA']:.0f}A) | 🔌 استهلاك:{cons:.0f} (متوقع≈{expect['expC']:.0f})<br>
  <small>تفصيل المخاطر: قاطع {parts['r_breaker']*100:.0f}% | استهلاك {parts['r_consumption']*100:.0f}% | شذوذ {parts['r_anomaly']*100:.0f}% | نباتي {parts['r_green_inv']*100:.0f}%</small><br>
  <em>مبررات القرار:</em> {why_html}<br>
  <a href="https://maps.google.com?q={lat},{lon}">📍 الموقع</a>
</div>""", unsafe_allow_html=True)

                results.append([meter, pr, score, det.edge_distance_m, det.area_m2,
                                cons, br, off, lat, lon, det.intensity, det.cohesion, det.peakiness, why_html])
                col_i += 1

            except Exception as e:
                st.warning(f"⚠️ خطأ في العداد {row.get('Subscription','?')}: {e}")
            finally:
                prog.progress(i/n)

        if results:
            res_df=pd.DataFrame(results, columns=[
                "Subscription","priority","risk_score","edge_distance_m","area_m2",
                "consumption","breaker","office","lat","lon",
                "intensity","cohesion","peakiness","why"
            ])
            st.sidebar.download_button("📥 نتائج Excel", data=save_results_excel(res_df), file_name="results.xlsx")

        st.sidebar.success("✅ اكتمل التحليل")
        st.stop()

st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس | 00966553339838 | ")
