# 🔧 نظام اكتشاف حالات الفاقد الكهربائي للفئة الزراعية - النسخة النهائية

import os
os.environ\["YOLO\_VERBOSE"] = "False"
os.environ\["OPENCV\_IO\_ENABLE\_OPENEXR"] = "0"

import math
import io
import base64
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import streamlit as st
import joblib
from ultralytics import YOLO
import urllib.parse

# ------------------------- إعدادات عامة -------------------------

st.set\_page\_config(
page\_title="نظام اكتشاف حالات الفاقد للفئة الزراعية",
layout="wide",
page\_icon="🌾"
)

# ------------------------- المسارات الرئيسية -------------------------

BASE\_DIR = os.path.dirname(os.path.abspath(**file**))
IMG\_DIR = os.path.join(BASE\_DIR, "images")
DETECTED\_DIR = os.path.join(BASE\_DIR, "DETECTED\_FIELDS")
OUTPUT\_FOLDER = os.path.join(BASE\_DIR, "output")
MODEL\_PATH = os.path.join(BASE\_DIR, "models", "best.pt")
ML\_MODEL\_PATH = os.path.join(BASE\_DIR, "models", "isolation\_model.joblib")
SCALER\_PATH = os.path.join(BASE\_DIR, "models", "isolation\_scaler.joblib")
CALIBRATION\_FACTOR = 0.6695

for path in \[IMG\_DIR, DETECTED\_DIR, OUTPUT\_FOLDER]:
os.makedirs(path, exist\_ok=True)

@st.cache\_resource
def load\_models():
model\_yolo = YOLO(MODEL\_PATH)
model\_ml = joblib.load(ML\_MODEL\_PATH)
scaler = joblib.load(SCALER\_PATH)
return model\_yolo, model\_ml, scaler

def download\_image(lat, lon, meter\_id):
img\_path = os.path.join(IMG\_DIR, f"{meter\_id}.png")
if os.path.exists(img\_path):
return img\_path
url = "[https://maps.googleapis.com/maps/api/staticmap](https://maps.googleapis.com/maps/api/staticmap)"
params = {
"center": f"{lat},{lon}",
"zoom": 16,
"size": "640x640",
"maptype": "satellite",
"markers": f"color\:red|label\:X|{lat},{lon}",
"key": "YOUR\_API\_KEY"
}
try:
r = requests.get(url, params=params, timeout=15)
if r.status\_code == 200:
with open(img\_path, "wb") as f:
f.write(r.content)
return img\_path
except:
pass
return None

def detect\_field(img\_path, lat, meter\_id, model\_yolo):
image = Image.open(img\_path).convert("RGB")
results = model\_yolo.predict(source=image, imgsz=640, conf=0.5)\[0]
if not results.boxes:
return None, None, None
box = results.boxes\[0].xyxy\[0].cpu().numpy()
conf = float(results.boxes\[0].conf.cpu().numpy())
if conf < 0.5:
return None, None, None
scale = 156543.03392 \* math.cos(math.radians(lat)) / (2 \*\* 16)
area = abs(box\[2] - box\[0]) \* abs(box\[3] - box\[1]) \* (scale \*\* 2)
corrected\_area = area \* CALIBRATION\_FACTOR
if corrected\_area < 5000:
return None, None, None
draw = ImageDraw\.Draw(image)
draw\.rectangle(box.tolist(), outline="green", width=3)
out\_path = os.path.join(DETECTED\_DIR, f"{meter\_id}.png")
image.save(out\_path)
return round(conf \* 100, 2), out\_path, int(corrected\_area)

# المعادلات النهائية المعتمدة

def compute\_final\_confidence(area, breaker, consumption, yolo\_confidence):
expected\_consumption = area \* 1
expected\_breaker = area / 500

```
consumption_ratio = consumption / expected_consumption
breaker_ratio = breaker / expected_breaker

consumption_risk = max(0, (1 - consumption_ratio)) * 100
breaker_risk = max(0, (1 - breaker_ratio)) * 100

case_risk = (consumption_risk * 0.7) + (breaker_risk * 0.3)

final_confidence = (case_risk * 0.6) + (100 - case_risk) * (yolo_confidence / 100) * 0.4

if final_confidence >= 80:
    priority = "🔴 قصوى"
    color = "crimson"
elif final_confidence >= 60:
    priority = "🟠 عالية"
    color = "orange"
elif final_confidence >= 40:
    priority = "🟡 تنبيه"
    color = "gold"
else:
    priority = "🟢 طبيعي"
    color = "green"

return round(final_confidence, 2), priority, color
```

# =============================

# واجهة المستخدم

# =============================

st.title("🌾 نظام اكتشاف حالات الفاقد الكهربائي للفئة الزراعية")
uploaded\_file = st.file\_uploader("📁 رفع ملف البيانات (Excel)", type=\["xlsx"])

if uploaded\_file:
df = pd.read\_excel(uploaded\_file)
df.columns = df.columns.str.lower().str.strip()

```
model_yolo, model_ml, scaler = load_models()

for idx, row in df.iterrows():
    meter_id, lat, lon = row["subscription"], row["y"], row["x"]
    breaker, consumption, office = row["breaker"], row["consumption"], row["office"]
    img_path = download_image(lat, lon, meter_id)
    if not img_path:
        continue
    yolo_conf, img_detected, area = detect_field(img_path, lat, meter_id, model_yolo)
    if yolo_conf is None or area is None:
        continue

    final_confidence, priority, color = compute_final_confidence(area, breaker, consumption, yolo_conf)

    with open(img_detected, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    st.markdown(f'''
    <div style="border:2px solid {color};padding:15px;border-radius:10px;margin-bottom:20px;">
        <img src="data:image/png;base64,{img_data}" width="300" />
        <p>🔢 العداد: {meter_id}</p>
        <p>📐 المساحة: {area:,} م²</p>
        <p>💡 الاستهلاك: {consumption:,} ك.و.س</p>
        <p>⚡ القاطع: {breaker} أمبير</p>
        <p>📊 نسبة الثقة: {final_confidence}%</p>
        <p>🚨 الأولوية: {priority}</p>
    </div>
    ''', unsafe_allow_html=True)
```
