import streamlit as st
import torch
import torch.nn as nn
import timm
import pickle
import json
import numpy as np
import io
import time
from pathlib import Path
from PIL import Image
from torchvision import transforms

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="FreshVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Model paths ──────────────────────────────────────────
MODEL_DIR  = Path("./model_weights")
MODEL_PATH = MODEL_DIR / "freshvision_model.pth"
OOD_PATH   = MODEL_DIR / "ood_detector.pkl"
INFO_PATH  = MODEL_DIR / "classes.json"

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# ── Load classes.json first (needed to build model) ──────
def load_class_info():
    if not INFO_PATH.exists():
        return None, None, None
    with open(INFO_PATH, "r") as f:
        data = json.load(f)
    classes      = data["classes"]          # list of 16 class keys
    display      = data["display"]          # dict of display info
    num_classes  = data.get("num_classes", len(classes))
    return classes, display, num_classes

CLASSES, CLASS_DISPLAY, NUM_CLASSES = load_class_info() or ([], {}, 16)

# Only show these 3 fruits in the UI chips (model still detects all 8)
UI_DISPLAY_ITEMS = ["Apple", "Banana", "Orange"]

def get_unique_items():
    seen = set()
    items = []
    for cls in (CLASSES or []):
        info = (CLASS_DISPLAY or {}).get(cls, {})
        name  = info.get("name", cls)
        emoji = info.get("emoji", "")
        if name not in seen and name in UI_DISPLAY_ITEMS:
            seen.add(name)
            items.append(f"{emoji} {name}")
    return items

TRAINED_ITEMS = get_unique_items()

# ── Model class (must match training exactly) ─────────────
class FreshVisionModel(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.backbone   = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        feat_dim        = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def get_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.classifier(self.backbone(x))

# ── Load model (cached) ───────────────────────────────────
@st.cache_resource
def load_everything():
    errors = []

    # Must have classes.json to know num_classes
    if not INFO_PATH.exists():
        errors.append(f"❌ classes.json not found: {INFO_PATH}")
        return None, None, errors

    classes, display, num_classes = load_class_info()

    if not MODEL_PATH.exists():
        errors.append(f"❌ Model file not found: {MODEL_PATH}")
        return None, None, errors

    # Build model with correct output size
    model = FreshVisionModel(num_classes=num_classes).to(DEVICE)
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
    except Exception as e:
        return None, None, [f"❌ Model load error: {e}"]

    # Load OOD detector
    ood = None
    if OOD_PATH.exists():
        with open(OOD_PATH, "rb") as f:
            ood = pickle.load(f)
    else:
        errors.append("⚠ ood_detector.pkl not found — OOD rejection disabled")

    return model, ood, errors

# ── Preprocess ────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def mahal_distance(feat_np, ood):
    delta = feat_np - ood["mean"]
    left  = delta @ ood["precision"]
    return float(np.sqrt(max(np.dot(left, delta), 0)))

def predict(image: Image.Image, model, ood):
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model.get_features(tensor)
        feat_np  = features.cpu().numpy()[0]

        # OOD check
        if ood is not None:
            dist      = mahal_distance(feat_np, ood)
            threshold = ood["threshold"]
            if dist > threshold:
                return {
                    "rejected":  True,
                    "distance":  round(dist, 2),
                    "threshold": round(threshold, 2),
                }

        # Classify
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        conf   = float(probs.max().item()) * 100
        idx    = int(probs.argmax().item())
        cls    = CLASSES[idx]
        disp   = CLASS_DISPLAY[cls]

        top3_idx = probs.topk(min(3, len(CLASSES))).indices.cpu().numpy()
        top3 = [{
            "name":       CLASS_DISPLAY[CLASSES[i]]["name"],
            "status":     CLASS_DISPLAY[CLASSES[i]]["status"],
            "emoji":      CLASS_DISPLAY[CLASSES[i]]["emoji"],
            "confidence": round(float(probs[i].item()) * 100, 1),
            "color":      CLASS_DISPLAY[CLASSES[i]]["color"],
        } for i in top3_idx]

    return {
        "rejected":   False,
        "class":      cls,
        "name":       disp["name"],
        "status":     disp["status"],
        "emoji":      disp["emoji"],
        "color":      disp["color"],
        "confidence": round(conf, 1),
        "is_fresh":   disp["status"] == "Fresh",
        "top3":       top3,
    }

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&family=Space+Mono&display=swap');

html,body,.stApp{background:#030d03!important;color:#f0fdf4!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem!important;max-width:860px!important;margin:0 auto!important;}

.stApp::before{
    content:'';position:fixed;inset:0;
    background:
        radial-gradient(ellipse at 15% 25%,rgba(34,197,94,.13) 0%,transparent 55%),
        radial-gradient(ellipse at 85% 75%,rgba(74,222,128,.08) 0%,transparent 55%),
        radial-gradient(ellipse at 50% 50%,rgba(16,185,129,.05) 0%,transparent 70%);
    animation:bgPulse 7s ease-in-out infinite;pointer-events:none;z-index:0;
}
@keyframes bgPulse{0%,100%{opacity:1}50%{opacity:.5}}

.particle{
    position:fixed;border-radius:50%;pointer-events:none;
    animation:floatUp linear infinite;opacity:0;z-index:0;
}
@keyframes floatUp{
    0%{transform:translateY(100vh) scale(0);opacity:0}
    10%{opacity:.5}90%{opacity:.2}
    100%{transform:translateY(-120px) scale(1);opacity:0}
}

.hero{text-align:center;padding:36px 0 24px;position:relative;z-index:10;}
.hero-title{
    font-family:'Syne',sans-serif;font-size:clamp(40px,8vw,72px);font-weight:800;
    background:linear-gradient(135deg,#22c55e 0%,#86efac 50%,#4ade80 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    margin:0;line-height:1.1;
    filter:drop-shadow(0 0 40px rgba(34,197,94,.5));
    animation:titleGlow 3s ease-in-out infinite;
}
@keyframes titleGlow{
    0%,100%{filter:drop-shadow(0 0 30px rgba(34,197,94,.4))}
    50%{filter:drop-shadow(0 0 70px rgba(34,197,94,.9))}
}
.hero-sub{
    font-family:'Space Mono',monospace;font-size:11px;
    color:rgba(134,239,172,.6);letter-spacing:.35em;margin-top:10px;
}

.ring-wrap{
    width:100px;height:100px;margin:16px auto;
    position:relative;display:flex;align-items:center;justify-content:center;
}
.ring{
    position:absolute;border-radius:50%;
    border:2px solid rgba(34,197,94,.35);
    animation:spin 5s linear infinite;
}
.ring2{
    position:absolute;border-radius:50%;
    border:1px solid rgba(74,222,128,.2);
    animation:spin 8s linear infinite reverse;
}
@keyframes spin{to{transform:rotate(360deg)}}
.ring-emoji{font-size:38px;position:relative;z-index:2;}

.glass{
    background:rgba(8,20,8,.82);
    backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
    border:1px solid rgba(34,197,94,.2);border-radius:20px;padding:24px;
    margin-bottom:16px;position:relative;z-index:10;
    box-shadow:0 8px 32px rgba(0,0,0,.5),inset 0 1px 0 rgba(34,197,94,.08);
}
.sec-label{
    font-family:'Space Mono',monospace;font-size:10px;
    color:rgba(134,239,172,.55);letter-spacing:.25em;margin-bottom:14px;
}

.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;}
.chip{
    display:inline-flex;align-items:center;gap:6px;
    background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.25);
    border-radius:100px;padding:6px 14px;
    font-family:'Space Mono',monospace;font-size:11px;color:rgba(134,239,172,.9);
}

.warn{
    background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.25);
    border-radius:10px;padding:10px 16px;
    font-family:'Space Mono',monospace;font-size:10px;
    color:rgba(239,68,68,.8);letter-spacing:.05em;
}

.stFileUploader>div{
    background:rgba(8,20,8,.6)!important;
    border:2px dashed rgba(34,197,94,.3)!important;border-radius:16px!important;
}
.stFileUploader>div:hover{
    border-color:rgba(34,197,94,.6)!important;
    background:rgba(34,197,94,.04)!important;
}

.stCameraInput>div{border-radius:16px!important;border:2px solid rgba(34,197,94,.3)!important;overflow:hidden!important;}

.stButton>button{
    width:100%;background:linear-gradient(135deg,#15803d,#22c55e)!important;
    color:#fff!important;font-family:'Syne',sans-serif!important;
    font-weight:800!important;font-size:15px!important;letter-spacing:.15em!important;
    border:none!important;border-radius:12px!important;padding:18px!important;
    box-shadow:0 0 28px rgba(34,197,94,.35)!important;transition:all .2s!important;
}
.stButton>button:hover{box-shadow:0 0 50px rgba(34,197,94,.65)!important;transform:translateY(-2px)!important;}

.scanbar{
    height:3px;margin:10px 0;border-radius:2px;
    background:linear-gradient(90deg,transparent,#22c55e,transparent);
    box-shadow:0 0 14px #22c55e;
    animation:scanPulse 1.2s ease-in-out infinite;
}
@keyframes scanPulse{0%,100%{opacity:.4;transform:scaleX(.6)}50%{opacity:1;transform:scaleX(1)}}

.res-fresh{
    background:linear-gradient(135deg,rgba(34,197,94,.15),rgba(74,222,128,.04));
    border:2px solid rgba(34,197,94,.6);border-radius:22px;padding:36px 28px;
    text-align:center;animation:glowG 2.5s ease-in-out infinite;
    position:relative;z-index:10;margin-bottom:14px;
}
@keyframes glowG{0%,100%{box-shadow:0 0 30px rgba(34,197,94,.2)}50%{box-shadow:0 0 65px rgba(34,197,94,.55)}}

.res-rotten{
    background:linear-gradient(135deg,rgba(239,68,68,.15),rgba(220,38,38,.04));
    border:2px solid rgba(239,68,68,.6);border-radius:22px;padding:36px 28px;
    text-align:center;animation:glowR 2.5s ease-in-out infinite;
    position:relative;z-index:10;margin-bottom:14px;
}
@keyframes glowR{0%,100%{box-shadow:0 0 30px rgba(239,68,68,.2)}50%{box-shadow:0 0 65px rgba(239,68,68,.55)}}

.res-reject{
    background:linear-gradient(135deg,rgba(234,179,8,.12),rgba(202,138,4,.04));
    border:2px solid rgba(234,179,8,.5);border-radius:22px;padding:36px 28px;
    text-align:center;position:relative;z-index:10;margin-bottom:14px;
}

.res-emoji{font-size:82px;display:block;line-height:1;margin-bottom:14px;}
.res-name{font-family:'Syne',sans-serif;font-size:40px;font-weight:800;color:#f0fdf4;margin:0 0 8px;}
.res-st-f{font-family:'Syne',sans-serif;font-size:26px;font-weight:700;color:#22c55e;filter:drop-shadow(0 0 12px #22c55e);display:block;margin-bottom:14px;}
.res-st-r{font-family:'Syne',sans-serif;font-size:26px;font-weight:700;color:#ef4444;filter:drop-shadow(0 0 12px #ef4444);display:block;margin-bottom:14px;}
.conf-lbl{font-family:'Space Mono',monospace;font-size:10px;color:rgba(255,255,255,.4);letter-spacing:.2em;margin-bottom:6px;}
.conf-track{background:rgba(255,255,255,.08);border-radius:6px;height:8px;overflow:hidden;margin-bottom:18px;}
.conf-g{height:100%;border-radius:6px;background:linear-gradient(90deg,#15803d,#22c55e);box-shadow:0 0 8px #22c55e;}
.conf-r{height:100%;border-radius:6px;background:linear-gradient(90deg,#991b1b,#ef4444);box-shadow:0 0 8px #ef4444;}
.res-msg{font-family:'DM Sans',sans-serif;font-size:15px;color:rgba(240,253,244,.7);}

.t3-lbl{font-family:'Space Mono',monospace;font-size:10px;color:rgba(134,239,172,.5);letter-spacing:.2em;margin:14px 0 8px;position:relative;z-index:10;}
.t3-row{
    display:flex;justify-content:space-between;align-items:center;
    background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);
    border-radius:10px;padding:10px 14px;margin-bottom:6px;
    font-family:'DM Sans',sans-serif;font-size:13px;color:#d1fae5;
    position:relative;z-index:10;
}

.err-box{
    background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.4);
    border-radius:14px;padding:24px;text-align:center;margin-top:12px;
    position:relative;z-index:10;
}

.stCheckbox{color:rgba(134,239,172,.8)!important;}
p,span,div,h1,h2,h3,label{color:#f0fdf4;}
.stMarkdown p{color:#f0fdf4!important;}
[data-testid="stDeprecationWarning"]{display:none!important;}
</style>

<div class="particle" style="width:6px;height:6px;left:10vw;background:#22c55e;animation-duration:12s;animation-delay:0s;"></div>
<div class="particle" style="width:4px;height:4px;left:25vw;background:#4ade80;animation-duration:9s;animation-delay:2s;"></div>
<div class="particle" style="width:8px;height:8px;left:45vw;background:#f97316;animation-duration:15s;animation-delay:1s;"></div>
<div class="particle" style="width:5px;height:5px;left:65vw;background:#fbbf24;animation-duration:11s;animation-delay:4s;"></div>
<div class="particle" style="width:7px;height:7px;left:80vw;background:#22c55e;animation-duration:13s;animation-delay:0.5s;"></div>
<div class="particle" style="width:4px;height:4px;left:90vw;background:#86efac;animation-duration:10s;animation-delay:3s;"></div>
<div class="particle" style="width:6px;height:6px;left:35vw;background:#4ade80;animation-duration:14s;animation-delay:6s;"></div>
<div class="particle" style="width:5px;height:5px;left:55vw;background:#22c55e;animation-duration:8s;animation-delay:2.5s;"></div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────
model, ood, load_errors = load_everything()

# ── HERO ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="ring-wrap">
        <div class="ring"  style="width:90px;height:90px;"></div>
        <div class="ring2" style="width:70px;height:70px;"></div>
        <span class="ring-emoji">🌿</span>
    </div>
    <h1 class="hero-title">FreshVision AI</h1>
    <p class="hero-sub">INTELLIGENT FRESHNESS DETECTION · DEEP LEARNING</p>
</div>
""", unsafe_allow_html=True)

# ── Model status ──────────────────────────────────────────
if model is None:
    st.markdown("""
    <div class="err-box">
        <p style="font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:#ef4444;">
            ⚠ Model Not Found
        </p>
        <p style="font-family:Space Mono,monospace;font-size:11px;color:rgba(239,68,68,.8);">
            Run the Training Notebook on Google Colab first.<br>
            Then place these files in <b>model_weights/</b> folder:<br><br>
            freshvision_model.pth &nbsp;·&nbsp; ood_detector.pkl &nbsp;·&nbsp; classes.json
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Show minor warnings
for w in load_errors:
    if "⚠" in w:
        st.markdown(f'<div style="background:rgba(234,179,8,.1);border:1px solid rgba(234,179,8,.3);border-radius:8px;padding:10px 14px;font-family:Space Mono,monospace;font-size:10px;color:rgba(234,179,8,.8);margin-bottom:12px;position:relative;z-index:10;">{w}</div>', unsafe_allow_html=True)

# ── Trained items ─────────────────────────────────────────
chips = "".join(f'<span class="chip">{item}</span>' for item in TRAINED_ITEMS)
st.markdown(f"""
<div class="glass">
    <div class="sec-label">✦ DETECTS THESE 3 FRUITS</div>
    <div class="chips">{chips}</div>
    <div class="warn">
        ⊗ &nbsp;Anything else → REJECTED — no wrong prediction
    </div>
</div>
""", unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="sec-label">📸 UPLOAD IMAGE OR USE CAMERA</div>', unsafe_allow_html=True)

use_cam = st.checkbox("📷 Use webcam instead")
img_src = None

if use_cam:
    cam = st.camera_input(" ", label_visibility="collapsed")
    if cam: img_src = cam
else:
    upl = st.file_uploader(" ", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    if upl: img_src = upl

if img_src:
    image = Image.open(img_src).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("🔍  ANALYZE FRESHNESS"):
        st.markdown('<div class="scanbar"></div>', unsafe_allow_html=True)

        prog = st.progress(0)
        ph   = st.empty()
        steps = [(20,"Preprocessing image..."),(45,"Extracting deep features..."),
                 (70,"Running OOD check..."),(90,"Classifying freshness..."),(100,"Done!")]

        for pct, msg in steps:
            ph.markdown(f'<p style="font-family:Space Mono,monospace;font-size:11px;color:rgba(134,239,172,.7);text-align:center;">{msg}</p>', unsafe_allow_html=True)
            prog.progress(pct)
            time.sleep(0.32)

        ph.empty()
        prog.empty()

        result = predict(image, model, ood)

        # ── REJECTED ──────────────────────────────────────
        if result["rejected"]:
            trained_list = " &nbsp;·&nbsp; ".join([i.split(" ", 1)[1] for i in TRAINED_ITEMS])
            st.markdown(f"""
            <div class="res-reject">
                <span class="res-emoji">⚠️</span>
                <p class="res-name" style="color:#fbbf24;">UNKNOWN ITEM</p>
                <span style="font-family:Space Mono,monospace;font-size:13px;
                    color:rgba(251,191,36,.8);display:block;margin-bottom:16px;">
                    This item is NOT in my training data.<br>
                    I will NOT make a prediction for it.
                </span>
                <div style="font-family:Space Mono,monospace;font-size:10px;
                    color:rgba(255,255,255,.4);line-height:2;">
                    I ONLY IDENTIFY:<br>
                    {trained_list}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── PREDICTION ────────────────────────────────────
        else:
            is_fresh = result["is_fresh"]
            conf     = result["confidence"]
            name     = result["name"]
            status   = result["status"]
            emoji    = result["emoji"]
            color    = result["color"]
            top3     = result["top3"]

            card  = "res-fresh"  if is_fresh else "res-rotten"
            scls  = "res-st-f"   if is_fresh else "res-st-r"
            stext = "✅ FRESH"    if is_fresh else "❌ ROTTEN"
            ccls  = "conf-g"     if is_fresh else "conf-r"
            msg   = (f"This {name} looks {status.lower()}. "
                     f"Safe to {'eat' if is_fresh else 'discard'}!")

            st.markdown(f"""
            <div class="{card}">
                <span class="res-emoji">{emoji}</span>
                <p class="res-name">{name}</p>
                <span class="{scls}">{stext}</span>
                <div class="conf-lbl">CONFIDENCE: {conf:.1f}%</div>
                <div class="conf-track">
                    <div class="{ccls}" style="width:{conf}%"></div>
                </div>
                <p class="res-msg">{msg}</p>
            </div>
            """, unsafe_allow_html=True)

            if top3:
                st.markdown('<div class="t3-lbl">TOP PREDICTIONS</div>', unsafe_allow_html=True)
                for item in top3:
                    st.markdown(f"""
                    <div class="t3-row">
                        <span>{item['emoji']} {item['name']} — {item['status']}</span>
                        <span style="color:{item['color']};font-family:Space Mono,monospace;font-size:13px;font-weight:700;">
                            {item['confidence']}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
