import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import os
import gdown

st.set_page_config(page_title="Car Light ML", page_icon="🔦", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
.subtitle { font-size: 1.05rem; color: #666; margin-top: 0; margin-bottom: 2rem; }
.prediction-card { background: #f8f8f6; border-radius: 12px; padding: 1.5rem; border: 1px solid #e5e5e3; margin: 1rem 0; }
.pred-label { font-size: 0.75rem; color: #999; text-transform: uppercase; letter-spacing: 0.05em; }
.pred-car { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }
.pred-confidence { font-size: 2rem; font-weight: 700; color: #E08A1E; }
.conf-label { font-size: 0.85rem; color: #999; margin-left: 0.25rem; }
.car-pill { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; background: #f0ede6; color: #8a7a5a; margin: 3px; }
.bar-row { display: flex; align-items: center; margin: 4px 0; font-size: 0.82rem; }
.bar-name { width: 220px; text-align: right; padding-right: 10px; color: #555; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-track { flex: 1; height: 10px; background: #eee; border-radius: 5px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 5px; }
.bar-fill-top { background: #E08A1E; }
.bar-fill-other { background: #ccc; }
.bar-pct { width: 50px; text-align: right; padding-left: 8px; color: #555; font-variant-numeric: tabular-nums; }
.footer { text-align: center; font-size: 0.8rem; color: #bbb; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "best_car_light_model.pth"
GDRIVE_FILE_ID = "1HyXCkf9toZP-qwTsAiT61-sS6j3gqc-r"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first time only)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH):
        st.error("Failed to download model. Check the Google Drive link is set to 'Anyone with the link'.")
        st.stop()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
    num_classes = state_dict['fc.weight'].shape[0]
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model, num_classes

@st.cache_data
def get_class_names():
    if os.path.exists("class_names.txt"):
        with open("class_names.txt", "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return None

def format_car_name(folder_name):
    parts = folder_name.split('_')
    formatted = []
    for part in parts:
        if part.upper() in ['BMW','VW','GMC','AMG','GT','SV','SI','SE','XLE','AWD','FWD','RWD','4WD',
                            'CRV','HRV','MDX','GLA','GLE','GLC','EQE','RAV4','IX','M3','X3','X4',
                            'X5','X7','Z4','K5','Q8','QX50','QX60','QX80','XC90','XT4','GV80',
                            'ES','IS','UX','RX']:
            formatted.append(part.upper())
        else:
            formatted.append(part.capitalize())
    return ' '.join(formatted)

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(model, image, class_names):
    input_tensor = eval_transforms(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    probs, indices = torch.sort(probabilities, descending=True)
    results = []
    for i in range(len(class_names)):
        idx = indices[i].item()
        results.append({'name': class_names[idx], 'display_name': format_car_name(class_names[idx]), 'prob': probs[i].item() * 100})
    return results

def main():
    st.markdown('<div class="main-title">🔦 Car Light ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a close-up headlight photo and the model will identify the car</div>', unsafe_allow_html=True)
    try:
        model, num_classes = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    class_names = get_class_names()
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
        st.warning("class_names.txt not found. Add it to the repo for proper car names.")
    uploaded = st.file_uploader("Drag a headlight photo here or click to browse", type=["jpg","jpeg","png"], help="Works best with close-up headlight photos taken from ~2-3 feet away")
    if uploaded is not None:
        image = Image.open(uploaded).convert('RGB')
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.image(image, use_container_width=True)
        results = predict(model, image, class_names)
        top = results[0]
        with col2:
            st.markdown(f'<div class="prediction-card"><div class="pred-label">Prediction</div><div class="pred-car">{top["display_name"]}</div><div><span class="pred-confidence">{top["prob"]:.1f}%</span><span class="conf-label">confidence</span></div></div>', unsafe_allow_html=True)
        st.markdown("**All predictions** (top 10)")
        bars_html = ""
        for i, r in enumerate(results[:10]):
            fill_class = "bar-fill-top" if i == 0 else "bar-fill-other"
            width = max(r['prob'], 0.5)
            bars_html += f'<div class="bar-row"><div class="bar-name">{r["display_name"]}</div><div class="bar-track"><div class="bar-fill {fill_class}" style="width: {width}%"></div></div><div class="bar-pct">{r["prob"]:.1f}%</div></div>'
        st.markdown(bars_html, unsafe_allow_html=True)
        with st.expander(f"See all {len(class_names)} classes"):
            for r in results:
                if r['prob'] >= 0.1:
                    st.text(f"  {r['display_name']:45s} {r['prob']:.1f}%")
    else:
        st.markdown("---")
        st.markdown(f"**Currently recognizes {num_classes} cars**")
        pills = ""
        for name in sorted(class_names):
            pills += f'<span class="car-pill">{format_car_name(name)}</span>'
        st.markdown(pills, unsafe_allow_html=True)
    st.markdown('<div class="footer">Built by Rus · ResNet18 transfer learning · 2026</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
