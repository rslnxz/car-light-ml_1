import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import os
import gdown

st.set_page_config(page_title="LightID — Car Recognition from Headlights", page_icon="💡", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero section */
.hero {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
}
.logo-container {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 72px;
    height: 72px;
    border-radius: 18px;
    background: linear-gradient(135deg, #E08A1E 0%, #F5B041 50%, #E08A1E 100%);
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(224, 138, 30, 0.3);
}
.logo-icon {
    font-size: 36px;
    filter: brightness(10);
}
.app-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #1a1a1a;
    margin: 0;
    letter-spacing: -0.02em;
}
.app-tagline {
    font-size: 1.1rem;
    color: #888;
    margin: 0.25rem 0 0;
}
.app-description {
    max-width: 520px;
    margin: 1.2rem auto 0;
    font-size: 0.92rem;
    color: #666;
    line-height: 1.6;
}
.divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #E08A1E, #F5B041);
    border-radius: 2px;
    margin: 1.5rem auto;
}

/* Upload area */
.upload-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #333;
    margin-bottom: 0.5rem;
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(135deg, #faf8f5 0%, #f5f0e8 100%);
    border-radius: 16px;
    padding: 1.8rem;
    border: 1px solid #e8e0d0;
    margin: 1rem 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.pred-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    color: #b0a080;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}
.pred-car {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #1a1a1a;
    margin: 0.3rem 0 0.6rem;
}
.pred-confidence {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #E08A1E, #D4740E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.conf-label {
    font-size: 0.85rem;
    color: #b0a080;
    margin-left: 0.3rem;
}

/* Bars */
.bars-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #444;
    margin: 1.5rem 0 0.8rem;
}
.bar-row {
    display: flex;
    align-items: center;
    margin: 6px 0;
    font-size: 0.82rem;
}
.bar-name {
    width: 200px;
    text-align: right;
    padding-right: 12px;
    color: #666;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 0.8rem;
}
.bar-track {
    flex: 1;
    height: 12px;
    background: #f0ece4;
    border-radius: 6px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.5s ease;
}
.bar-fill-top {
    background: linear-gradient(90deg, #E08A1E, #F5B041);
}
.bar-fill-other {
    background: #d5cfc5;
}
.bar-pct {
    width: 50px;
    text-align: right;
    padding-left: 10px;
    color: #888;
    font-variant-numeric: tabular-nums;
    font-size: 0.8rem;
}

/* How it works */
.how-section {
    background: #faf8f5;
    border-radius: 16px;
    padding: 1.8rem;
    border: 1px solid #e8e0d0;
    margin: 2rem 0 1rem;
}
.how-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #333;
    margin: 0 0 1rem;
}
.how-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin: 0.8rem 0;
}
.step-num {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, #E08A1E, #F5B041);
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.step-text {
    font-size: 0.9rem;
    color: #555;
    line-height: 1.5;
    padding-top: 3px;
}

/* Stats */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin: 1.5rem 0;
}
.stat {
    text-align: center;
}
.stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #E08A1E;
}
.stat-label {
    font-size: 0.75rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.78rem;
    color: #bbb;
    margin-top: 3rem;
    padding-bottom: 1rem;
}
.footer a {
    color: #E08A1E;
    text-decoration: none;
}
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
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="logo-container"><span class="logo-icon">💡</span></div>
        <h1 class="app-name">LightID</h1>
        <p class="app-tagline">Identify any car from its headlights</p>
        <p class="app-description">
            LightID uses a deep learning model trained on 1,500+ real headlight photographs
            to recognize car make, model, and year from a single close-up photo.
            Built with PyTorch and ResNet18 transfer learning.
        </p>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    try:
        model, num_classes = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    class_names = get_class_names()
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
        st.warning("class_names.txt not found. Add it to the repo for proper car names.")

    # Upload
    st.markdown('<div class="upload-label">Upload a headlight photo</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "upload",
        type=["jpg","jpeg","png"],
        help="Works best with close-up headlight photos taken from ~2-3 feet away",
        label_visibility="collapsed"
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert('RGB')
        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.image(image, use_container_width=True)
        results = predict(model, image, class_names)
        top = results[0]
        with col2:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="pred-label">Identified as</div>
                <div class="pred-car">{top['display_name']}</div>
                <div>
                    <span class="pred-confidence">{top['prob']:.1f}%</span>
                    <span class="conf-label">confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="bars-header">Top predictions</div>', unsafe_allow_html=True)
        bars_html = ""
        for i, r in enumerate(results[:8]):
            fill_class = "bar-fill-top" if i == 0 else "bar-fill-other"
            width = max(r['prob'], 0.5)
            bars_html += f'<div class="bar-row"><div class="bar-name">{r["display_name"]}</div><div class="bar-track"><div class="bar-fill {fill_class}" style="width: {width}%"></div></div><div class="bar-pct">{r["prob"]:.1f}%</div></div>'
        st.markdown(bars_html, unsafe_allow_html=True)

        with st.expander(f"All {len(class_names)} classes"):
            for r in results:
                if r['prob'] >= 0.1:
                    st.text(f"  {r['display_name']:45s} {r['prob']:.1f}%")

    else:
        # Stats
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat">
                <div class="stat-num">{num_classes}</div>
                <div class="stat-label">Car Models</div>
            </div>
            <div class="stat">
                <div class="stat-num">1,540</div>
                <div class="stat-label">Training Photos</div>
            </div>
            <div class="stat">
                <div class="stat-num">94.8%</div>
                <div class="stat-label">Accuracy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # How it works
        st.markdown("""
        <div class="how-section">
            <div class="how-title">How it works</div>
            <div class="how-step">
                <div class="step-num">1</div>
                <div class="step-text">Take a close-up photo of a car's headlight from about 2-3 feet away. Make sure the headlight fills most of the frame.</div>
            </div>
            <div class="how-step">
                <div class="step-num">2</div>
                <div class="step-text">Upload the photo above. The model processes it in seconds using a ResNet18 neural network trained on real dealership and parking lot headlight photos.</div>
            </div>
            <div class="how-step">
                <div class="step-num">3</div>
                <div class="step-text">Get the prediction with confidence scores for each possible car model, showing how certain the AI is about its answer.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        Built by Rus Surmin · ResNet18 Transfer Learning · PyTorch · 2026<br>
        <a href="https://github.com/rslnxz" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
