import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import os
import json
import gdown

st.set_page_config(page_title="LightID — Car Recognition from Headlights", page_icon="💡", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero { text-align: center; padding: 2.5rem 1rem 1rem; }
.logo-container {
    display: inline-flex; align-items: center; justify-content: center;
    width: 80px; height: 80px; border-radius: 20px;
    background: linear-gradient(135deg, #E08A1E 0%, #F5B041 50%, #E08A1E 100%);
    margin-bottom: 1.2rem; box-shadow: 0 4px 24px rgba(224, 138, 30, 0.35);
}
.logo-icon { font-size: 40px; filter: brightness(10); }
.app-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem; font-weight: 700; margin: 0; letter-spacing: -0.02em;
}
.app-tagline { font-size: 1.25rem; color: #777; margin: 0.4rem 0 0; }
.app-description {
    max-width: 560px; margin: 1.2rem auto 0;
    font-size: 1rem; color: #666; line-height: 1.7; text-align: center;
}
.divider {
    width: 60px; height: 3px;
    background: linear-gradient(90deg, #E08A1E, #F5B041);
    border-radius: 2px; margin: 2rem auto;
}

.how-section {
    background: #faf8f5; border-radius: 16px; padding: 2rem;
    border: 1px solid #e8e0d0; margin: 0 0 2rem;
}
.how-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem; font-weight: 600; color: #333;
    margin: 0 0 1.2rem; text-align: center;
}
.how-steps { display: flex; gap: 1.5rem; justify-content: center; flex-wrap: wrap; }
.how-card { flex: 1; min-width: 160px; max-width: 220px; text-align: center; padding: 0.5rem; }
.step-num {
    width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, #E08A1E, #F5B041);
    color: white; font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem; font-weight: 700;
    display: inline-flex; align-items: center; justify-content: center;
    margin-bottom: 0.6rem;
}
.step-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem; font-weight: 600; color: #333; margin-bottom: 0.3rem;
}
.step-text { font-size: 0.82rem; color: #777; line-height: 1.5; }

.stats-row {
    display: flex; justify-content: center; gap: 3rem; margin: 0 0 2rem;
    padding: 1.5rem; background: linear-gradient(135deg, #faf8f5, #f5f0e8);
    border-radius: 16px; border: 1px solid #e8e0d0;
}
.stat { text-align: center; }
.stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; color: #E08A1E;
}
.stat-label {
    font-size: 0.72rem; color: #999;
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.2rem;
}

.brands-section { text-align: center; margin: 0 0 2rem; }
.brands-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem; font-weight: 600; color: #999;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem;
}
.brand-grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
.brand-tag {
    padding: 6px 14px; border-radius: 8px; font-size: 0.82rem;
    font-weight: 500; background: #f5f2ec; color: #8a7a5a; border: 1px solid #e8e0d0;
}

.upload-section { text-align: center; margin: 0 0 1rem; }
.upload-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem; font-weight: 600; color: #333; margin-bottom: 0.5rem;
}

.prediction-card {
    background: linear-gradient(135deg, #faf8f5 0%, #f5f0e8 100%);
    border-radius: 16px; padding: 1.8rem;
    border: 1px solid #e8e0d0; margin: 1rem 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.pred-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem; color: #b0a080;
    text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
}
.pred-car {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem; font-weight: 700; color: #1a1a1a; margin: 0.3rem 0 0.6rem;
}
.pred-confidence {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #E08A1E, #D4740E);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.conf-label { font-size: 0.9rem; color: #b0a080; margin-left: 0.3rem; }

.low-conf {
    background: #fff8e1; border: 1px solid #ffe082; border-radius: 10px;
    padding: 0.8rem 1rem; margin: 0.8rem 0; font-size: 0.85rem; color: #8d6e00;
}

.bars-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem; font-weight: 600; color: #444; margin: 1.8rem 0 0.8rem;
}
.bar-row { display: flex; align-items: center; margin: 6px 0; font-size: 0.82rem; }
.bar-name {
    width: 200px; text-align: right; padding-right: 12px; color: #666;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.82rem;
}
.bar-track { flex: 1; height: 14px; background: #f0ece4; border-radius: 7px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 7px; }
.bar-fill-top { background: linear-gradient(90deg, #E08A1E, #F5B041); }
.bar-fill-other { background: #d5cfc5; }
.bar-pct {
    width: 50px; text-align: right; padding-left: 10px; color: #888;
    font-variant-numeric: tabular-nums; font-size: 0.82rem;
}

.about-section {
    background: #faf8f5; border-radius: 16px; padding: 2rem;
    border: 1px solid #e8e0d0; margin: 2rem 0;
}
.about-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem; font-weight: 600; color: #333; margin: 0 0 0.8rem;
}
.about-text { font-size: 0.92rem; color: #666; line-height: 1.7; }

/* Analysis page */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700; color: #333; margin: 1.5rem 0 0.5rem;
}
.section-sub { font-size: 0.9rem; color: #888; margin-bottom: 1.2rem; }

.comparison-table {
    width: 100%; border-collapse: collapse; margin: 1rem 0;
    font-size: 0.9rem;
}
.comparison-table th {
    font-family: 'Space Grotesk', sans-serif;
    text-align: left; padding: 10px 16px;
    background: linear-gradient(135deg, #E08A1E, #F5B041);
    color: white; font-weight: 600; font-size: 0.85rem;
}
.comparison-table td {
    padding: 10px 16px; border-bottom: 1px solid #eee;
}
.comparison-table tr:hover td { background: #faf8f5; }

.confusion-item {
    display: flex; align-items: center; padding: 8px 0;
    border-bottom: 1px solid #f0ece4;
}
.confusion-arrow { color: #E08A1E; font-weight: 700; margin: 0 10px; font-size: 1.1rem; }
.confusion-count {
    background: #fff0d4; color: #c47a00; padding: 2px 10px;
    border-radius: 12px; font-size: 0.8rem; font-weight: 600; margin-left: auto;
}

.class-acc-row {
    display: flex; align-items: center; margin: 4px 0; font-size: 0.85rem;
}
.class-acc-name {
    width: 220px; text-align: right; padding-right: 12px; color: #555;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.class-acc-track { flex: 1; height: 16px; background: #f0ece4; border-radius: 8px; overflow: hidden; }
.class-acc-fill { height: 100%; border-radius: 8px; }
.class-acc-fill-good { background: linear-gradient(90deg, #4CAF50, #66BB6A); }
.class-acc-fill-ok { background: linear-gradient(90deg, #FFC107, #FFD54F); }
.class-acc-fill-bad { background: linear-gradient(90deg, #f44336, #ef5350); }
.class-acc-pct {
    width: 60px; text-align: right; padding-left: 10px; color: #888;
    font-variant-numeric: tabular-nums;
}

.footer {
    text-align: center; font-size: 0.78rem; color: #bbb;
    margin-top: 2rem; padding-bottom: 1.5rem;
}
.footer a { color: #E08A1E; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "best_car_light_model.pth"
GDRIVE_MODEL_ID = "1HyXCkf9toZP-qwTsAiT61-sS6j3gqc-r"
CONFUSION_PATH = "confusion_matrix.json"
GDRIVE_CONFUSION_ID = "1rNWaQX298bVdrU88fLfH3MiSci_0I85q"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first time only)..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}", MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH):
        st.error("Failed to download model.")
        st.stop()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
    num_classes = state_dict['fc.weight'].shape[0]
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model, num_classes

@st.cache_data
def load_confusion_data():
    if not os.path.exists(CONFUSION_PATH):
        try:
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_CONFUSION_ID}", CONFUSION_PATH, quiet=True)
        except:
            return None
    if os.path.exists(CONFUSION_PATH):
        with open(CONFUSION_PATH, 'r') as f:
            return json.load(f)
    return None

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
                            'ES','IS','UX','RX','DCT']:
            formatted.append(part.upper())
        else:
            formatted.append(part.capitalize())
    return ' '.join(formatted)

def get_brands(class_names):
    brands = set()
    brand_map = {
        'bmw': 'BMW', 'toyota': 'Toyota', 'honda': 'Honda', 'ford': 'Ford',
        'jeep': 'Jeep', 'mercedes': 'Mercedes-Benz', 'lexus': 'Lexus',
        'hyundai': 'Hyundai', 'kia': 'Kia', 'cadillac': 'Cadillac',
        'chevy': 'Chevrolet', 'dodge': 'Dodge', 'vw': 'Volkswagen',
        'porsche': 'Porsche', 'land': 'Land Rover', 'subaru': 'Subaru',
        'infiniti': 'Infiniti', 'genesis': 'Genesis', 'acura': 'Acura',
        'nissan': 'Nissan', 'gmc': 'GMC', 'buick': 'Buick',
        'mitsubishi': 'Mitsubishi', 'volvo': 'Volvo', 'audi': 'Audi',
    }
    for name in class_names:
        first = name.split('_')[0]
        if first in brand_map:
            brands.add(brand_map[first])
    return sorted(brands)

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

def page_identify(model, num_classes, class_names):
    st.markdown('<div class="upload-section"><div class="upload-label">Upload a headlight photo</div></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("upload", type=["jpg","jpeg","png"],
        help="Close-up headlight photos work best", label_visibility="collapsed")

    if uploaded is not None:
        image = Image.open(uploaded).convert('RGB')
        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.image(image, use_container_width=True)
        results = predict(model, image, class_names)
        top = results[0]
        with col2:
            st.markdown(f'<div class="prediction-card"><div class="pred-label">Identified as</div><div class="pred-car">{top["display_name"]}</div><div><span class="pred-confidence">{top["prob"]:.1f}%</span><span class="conf-label">confidence</span></div></div>', unsafe_allow_html=True)
            if top['prob'] < 60:
                st.markdown('<div class="low-conf">⚠️ Low confidence — this car may not be in our database, or try a closer crop of the headlight.</div>', unsafe_allow_html=True)

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
        st.markdown("""
        <div class="how-section">
            <div class="how-title">How it works</div>
            <div class="how-steps">
                <div class="how-card"><div class="step-num">1</div><div class="step-title">Photograph</div><div class="step-text">Take a close-up photo of any car's headlight from 2-3 feet away</div></div>
                <div class="how-card"><div class="step-num">2</div><div class="step-title">Upload</div><div class="step-text">Drop the photo above and the neural network processes it in seconds</div></div>
                <div class="how-card"><div class="step-num">3</div><div class="step-title">Identify</div><div class="step-text">Get the car's make, model, and year with a confidence score</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stats-row">
            <div class="stat"><div class="stat-num">{num_classes}</div><div class="stat-label">Car Models</div></div>
            <div class="stat"><div class="stat-num">1,540</div><div class="stat-label">Training Photos</div></div>
            <div class="stat"><div class="stat-num">96.1%</div><div class="stat-label">Test Accuracy</div></div>
        </div>
        """, unsafe_allow_html=True)

        brands = get_brands(class_names)
        if brands:
            brand_tags = ''.join([f'<span class="brand-tag">{b}</span>' for b in brands])
            st.markdown(f'<div class="brands-section"><div class="brands-title">Supported Brands</div><div class="brand-grid">{brand_tags}</div></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="about-section">
            <div class="about-title">About this project</div>
            <div class="about-text">
                LightID was built as an independent research project to explore how convolutional neural networks
                can learn to distinguish between cars using only their headlight designs. Every car has a unique
                headlight signature — the shape of the LED elements, the curve of the housing, the pattern of
                the daytime running lights — and this model learns to recognize those patterns.
                <br><br>
                The dataset was collected entirely by hand: photographing headlights at dealerships and parking lots,
                then training a ResNet18 model using transfer learning from ImageNet. No scraped or AI-generated
                images were used — every photo in the training set is an original photograph.
            </div>
        </div>
        """, unsafe_allow_html=True)

def page_analysis(confusion_data):
    st.markdown('<div class="section-header">Model Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A deeper look at how the model performs across different car models</div>', unsafe_allow_html=True)

    if confusion_data is None:
        st.warning("Confusion matrix data not available.")
        return

    # Before / After comparison
    st.markdown('<div class="section-header">📈 Project Evolution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How the model improved from initial proof of concept to final version</div>', unsafe_allow_html=True)

    st.markdown("""
    <table class="comparison-table">
        <tr><th>Metric</th><th>POC (July 2026)</th><th>Final (Nov 2026)</th></tr>
        <tr><td>Car models</td><td>5 (all SUVs)</td><td>62 (mixed types)</td></tr>
        <tr><td>Training images</td><td>98</td><td>1,540</td></tr>
        <tr><td>Data source</td><td>Personal phone photos</td><td>Dealerships + parking lots</td></tr>
        <tr><td>Architecture</td><td>Custom 3-layer CNN + ResNet18</td><td>ResNet18 transfer learning</td></tr>
        <tr><td>Train/Val/Test split</td><td>80/20 (no validation)</td><td>70/15/15</td></tr>
        <tr><td>Test accuracy</td><td>90% (unreliable, 20 test images)</td><td>96.1% (231 test images)</td></tr>
        <tr><td>External image test</td><td>Failed (96% confident, wrong car)</td><td>Reliable</td></tr>
        <tr><td>Deployment</td><td>Colab only</td><td>Live web app</td></tr>
    </table>
    """, unsafe_allow_html=True)

    # Top confusions
    st.markdown('<div class="section-header">🔴 Most Confused Pairs</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Cars the model most often mistakes for each other — usually ones with similar headlight designs</div>', unsafe_allow_html=True)

    confusions = confusion_data.get('top_confusions', [])
    if confusions:
        for c in confusions[:10]:
            true_name = format_car_name(c['true'])
            pred_name = format_car_name(c['predicted'])
            st.markdown(f"""
            <div class="confusion-item">
                <span>{true_name}</span>
                <span class="confusion-arrow">→</span>
                <span>{pred_name}</span>
                <span class="confusion-count">{c['count']}×</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No confusions found — the model got everything right on the test set.")

    # Per-class accuracy
    st.markdown('<div class="section-header">📊 Per-Class Accuracy</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How well the model performs on each individual car model</div>', unsafe_allow_html=True)

    per_class = confusion_data.get('per_class', [])
    per_class_sorted = sorted(per_class, key=lambda x: x['accuracy'], reverse=True)

    # Filter out classes with 0 test images
    has_data = [p for p in per_class_sorted if p['total'] > 0]
    no_data = [p for p in per_class_sorted if p['total'] == 0]

    bars_html = ""
    for p in has_data:
        if p['accuracy'] >= 80:
            fill_class = "class-acc-fill-good"
        elif p['accuracy'] >= 50:
            fill_class = "class-acc-fill-ok"
        else:
            fill_class = "class-acc-fill-bad"
        display = format_car_name(p['name'])
        bars_html += f'<div class="class-acc-row"><div class="class-acc-name">{display}</div><div class="class-acc-track"><div class="class-acc-fill {fill_class}" style="width: {max(p["accuracy"], 2)}%"></div></div><div class="class-acc-pct">{p["accuracy"]}%</div></div>'
    st.markdown(bars_html, unsafe_allow_html=True)

    if no_data:
        with st.expander(f"{len(no_data)} classes with no test images"):
            st.markdown("These classes had too few images — all went to training, none left for testing. Not a failure, just need more photos.")
            for p in no_data:
                st.text(f"  {format_car_name(p['name'])}")

    # Key insight
    st.markdown("""
    <div class="about-section">
        <div class="about-title">Key Insights</div>
        <div class="about-text">
            The biggest confusion is between the BMW X5 and Toyota 4Runner — a pattern that persisted from the
            5-class proof of concept through to the 62-class final model. This makes sense: both are large SUVs
            with similar rectangular LED headlight shapes. Solving this would likely require more training images
            of both models from varied angles and lighting conditions.
            <br><br>
            Classes showing 0% accuracy with 0 test images (like the Mercedes G550 and Volvo XC90) aren't failures —
            they simply had too few total images for any to land in the test split. Adding more photos of these
            cars would fix this.
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="logo-container"><span class="logo-icon">💡</span></div>
        <h1 class="app-name">LightID</h1>
        <p class="app-tagline">Identify any car from its headlights</p>
        <p class="app-description">
            A deep learning model trained on 1,500+ real-world headlight photographs
            to recognize car make, model, and year from a single close-up photo.
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

    confusion_data = load_confusion_data()

    # Tabs
    tab1, tab2 = st.tabs(["🔦 Identify", "📊 Analysis"])

    with tab1:
        page_identify(model, num_classes, class_names)

    with tab2:
        page_analysis(confusion_data)

    # Footer
    st.markdown("""
    <div class="footer">
        Built by Rus Surmin · PyTorch · ResNet18 · 2026<br>
        <a href="https://github.com/rslnxz" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
