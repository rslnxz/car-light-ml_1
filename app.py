import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import os
import json
import gdown

st.set_page_config(page_title="CarID — Identify Cars from Headlights", page_icon="🚗", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp, .main, .block-container,
[data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stBottom"], header, section {
    font-family: 'Inter', sans-serif;
    background-color: #0e0e0e !important; color: #e0e0e0 !important;
}
* { color: inherit; }
a, a:hover, a:visited, a:active { color: #E08A1E !important; }
p, span, div, li, td, th, label, h1, h2, h3, h4, h5, h6 { color: #e0e0e0 !important; }
.stFileUploader, .stFileUploader * { color: #e0e0e0 !important; }
[data-testid="stFileUploaderDropzone"] { background: #1a1a1a !important; border-color: #333 !important; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #E08A1E !important; }
[data-testid="stFileUploaderDropzone"] * { color: #999 !important; }
button { color: #e0e0e0 !important; background: #222 !important; border-color: #444 !important; }
button:hover { background: #333 !important; color: #fff !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; gap: 0; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #888 !important; border-bottom: 2px solid transparent !important; padding: 10px 20px !important; }
.stTabs [aria-selected="true"] { color: #E08A1E !important; border-bottom: 2px solid #E08A1E !important; }
.stTabs [data-baseweb="tab"]:hover { color: #ccc !important; }
[data-testid="stExpander"] { background: #1a1a1a !important; border-color: #333 !important; border-radius: 12px !important; }
[data-testid="stExpander"] * { color: #ccc !important; }
::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-track { background: #111; } ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }

.hero { text-align: center; padding: 2.5rem 1rem 1rem; }
.logo-container { display: inline-flex; align-items: center; justify-content: center; width: 80px; height: 80px; border-radius: 20px; background: linear-gradient(135deg, #E08A1E 0%, #F5B041 50%, #E08A1E 100%); margin-bottom: 1.2rem; box-shadow: 0 4px 24px rgba(224, 138, 30, 0.25); position: relative; }
.logo-car { font-size: 36px; } .logo-q { position: absolute; top: -8px; right: -8px; font-size: 20px; font-weight: 700; background: #0e0e0e; border: 2px solid #E08A1E; width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #E08A1E !important; }
.app-name { font-family: 'Space Grotesk', sans-serif; font-size: 3.2rem; font-weight: 700; margin: 0 auto; letter-spacing: -0.02em; color: #ffffff !important; text-align: center !important; display: block; width: 100%; }
.app-tagline { font-size: 1.25rem; color: #888 !important; margin: 0.4rem auto 0; text-align: center !important; display: block; width: 100%; }
.app-description { max-width: 560px; margin: 1.2rem auto 0; font-size: 1rem; color: #777 !important; line-height: 1.7; text-align: center !important; display: block; }
.divider { width: 60px; height: 3px; background: linear-gradient(90deg, #E08A1E, #F5B041); border-radius: 2px; margin: 2rem auto; }

.how-section { background: #161616; border-radius: 16px; padding: 2rem; border: 1px solid #2a2a2a; margin: 0 auto 2rem; max-width: 700px; }
.how-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 600; color: #eee !important; margin: 0 0 1.2rem; text-align: center; }
.how-steps { display: flex; gap: 1.5rem; justify-content: center; flex-wrap: wrap; }
.how-card { flex: 1; min-width: 160px; max-width: 220px; text-align: center; padding: 0.5rem; }
.step-num { width: 36px; height: 36px; border-radius: 50%; background: linear-gradient(135deg, #E08A1E, #F5B041); color: #fff !important; font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 0.6rem; }
.step-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 600; color: #ddd !important; margin-bottom: 0.3rem; }
.step-text { font-size: 0.82rem; color: #888 !important; line-height: 1.5; }

.stats-row { display: flex; justify-content: center; gap: 2.5rem; margin: 0 auto 2rem; padding: 1.5rem; background: #161616; border-radius: 16px; border: 1px solid #2a2a2a; max-width: 700px; flex-wrap: wrap; }
.stat { text-align: center; } .stat-num { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: #E08A1E !important; } .stat-label { font-size: 0.72rem; color: #666 !important; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.2rem; }

.brands-section { text-align: center; margin: 0 auto 2rem; max-width: 700px; }
.brands-title { font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600; color: #666 !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem; }
.brand-grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
.brand-tag { padding: 6px 14px; border-radius: 8px; font-size: 0.82rem; font-weight: 500; background: #1a1a1a; color: #bba060 !important; border: 1px solid #2a2a2a; }

.upload-section { text-align: center; margin: 0 auto 1rem; max-width: 700px; }
.upload-label { font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 600; color: #eee !important; margin-bottom: 0.5rem; }

.prediction-card { background: #161616; border-radius: 16px; padding: 1.8rem; border: 1px solid #2a2a2a; margin: 1rem 0; box-shadow: 0 2px 12px rgba(0,0,0,0.3); }
.pred-label { font-family: 'Space Grotesk', sans-serif; font-size: 0.72rem; color: #777 !important; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
.pred-car { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; color: #ffffff !important; margin: 0.3rem 0 0.6rem; }
.pred-confidence { font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700; background: linear-gradient(135deg, #E08A1E, #F5B041); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.conf-label { font-size: 0.9rem; color: #777 !important; margin-left: 0.3rem; }

.low-conf { background: #2a2200; border: 1px solid #5a4a00; border-radius: 10px; padding: 0.8rem 1rem; margin: 0.8rem 0; font-size: 0.85rem; color: #ffc107 !important; }

/* Also possible section */
.also-section { background: #161616; border-radius: 16px; padding: 1.2rem 1.5rem; border: 1px solid #2a2a2a; margin: 1rem 0; }
.also-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 600; color: #E08A1E !important; margin-bottom: 0.6rem; }
.also-subtitle { font-size: 0.72rem; color: #666 !important; text-transform: uppercase; letter-spacing: 0.05em; margin: 0.6rem 0 0.3rem; }
.also-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.also-tag { padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; background: #1e1e1e; color: #ccc !important; border: 1px solid #333; }
.also-tag-sibling { background: #1a1500; color: #E08A1E !important; border: 1px solid #3a2a00; }

.bars-header { font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600; color: #ccc !important; margin: 1.8rem 0 0.8rem; text-align: center; }
.bar-row { display: flex; align-items: center; margin: 6px auto; font-size: 0.82rem; max-width: 700px; }
.bar-name { width: 200px; text-align: right; padding-right: 12px; color: #999 !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.82rem; }
.bar-track { flex: 1; height: 14px; background: #222; border-radius: 7px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 7px; } .bar-fill-top { background: linear-gradient(90deg, #E08A1E, #F5B041); } .bar-fill-other { background: #444; }
.bar-pct { width: 50px; text-align: right; padding-left: 10px; color: #777 !important; font-variant-numeric: tabular-nums; font-size: 0.82rem; }

.about-section { background: #161616; border-radius: 16px; padding: 2rem; border: 1px solid #2a2a2a; margin: 2rem auto; max-width: 700px; }
.about-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 600; color: #eee !important; margin: 0 0 0.8rem; text-align: center; }
.about-text { font-size: 0.92rem; color: #888 !important; line-height: 1.7; text-align: center; }

.section-header { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; color: #eee !important; margin: 1.5rem 0 0.5rem; text-align: center; }
.section-sub { font-size: 0.9rem; color: #777 !important; margin-bottom: 1.2rem; text-align: center; }

.comparison-table { width: 100%; max-width: 700px; border-collapse: collapse; margin: 1rem auto; font-size: 0.9rem; }
.comparison-table th { font-family: 'Space Grotesk', sans-serif; text-align: left; padding: 10px 16px; background: linear-gradient(135deg, #E08A1E, #D4740E) !important; color: #fff !important; font-weight: 600; font-size: 0.85rem; }
.comparison-table td { padding: 10px 16px; border-bottom: 1px solid #222; color: #ccc !important; }
.comparison-table tr:hover td { background: #1a1a1a; }

.confusion-item { display: flex; align-items: center; padding: 8px 0; max-width: 700px; margin: 0 auto; border-bottom: 1px solid #222; color: #ccc !important; }
.confusion-arrow { color: #E08A1E !important; font-weight: 700; margin: 0 10px; font-size: 1.1rem; }
.confusion-count { background: #2a1f00; color: #E08A1E !important; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; margin-left: auto; }

.class-acc-row { display: flex; align-items: center; margin: 4px auto; font-size: 0.85rem; max-width: 700px; }
.class-acc-name { width: 220px; text-align: right; padding-right: 12px; color: #999 !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.class-acc-track { flex: 1; height: 16px; background: #222; border-radius: 8px; overflow: hidden; }
.class-acc-fill { height: 100%; border-radius: 8px; }
.class-acc-fill-good { background: linear-gradient(90deg, #2e7d32, #4CAF50); }
.class-acc-fill-ok { background: linear-gradient(90deg, #f57f17, #FFC107); }
.class-acc-fill-bad { background: linear-gradient(90deg, #c62828, #f44336); }
.class-acc-pct { width: 60px; text-align: right; padding-left: 10px; color: #777 !important; font-variant-numeric: tabular-nums; }

.insight-section { background: #161616; border-radius: 16px; padding: 2rem; border: 1px solid #2a2a2a; margin: 2rem auto; max-width: 700px; }
.insight-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 600; color: #eee !important; margin: 0 0 0.8rem; text-align: center; }
.insight-text { font-size: 0.92rem; color: #888 !important; line-height: 1.7; text-align: center; }

.footer { text-align: center; font-size: 0.78rem; color: #555 !important; margin-top: 2rem; padding-bottom: 1.5rem; }
.footer a { color: #E08A1E !important; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "best_car_light_model.pth"
GDRIVE_MODEL_ID = "1HyXCkf9toZP-qwTsAiT61-sS6j3gqc-r"
CONFUSION_PATH = "confusion_matrix.json"
GDRIVE_CONFUSION_ID = "1rNWaQX298bVdrU88fLfH3MiSci_0I85q"

HEADLIGHT_SIBLINGS = {
    "bmw_x5_40i_2021": {"trims": ["X5 xDrive45e", "X5 M50i", "X5 M Competition"], "siblings": []},
    "toyota_4runner_5gen_2010_2024": {"trims": ["4Runner SR5", "4Runner TRD Off-Road", "4Runner TRD Pro", "4Runner Limited"], "siblings": []},
    "honda_crv_6gen_2023_2025": {"trims": ["CR-V EX", "CR-V Sport", "CR-V Sport Touring", "CR-V Hybrid"], "siblings": []},
    "ford_escape_3gen_2013_2016": {"trims": ["Escape S", "Escape SE", "Escape Titanium"], "siblings": []},
    "jeep_grand_cherokee_wl_5gen_2022": {"trims": ["Grand Cherokee Limited", "Grand Cherokee Overland", "Grand Cherokee Summit"], "siblings": ["Grand Cherokee L", "Grand Cherokee 4xe"]},
    "land_rover_defender_110_2023": {"trims": ["Defender 110 SE", "Defender 110 X", "Defender 110 V8"], "siblings": ["Defender 90", "Defender 130"]},
    "subaru_outback_2025": {"trims": ["Outback Premium", "Outback Limited", "Outback Touring", "Outback Onyx", "Outback Wilderness"], "siblings": []},
    "mercedes_gle_350_2024": {"trims": ["GLE 450", "GLE 580"], "siblings": ["GLE 53 AMG", "GLE 63 AMG"]},
    "toyota_venza_2022": {"trims": ["Venza LE", "Venza XLE", "Venza Limited"], "siblings": []},
    "bmw_ix_m60_2023": {"trims": ["iX xDrive40", "iX xDrive50"], "siblings": []},
    "jeep_wrangler_rubicon_2021": {"trims": ["Wrangler Sport", "Wrangler Sahara", "Wrangler Willys"], "siblings": ["Wrangler 4xe"]},
    "mercedes_eqe_350_2023": {"trims": ["EQE 350+", "EQE 500"], "siblings": ["EQE SUV"]},
    "toyota_tundra_2024": {"trims": ["Tundra SR5", "Tundra Limited", "Tundra Platinum", "Tundra TRD Pro", "Tundra Capstone"], "siblings": ["Tundra i-Force MAX Hybrid"]},
    "cadillac_xt4_2021": {"trims": ["XT4 Luxury", "XT4 Premium Luxury", "XT4 Sport"], "siblings": []},
    "ford_expedition_2021": {"trims": ["Expedition Limited", "Expedition King Ranch", "Expedition Platinum"], "siblings": ["Expedition MAX", "Lincoln Navigator"]},
    "toyota_rav4_2024": {"trims": ["RAV4 LE", "RAV4 XLE", "RAV4 Adventure", "RAV4 TRD Off-Road", "RAV4 Limited"], "siblings": ["RAV4 Hybrid", "RAV4 Prime"]},
    "vw_atlas_2023": {"trims": ["Atlas S", "Atlas SE", "Atlas SEL", "Atlas SEL Premium"], "siblings": []},
    "cadillac_escalade_2024": {"trims": ["Escalade Luxury", "Escalade Premium Luxury", "Escalade Sport", "Escalade Platinum"], "siblings": ["Escalade ESV", "Escalade-V"]},
    "chevy_tahoe_2024": {"trims": ["Tahoe LS", "Tahoe LT", "Tahoe Z71", "Tahoe Premier", "Tahoe High Country"], "siblings": ["Chevrolet Suburban"]},
    "ford_explorer_2022": {"trims": ["Explorer XLT", "Explorer Limited", "Explorer ST", "Explorer Timberline", "Explorer Platinum"], "siblings": ["Lincoln Aviator"]},
    "hyundai_elantra_2023": {"trims": ["Elantra SE", "Elantra SEL", "Elantra Limited"], "siblings": ["Elantra N", "Elantra Hybrid"]},
    "lexus_ux_300h_2025": {"trims": ["UX 250h"], "siblings": []},
    "toyota_corolla_2022": {"trims": ["Corolla LE", "Corolla SE", "Corolla XSE"], "siblings": ["Corolla Hybrid", "Corolla Hatchback"]},
    "toyota_avalon_2014": {"trims": ["Avalon XLE", "Avalon Limited", "Avalon Touring"], "siblings": ["Avalon Hybrid"]},
    "mercedes_cclass_300_2024": {"trims": ["C 300 4MATIC"], "siblings": ["C 43 AMG", "C 63 AMG"]},
    "bmw_z4_2013": {"trims": ["Z4 sDrive28i", "Z4 sDrive35i", "Z4 sDrive35is"], "siblings": []},
    "dodge_challenger": {"trims": ["Challenger SXT", "Challenger GT", "Challenger R/T", "Challenger R/T Scat Pack", "Challenger SRT Hellcat"], "siblings": ["Challenger SRT Demon"]},
    "porsche_panamera": {"trims": ["Panamera 4", "Panamera 4S", "Panamera GTS", "Panamera Turbo", "Panamera Turbo S"], "siblings": ["Panamera Sport Turismo", "Panamera E-Hybrid"]},
    "lexus_rx": {"trims": ["RX 350", "RX 350h", "RX 500h"], "siblings": ["RX 450h+"]},
    "nissan_titan_sv": {"trims": ["Titan S", "Titan Pro-4X", "Titan Platinum Reserve"], "siblings": ["Titan XD"]},
    "bmw_x4": {"trims": ["X4 xDrive30i", "X4 M40i"], "siblings": ["X4 M", "X4 M Competition"]},
    "kia_sorento": {"trims": ["Sorento LX", "Sorento S", "Sorento EX", "Sorento SX Prestige"], "siblings": ["Sorento Hybrid", "Sorento PHEV"]},
    "hyundai_santa_fe": {"trims": ["Santa Fe SE", "Santa Fe SEL", "Santa Fe Limited", "Santa Fe Calligraphy"], "siblings": ["Santa Fe Hybrid", "Santa Fe PHEV"]},
    "infiniti_qx80": {"trims": ["QX80 Luxe", "QX80 Premium Select", "QX80 Sensory"], "siblings": ["Nissan Armada"]},
    "toyota_highlander": {"trims": ["Highlander LE", "Highlander XLE", "Highlander Limited", "Highlander Platinum"], "siblings": ["Highlander Hybrid"]},
    "genesis_gv80": {"trims": ["GV80 2.5T", "GV80 3.5T"], "siblings": ["GV80 Coupe"]},
    "mitsubishi_lancer_gt": {"trims": ["Lancer ES", "Lancer SE"], "siblings": ["Lancer Sportback"]},
    "kia_optima": {"trims": ["Optima LX", "Optima S", "Optima EX", "Optima SX"], "siblings": ["Optima Hybrid"]},
    "honda_odyssey": {"trims": ["Odyssey EX", "Odyssey EX-L", "Odyssey Sport", "Odyssey Touring", "Odyssey Elite"], "siblings": []},
    "buick_envista": {"trims": ["Envista Preferred", "Envista Sport Touring", "Envista Avenir"], "siblings": []},
    "mercedes_amg_a35": {"trims": ["A 220", "A 250"], "siblings": ["A 45 AMG"]},
    "kia_sportage": {"trims": ["Sportage LX", "Sportage EX", "Sportage SX Prestige"], "siblings": ["Sportage Hybrid", "Sportage PHEV", "Hyundai Tucson"]},
    "gmc_yukon": {"trims": ["Yukon SLE", "Yukon SLT", "Yukon AT4", "Yukon Denali", "Yukon Denali Ultimate"], "siblings": ["Yukon XL"]},
    "honda_hrv_sport": {"trims": ["HR-V LX", "HR-V EX-L"], "siblings": []},
    "bmw_x3_2017": {"trims": ["X3 sDrive30i", "X3 xDrive30i", "X3 M40i"], "siblings": ["X3 M", "X3 M Competition"]},
    "vw_tiguan_2025": {"trims": ["Tiguan S", "Tiguan SE", "Tiguan SEL", "Tiguan SEL R-Line"], "siblings": []},
    "mercedes_gla_250_2021": {"trims": ["GLA 250 4MATIC"], "siblings": ["GLA 35 AMG", "GLA 45 AMG"]},
    "honda_civic_si_2024": {"trims": ["Civic Sport", "Civic EX", "Civic Touring"], "siblings": ["Civic Type R", "Civic Hybrid", "Civic Hatchback"]},
    "kia_k5_gt_2026": {"trims": ["K5 LX", "K5 LXS", "K5 GT-Line", "K5 EX"], "siblings": []},
    "vw_atlas_cross_sport_2021": {"trims": ["Atlas Cross Sport S", "Atlas Cross Sport SE", "Atlas Cross Sport SEL"], "siblings": []},
    "bmw_m3_2025": {"trims": ["M3 Competition", "M3 Competition xDrive"], "siblings": ["3 Series (G20)", "M3 CS"]},
    "lexus_is_250_2014": {"trims": ["IS 350", "IS 300"], "siblings": ["IS 300h"]},
    "lexus_es_350_2025": {"trims": ["ES 250"], "siblings": ["ES 300h"]},
    "acura_mdx_2026": {"trims": ["MDX Technology", "MDX A-Spec", "MDX Advance"], "siblings": ["MDX Type S"]},
    "infiniti_qx60_2023": {"trims": ["QX60 Pure", "QX60 Luxe", "QX60 Sensory", "QX60 Autograph"], "siblings": []},
    "bmw_x7_2022": {"trims": ["X7 xDrive40i", "X7 M60i"], "siblings": ["Alpina XB7"]},
    "infiniti_qx50_2019": {"trims": ["QX50 Pure", "QX50 Luxe", "QX50 Essential", "QX50 Sensory"], "siblings": []},
    "volvo_xc90_2020": {"trims": ["XC90 T5", "XC90 T6", "XC90 T8 Recharge"], "siblings": []},
    "audi_q8_2021": {"trims": ["Q8 55 TFSI"], "siblings": ["SQ8", "RS Q8"]},
    "mercedes_g550_2017": {"trims": ["G 500"], "siblings": ["G 63 AMG"]},
    "chevy_traverse_2024": {"trims": ["Traverse LS", "Traverse LT", "Traverse RS", "Traverse Z71", "Traverse Premier"], "siblings": ["Buick Enclave"]},
    "hyundai_palisade_2022": {"trims": ["Palisade SE", "Palisade SEL", "Palisade Limited", "Palisade Calligraphy"], "siblings": ["Kia Telluride"]},
}

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first time only)..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}", MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH):
        st.error("Failed to download model."); st.stop()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
    num_classes = state_dict['fc.weight'].shape[0]
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict); model.eval()
    return model, num_classes

@st.cache_data
def load_confusion_data():
    if not os.path.exists(CONFUSION_PATH):
        try: gdown.download(f"https://drive.google.com/uc?id={GDRIVE_CONFUSION_ID}", CONFUSION_PATH, quiet=True)
        except: return None
    if os.path.exists(CONFUSION_PATH):
        with open(CONFUSION_PATH, 'r') as f: return json.load(f)
    return None

@st.cache_data
def get_class_names():
    if os.path.exists("class_names.txt"):
        with open("class_names.txt", "r") as f: return [line.strip() for line in f.readlines() if line.strip()]
    return None

def format_car_name(folder_name):
    parts = folder_name.split('_')
    formatted = []
    for part in parts:
        if part.upper() in ['BMW','VW','GMC','AMG','GT','SV','SI','SE','XLE','AWD','FWD','RWD','4WD','CRV','HRV','MDX','GLA','GLE','GLC','EQE','RAV4','IX','M3','X3','X4','X5','X7','Z4','K5','Q8','QX50','QX60','QX80','XC90','XT4','GV80','ES','IS','UX','RX','DCT']:
            formatted.append(part.upper())
        else: formatted.append(part.capitalize())
    return ' '.join(formatted)

def get_brands(class_names):
    brands = set()
    brand_map = {'bmw':'BMW','toyota':'Toyota','honda':'Honda','ford':'Ford','jeep':'Jeep','mercedes':'Mercedes-Benz','lexus':'Lexus','hyundai':'Hyundai','kia':'Kia','cadillac':'Cadillac','chevy':'Chevrolet','dodge':'Dodge','vw':'Volkswagen','porsche':'Porsche','land':'Land Rover','subaru':'Subaru','infiniti':'Infiniti','genesis':'Genesis','acura':'Acura','nissan':'Nissan','gmc':'GMC','buick':'Buick','mitsubishi':'Mitsubishi','volvo':'Volvo','audi':'Audi'}
    for name in class_names:
        first = name.split('_')[0]
        if first in brand_map: brands.add(brand_map[first])
    return sorted(brands)

def count_identifiable():
    total = len(HEADLIGHT_SIBLINGS)
    for v in HEADLIGHT_SIBLINGS.values():
        total += len(v['trims']) + len(v['siblings'])
    return total

eval_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

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
    uploaded = st.file_uploader("upload", type=["jpg","jpeg","png"], help="Close-up headlight photos work best", label_visibility="collapsed")

    if uploaded is not None:
        image = Image.open(uploaded).convert('RGB')
        col1, col2 = st.columns([1, 1.4])
        with col1: st.image(image, use_container_width=True)
        results = predict(model, image, class_names)
        top = results[0]
        with col2:
            st.markdown(f'<div class="prediction-card"><div class="pred-label">Identified as</div><div class="pred-car">{top["display_name"]}</div><div><span class="pred-confidence">{top["prob"]:.1f}%</span><span class="conf-label">confidence</span></div></div>', unsafe_allow_html=True)
            if top['prob'] < 60:
                st.markdown('<div class="low-conf">⚠️ Low confidence — this car may not be in our database, or try a closer crop of the headlight.</div>', unsafe_allow_html=True)

        # Also possible section
        sibling_data = HEADLIGHT_SIBLINGS.get(top['name'], None)
        if sibling_data:
            trims = sibling_data['trims']
            siblings = sibling_data['siblings']
            if trims or siblings:
                html = '<div class="also-section"><div class="also-title">Other possible cars with same headlights</div>'
                if trims:
                    html += '<div class="also-subtitle">Same model, different trims</div><div class="also-tags">'
                    for t in trims: html += f'<span class="also-tag">{t}</span>'
                    html += '</div>'
                if siblings:
                    html += '<div class="also-subtitle">Related models</div><div class="also-tags">'
                    for s in siblings: html += f'<span class="also-tag also-tag-sibling">{s}</span>'
                    html += '</div>'
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)

        st.markdown('<div class="bars-header">Top predictions</div>', unsafe_allow_html=True)
        bars_html = ""
        for i, r in enumerate(results[:8]):
            fill_class = "bar-fill-top" if i == 0 else "bar-fill-other"
            width = max(r['prob'], 0.5)
            bars_html += f'<div class="bar-row"><div class="bar-name">{r["display_name"]}</div><div class="bar-track"><div class="bar-fill {fill_class}" style="width: {width}%"></div></div><div class="bar-pct">{r["prob"]:.1f}%</div></div>'
        st.markdown(bars_html, unsafe_allow_html=True)

        with st.expander(f"All {len(class_names)} classes"):
            for r in results:
                if r['prob'] >= 0.1: st.text(f"  {r['display_name']:45s} {r['prob']:.1f}%")
    else:
        identifiable = count_identifiable()
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
            <div class="stat"><div class="stat-num">{identifiable}</div><div class="stat-label">Cars Identifiable</div></div>
            <div class="stat"><div class="stat-num">25</div><div class="stat-label">Brands</div></div>
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
                CarID uses a convolutional neural network (ResNet18) to identify cars from close-up headlight
                photographs. The model was trained on 1,540 original photos collected by hand at dealerships
                and parking lots — no scraped or AI-generated images.
                <br><br>
                Every car has a unique headlight signature: the shape of the LED elements, the curve of
                the housing, and the pattern of the daytime running lights. The model learns these visual
                patterns through transfer learning, starting from a network pre-trained on 1.2 million
                general images (ImageNet) and fine-tuning it specifically for headlight recognition.
                <br><br>
                When the model identifies a car, it also shows other trims and related models that share
                the same headlight design — effectively expanding coverage from 62 trained classes to
                over 340 identifiable vehicles.
            </div>
        </div>
        """, unsafe_allow_html=True)

def page_analysis(confusion_data):
    st.markdown('<div class="section-header">Model Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A deeper look at how the model performs across different car models</div>', unsafe_allow_html=True)
    if confusion_data is None:
        st.warning("Confusion matrix data not available."); return

    st.markdown('<div class="section-header">📈 Project Evolution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How the model improved from proof of concept to final version</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="comparison-table">
        <tr><th>Metric</th><th>POC (July 2026)</th><th>Final (Nov 2026)</th></tr>
        <tr><td>Car models</td><td>5 (all SUVs)</td><td>62 (mixed types)</td></tr>
        <tr><td>Total identifiable</td><td>5</td><td>342 (incl. trims & siblings)</td></tr>
        <tr><td>Training images</td><td>98</td><td>1,540</td></tr>
        <tr><td>Data source</td><td>Personal phone photos</td><td>Dealerships + parking lots</td></tr>
        <tr><td>Architecture</td><td>Custom 3-layer CNN + ResNet18</td><td>ResNet18 transfer learning</td></tr>
        <tr><td>Train/Val/Test split</td><td>80/20 (no validation)</td><td>70/15/15</td></tr>
        <tr><td>Test accuracy</td><td>90% (unreliable, 20 test images)</td><td>96.1% (231 test images)</td></tr>
        <tr><td>External image test</td><td>Failed (96% confident, wrong car)</td><td>Reliable</td></tr>
        <tr><td>Deployment</td><td>Colab only</td><td>Live web app</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Per-Class Accuracy</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How well the model performs on each car model</div>', unsafe_allow_html=True)
    per_class = confusion_data.get('per_class', [])
    has_data = sorted([p for p in per_class if p['total'] > 0], key=lambda x: x['accuracy'], reverse=True)
    bars_html = ""
    for p in has_data:
        fill_class = "class-acc-fill-good" if p['accuracy'] >= 80 else ("class-acc-fill-ok" if p['accuracy'] >= 50 else "class-acc-fill-bad")
        bars_html += f'<div class="class-acc-row"><div class="class-acc-name">{format_car_name(p["name"])}</div><div class="class-acc-track"><div class="class-acc-fill {fill_class}" style="width: {max(p["accuracy"], 2)}%"></div></div><div class="class-acc-pct">{p["accuracy"]}%</div></div>'
    st.markdown(bars_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-section">
        <div class="insight-title">Key Insight</div>
        <div class="insight-text">
            The biggest confusion is between the BMW X5 and Toyota 4Runner — a pattern that persisted
            from the 5-class proof of concept through to the 62-class final model. Solving this would 
            require more training images of both models from varied angles and lighting conditions.
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown("""
    <div class="hero">
        <div class="logo-container"><span class="logo-car">🚗</span><span class="logo-q">?</span></div>
        <h1 class="app-name">CarID</h1>
        <p class="app-tagline">Identify any car from its headlights</p>
        <p class="app-description">
            A deep learning model trained on 1,500+ real-world headlight photographs
            to recognize car make, model, and year from a single close-up photo.
        </p>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)
    try: model, num_classes = load_model()
    except Exception as e: st.error(f"Failed to load model: {e}"); st.stop()
    class_names = get_class_names()
    if class_names is None: class_names = [f"class_{i}" for i in range(num_classes)]
    confusion_data = load_confusion_data()
    tab1, tab2 = st.tabs(["🚗 Identify", "📊 Analysis"])
    with tab1: page_identify(model, num_classes, class_names)
    with tab2: page_analysis(confusion_data)
    st.markdown('<div class="footer">Built by Rus Surmin · PyTorch · ResNet18 · 2026<br><a href="https://github.com/rslnxz" target="_blank">GitHub</a></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
