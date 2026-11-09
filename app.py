import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import os, json, gdown
from datetime import datetime

st.set_page_config(page_title="CarID", page_icon="🚗", layout="centered")
if 'history' not in st.session_state: st.session_state.history = []
if 'pred_count' not in st.session_state: st.session_state.pred_count = 0

LOGO_B64 = "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"
MODEL_PATH = "best_car_light_model.pth"
GDRIVE_MODEL_ID = "1HyXCkf9toZP-qwTsAiT61-sS6j3gqc-r"
CONFUSION_PATH = "confusion_matrix.json"
GDRIVE_CONFUSION_ID = "1rNWaQX298bVdrU88fLfH3MiSci_0I85q"

SIBLINGS = {
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

# --- CSS: same working dark mode + 3 targeted fixes ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp, .main, .block-container,
[data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stBottom"], header, section {
    font-family: 'Inter', sans-serif !important;
    background-color: #0e0e0e !important; color: #e0e0e0 !important;
}
* { color: inherit; }
a, a:hover, a:visited, a:active { color: #E08A1E !important; }
p, span, div, li, td, th, label, h1, h2, h3, h4, h5, h6 { color: #e0e0e0 !important; }
.block-container { max-width: 750px !important; margin: 0 auto !important; padding-left: 1rem !important; padding-right: 1rem !important; }
[data-testid="stMarkdownContainer"] { width: 100% !important; }
.stFileUploader, .stFileUploader * { color: #e0e0e0 !important; }
[data-testid="stFileUploaderDropzone"] { background: #1a1a1a !important; border-color: #333 !important; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #E08A1E !important; }
[data-testid="stFileUploaderDropzone"] * { color: #999 !important; }
button { color: #e0e0e0 !important; background: #222 !important; border-color: #444 !important; }
button:hover { background: #333 !important; color: #fff !important; }
[data-testid="stExpander"] { background: #1a1a1a !important; border-color: #333 !important; border-radius: 12px !important; }
[data-testid="stExpander"] * { color: #ccc !important; }
::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-track { background: #111; } ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }

/* FIX 1: Tabs - centered and bigger */
.stTabs [data-baseweb="tab-list"] { justify-content: center !important; gap: 8px !important; }
.stTabs [data-baseweb="tab"] { padding: 14px 36px !important; font-size: 1.1rem !important; font-weight: 600 !important; }
.stTabs [aria-selected="true"] { color: #E08A1E !important; border-bottom-color: #E08A1E !important; }

/* FIX 2: Radio buttons - centered and bigger */
.stRadio > div[role="radiogroup"] { justify-content: center !important; gap: 12px !important; }
.stRadio [data-testid="stMarkdownContainer"] p { font-size: 1.05rem !important; font-weight: 500 !important; }

.hero-wrap { width: 100%; display: flex; flex-direction: column; align-items: center; text-align: center; padding: 2.5rem 0 1rem; }
.logo-img { width: 150px; height: 150px; margin: 0 auto 1.5rem; display: block; }
.hero-name { font-family: 'Space Grotesk', sans-serif; font-size: 3.2rem; font-weight: 700; color: #ffffff !important; margin: 0; text-align: center; width: 100%; }
.hero-tagline { font-size: 1.25rem; color: #888 !important; margin: 0.4rem 0 0; text-align: center; width: 100%; }
.hero-desc { max-width: 520px; margin: 1.2rem auto 0; font-size: 1rem; color: #777 !important; line-height: 1.7; text-align: center; width: 100%; }
.divider { width: 60px; height: 3px; background: linear-gradient(90deg, #E08A1E, #F5B041); border-radius: 2px; margin: 2rem auto; }

.card { background: #161616; border-radius: 16px; padding: 2rem; border: 1px solid #2a2a2a; margin: 0 auto 2rem; width: 100%; box-sizing: border-box; }
.card-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 600; color: #eee !important; margin: 0 0 1.2rem; text-align: center; }
.card-text { font-size: 0.92rem; color: #888 !important; line-height: 1.7; text-align: center; }
.how-steps { display: flex; gap: 1.5rem; justify-content: center; flex-wrap: wrap; }
.how-card { flex: 1; min-width: 160px; max-width: 220px; text-align: center; padding: 0.5rem; }
.step-num { width: 36px; height: 36px; border-radius: 50%; background: linear-gradient(135deg, #E08A1E, #F5B041); color: #fff !important; font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 0.6rem; }
.step-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 600; color: #ddd !important; margin-bottom: 0.3rem; }
.step-text { font-size: 0.82rem; color: #888 !important; line-height: 1.5; }
.stats-row { display: flex; justify-content: center; gap: 2.5rem; padding: 1.5rem; background: #161616; border-radius: 16px; border: 1px solid #2a2a2a; margin: 0 auto 2rem; flex-wrap: wrap; }
.stat { text-align: center; } .stat-num { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: #E08A1E !important; } .stat-label { font-size: 0.72rem; color: #666 !important; text-transform: uppercase; letter-spacing: 0.06em; }
.brands-wrap { text-align: center; margin: 0 auto 2rem; }
.brands-title { font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600; color: #666 !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem; text-align: center; }
.brand-grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
.brand-tag { padding: 6px 14px; border-radius: 8px; font-size: 0.82rem; font-weight: 500; background: #1a1a1a; color: #bba060 !important; border: 1px solid #2a2a2a; }
.upload-label { font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 600; color: #eee !important; text-align: center; margin-bottom: 0.5rem; }
.pred-card { background: #161616; border-radius: 16px; padding: 1.8rem; border: 1px solid #2a2a2a; margin: 1rem 0; box-shadow: 0 2px 12px rgba(0,0,0,0.3); }
.pred-label { font-family: 'Space Grotesk', sans-serif; font-size: 0.72rem; color: #777 !important; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
.pred-car { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; color: #ffffff !important; margin: 0.3rem 0 0.6rem; }
.pred-conf { font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700; background: linear-gradient(135deg, #E08A1E, #F5B041); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.conf-lbl { font-size: 0.9rem; color: #777 !important; margin-left: 0.3rem; }
.low-conf { background: #2a2200; border: 1px solid #5a4a00; border-radius: 10px; padding: 0.8rem 1rem; margin: 0.8rem 0; font-size: 0.85rem; color: #ffc107 !important; }
.also-box { background: #161616; border-radius: 16px; padding: 1.2rem 1.5rem; border: 1px solid #2a2a2a; margin: 1rem 0; }
.also-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 600; color: #E08A1E !important; margin-bottom: 0.6rem; }
.also-sub { font-size: 0.72rem; color: #666 !important; text-transform: uppercase; letter-spacing: 0.05em; margin: 0.6rem 0 0.3rem; }
.also-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.also-tag { padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; background: #1e1e1e; color: #ccc !important; border: 1px solid #333; }
.also-tag-s { background: #1a1500; color: #E08A1E !important; border: 1px solid #3a2a00; }
.conf-guide { display: flex; gap: 0.8rem; justify-content: center; flex-wrap: wrap; margin: 1rem 0; }
.conf-level { text-align: center; padding: 0.6rem 1rem; border-radius: 10px; background: #161616; border: 1px solid #2a2a2a; flex: 1; min-width: 120px; max-width: 200px; }
.conf-range { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1rem; }
.cr-hi { color: #4CAF50 !important; } .cr-mid { color: #FFC107 !important; } .cr-lo { color: #f44336 !important; }
.conf-meaning { font-size: 0.75rem; color: #888 !important; }
.bars-title { font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600; color: #ccc !important; margin: 1.8rem 0 0.8rem; text-align: center; }
.bar-row { display: flex; align-items: center; margin: 6px 0; font-size: 0.82rem; }
.bar-name { width: 200px; text-align: right; padding-right: 12px; color: #999 !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex-shrink: 0; }
.bar-track { flex: 1; height: 14px; background: #222; border-radius: 7px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 7px; }
.bar-top { background: linear-gradient(90deg, #E08A1E, #F5B041); }
.bar-other { background: #444; }
.bar-pct { width: 50px; text-align: right; padding-left: 10px; color: #777 !important; font-variant-numeric: tabular-nums; flex-shrink: 0; }
.hist-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem; font-weight: 600; color: #777 !important; margin: 1.5rem 0 0.6rem; text-align: center; }
.hist-item { display: flex; align-items: center; padding: 8px 12px; background: #161616; border: 1px solid #2a2a2a; border-radius: 10px; margin: 4px 0; gap: 10px; }
.hist-car { flex: 1; font-size: 0.85rem; color: #ddd !important; font-weight: 500; }
.hist-conf { font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 700; color: #E08A1E !important; }
.hist-time { font-size: 0.72rem; color: #555 !important; }
.session-counter { text-align: center; margin: 1rem 0; font-size: 0.82rem; color: #555 !important; }
.counter-num { color: #E08A1E !important; font-family: 'Space Grotesk', sans-serif; font-weight: 700; }
.a-header { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; color: #eee !important; margin: 1.5rem 0 0.5rem; text-align: center; }
.a-sub { font-size: 0.9rem; color: #777 !important; margin-bottom: 1.2rem; text-align: center; }
.cmp-table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }
.cmp-table th { font-family: 'Space Grotesk', sans-serif; text-align: left; padding: 10px 16px; background: linear-gradient(135deg, #E08A1E, #D4740E) !important; color: #fff !important; font-weight: 600; font-size: 0.85rem; }
.cmp-table td { padding: 10px 16px; border-bottom: 1px solid #222; color: #ccc !important; }
/* FIX 3: Per-class accuracy - narrower names */
.ca-row { display: flex; align-items: center; margin: 4px 0; font-size: 0.85rem; }
.ca-name { width: 100px; text-align: right; padding-right: 6px; color: #999 !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex-shrink: 0; font-size: 0.75rem; }
.ca-track { flex: 1; height: 16px; background: #222; border-radius: 8px; overflow: hidden; }
.ca-fill { height: 100%; border-radius: 8px; }
.ca-good { background: linear-gradient(90deg, #2e7d32, #4CAF50); }
.ca-ok { background: linear-gradient(90deg, #f57f17, #FFC107); }
.ca-bad { background: linear-gradient(90deg, #c62828, #f44336); }
.ca-pct { width: 55px; text-align: right; padding-left: 8px; color: #777 !important; font-variant-numeric: tabular-nums; flex-shrink: 0; }
.arch-flow { display: flex; align-items: center; justify-content: center; gap: 0; flex-wrap: wrap; margin: 1rem 0; }
.arch-node { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px; padding: 10px 14px; text-align: center; min-width: 80px; }
.arch-hl { background: #1a1500; border-color: #3a2a00; }
.arch-label { font-family: 'Space Grotesk', sans-serif; font-size: 0.75rem; font-weight: 600; color: #ddd !important; }
.arch-detail { font-size: 0.65rem; color: #777 !important; margin-top: 2px; }
.arch-arrow { color: #E08A1E !important; font-size: 1.2rem; padding: 0 4px; }
.footer { text-align: center; font-size: 0.78rem; color: #555 !important; margin-top: 2rem; padding-bottom: 1.5rem; }
.footer a { color: #E08A1E !important; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}", MODEL_PATH, quiet=False)
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    nc = sd['fc.weight'].shape[0]; m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, nc); m.load_state_dict(sd); m.eval(); return m, nc

@st.cache_data
def load_confusion():
    if not os.path.exists(CONFUSION_PATH):
        try: gdown.download(f"https://drive.google.com/uc?id={GDRIVE_CONFUSION_ID}", CONFUSION_PATH, quiet=True)
        except: return None
    if os.path.exists(CONFUSION_PATH):
        with open(CONFUSION_PATH) as f: return json.load(f)
    return None

@st.cache_data
def get_classes():
    if os.path.exists("class_names.txt"):
        with open("class_names.txt") as f: return [l.strip() for l in f if l.strip()]
    return None

def fmt(n):
    ups = {'BMW','VW','GMC','AMG','GT','SV','SI','SE','XLE','AWD','FWD','RWD','4WD','CRV','HRV','MDX','GLA','GLE','GLC','EQE','RAV4','IX','M3','X3','X4','X5','X7','Z4','K5','Q8','QX50','QX60','QX80','XC90','XT4','GV80','ES','IS','UX','RX','DCT'}
    return ' '.join(p.upper() if p.upper() in ups else p.capitalize() for p in n.split('_'))

def get_brands(cn):
    bm = {'bmw':'BMW','toyota':'Toyota','honda':'Honda','ford':'Ford','jeep':'Jeep','mercedes':'Mercedes-Benz','lexus':'Lexus','hyundai':'Hyundai','kia':'Kia','cadillac':'Cadillac','chevy':'Chevrolet','dodge':'Dodge','vw':'Volkswagen','porsche':'Porsche','land':'Land Rover','subaru':'Subaru','infiniti':'Infiniti','genesis':'Genesis','acura':'Acura','nissan':'Nissan','gmc':'GMC','buick':'Buick','mitsubishi':'Mitsubishi','volvo':'Volvo','audi':'Audi'}
    return sorted({bm[n.split('_')[0]] for n in cn if n.split('_')[0] in bm})

def total_id():
    t = len(SIBLINGS)
    for v in SIBLINGS.values(): t += len(v['trims']) + len(v['siblings'])
    return t

tx = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def predict(model, img, cn):
    with torch.no_grad():
        pr = torch.nn.functional.softmax(model(tx(img).unsqueeze(0))[0], dim=0)
    ps, ix = torch.sort(pr, descending=True)
    return [{'name': cn[ix[i].item()], 'dn': fmt(cn[ix[i].item()]), 'p': ps[i].item()*100} for i in range(len(cn))]

def page_identify(model, nc, cn):
    st.markdown('<div class="upload-label">Upload a headlight photo</div>', unsafe_allow_html=True)
    mode = st.radio("Input mode", ["📁 Upload file", "📷 Take photo"], horizontal=True, label_visibility="collapsed")
    img = None
    if "Take photo" in mode:
        cam = st.camera_input("Take a photo", label_visibility="collapsed")
        if cam: img = Image.open(cam).convert('RGB')
    else:
        up = st.file_uploader("Upload headlight", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if up: img = Image.open(up).convert('RGB')
    if img:
        c1, c2 = st.columns([1, 1.4])
        with c1: st.image(img, use_container_width=True)
        res = predict(model, img, cn); top = res[0]
        st.session_state.pred_count += 1
        st.session_state.history.insert(0, {'car': top['dn'], 'conf': top['p'], 'time': datetime.now().strftime('%H:%M')})
        st.session_state.history = st.session_state.history[:5]
        with c2:
            st.markdown(f'<div class="pred-card"><div class="pred-label">Identified as</div><div class="pred-car">{top["dn"]}</div><div><span class="pred-conf">{top["p"]:.1f}%</span><span class="conf-lbl">confidence</span></div></div>', unsafe_allow_html=True)
            if top['p'] < 60: st.markdown('<div class="low-conf">⚠️ Low confidence — this car may not be in our database, or try a closer crop.</div>', unsafe_allow_html=True)
        sd = SIBLINGS.get(top['name'])
        if sd and (sd['trims'] or sd['siblings']):
            h = '<div class="also-box"><div class="also-title">Other possible cars with same headlights</div>'
            if sd['trims']:
                h += '<div class="also-sub">Same model, different trims</div><div class="also-tags">'
                for t in sd['trims']: h += f'<span class="also-tag">{t}</span>'
                h += '</div>'
            if sd['siblings']:
                h += '<div class="also-sub">Related models</div><div class="also-tags">'
                for s in sd['siblings']: h += f'<span class="also-tag also-tag-s">{s}</span>'
                h += '</div>'
            st.markdown(h + '</div>', unsafe_allow_html=True)
        st.markdown("""<div class="conf-guide"><div class="conf-level"><div class="conf-range cr-hi">90%+</div><div class="conf-meaning">Very likely correct</div></div><div class="conf-level"><div class="conf-range cr-mid">60-90%</div><div class="conf-meaning">Probably right</div></div><div class="conf-level"><div class="conf-range cr-lo">&lt;60%</div><div class="conf-meaning">Uncertain</div></div></div>""", unsafe_allow_html=True)
        st.markdown('<div class="bars-title">Top predictions</div>', unsafe_allow_html=True)
        bh = ""
        for i, r in enumerate(res[:8]):
            fc = "bar-top" if i == 0 else "bar-other"
            bh += f'<div class="bar-row"><div class="bar-name">{r["dn"]}</div><div class="bar-track"><div class="bar-fill {fc}" style="width:{max(r["p"],0.5)}%"></div></div><div class="bar-pct">{r["p"]:.1f}%</div></div>'
        st.markdown(bh, unsafe_allow_html=True)
        with st.expander(f"All {len(cn)} classes"):
            for r in res:
                if r['p'] >= 0.1: st.text(f"  {r['dn']:45s} {r['p']:.1f}%")
        if len(st.session_state.history) > 1:
            hh = '<div class="hist-title">Recent predictions</div>'
            for item in st.session_state.history[1:]:
                hh += f'<div class="hist-item"><div class="hist-car">{item["car"]}</div><div class="hist-conf">{item["conf"]:.1f}%</div><div class="hist-time">{item["time"]}</div></div>'
            st.markdown(hh, unsafe_allow_html=True)
        st.markdown(f'<div class="session-counter"><span class="counter-num">{st.session_state.pred_count}</span> predictions this session</div>', unsafe_allow_html=True)
    else:
        nid = total_id()
        st.markdown(f"""<div class="card"><div class="card-title">How it works</div><div class="how-steps"><div class="how-card"><div class="step-num">1</div><div class="step-title">Photograph</div><div class="step-text">Take a close-up photo of any car's headlight from 2-3 feet away</div></div><div class="how-card"><div class="step-num">2</div><div class="step-title">Upload</div><div class="step-text">Drop the photo above or snap one with your camera</div></div><div class="how-card"><div class="step-num">3</div><div class="step-title">Identify</div><div class="step-text">Get the car's make, model, and year with a confidence score</div></div></div></div>
        <div class="stats-row"><div class="stat"><div class="stat-num">{nid}</div><div class="stat-label">Cars Identifiable</div></div><div class="stat"><div class="stat-num">25</div><div class="stat-label">Brands</div></div><div class="stat"><div class="stat-num">1,540</div><div class="stat-label">Training Photos</div></div><div class="stat"><div class="stat-num">96.1%</div><div class="stat-label">Test Accuracy</div></div></div>""", unsafe_allow_html=True)
        bl = get_brands(cn)
        if bl:
            bt = ''.join(f'<span class="brand-tag">{b}</span>' for b in bl)
            st.markdown(f'<div class="brands-wrap"><div class="brands-title">Supported Brands</div><div class="brand-grid">{bt}</div></div>', unsafe_allow_html=True)
        st.markdown("""<div class="card"><div class="card-title">About this project</div><div class="card-text">CarID uses a convolutional neural network (ResNet18) to identify cars from close-up headlight photographs. The model was trained on 1,540 original photos collected by hand at dealerships and parking lots — no scraped or AI-generated images.<br><br>Every car has a unique headlight signature: the shape of the LED elements, the curve of the housing, and the pattern of the daytime running lights. The model learns these patterns through transfer learning from ImageNet.<br><br>When the model identifies a car, it also shows other trims and related models that share the same headlight design — expanding from 62 trained classes to over 340 identifiable vehicles.</div></div>""", unsafe_allow_html=True)

def page_analysis(cd):
    st.markdown('<div class="a-header">Model Analysis</div><div class="a-sub">How the model performs across car models</div>', unsafe_allow_html=True)
    st.markdown("""<div class="a-header">🧠 Model Architecture</div><div class="a-sub">How a headlight photo becomes a prediction</div><div class="arch-flow"><div class="arch-node"><div class="arch-label">📷 Photo</div><div class="arch-detail">Input</div></div><div class="arch-arrow">→</div><div class="arch-node"><div class="arch-label">Resize</div><div class="arch-detail">224×224</div></div><div class="arch-arrow">→</div><div class="arch-node arch-hl"><div class="arch-label">ResNet18</div><div class="arch-detail">18 layers, 11M params</div></div><div class="arch-arrow">→</div><div class="arch-node"><div class="arch-label">62 outputs</div><div class="arch-detail">one per class</div></div><div class="arch-arrow">→</div><div class="arch-node"><div class="arch-label">Softmax</div><div class="arch-detail">probabilities</div></div><div class="arch-arrow">→</div><div class="arch-node arch-hl"><div class="arch-label">Trim lookup</div><div class="arch-detail">→ 342 cars</div></div></div>""", unsafe_allow_html=True)
    if cd is None: st.warning("Confusion matrix data not available."); return
    st.markdown("""<div class="a-header">📈 Project Evolution</div><table class="cmp-table"><tr><th>Metric</th><th>POC (Jul 2026)</th><th>Final (Nov 2026)</th></tr><tr><td>Car models</td><td>5</td><td>62</td></tr><tr><td>Total identifiable</td><td>5</td><td>342</td></tr><tr><td>Training images</td><td>98</td><td>1,540</td></tr><tr><td>Data source</td><td>Phone photos</td><td>Dealerships + parking lots</td></tr><tr><td>Architecture</td><td>Custom CNN + ResNet18</td><td>ResNet18 transfer learning</td></tr><tr><td>Split</td><td>80/20</td><td>70/15/15</td></tr><tr><td>Test accuracy</td><td>90% (20 images)</td><td>96.1% (231 images)</td></tr><tr><td>Deployment</td><td>Colab only</td><td>Live web app</td></tr></table>""", unsafe_allow_html=True)
    st.markdown('<div class="a-header">📊 Per-Class Accuracy</div>', unsafe_allow_html=True)
    pc = cd.get('per_class', [])
    hd = sorted([p for p in pc if p['total'] > 0], key=lambda x: x['accuracy'], reverse=True)
    bh = ""
    for p in hd:
        fc = "ca-good" if p['accuracy'] >= 80 else ("ca-ok" if p['accuracy'] >= 50 else "ca-bad")
        bh += f'<div class="ca-row"><div class="ca-name">{fmt(p["name"])}</div><div class="ca-track"><div class="ca-fill {fc}" style="width:{max(p["accuracy"],2)}%"></div></div><div class="ca-pct">{p["accuracy"]}%</div></div>'
    st.markdown(bh, unsafe_allow_html=True)
    st.markdown("""<div class="card" style="margin-top:2rem;"><div class="card-title">Key Insight</div><div class="card-text">The biggest confusion is between the BMW X5 and Toyota 4Runner — a pattern that persisted from the 5-class proof of concept through to the 62-class final model. Solving this would require more training images of both models from varied angles and lighting conditions.</div></div>""", unsafe_allow_html=True)

def main():
    st.markdown(f"""<div class="hero-wrap"><img class="logo-img" src="data:image/png;base64,{LOGO_B64}" alt="CarID"/><div class="hero-name">CarID</div><div class="hero-tagline">Identify any car from its headlights</div><div class="hero-desc">A deep learning model trained on 1,500+ real-world headlight photographs to recognize car make, model, and year from a single close-up photo.</div></div><div class="divider"></div>""", unsafe_allow_html=True)
    try: model, nc = load_model()
    except Exception as e: st.error(f"Failed to load model: {e}"); st.stop()
    cn = get_classes()
    if cn is None: cn = [f"class_{i}" for i in range(nc)]
    cd = load_confusion()
    t1, t2 = st.tabs(["🚗 Identify", "📊 Analysis"])
    with t1: page_identify(model, nc, cn)
    with t2: page_analysis(cd)
    st.markdown('<div class="footer">Built by Rus Surmin · PyTorch · ResNet18 · 2026<br><a href="https://github.com/rslnxz">GitHub</a></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
