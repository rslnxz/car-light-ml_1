import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import os
import json
import gdown
from datetime import datetime

st.set_page_config(page_title="CarID — Identify Cars from Headlights", page_icon="🚗", layout="centered")

# Session state
if 'history' not in st.session_state: st.session_state.history = []
if 'pred_count' not in st.session_state: st.session_state.pred_count = 0

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
[data-testid="stCameraInputButton"] button { color: #e0e0e0 !important; background: #1a1a1a !important; border-color: #333 !important; }
button { color: #e0e0e0 !important; background: #222 !important; border-color: #444 !important; }
button:hover { background: #333 !important; color: #fff !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; gap: 0; justify-content: center !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #888 !important; border-bottom: 2px solid transparent !important; padding: 10px 20px !important; }
.stTabs [aria-selected="true"] { color: #E08A1E !important; border-bottom: 2px solid #E08A1E !important; }
.stTabs [data-baseweb="tab"]:hover { color: #ccc !important; }
[data-testid="stExpander"] { background: #1a1a1a !important; border-color: #333 !important; border-radius: 12px !important; }
[data-testid="stExpander"] * { color: #ccc !important; }
::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-track { background: #111; } ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
/* Radio buttons for input mode */
.stRadio > div { display: flex !important; justify-content: center !important; gap: 1rem !important; }
.stRadio label { color: #ccc !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #ccc !important; }

.hero-wrap { width: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 2.5rem 0 1rem; }
.logo-img { width: 160px; height: 160px; margin: 0 auto 1.5rem; filter: invert(1); display: block; }
.hero-name { font-family: 'Space Grotesk', sans-serif; font-size: 3.2rem; font-weight: 700; letter-spacing: -0.02em; color: #ffffff !important; margin: 0; text-align: center; width: 100%; }
.hero-tagline { font-size: 1.25rem; color: #888 !important; margin: 0.4rem 0 0; text-align: center; width: 100%; }
.hero-desc { max-width: 520px; margin: 1.2rem auto 0; font-size: 1rem; color: #777 !important; line-height: 1.7; text-align: center; width: 100%; }
.divider { width: 60px; height: 3px; background: linear-gradient(90deg, #E08A1E, #F5B041); border-radius: 2px; margin: 2rem auto; }

.card { background: #161616; border-radius: 16px; padding: 2rem; border: 1px solid #2a2a2a; margin: 0 auto 2rem; width: 100%; box-sizing: border-box; }
.card-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 600; color: #eee !important; margin: 0 0 1.2rem; text-align: center; width: 100%; }
.card-text { font-size: 0.92rem; color: #888 !important; line-height: 1.7; text-align: center; width: 100%; }

.how-steps { display: flex; gap: 1.5rem; justify-content: center; flex-wrap: wrap; width: 100%; }
.how-card { flex: 1; min-width: 160px; max-width: 220px; text-align: center; padding: 0.5rem; }
.step-num { width: 36px; height: 36px; border-radius: 50%; background: linear-gradient(135deg, #E08A1E, #F5B041); color: #fff !important; font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 0.6rem; }
.step-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 600; color: #ddd !important; margin-bottom: 0.3rem; }
.step-text { font-size: 0.82rem; color: #888 !important; line-height: 1.5; }

.stats-row { display: flex; justify-content: center; gap: 2.5rem; padding: 1.5rem; background: #161616; border-radius: 16px; border: 1px solid #2a2a2a; margin: 0 auto 2rem; width: 100%; box-sizing: border-box; flex-wrap: wrap; }
.stat { text-align: center; } .stat-num { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: #E08A1E !important; } .stat-label { font-size: 0.72rem; color: #666 !important; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.2rem; }

.brands-wrap { text-align: center; margin: 0 auto 2rem; width: 100%; }
.brands-title { font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600; color: #666 !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem; text-align: center; width: 100%; }
.brand-grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; width: 100%; }
.brand-tag { padding: 6px 14px; border-radius: 8px; font-size: 0.82rem; font-weight: 500; background: #1a1a1a; color: #bba060 !important; border: 1px solid #2a2a2a; }

.upload-wrap { text-align: center; margin: 0 auto 1rem; width: 100%; }
.upload-label { font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 600; color: #eee !important; margin-bottom: 0.5rem; text-align: center; width: 100%; }

.pred-card { background: #161616; border-radius: 16px; padding: 1.8rem; border: 1px solid #2a2a2a; margin: 1rem 0; box-shadow: 0 2px 12px rgba(0,0,0,0.3); }
.pred-label { font-family: 'Space Grotesk', sans-serif; font-size: 0.72rem; color: #777 !important; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
.pred-car { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; color: #ffffff !important; margin: 0.3rem 0 0.6rem; }
.pred-confidence { font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700; background: linear-gradient(135deg, #E08A1E, #F5B041); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.conf-label { font-size: 0.9rem; color: #777 !important; margin-left: 0.3rem; }
.low-conf { background: #2a2200; border: 1px solid #5a4a00; border-radius: 10px; padding: 0.8rem 1rem; margin: 0.8rem 0; font-size: 0.85rem; color: #ffc107 !important; }

.also-box { background: #161616; border-radius: 16px; padding: 1.2rem 1.5rem; border: 1px solid #2a2a2a; margin: 1rem 0; }
.also-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 600; color: #E08A1E !important; margin-bottom: 0.6rem; }
.also-sub { font-size: 0.72rem; color: #666 !important; text-transform: uppercase; letter-spacing: 0.05em; margin: 0.6rem 0 0.3rem; }
.also-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.also-tag { padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; background: #1e1e1e; color: #ccc !important; border: 1px solid #333; }
.also-tag-sib { background: #1a1500; color: #E08A1E !important; border: 1px solid #3a2a00; }

.bars-title { font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600; color: #ccc !important; margin: 1.8rem 0 0.8rem; text-align: center; width: 100%; }
.bar-row { display: flex; align-items: center; margin: 6px 0; font-size: 0.82rem; width: 100%; }
.bar-name { width: 200px; text-align: right; padding-right: 12px; color: #999 !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.82rem; flex-shrink: 0; }
.bar-track { flex: 1; height: 14px; background: #222; border-radius: 7px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 7px; } .bar-fill-top { background: linear-gradient(90deg, #E08A1E, #F5B041); } .bar-fill-other { background: #444; }
.bar-pct { width: 50px; text-align: right; padding-left: 10px; color: #777 !important; font-variant-numeric: tabular-nums; font-size: 0.82rem; flex-shrink: 0; }

/* Confidence guide */
.conf-guide { display: flex; gap: 0.8rem; justify-content: center; flex-wrap: wrap; margin: 1rem 0; width: 100%; }
.conf-level { text-align: center; padding: 0.6rem 1rem; border-radius: 10px; background: #161616; border: 1px solid #2a2a2a; flex: 1; min-width: 120px; max-width: 200px; }
.conf-range { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1rem; margin-bottom: 0.2rem; }
.conf-range-high { color: #4CAF50 !important; } .conf-range-mid { color: #FFC107 !important; } .conf-range-low { color: #f44336 !important; }
.conf-meaning { font-size: 0.75rem; color: #888 !important; }

/* History */
.hist-wrap { margin: 1.5rem 0; width: 100%; }
.hist-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem; font-weight: 600; color: #777 !important; margin-bottom: 0.6rem; text-align: center; }
.hist-item { display: flex; align-items: center; padding: 8px 12px; background: #161616; border: 1px solid #2a2a2a; border-radius: 10px; margin: 4px 0; gap: 10px; }
.hist-car { flex: 1; font-size: 0.85rem; color: #ddd !important; font-weight: 500; }
.hist-conf { font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 700; color: #E08A1E !important; }
.hist-time { font-size: 0.72rem; color: #555 !important; }

/* Session counter */
.session-counter { text-align: center; margin: 1rem 0; font-size: 0.82rem; color: #555 !important; }
.counter-num { color: #E08A1E !important; font-family: 'Space Grotesk', sans-serif; font-weight: 700; }

/* Architecture diagram */
.arch-flow { display: flex; align-items: center; justify-content: center; gap: 0; flex-wrap: wrap; margin: 1rem 0; width: 100%; }
.arch-node { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px; padding: 10px 14px; text-align: center; min-width: 90px; }
.arch-node-highlight { background: #1a1500; border-color: #3a2a00; }
.arch-label { font-family: 'Space Grotesk', sans-serif; font-size: 0.75rem; font-weight: 600; color: #ddd !important; }
.arch-detail { font-size: 0.65rem; color: #777 !important; margin-top: 2px; }
.arch-arrow { color: #E08A1E !important; font-size: 1.2rem; padding: 0 4px; flex-shrink: 0; }

/* Analysis */
.a-header { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; color: #eee !important; margin: 1.5rem 0 0.5rem; text-align: center; width: 100%; }
.a-sub { font-size: 0.9rem; color: #777 !important; margin-bottom: 1.2rem; text-align: center; width: 100%; }
.cmp-table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }
.cmp-table th { font-family: 'Space Grotesk', sans-serif; text-align: left; padding: 10px 16px; background: linear-gradient(135deg, #E08A1E, #D4740E) !important; color: #fff !important; font-weight: 600; font-size: 0.85rem; }
.cmp-table td { padding: 10px 16px; border-bottom: 1px solid #222; color: #ccc !important; }
.cmp-table tr:hover td { background: #1a1a1a; }

.ca-row { display: flex; align-items: center; margin: 4px 0; font-size: 0.85rem; width: 100%; }
.ca-name { width: 130px; text-align: right; padding-right: 8px; color: #999 !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex-shrink: 0; font-size: 0.78rem; }
.ca-track { flex: 1; height: 16px; background: #222; border-radius: 8px; overflow: hidden; }
.ca-fill { height: 100%; border-radius: 8px; }
.ca-fill-good { background: linear-gradient(90deg, #2e7d32, #4CAF50); }
.ca-fill-ok { background: linear-gradient(90deg, #f57f17, #FFC107); }
.ca-fill-bad { background: linear-gradient(90deg, #c62828, #f44336); }
.ca-pct { width: 60px; text-align: right; padding-left: 10px; color: #777 !important; font-variant-numeric: tabular-nums; flex-shrink: 0; }

.footer { text-align: center; font-size: 0.78rem; color: #555 !important; margin-top: 2rem; padding-bottom: 1.5rem; width: 100%; }
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
        with st.spinner("Downloading model..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}", MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH): st.error("Failed to download model."); st.stop()
    sd = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
    nc = sd['fc.weight'].shape[0]
    m = torchvision.models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, nc)
    m.load_state_dict(sd); m.eval(); return m, nc

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
        with open("class_names.txt", "r") as f: return [l.strip() for l in f if l.strip()]
    return None

def fmt(name):
    ups = {'BMW','VW','GMC','AMG','GT','SV','SI','SE','XLE','AWD','FWD','RWD','4WD','CRV','HRV','MDX','GLA','GLE','GLC','EQE','RAV4','IX','M3','X3','X4','X5','X7','Z4','K5','Q8','QX50','QX60','QX80','XC90','XT4','GV80','ES','IS','UX','RX','DCT'}
    return ' '.join(p.upper() if p.upper() in ups else p.capitalize() for p in name.split('_'))

def get_brands(cn):
    bm = {'bmw':'BMW','toyota':'Toyota','honda':'Honda','ford':'Ford','jeep':'Jeep','mercedes':'Mercedes-Benz','lexus':'Lexus','hyundai':'Hyundai','kia':'Kia','cadillac':'Cadillac','chevy':'Chevrolet','dodge':'Dodge','vw':'Volkswagen','porsche':'Porsche','land':'Land Rover','subaru':'Subaru','infiniti':'Infiniti','genesis':'Genesis','acura':'Acura','nissan':'Nissan','gmc':'GMC','buick':'Buick','mitsubishi':'Mitsubishi','volvo':'Volvo','audi':'Audi'}
    return sorted({bm[n.split('_')[0]] for n in cn if n.split('_')[0] in bm})

def count_id():
    t = len(HEADLIGHT_SIBLINGS)
    for v in HEADLIGHT_SIBLINGS.values(): t += len(v['trims']) + len(v['siblings'])
    return t

tx = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def predict(model, image, cn):
    inp = tx(image).unsqueeze(0)
    with torch.no_grad():
        out = model(inp)
        pr = torch.nn.functional.softmax(out[0], dim=0)
    ps, ix = torch.sort(pr, descending=True)
    return [{'name': cn[ix[i].item()], 'display_name': fmt(cn[ix[i].item()]), 'prob': ps[i].item()*100} for i in range(len(cn))]

LOGO_B64 = "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"

def page_identify(model, nc, cn):
    st.markdown('<div class="upload-wrap"><div class="upload-label">Upload a headlight photo</div></div>', unsafe_allow_html=True)

    mode = st.radio("Input method", ["📁 Upload file", "📷 Take photo"], horizontal=True, label_visibility="collapsed")
    img = None
    if mode == "📷 Take photo":
        cam = st.camera_input("Take a photo", label_visibility="collapsed")
        if cam: img = Image.open(cam).convert('RGB')
    else:
        up = st.file_uploader("upload", type=["jpg","jpeg","png"], help="Close-up headlight photos work best", label_visibility="collapsed")
        if up: img = Image.open(up).convert('RGB')

    if img:
        c1, c2 = st.columns([1, 1.4])
        with c1: st.image(img, use_container_width=True)
        res = predict(model, img, cn); top = res[0]
        st.session_state.pred_count += 1
        st.session_state.history.insert(0, {'car': top['display_name'], 'conf': top['prob'], 'time': datetime.now().strftime('%H:%M')})
        if len(st.session_state.history) > 5: st.session_state.history = st.session_state.history[:5]

        with c2:
            st.markdown(f'<div class="pred-card"><div class="pred-label">Identified as</div><div class="pred-car">{top["display_name"]}</div><div><span class="pred-confidence">{top["prob"]:.1f}%</span><span class="conf-label">confidence</span></div></div>', unsafe_allow_html=True)
            if top['prob'] < 60:
                st.markdown('<div class="low-conf">⚠️ Low confidence — this car may not be in our database, or try a closer crop of the headlight.</div>', unsafe_allow_html=True)

        sd = HEADLIGHT_SIBLINGS.get(top['name'])
        if sd and (sd['trims'] or sd['siblings']):
            h = '<div class="also-box"><div class="also-title">Other possible cars with same headlights</div>'
            if sd['trims']:
                h += '<div class="also-sub">Same model, different trims</div><div class="also-tags">'
                for t in sd['trims']: h += f'<span class="also-tag">{t}</span>'
                h += '</div>'
            if sd['siblings']:
                h += '<div class="also-sub">Related models</div><div class="also-tags">'
                for s in sd['siblings']: h += f'<span class="also-tag also-tag-sib">{s}</span>'
                h += '</div>'
            st.markdown(h + '</div>', unsafe_allow_html=True)

        # Confidence guide
        st.markdown("""
        <div class="conf-guide">
            <div class="conf-level"><div class="conf-range conf-range-high">90%+</div><div class="conf-meaning">Very likely correct</div></div>
            <div class="conf-level"><div class="conf-range conf-range-mid">60–90%</div><div class="conf-meaning">Probably right, verify</div></div>
            <div class="conf-level"><div class="conf-range conf-range-low">Below 60%</div><div class="conf-meaning">Uncertain</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="bars-title">Top predictions</div>', unsafe_allow_html=True)
        bh = ""
        for i, r in enumerate(res[:8]):
            fc = "bar-fill-top" if i == 0 else "bar-fill-other"
            bh += f'<div class="bar-row"><div class="bar-name">{r["display_name"]}</div><div class="bar-track"><div class="bar-fill {fc}" style="width:{max(r["prob"],0.5)}%"></div></div><div class="bar-pct">{r["prob"]:.1f}%</div></div>'
        st.markdown(bh, unsafe_allow_html=True)

        with st.expander(f"All {len(cn)} classes"):
            for r in res:
                if r['prob'] >= 0.1: st.text(f"  {r['display_name']:45s} {r['prob']:.1f}%")

        # History
        if len(st.session_state.history) > 1:
            hh = '<div class="hist-wrap"><div class="hist-title">Recent predictions</div>'
            for item in st.session_state.history[1:]:
                hh += f'<div class="hist-item"><div class="hist-car">{item["car"]}</div><div class="hist-conf">{item["conf"]:.1f}%</div><div class="hist-time">{item["time"]}</div></div>'
            st.markdown(hh + '</div>', unsafe_allow_html=True)

        # Session counter
        st.markdown(f'<div class="session-counter"><span class="counter-num">{st.session_state.pred_count}</span> predictions this session</div>', unsafe_allow_html=True)

    else:
        nid = count_id()
        st.markdown(f"""
        <div class="card">
            <div class="card-title">How it works</div>
            <div class="how-steps">
                <div class="how-card"><div class="step-num">1</div><div class="step-title">Photograph</div><div class="step-text">Take a close-up photo of any car's headlight from 2-3 feet away</div></div>
                <div class="how-card"><div class="step-num">2</div><div class="step-title">Upload</div><div class="step-text">Drop the photo above or use your camera to snap one live</div></div>
                <div class="how-card"><div class="step-num">3</div><div class="step-title">Identify</div><div class="step-text">Get the car's make, model, and year with a confidence score</div></div>
            </div>
        </div>
        <div class="stats-row">
            <div class="stat"><div class="stat-num">{nid}</div><div class="stat-label">Cars Identifiable</div></div>
            <div class="stat"><div class="stat-num">25</div><div class="stat-label">Brands</div></div>
            <div class="stat"><div class="stat-num">1,540</div><div class="stat-label">Training Photos</div></div>
            <div class="stat"><div class="stat-num">96.1%</div><div class="stat-label">Test Accuracy</div></div>
        </div>
        """, unsafe_allow_html=True)
        brands = get_brands(cn)
        if brands:
            bt = ''.join([f'<span class="brand-tag">{b}</span>' for b in brands])
            st.markdown(f'<div class="brands-wrap"><div class="brands-title">Supported Brands</div><div class="brand-grid">{bt}</div></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <div class="card-title">About this project</div>
            <div class="card-text">
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
                the same headlight design — expanding coverage from 62 trained classes to over 340
                identifiable vehicles.
            </div>
        </div>
        """, unsafe_allow_html=True)

def page_analysis(cd):
    st.markdown('<div class="a-header">Model Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="a-sub">A deeper look at how the model performs</div>', unsafe_allow_html=True)

    # Architecture diagram
    st.markdown('<div class="a-header">🧠 Model Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="a-sub">How a headlight photo becomes a car prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="arch-flow">
        <div class="arch-node"><div class="arch-label">📷 Photo</div><div class="arch-detail">Input image</div></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node"><div class="arch-label">Resize</div><div class="arch-detail">224 × 224 px</div></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node arch-node-highlight"><div class="arch-label">ResNet18</div><div class="arch-detail">18 layers, 11M params</div></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node"><div class="arch-label">62 outputs</div><div class="arch-detail">one per class</div></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node"><div class="arch-label">Softmax</div><div class="arch-detail">% probabilities</div></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node arch-node-highlight"><div class="arch-label">Trim lookup</div><div class="arch-detail">→ 342 cars</div></div>
    </div>
    """, unsafe_allow_html=True)

    if cd is None: st.warning("Confusion matrix data not available."); return

    # Evolution table
    st.markdown('<div class="a-header">📈 Project Evolution</div>', unsafe_allow_html=True)
    st.markdown('<div class="a-sub">From proof of concept to final version</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="cmp-table">
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

    # Per-class accuracy
    st.markdown('<div class="a-header">📊 Per-Class Accuracy</div>', unsafe_allow_html=True)
    st.markdown('<div class="a-sub">How well the model performs on each car model</div>', unsafe_allow_html=True)
    pc = cd.get('per_class', [])
    hd = sorted([p for p in pc if p['total'] > 0], key=lambda x: x['accuracy'], reverse=True)
    bh = ""
    for p in hd:
        fc = "ca-fill-good" if p['accuracy'] >= 80 else ("ca-fill-ok" if p['accuracy'] >= 50 else "ca-fill-bad")
        bh += f'<div class="ca-row"><div class="ca-name">{fmt(p["name"])}</div><div class="ca-track"><div class="ca-fill {fc}" style="width:{max(p["accuracy"],2)}%"></div></div><div class="ca-pct">{p["accuracy"]}%</div></div>'
    st.markdown(bh, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:2rem;">
        <div class="card-title">Key Insight</div>
        <div class="card-text">
            The biggest confusion is between the BMW X5 and Toyota 4Runner — a pattern that persisted
            from the 5-class proof of concept through to the 62-class final model. Solving this would
            require more training images of both models from varied angles and lighting conditions.
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown(f"""
    <div class="hero-wrap">
        <img class="logo-img" src="data:image/png;base64,{LOGO_B64}" alt="CarID"/>
        <div class="hero-name">CarID</div>
        <div class="hero-tagline">Identify any car from its headlights</div>
        <div class="hero-desc">A deep learning model trained on 1,500+ real-world headlight photographs to recognize car make, model, and year from a single close-up photo.</div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)
    try: model, nc = load_model()
    except Exception as e: st.error(f"Failed to load model: {e}"); st.stop()
    cn = get_class_names()
    if cn is None: cn = [f"class_{i}" for i in range(nc)]
    cd = load_confusion_data()
    t1, t2 = st.tabs(["🚗 Identify", "📊 Analysis"])
    with t1: page_identify(model, nc, cn)
    with t2: page_analysis(cd)
    st.markdown('<div class="footer">Built by Rus Surmin · PyTorch · ResNet18 · 2026<br><a href="https://github.com/rslnxz" target="_blank">GitHub</a></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
