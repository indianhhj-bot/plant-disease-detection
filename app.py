
import json
import html
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: #f5faf5;
    color: #203b2b;
}

[data-testid="stHeader"] {
    background: #f5faf5;
}

[data-testid="stSidebar"] {
    background: #e5f1e6;
}

[data-testid="stSidebar"] * {
    color: #203b2b !important;
}

.hero {
    padding: 30px 32px;
    border-radius: 20px;
    background: linear-gradient(120deg, #174c32, #318451);
    color: #ffffff;
    margin-bottom: 24px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 2.3rem;
}

.hero p {
    color: #f0fff3 !important;
    font-size: 1.05rem;
}

.info-card {
    padding: 20px;
    background: #edf7ee;
    border: 1px solid #cfe3d1;
    border-radius: 16px;
    margin: 12px 0;
}

.info-card h3 {
    color: #205b39 !important;
}

div.stButton > button {
    background: #237547;
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 650;
}

div.stButton > button:hover {
    background: #185b36;
    color: #ffffff !important;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px solid #c9ddcc;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] small {
    color: #203b2b !important;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    padding: 16px;
    border: 1px solid #dceade;
    border-radius: 14px;
}

.small-note {
    color: #506b58 !important;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)
