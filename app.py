
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="PlantCare AI | Plant Health Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CUSTOM GREEN THEME ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f4faf4 0%, #ffffff 48%);
        color: #193b2a;
    }

    [data-testid="stHeader"] {
        background: rgba(244, 250, 244, 0.85);
    }

    [data-testid="stSidebar"] {
        background: #e8f3e9;
        border-right: 1px solid #d4e6d7;
    }

    [data-testid="stSidebar"] * {
        color: #193b2a;
    }

    .hero {
        padding: 30px 32px;
        border-radius: 22px;
        background: linear-gradient(120deg, #174c32, #2e8050);
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 2.35rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e4f4e8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .section-title {
        color: #205b39;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 8px 0 14px 0;
    }

    .result-card {
        padding: 22px;
        background: #edf7ee;
        border: 1px solid #cfe6d2;
        border-radius: 18px;
        margin: 10px 0 18px 0;
    }

    .result-label {
        color: #477451;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }

    .result-name {
        color: #174c32;
        font-size: 1.5rem;
        font-weight: 750;
        overflow-wrap: anywhere;
    }

    .small-note {
        color: #55715e;
        font-size: 0.88rem;
    }

    div.stButton > button {
        background: #237547;
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 46px;
        font-weight: 650;
        width: 100%;
    }

    div.stButton > button:hover {
        background: #185b36;
        color: white;
        border: none;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 16px;
        border: 1px solid #dceade;
        border-radius: 14px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_model.keras")
    with open("class_names.json", "r", encoding="utf-8") as file:
        class_names = json.load(file)
    return model, class_names

def format_class_name(name):
    return (
        name.replace("___", " — ")
            .replace("__", " ")
            .replace("_", " ")
            .strip()
    )

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 🌿 PlantCare AI")
    st.markdown("### Your plant health assistant")
    st.divider()

    st.markdown("**How to use**")
    st.markdown(
        "1. Choose a clear leaf photo.\n"
        "2. Preview the image.\n"
        "3. Click **Analyze leaf**.\n"
        "4. Review the AI prediction."
    )

    st.divider()
    st.markdown("**Photo tips**")
    st.write("• Use good lighting.")
    st.write("• Keep the leaf in focus.")
    st.write("• Avoid blurry or very dark photos.")
    st.divider()
    st.caption("Student project · Deep learning demonstration")

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🌱 PlantCare AI</h1>
    <p>Understand your plant leaf with an AI-powered image prediction.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">🔎 Analyze a plant leaf</div>',
    unsafe_allow_html=True,
)
st.write(
    "Upload a leaf photograph to see which plant-health category "
    "your trained model predicts."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG.",
)

# ---------------- IMAGE AND PREDICTION ----------------
if uploaded_file is not None:
    try:
        image = ImageOps.exif_transpose(
            Image.open(uploaded_file)
        ).convert("RGB")

        image_col, result_col = st.columns([1, 1], gap="large")

        with image_col:
            st.markdown(
                '<div class="section-title">📷 Your leaf photo</div>',
                unsafe_allow_html=True,
            )
            st.image(image, use_container_width=True)
            st.caption(
                f"Image dimensions: {image.width} × {image.height} pixels"
            )

        with result_col:
            st.markdown(
                '<div class="section-title">🌿 AI analysis</div>',
                unsafe_allow_html=True,
            )
            st.write(
                "When you're ready, analyze the photo to display "
                "the model's prediction."
            )

            analyze = st.button(
                "🌱 Analyze leaf",
                type="primary",
                use_container_width=True,
            )

            if analyze:
                try:
                    with st.spinner("Analyzing your leaf photo..."):
                        model, class_names = load_model()

                        resized_image = image.resize((160, 160))
                        image_array = np.asarray(
                            resized_image, dtype=np.float32
                        )[None, ...]

                        probabilities = model.predict(
                            image_array, verbose=0
                        )[0]

                        if len(probabilities) != len(class_names):
                            raise ValueError(
                                "The model output does not match "
                                "the saved class labels."
                            )

                        best_index = int(np.argmax(probabilities))
                        best_score = float(probabilities[best_index])
                        predicted_name = format_class_name(
                            class_names[best_index]
                        )

                        top_indices = np.argsort(probabilities)[::-1][:3]

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">PREDICTED CATEGORY</div>
                            <div class="result-name">{predicted_name}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.metric(
                        "Model score",
                        f"{best_score:.2%}",
                    )
                    st.progress(
                        max(0.0, min(1.0, best_score)),
                        text="Top prediction score",
                    )

                    with st.expander("View the top 3 model predictions"):
                        for rank, idx in enumerate(top_indices, start=1):
                            st.write(
                                f"**{rank}.** "
                                f"{format_class_name(class_names[int(idx)])}"
                            )
                            st.progress(
                                max(0.0, min(1.0, float(probabilities[idx]))),
                                text=f"{float(probabilities[idx]):.2%}",
                            )

                    st.info(
                        "This is an experimental model prediction, not a "
                        "confirmed plant diagnosis. Model scores are not "
                        "guaranteed probabilities of correctness. Check the "
                        "leaf and seek expert advice if needed."
                    )

                except Exception as error:
                    st.error(
                        "The image could not be analyzed. "
                        "Please check the app logs for technical details."
                    )
                    st.exception(error)

    except Exception:
        st.error(
            "We couldn't open this image. Please upload a valid "
            "JPG, JPEG or PNG photograph."
        )

else:
    st.markdown("""
    <div class="result-card">
        <div class="result-name">Your plant-care journey starts here 🌿</div>
        <p class="small-note">
            Upload a clear photo of a plant leaf to begin.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    '<p class="small-note" style="text-align:center;">'
    'PlantCare AI · Educational demonstration using deep learning'
    '</p>',
    unsafe_allow_html=True,
)
