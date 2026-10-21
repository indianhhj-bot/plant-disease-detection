
import json
import html
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ============== PAGE SETTINGS ==============
st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============== PROFESSIONAL GREEN THEME ==============
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
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(120deg, #174c32, #318451);
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white !important;
    font-size: 2.4rem;
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
[data-testid="stFileUploader"] {
    background: white;
    border: 1px solid #c9ddcc;
    border-radius: 12px;
    padding: 12px;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] small {
    color: #203b2b !important;
}
div.stButton > button {
    background: #237547;
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 650;
}
div.stButton > button:hover {
    background: #185b36;
    color: white !important;
}
div[data-testid="stMetric"] {
    background: white;
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


# ============== LOAD TRAINED MODEL ==============
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_model.keras")
    with open("class_names.json", "r", encoding="utf-8") as file:
        class_names = json.load(file)
    return model, class_names


def format_name(name):
    return (
        name.replace("___", " — ")
        .replace("__", " ")
        .replace("_", " ")
        .strip()
    )


# ============== DISEASE INFORMATION ==============
DISEASE_INFO = {
    "healthy": {
        "description": "The model classified this leaf as belonging to a healthy category.",
        "symptoms": [
            "No obvious disease pattern was identified in this photo.",
            "Leaves may show their usual color and shape."
        ],
        "causes": [
            "The model did not identify a recognizable disease pattern."
        ],
        "care": [
            "Continue suitable watering and plant care.",
            "Inspect the whole plant regularly; one leaf photo cannot confirm overall health."
        ],
    },
    "bacterial_spot": {
        "description": "Bacterial spot can affect leaves and sometimes stems or fruit.",
        "symptoms": [
            "Small dark or water-soaked-looking spots.",
            "Yellowing around damaged areas.",
            "Spots may merge as damage progresses."
        ],
        "causes": [
            "Bacterial pathogens; the exact species depends on the plant.",
            "Splashing water and wet foliage can contribute to spread."
        ],
        "care": [
            "Avoid overhead watering where possible.",
            "Clean gardening tools and avoid handling wet plants.",
            "Seek local agricultural advice if symptoms spread."
        ],
    },
    "early_blight": {
        "description": "Early blight is a disease commonly associated with tomato and potato plants.",
        "symptoms": [
            "Brown spots that may have concentric rings.",
            "Yellowing around leaf spots.",
            "Older leaves may be affected first."
        ],
        "causes": [
            "Fungal pathogens associated with early blight.",
            "Infected plant debris and prolonged leaf wetness can contribute."
        ],
        "care": [
            "Water near the soil rather than over the leaves.",
            "Improve airflow around plants.",
            "Remove badly affected leaves when appropriate and seek crop-specific advice."
        ],
    },
    "late_blight": {
        "description": "Late blight is a potentially serious disease of potato and tomato that can progress rapidly.",
        "symptoms": [
            "Irregular dark or water-soaked-looking patches.",
            "Pale growth may appear near lesions in humid conditions.",
            "Rapid leaf or stem damage may occur."
        ],
        "causes": [
            "The oomycete pathogen Phytophthora infestans.",
            "Cool, humid conditions and prolonged leaf wetness can favor outbreaks."
        ],
        "care": [
            "Seek prompt agricultural advice if late blight is suspected.",
            "Avoid moving wet foliage or contaminated tools between plants.",
            "Follow local guidance for handling infected plant material and suitable treatments."
        ],
    },
    "septoria": {
        "description": "Septoria leaf spot is a fungal disease often seen on tomato leaves.",
        "symptoms": [
            "Small, round spots with pale or gray centers.",
            "Dark margins and tiny dark specks may appear.",
            "Affected leaves may yellow and dry."
        ],
        "causes": [
            "A Septoria fungal pathogen.",
            "Rain splash, wet foliage, and infected plant debris can contribute to spread."
        ],
        "care": [
            "Avoid overhead watering and improve airflow.",
            "Remove affected lower leaves where appropriate.",
            "Seek advice if symptoms continue to spread."
        ],
    },
    "yellow_leaf_curl": {
        "description": "Tomato yellow leaf curl is a viral disease that can affect tomato growth.",
        "symptoms": [
            "Upward curling or cupping of leaves.",
            "Yellowing and reduced leaf size.",
            "Stunted growth or reduced fruit production."
        ],
        "causes": [
            "Tomato yellow leaf curl virus.",
            "The virus is commonly transmitted by whiteflies."
        ],
        "care": [
            "Inspect plants for whiteflies.",
            "Use suitable vector-management practices based on local advice.",
            "Seek agricultural guidance; ordinary fungicides do not cure viral disease."
        ],
    },
    "leaf_mold": {
        "description": "Leaf mold is a fungal disease commonly associated with humid growing conditions.",
        "symptoms": [
            "Yellow patches on the upper leaf surface.",
            "Olive or grayish growth may appear underneath leaves.",
            "Severe cases may cause leaves to dry."
        ],
        "causes": [
            "A leaf-mold fungus favored by high humidity and leaf moisture."
        ],
        "care": [
            "Improve ventilation around plants.",
            "Avoid wetting foliage during watering.",
            "Remove severely affected leaves when appropriate and seek crop-specific advice."
        ],
    },
    "spider_mite": {
        "description": "Spider mites are tiny plant-feeding pests that can damage leaves.",
        "symptoms": [
            "Fine pale speckles on leaves.",
            "Yellowing or bronzing.",
            "Fine webbing may be visible in severe infestations."
        ],
        "causes": [
            "Feeding by spider mites, which can multiply quickly in hot, dry conditions."
        ],
        "care": [
            "Inspect the undersides of leaves closely.",
            "Check nearby plants for similar damage.",
            "Use plant-appropriate pest management if mites are confirmed."
        ],
    },
    "target_spot": {
        "description": "Target spot can produce circular or irregular lesions on leaves.",
        "symptoms": [
            "Brown lesions, sometimes with concentric rings.",
            "Spots may merge into larger damaged areas.",
            "Yellowing or leaf loss can occur."
        ],
        "causes": [
            "A fungal pathogen associated with target spot.",
            "Warm, humid conditions may favor disease development."
        ],
        "care": [
            "Improve airflow and avoid prolonged leaf wetness.",
            "Remove affected material where appropriate.",
            "Confirm the likely cause before selecting treatment."
        ],
    },
    "powdery_mildew": {
        "description": "Powdery mildew often produces a white, powder-like coating on leaves.",
        "symptoms": [
            "White or gray powdery patches.",
            "Leaf distortion or yellowing.",
            "Affected foliage may dry or drop."
        ],
        "causes": [
            "Powdery mildew fungi; different species affect different plants."
        ],
        "care": [
            "Improve airflow and avoid overcrowding.",
            "Remove severely affected leaves where appropriate.",
            "Use only a suitable, labeled treatment when advised."
        ],
    },
    "rust": {
        "description": "Rust diseases are fungal infections that may produce colored pustules on leaves.",
        "symptoms": [
            "Orange, reddish, yellow, or brown powdery spots.",
            "Pustules may be more visible on the underside of leaves.",
            "Severe infections may cause leaf yellowing or drop."
        ],
        "causes": [
            "Rust fungi; the species depends on the host plant."
        ],
        "care": [
            "Remove heavily affected leaves where appropriate.",
            "Avoid handling plants while foliage is wet.",
            "Confirm the disease and plant species before choosing treatment."
        ],
    },
    "mosaic": {
        "description": "Mosaic-like leaf patterns may occur with some plant virus infections, but other problems can look similar.",
        "symptoms": [
            "Irregular light and dark green or yellow patches.",
            "Leaf distortion or reduced growth may occur."
        ],
        "causes": [
            "Possible viral infection; symptoms alone do not identify the exact virus.",
            "Nutrient and environmental problems can sometimes look similar."
        ],
        "care": [
            "Clean tools after handling suspect plants.",
            "Avoid propagating material from plants suspected of viral infection.",
            "Seek expert advice before removing plants or selecting treatment."
        ],
    },
    "default": {
        "description": "Specific guidance for this predicted class is not available in the app.",
        "symptoms": [
            "Compare the prediction with visible signs on the actual plant.",
            "Inspect both leaf surfaces, stems, and nearby plants."
        ],
        "causes": [
            "Spots, curling, or discoloration can result from disease, pests, nutrient issues, or environmental stress."
        ],
        "care": [
            "Monitor symptoms and take clear photos over time.",
            "Avoid applying pesticides or fungicides based only on the AI prediction.",
            "Seek local agricultural or plant-specialist advice if the plant worsens."
        ],
    },
}


def get_info(class_name):
    label = class_name.lower().replace("-", "_").replace(" ", "_")

    if "healthy" in label:
        return DISEASE_INFO["healthy"]
    if "yellow" in label and "curl" in label:
        return DISEASE_INFO["yellow_leaf_curl"]
    if "mosaic" in label or "virus" in label:
        return DISEASE_INFO["mosaic"]
    if "late_blight" in label:
        return DISEASE_INFO["late_blight"]
    if "early_blight" in label:
        return DISEASE_INFO["early_blight"]
    if "septoria" in label:
        return DISEASE_INFO["septoria"]
    if "bacterial_spot" in label:
        return DISEASE_INFO["bacterial_spot"]
    if "leaf_mold" in label or "leaf_mould" in label:
        return DISEASE_INFO["leaf_mold"]
    if "spider_mite" in label:
        return DISEASE_INFO["spider_mite"]
    if "target_spot" in label:
        return DISEASE_INFO["target_spot"]
    if "powdery_mildew" in label:
        return DISEASE_INFO["powdery_mildew"]
    if "rust" in label:
        return DISEASE_INFO["rust"]
    return DISEASE_INFO["default"]


# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("# 🌿 PlantCare AI")
    st.write("An educational plant-leaf screening tool.")
    st.divider()
    st.markdown("### How to use")
    st.write(
        "1. Upload a clear leaf photo.\n"
        "2. Click Analyze leaf.\n"
        "3. Review the prediction and guidance."
    )
    st.divider()
    st.markdown("### Photo tips")
    st.write("- Use good lighting.")
    st.write("- Keep the leaf in focus.")
    st.write("- Avoid blurry images.")
    st.divider()
    st.caption("AI screening is not a confirmed diagnosis.")


# ============== HEADER ==============
st.markdown("""
<div class="hero">
    <h1>🌱 PlantCare AI</h1>
    <p>AI-powered leaf screening with educational plant-care guidance.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("🔎 Analyze a plant leaf")
st.write(
    "Upload a clear leaf photograph to see the model's predicted "
    "category and general information about symptoms and care."
)

uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
)

# ============== IMAGE ANALYSIS ==============
if uploaded is not None:
    try:
        image = ImageOps.exif_transpose(
            Image.open(uploaded)
        ).convert("RGB")

        image_col, result_col = st.columns([1, 1], gap="large")

        with image_col:
            st.markdown("### 📷 Uploaded leaf")
            st.image(image, use_container_width=True)
            st.caption(
                f"Image dimensions: {image.width} × {image.height} pixels"
            )

        with result_col:
            st.markdown("### 🌿 AI analysis")
            if st.button("🌱 Analyze leaf", type="primary"):
                try:
                    with st.spinner("Analyzing your leaf photo..."):
                        model, class_names = load_model()
                        resized = image.resize((160, 160))
                        array = np.asarray(
                            resized, dtype=np.float32
                        )[None, ...]

                        scores = model.predict(array, verbose=0)[0]

                        if len(scores) != len(class_names):
                            raise ValueError(
                                "Model output and class labels do not match."
                            )

                        best_index = int(np.argmax(scores))
                        score = float(scores[best_index])
                        predicted_class = class_names[best_index]
                        info = get_info(predicted_class)

                    st.markdown("#### Predicted category")
                    st.markdown(
                        f"""
                        <div class="info-card">
                            <h3>{html.escape(format_name(predicted_class))}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.metric("Model score", f"{score:.2%}")
                    st.progress(
                        max(0.0, min(1.0, score)),
                        text="Score for the top prediction",
                    )

                    st.warning(
                        "Uncertainty warning: this is an AI prediction, "
                        "not a confirmed diagnosis. The score is not a "
                        "guarantee of correctness, and different problems "
                        "can cause similar leaf symptoms."
                    )

                    st.markdown("## 🌱 About this condition")
                    st.write(info["description"])

                    st.markdown("## 👀 Symptoms to check")
                    for item in info["symptoms"]:
                        st.markdown(f"- {item}")

                    st.markdown("## 🔬 Possible causes")
                    for item in info["causes"]:
                        st.markdown(f"- {item}")

                    st.markdown("## 🌿 Care suggestions")
                    for item in info["care"]:
                        st.markdown(f"- {item}")

                    with st.expander("View top 3 model predictions"):
                        for idx in np.argsort(scores)[::-1][:3]:
                            idx = int(idx)
                            st.write(
                                f"**{format_name(class_names[idx])}** — "
                                f"{float(scores[idx]):.2%}"
                            )
                            st.progress(
                                max(0.0, min(1.0, float(scores[idx])))
                            )

                    st.info(
                        "Care guidance is general educational information. "
                        "Confirm the plant species and condition before "
                        "treatment. Follow local agricultural advice and "
                        "all product labels."
                    )

                except Exception as error:
                    st.error("The image could not be analyzed.")
                    st.exception(error)

    except Exception:
        st.error("Please upload a valid JPG, JPEG, or PNG image.")

else:
    st.markdown("""
    <div class="info-card">
        <h3>🌿 Start with a leaf photo</h3>
        <p>Upload a clear image to receive a model prediction and general
        plant-care information.</p>
    </div>
    """, unsafe_allow_html=True)

# ============== FOOTER ==============
st.divider()
st.caption(
    "PlantCare AI | Student project for educational use. "
    "Not a substitute for professional plant diagnosis."
)
