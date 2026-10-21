
import json
import html
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- GREEN THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f1f8f1, #ffffff 55%);
    color: #193b2a;
}
[data-testid="stSidebar"] {
    background: #e5f1e6;
}
.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(120deg, #174c32, #318451);
    color: white;
    margin-bottom: 24px;
}
.hero h1 { color: white; font-size: 2.3rem; }
.hero p { color: #e5f5e9; font-size: 1.05rem; }
.info-card {
    padding: 20px;
    background: #f0f8f0;
    border: 1px solid #d1e6d3;
    border-radius: 16px;
    margin: 12px 0;
}
.info-card h3 { color: #205b39; }
div.stButton > button {
    background: #237547;
    color: white;
    border-radius: 10px;
    border: none;
    min-height: 45px;
}
div.stButton > button:hover {
    background: #185b36;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_model.keras")
    with open("class_names.json", "r", encoding="utf-8") as f:
        names = json.load(f)
    return model, names


def format_name(name):
    return name.replace("___", " — ").replace("__", " ").replace("_", " ").strip()


# ---------------- CONDITION INFORMATION ----------------
# General educational information. Confirm the diagnosis before treatment.
DISEASE_INFO = {
    "healthy": {
        "status": "healthy",
        "description": "The model classified this leaf as belonging to a healthy category.",
        "symptoms": [
            "Leaves generally have their normal color and shape.",
            "No obvious disease spots may be visible in the uploaded area."
        ],
        "causes": [
            "No disease pattern was identified by the model in this image."
        ],
        "care": [
            "Continue appropriate watering, nutrition, and routine plant inspection.",
            "Check other leaves and the whole plant; one photo cannot establish overall plant health."
        ],
    },
    "bacterial_spot": {
        "status": "disease",
        "description": "Bacterial spot can affect leaves and fruit, producing small damaged areas.",
        "symptoms": [
            "Small dark or water-soaked-looking spots.",
            "Yellowing around spots or damaged leaf tissue.",
            "In some plants, lesions may also occur on stems or fruit."
        ],
        "causes": [
            "Bacterial pathogens; the exact species depends on the host plant.",
            "Spread can be favored by splashing water, wet foliage, and contaminated tools."
        ],
        "care": [
            "Avoid working with plants while foliage is wet.",
            "Water near the soil rather than spraying the leaves.",
            "Clean tools and remove severely affected material where appropriate.",
            "Ask a local agricultural extension service about confirmed diagnosis and suitable controls."
        ],
    },
    "early_blight": {
        "status": "disease",
        "description": "Early blight is a fungal disease commonly associated with plants such as tomato and potato.",
        "symptoms": [
            "Brown leaf spots that may have concentric rings.",
            "Yellowing around affected areas.",
            "Older leaves may be affected first."
        ],
        "causes": [
            "Fungal pathogens associated with early blight.",
            "Disease can be favored by leaf wetness, warm conditions, and infected plant debris."
        ],
        "care": [
            "Avoid overhead watering and prolonged leaf wetness.",
            "Remove affected leaves if practical, without removing excessive healthy foliage.",
            "Dispose of diseased plant material appropriately.",
            "Use crop-specific fungicides only when suitable and according to the product label and local guidance."
        ],
    },
    "late_blight": {
        "status": "disease",
        "description": "Late blight is a potentially serious disease of potato and tomato that can progress quickly under favorable conditions.",
        "symptoms": [
            "Irregular dark or water-soaked-looking leaf patches.",
            "Pale or whitish growth may appear near lesions in humid conditions.",
            "Rapid leaf or stem damage can occur."
        ],
        "causes": [
            "The oomycete pathogen Phytophthora infestans.",
            "Cool, humid weather and prolonged leaf wetness can favor outbreaks."
        ],
        "care": [
            "Inspect nearby plants and seek prompt local agricultural advice if late blight is suspected.",
            "Avoid moving wet foliage or contaminated tools between plants.",
            "Remove and dispose of confirmed infected material according to local recommendations.",
            "Do not rely on this image prediction alone to select a treatment."
        ],
    },
    "septoria": {
        "status": "disease",
        "description": "Septoria leaf spot is a fungal leaf disease commonly affecting tomato.",
        "symptoms": [
            "Numerous small, circular spots.",
            "Spots may have pale or gray centers and darker margins.",
            "Tiny dark specks may be visible inside some lesions."
        ],
        "causes": [
            "A Septoria fungal pathogen.",
            "Rain splash, wet foliage, and infected plant debris can contribute to spread."
        ],
        "care": [
            "Avoid overhead watering and improve airflow around plants.",
            "Remove affected lower leaves when practical.",
            "Clean tools and manage plant debris appropriately.",
            "Seek crop-specific guidance if symptoms continue to spread."
        ],
    },
    "yellow_leaf_curl": {
        "status": "disease",
        "description": "Tomato yellow leaf curl is a viral disease that can seriously affect tomato growth.",
        "symptoms": [
            "Upward curling or cupping of leaves.",
            "Yellowing, reduced leaf size, or stunted growth.",
            "Poor flowering or fruit set may occur."
        ],
        "causes": [
            "Tomato yellow leaf curl virus.",
            "The virus is commonly transmitted by whiteflies."
        ],
        "care": [
            "Inspect plants for whiteflies and seek local advice on appropriate management.",
            "Remove suspected infected plants only in accordance with local crop guidance.",
            "Use healthy planting material and suitable vector-management practices.",
            "Viral diseases generally cannot be cured by ordinary fungicides."
        ],
    },
    "mosaic_virus": {
        "status": "disease",
        "description": "Mosaic virus diseases can cause irregular patterns of light and dark green or yellow on leaves.",
        "symptoms": [
            "Mottled or mosaic-like leaf coloration.",
            "Leaf distortion or reduced plant growth.",
            "Symptoms vary with the virus and plant species."
        ],
        "causes": [
            "A plant virus; this image classifier may not identify the exact virus.",
            "Transmission routes vary and may include contaminated tools, plant material, or insect vectors."
        ],
        "care": [
            "Clean tools and wash hands after handling suspect plants.",
            "Avoid propagating plants from suspected infected material.",
            "Check for pests and seek local diagnostic advice.",
            "Do not assume that every mottled leaf has a viral infection."
        ],
    },
    "leaf_mold": {
        "status": "disease",
        "description": "Leaf mold is a fungal disease often associated with tomato plants grown in humid conditions.",
        "symptoms": [
            "Pale yellow patches on the upper leaf surface.",
            "Olive or grayish fungal growth may appear on the underside.",
            "Severe infection can lead to leaf drying."
        ],
        "causes": [
            "A leaf-mold fungus favored by high humidity and prolonged moisture."
        ],
        "care": [
            "Improve ventilation and airflow around plants.",
            "Avoid wetting the foliage when watering.",
            "Remove severely affected leaves when appropriate.",
            "Use any disease-control product only if labeled for the crop and condition."
        ],
    },
    "spider_mites": {
        "status": "pest",
        "description": "Spider mites are tiny plant-feeding pests that can cause leaf discoloration and damage.",
        "symptoms": [
            "Fine pale speckling on leaves.",
            "Bronzing, yellowing, or drying in more severe cases.",
            "Fine webbing may be present."
        ],
        "causes": [
            "Feeding by spider mites, which can multiply quickly in hot, dry conditions."
        ],
        "care": [
            "Inspect the undersides of leaves with a magnifier.",
            "Check nearby plants for similar damage.",
            "Consider washing foliage with water where suitable for the plant.",
            "Use only plant-appropriate, locally approved mite controls if necessary."
        ],
    },
    "target_spot": {
        "status": "disease",
        "description": "Target spot can cause circular or irregular lesions on leaves and other plant parts.",
        "symptoms": [
            "Brown lesions, sometimes with concentric rings.",
            "Spots may merge into larger damaged areas.",
            "Leaf yellowing or loss may occur."
        ],
        "causes": [
            "A fungal pathogen associated with target spot; identification depends on the host plant.",
            "Warm, humid conditions can favor disease development."
        ],
        "care": [
            "Keep foliage as dry as practical and improve airflow.",
            "Remove severely affected material where appropriate.",
            "Inspect nearby plants and monitor whether spots are spreading.",
            "Obtain a crop-specific diagnosis before choosing treatment."
        ],
    },
    "powdery_mildew": {
        "status": "disease",
        "description": "Powdery mildew is a fungal disease that often creates a white, powder-like coating on leaves.",
        "symptoms": [
            "White or gray powdery patches.",
            "Leaf distortion or yellowing in some plants.",
            "Affected foliage may dry or drop."
        ],
        "causes": [
            "Powdery mildew fungi; different species affect different host plants."
        ],
        "care": [
            "Improve airflow and avoid overcrowding.",
            "Remove severely affected leaves where appropriate.",
            "Avoid applying home remedies that may burn foliage.",
            "Use a labeled treatment suitable for the plant if advised."
        ],
    },
    "rust": {
        "status": "disease",
        "description": "Rust diseases are fungal infections that may produce colored pustules on leaves.",
        "symptoms": [
            "Orange, reddish, yellow, or brown powdery spots.",
            "Pustules may be more visible on the underside of leaves.",
            "Severe infections may cause leaf yellowing and drop."
        ],
        "causes": [
            "Rust fungi, with different species affecting different plants."
        ],
        "care": [
            "Remove heavily affected leaves when practical.",
            "Avoid handling plants while foliage is wet.",
            "Improve airflow and monitor neighboring plants.",
            "Confirm the rust type before choosing a treatment."
        ],
    },
    "default": {
        "status": "uncertain",
        "description": "The model predicted this category, but detailed guidance for this exact class is not available in the app.",
        "symptoms": [
            "Compare the predicted category with the actual visible signs on the plant.",
            "Check both sides of the leaves, stems, and nearby plants."
        ],
        "causes": [
            "Leaf spots, discoloration, and curling can have multiple causes, including disease, pests, nutrient problems, or environmental stress."
        ],
        "care": [
            "Take clear photographs of the affected leaves and monitor changes.",
            "Avoid applying pesticides or fungicides based only on this AI result.",
            "Seek advice from a local agricultural extension service or plant specialist if symptoms worsen."
        ],
    },
}


def get_condition_info(class_name):
    """Match common dataset class names to educational guidance."""
    label = class_name.lower().replace("-", "_").replace(" ", "_")

    if "healthy" in label:
        return DISEASE_INFO["healthy"]

    if "yellow" in label and ("curl" in label or "leaf" in label):
        return DISEASE_INFO["yellow_leaf_curl"]

    if "mosaic" in label and "virus" in label:
        return DISEASE_INFO["mosaic_virus"]

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

    if "spider_mite" in label or "two_spotted_spider_mite" in label:
        return DISEASE_INFO["spider_mites"]

    if "target_spot" in label:
        return DISEASE_INFO["target_spot"]

    if "powdery_mildew" in label:
        return DISEASE_INFO["powdery_mildew"]

    if "rust" in label:
        return DISEASE_INFO["rust"]

    return DISEASE_INFO["default"]


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 🌿 PlantCare AI")
    st.write("An educational plant-leaf screening tool.")
    st.divider()
    st.markdown("### How to use")
    st.write("1. Upload a clear leaf photo.")
    st.write("2. Click Analyze leaf.")
    st.write("3. Review the prediction and guidance.")
    st.divider()
    st.markdown("### Photo tips")
    st.write("- Use good lighting.")
    st.write("- Keep the leaf in focus.")
    st.write("- Avoid blurry images.")
    st.divider()
    st.caption("AI screening is not a confirmed diagnosis.")

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🌱 PlantCare AI</h1>
    <p>AI-powered leaf screening with educational plant-care guidance.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("Analyze a plant leaf")
st.write(
    "Upload a photograph to see the model's predicted category "
    "and general information about common symptoms and care."
)

uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    try:
        image = ImageOps.exif_transpose(
            Image.open(uploaded)
        ).convert("RGB")

        image_col, result_col = st.columns([1, 1], gap="large")

        with image_col:
            st.markdown("### 📷 Uploaded leaf")
            st.image(image, use_container_width=True)
            st.caption(f"{image.width} × {image.height} pixels")

        with result_col:
            st.markdown("### 🔎 Analysis")
            st.write("Click below to run the model on this image.")

            if st.button("🌱 Analyze leaf", type="primary"):
                try:
                    with st.spinner("Analyzing the image..."):
                        model, class_names = load_model()

                        resized = image.resize((160, 160))
                        array = np.asarray(
                            resized, dtype=np.float32
                        )[None, ...]

                        scores = model.predict(array, verbose=0)[0]

                        if len(scores) != len(class_names):
                            raise ValueError(
                                "The model output and class labels do not match."
                            )

                        best_index = int(np.argmax(scores))
                        score = float(scores[best_index])
                        predicted_class = class_names[best_index]
                        condition = get_condition_info(predicted_class)

                    st.markdown("### 🌿 Prediction result")
                    st.markdown(
                        f"""
                        <div class="info-card">
                            <p><b>Predicted category</b></p>
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
                        "Uncertainty warning: this is a model prediction, "
                        "not a confirmed diagnosis. The score is not a "
                        "guarantee of correctness. Similar symptoms can have "
                        "different causes."
                    )

                    st.markdown("## 🌱 About this condition")
                    st.write(condition["description"])

                    st.markdown("## 👀 Symptoms to check")
                    for item in condition["symptoms"]:
                        st.markdown(f"- {item}")

                    st.markdown("## 🔬 Possible causes")
                    for item in condition["causes"]:
                        st.markdown(f"- {item}")

                    st.markdown("## 🌿 Care suggestions")
                    for item in condition["care"]:
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
                    st.error(
                        "The image could not be analyzed. "
                        "Check the app logs for technical details."
                    )
                    st.exception(error)

    except Exception:
        st.error("Please upload a valid JPG, JPEG, or PNG image.")

else:
    st.markdown("""
    <div class="info-card">
        <h3>Start with a leaf photo 🌿</h3>
        Upload a clear image to receive a model prediction and general
        plant-care information.
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption(
    "PlantCare AI | Student project for educational use. "
    "Not a substitute for professional plant diagnosis."
)
