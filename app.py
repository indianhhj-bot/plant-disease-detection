import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿"
)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "/content/plant_disease_model.keras"
    )
    with open("/content/class_names.json") as f:
        names = json.load(f)
    return model, names

st.title("🌿 PlantCare AI")
st.subheader("AI-Based Plant Disease Detection")
st.write("Upload a leaf photograph to obtain a prediction.")

uploaded = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    if st.button("Detect Disease"):
        model, names = load_model()
        image = image.resize((160, 160))
        array = np.asarray(image, dtype=np.float32)[None, ...]

        probabilities = model.predict(array, verbose=0)[0]
        index = int(np.argmax(probabilities))

        st.success("Prediction completed")
        st.write("**Predicted class:**", names[index].replace("___", " — "))
        st.metric("Model score", f"{probabilities[index]:.2%}")

        st.caption(
            "Experimental prediction only. This result is not a "
            "confirmed plant diagnosis."
        )
