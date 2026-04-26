import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(
    page_title="Skin Disease Detector",
    page_icon="🩺",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        'skin_model_best.h5',
        compile=False,
        custom_objects=None
    )
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = load_model()

# Disease names matching your training order
class_names = {
    0: 'Actinic Keratosis',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Nevus',
    6: 'Vascular Lesion'
}

# Disease descriptions
descriptions = {
    0: 'A rough, scaly patch on skin caused by years of sun exposure.',
    1: 'Most common form of skin cancer. Rarely spreads but needs treatment.',
    2: 'Non-cancerous growth. Generally harmless but monitor for changes.',
    3: 'Common benign skin growth. Usually harmless and painless.',
    4: 'Most dangerous form of skin cancer. Early detection is critical.',
    5: 'Common mole. Usually harmless but monitor for changes.',
    6: 'Lesion related to blood vessels. Usually benign.'
}

# Risk levels
risk = {
    0: '⚠️ Medium Risk',
    1: '🔴 High Risk',
    2: '🟢 Low Risk',
    3: '🟢 Low Risk',
    4: '🔴 High Risk',
    5: '🟢 Low Risk',
    6: '🟡 Low-Medium Risk'
}

# App header
st.title("🩺 Skin Disease Detection System")
st.write("AI-Based Medical Image Classification Using Deep Learning")
st.divider()

# Upload section
st.subheader("Upload a Skin Image")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:

    # Show uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=300)

    # Preprocess exactly like training
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner('Analyzing image...'):
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100

    # Show results
    st.divider()
    st.subheader("Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected Disease", class_names[predicted_class])
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")

    st.write(f"**Risk Level:** {risk[predicted_class]}")
    st.write(f"**About:** {descriptions[predicted_class]}")

    # Show all probabilities
    st.divider()
    st.subheader("All Disease Probabilities")
    for i, prob in enumerate(predictions[0]):
        st.progress(float(prob), text=f"{class_names[i]}: {prob*100:.1f}%")

    # Disclaimer
    st.divider()
    st.warning("⚠️ This is an AI screening tool only. Always consult a qualified dermatologist for proper medical diagnosis.")