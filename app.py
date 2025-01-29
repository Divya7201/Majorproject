import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("double_unet_pruned_model.h5", compile=False)

model = load_model()

# Streamlit UI
st.title("Double UNet Image Segmentation")
st.write("Upload an image for segmentation.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((256, 256))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Perform prediction
    prediction = model.predict(img_array)
    mask = (prediction[0] > 0.5).astype(np.uint8)  # Convert to binary mask

    # Convert mask to image
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    st.image(mask_img, caption="Predicted Segmentation Mask", use_column_width=True)

st.write("Upload an image and see the segmented output.")
