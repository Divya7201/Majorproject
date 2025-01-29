import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import skimage.transform

# Load the model
model = tf.keras.models.load_model('double_unet_pruned_model.h5', compile=False)

# Streamlit app
st.title("Nerve Segmentation with Double U-Net")

uploaded_file = st.file_uploader("Upload an Ultrasound Image", type=['png', 'jpg', 'jpeg', 'tif'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img = np.array(img)
    img_resized = skimage.transform.resize(img, (128, 128, 1), mode='constant', preserve_range=True)
    img_resized = np.expand_dims(img_resized, axis=0) / 255.0

    prediction = model.predict(img_resized)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.image(prediction.squeeze(), caption="Predicted Mask", use_column_width=True)
