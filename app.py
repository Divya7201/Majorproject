import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import skimage.transform
import h5py

# Function to load the model with error handling
def load_model():
    try:
        model = tf.keras.models.load_model('double_unet_pruned_model.h5')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

if model:
    # Print model input shape
    st.write(f"Model expected input shape: {model.input_shape}")

    # Streamlit app UI
    st.title("Nerve Segmentation with Double U-Net")

    uploaded_file = st.file_uploader("Upload an Ultrasound Image", type=['png', 'jpg', 'jpeg', 'tif'])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        img = np.array(img)

        # Resize the image to match model input shape
        img_resized = skimage.transform.resize(img, (128, 128), mode='constant', preserve_range=True)
        img_resized = np.expand_dims(img_resized, axis=-1)  # Ensure channel dimension
        img_resized = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize and add batch dimension

        # Make prediction
        prediction = model.predict(img_resized)

        # Display results
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.image(prediction.squeeze(), caption="Predicted Mask", use_column_width=True)
