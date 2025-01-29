import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the model (Make sure the model file is in the correct path)
@st.cache_resource
def load_trained_model():
    model = load_model('double_unet_pruned_model.h5', compile=False)
    return model

# Initialize model
model = load_trained_model()

# Streamlit app
st.title("Image Segmentation with U-Net")
st.write("Upload an image and get the segmented output.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_image)
    
    # Preprocess the image (resizing to the expected input size of the model)
    image = image.resize((256, 256))  # Assuming your model expects 256x256 input
    image_array = np.array(image) / 255.0  # Normalize the image (if required by your model)
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict segmentation mask
    with st.spinner("Processing..."):
        prediction = model.predict(image_array)
    
    # Process output (Assuming the model output is a mask of the same size as input)
    mask = prediction[0]  # Assuming model outputs a batch of predictions, take the first one
    
    # Display the results
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(mask, caption="Segmentation Output", use_column_width=True)
