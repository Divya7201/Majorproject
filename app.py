import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load the trained model (replace with your actual model path)
@st.cache_resource
def load_model():
    # Change the path to the location of your trained model
    model = tf.keras.models.load_model('ultrasound_nerve_segmentation_model.h5')
    return model

# Function to preprocess image (if needed)
def preprocess_image(image):
    image = image.convert('RGB')
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize image if required
    return image

# Function to postprocess model output (if needed)
def postprocess_output(output):
    output = np.squeeze(output, axis=0)  # Remove batch dimension
    output = np.argmax(output, axis=-1)  # Convert to class labels
    return output

# Set up Streamlit layout
st.title("Ultrasound Nerve Segmentation")

st.write("Upload an ultrasound image to segment the nerve.")

# File upload widget
uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Load the pre-trained model
    model = load_model()

    # Predict the segmentation mask
    with st.spinner('Segmenting the nerve...'):
        output = model.predict(preprocessed_image)

    # Postprocess and visualize the output
    segmented_image = postprocess_output(output)

    # Show the original and segmented image side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(segmented_image, cmap='gray')
    axes[1].set_title("Segmented Image")
    axes[1].axis('off')

    st.pyplot(fig)
