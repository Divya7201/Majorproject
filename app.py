import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io

# Fix model by removing 'groups' from Conv2DTranspose layers
def fix_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Convert model to JSON
    model_config = model.to_json()
    model_config_dict = json.loads(model_config)

    # Remove 'groups' argument from Conv2DTranspose layers
    for layer in model_config_dict["config"]["layers"]:
        if layer["class_name"] == "Conv2DTranspose" and "groups" in layer["config"]:
            del layer["config"]["groups"]

    # Recreate model from modified config
    new_model = tf.keras.models.model_from_json(json.dumps(model_config_dict))

    # Load weights separately
    new_model.load_weights(model_path)

    # Save the modified model
    fixed_model_path = "double_unet_fixed_model.h5"
    new_model.save(fixed_model_path)
    
    return fixed_model_path

# Load fixed model
model_path = fix_model("double_unet_pruned_model.h5")
model = tf.keras.models.load_model(model_path)

# Streamlit UI
st.title("Double UNet Image Segmentation")

st.write("Upload an image for segmentation")

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

    # Post-process output (assuming model outputs segmentation mask)
    mask = (prediction[0] > 0.5).astype(np.uint8)  # Thresholding

    # Convert mask to image
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    st.image(mask_img, caption="Predicted Segmentation Mask", use_column_width=True)

st.write("Upload an image and see the segmented output.")
