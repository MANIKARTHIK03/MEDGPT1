import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# load pre-trained model once (ImageNet weights as demo)
@st.cache_resource
def load_model():
    model = ResNet50(weights="imagenet")
    return model

def predict_image(img_file):
    model = load_model()
    img = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    return decoded
