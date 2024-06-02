import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Function to load the model
@st.cache_resource  # Use st.cache_resource for loading the model
def load_model():
    model = tf.keras.models.load_model('D:/GRP-34/Test10.hdf5')
    return model

# Load the model
with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Cervical Spine Image Classification
         """)

file = st.file_uploader("Please upload a cervical spine image (jpg or png)", type=[".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",".kytjpg"])

# Function to process and predict the image
def import_and_predict(image_data, model):
    # Resize the image to match the model's input size (224x224)
    input_size = (224, 224)
    image = ImageOps.fit(image_data, input_size, Image.LANCZOS)
    image = image.convert("RGB")  # Ensure the image is in RGB format
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    prediction = model.predict(image)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    # Make predictions
    predictions = import_and_predict(image, model)
    
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = ['Dislocation', 'Fracture', 'Normal']  # Replace with your actual class labels
    class_label = class_labels[predicted_class[0]]
    probability = predictions[0][predicted_class[0]]
    
    st.write("Prediction:", class_label)
    st.write("Confidence:", probability * 100, "%")
