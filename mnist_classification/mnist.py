import streamlit as st
import tensorflow as  tf
import numpy as np 
from PIL import Image

def load_model():
    try:
        model = tf.keras.models.load_model('mnist_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.title("🖼️ MNIST Digit Classifier")
st.markdown("""
Welcome to the MNIST digit classifier app!  
Upload a grayscale image of a digit, and the model will predict which digit (0-9) it is.  
*(Images should be similar to MNIST 28x28 format for best results).*
""")

uploaded_file = st.file_uploader("Choose a digit_image...", type=["png","jpg","jpeg"])

if uploaded_file is not None and model is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28,28))

    st.image(image, caption='Processed Image (28x28 Grayscale)', use_column_width=True)

    image_array = np.array(image).reshape(1,28,28,1) / 255.0

    with st.spinner("Classifying..."):
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        prediction_prob = np.max(prediction)


    st.markdown("## Prediction Result:")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### Predicted Digit: **{predicted_class}**")
    
    with col2:
        st.markdown(f"#### Confidence: **{prediction_prob*100:.2f}%**")
    
    st.success(f"Digit {predicted_class} with {prediction_prob*100:.2f}% confidence!")
    
    st.markdown("### Prediction Probabilities:")
    st.bar_chart(prediction[0])

else:
    if model is None:
        st.warning("Model is not loaded. Please ensure 'mnist_model.h5' is in correct directory.")
    else:
        st.info("Upload as image to get started!")


st.markdown("""
---
**Note:**  
Make sure the image you upload is a single handwritten digit on a clean background. The classifier is trained on MNIST-like images.
""")

