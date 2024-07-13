import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os
import tempfile
print(cv2.__version__)
# Load the trained model
model_path = 'D:\Activity Recognition\Activity-Recongnition-Web-Application\CNN_LSTM.h5'
activity_model = load_model(model_path)

# Define activity labels
activity_labels = ['WalkingWithDog', 'TaiChi', 'Swing', 'HorseRace']

# Function to preprocess image for CNN input
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict activity from a sequence of frames (video)
def predict_activity(frames):
    required_length = 20
    if len(frames) > required_length:
        frames = frames[:required_length]
    elif len(frames) < required_length:
        frames += [np.zeros_like(frames[0]) for _ in range(required_length - len(frames))]

    preprocessed_frames = [preprocess_image(frame) for frame in frames]
    preprocessed_frames = np.vstack(preprocessed_frames)

    # Assuming the custom CNN model is part of the loaded model, and features are passed to LSTM
    sequence_frames = np.expand_dims(preprocessed_frames, axis=0)

    prediction = activity_model.predict(sequence_frames)
    predicted_label = activity_labels[np.argmax(prediction)]
    
    return predicted_label

# Centered title
st.markdown("<h1 style='text-align: center;'>Activity Recognition Application</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a video to predict activity.</p>", unsafe_allow_html=True)

# File uploader for input video
uploaded_file = st.file_uploader("Select A Video File", type=["mp4"])

tmp_file_path = None
if uploaded_file is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    # OpenCV video capture from temporary file
    cap = cv2.VideoCapture(tmp_file_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()

    # Button to preview the video
    if st.button("Preview Video"):
        st.video(tmp_file_path, start_time=0, format="video/mp4")
    
    # Make prediction
    if st.button("Predict Activity"):
        with st.spinner('Predicting...'):
            label = predict_activity(frames)
            st.success(f"Predicted Activity: {label}")
            
            # Custom celebratory animation using HTML and CSS
            st.markdown("""
            <style>
            @keyframes celebrate {
                0% { transform: translateY(0); }
                50% { transform: translateY(-10px); }
                100% { transform: translateY(0); }
            }
            .celebrate-text {
                animation: celebrate 0.5s ease-in-out;
                color: #4CAF50;
                font-size: 24px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<p class="celebrate-text">🎉 🎊 Congratulations! 🎊 🎉</p>', unsafe_allow_html=True)
else:
    st.write("Please upload a video file.")

# Footer with name centered at the bottom
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px 0;
        background-color: #f1f1f1;
        color: #333333;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">Created by MUHAMMAD AFAQ</div>', unsafe_allow_html=True)

# Remove the temporary video file if it exists
if tmp_file_path and os.path.exists(tmp_file_path):
    os.remove(tmp_file_path)
