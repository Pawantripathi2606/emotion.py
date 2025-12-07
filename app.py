import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np

st.set_page_config(page_title="Emotion Detection App", layout="wide")

st.title("😊 Face Emotion Detection using Streamlit")
st.write("Real-time webcam-based emotion recognition")

run = st.checkbox('Start Webcam')

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Camera not detected")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Analyze emotion
    try:
        result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        # Put emotion text
        cv2.putText(rgb_frame, f'Emotion: {emotion}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except:
        cv2.putText(rgb_frame, "No face detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    FRAME_WINDOW.image(rgb_frame)

else:
    st.write("Webcam stopped")
    camera.release()
