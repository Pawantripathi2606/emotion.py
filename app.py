import streamlit as st
import cv2
import numpy as np
from fer.fer import FER   # FIXED IMPORT

st.set_page_config(page_title="Emotion Detection App", layout="wide")

st.title("😊 Real-Time Emotion Detection (Render & Local Compatible)")
st.write("Lightweight FER model used for emotion detection.")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

# FER Detector
detector = FER()

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Webcam not detected. Try refreshing.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = detector.detect_emotions(rgb_frame)

    if result:
        emotions = result[0]["emotions"]
        emotion = max(emotions, key=emotions.get)

        cv2.putText(rgb_frame, f"Emotion: {emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(rgb_frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    FRAME_WINDOW.image(rgb_frame)

else:
    camera.release()
    st.write("Webcam stopped.")
