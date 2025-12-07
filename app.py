import streamlit as st
import cv2
import numpy as np
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from fer.fer import FER   # important: this import works well with latest fer


st.set_page_config(page_title="Live Emotion Detection", layout="wide")

st.title("😊 Live Emotion Detection (Webcam + Streamlit WebRTC)")
st.write("Real-time facial emotion detection from your browser webcam.")


class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = FER()
        self.last_emotion = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert frame to numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")

        # Convert to RGB for FER
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect emotions
        results = self.detector.detect_emotions(rgb)

        if results:
            emotions = results[0]["emotions"]
            emotion = max(emotions, key=emotions.get)
            self.last_emotion = emotion

            # Draw label on frame
            cv2.putText(
                img,
                f"Emotion: {emotion}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                img,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Return frame back to WebRTC
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.sidebar.header("Controls")
st.sidebar.write("Allow camera access when browser asks.")


webrtc_ctx = webrtc_streamer(
    key="emotion-live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=EmotionProcessor,
    async_processing=True,
)

# Show current emotion text under video
if webrtc_ctx and webrtc_ctx.video_processor:
    emotion = webrtc_ctx.video_processor.last_emotion
    if emotion:
        st.markdown(f"### Current Detected Emotion: **{emotion.upper()}**")
    else:
        st.markdown("### Current Detected Emotion: _Detecting..._")
