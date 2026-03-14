import time
from collections import deque

import av
import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from utils.inference import GestureDetector
from utils.preprocessing import process_frame, create_hand_detector
from utils.speech import create_tts_audio_html
from utils.translation import SUPPORTED_LANGUAGES, translate_text
from utils.ui import display_result_box, inject_custom_css, render_header
from utils.constants import SEQUENCE_LENGTH


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Gesture Translator", layout="wide")

inject_custom_css()
render_header()


# ---------------- SIDEBAR ----------------
st.sidebar.title("Settings")

mode = st.sidebar.selectbox(
    "Recognition Mode",
    ["word", "phrase"],
    index=0,
)

language = st.sidebar.selectbox(
    "Translation Language",
    list(SUPPORTED_LANGUAGES.keys()),
    index=0,
)

enable_voice = st.sidebar.checkbox("Enable Voice", value=True)


# ---------------- SESSION STATE ----------------
if "prediction" not in st.session_state:
    st.session_state.prediction = ""

if "translation" not in st.session_state:
    st.session_state.translation = ""

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

if "last_spoken_text" not in st.session_state:
    st.session_state.last_spoken_text = ""


# ---------------- VIDEO PROCESSOR ----------------
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.detector = GestureDetector(mode=mode)
        self.latest_prediction = ""
        self.latest_translation = ""
        self.latest_confidence = 0.0

        # Create hand detector (works with both old and new MediaPipe)
        self.hands = create_hand_detector()

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            annotated_image, landmarks = process_frame(img, self.hands)
            self.sequence.append(landmarks)

            if len(self.sequence) == SEQUENCE_LENGTH:
                label, conf = self.detector.predict(list(self.sequence))

                if label:
                    translated = translate_text(label, language)
                    self.latest_prediction = label
                    self.latest_translation = translated
                    self.latest_confidence = conf

            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

        except Exception as e:
            # Fallback: return original frame even if processing fails
            img = frame.to_ndarray(format="bgr24")
            self.latest_prediction = "Error"
            self.latest_translation = str(e)
            self.latest_confidence = 0.0
            return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- WEBRTC CONFIG ----------------
# Use Metered TURN servers for reliable cloud deployment.
# Free credentials from https://www.metered.ca/stun-turn (or set your own in st.secrets).
def _get_ice_servers():
    try:
        # If Twilio or custom credentials are in secrets, use those
        if hasattr(st, "secrets") and "TURN_USERNAME" in st.secrets:
            return [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": st.secrets["TURN_URLS"],
                    "username": st.secrets["TURN_USERNAME"],
                    "credential": st.secrets["TURN_CREDENTIAL"],
                },
            ]
    except Exception:
        pass

    # Fallback: free OpenRelay TURN servers for reliable cloud connectivity
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:80?transport=tcp",
                "turn:openrelay.metered.ca:443",
                "turns:openrelay.metered.ca:443?transport=tcp",
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]


RTC_CONFIGURATION = RTCConfiguration({"iceServers": _get_ice_servers()})


# ---------------- UI LAYOUT ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Camera")

    webrtc_ctx = webrtc_streamer(
        key="gesture-translator",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=GestureProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Prediction")
    result_placeholder = st.empty()
    audio_placeholder = st.empty()

    if not webrtc_ctx.state.playing:
        result_placeholder.info("Click START and allow camera access.")
    else:
        result_placeholder.info("Show a gesture to the camera.")


# ---------------- READ RESULTS SAFELY OUTSIDE recv() ----------------
if webrtc_ctx.video_processor:
    processor = webrtc_ctx.video_processor

    # Small refresh loop so the side panel updates while camera is running
    for _ in range(200):
        if webrtc_ctx.video_processor:
            processor = webrtc_ctx.video_processor

            if processor.latest_prediction:
                st.session_state.prediction = processor.latest_prediction
                st.session_state.translation = processor.latest_translation
                st.session_state.confidence = processor.latest_confidence

                with col2:
                    display_result_box(
                        st.session_state.prediction,
                        st.session_state.translation,
                        st.session_state.confidence,
                    )

                    if (
                        enable_voice
                        and st.session_state.translation
                        and st.session_state.translation
                        != st.session_state.last_spoken_text
                    ):
                        audio_html = create_tts_audio_html(
                            st.session_state.translation,
                            language,
                        )
                        audio_placeholder.markdown(
                            audio_html, unsafe_allow_html=True
                        )
                        st.session_state.last_spoken_text = (
                            st.session_state.translation
                        )

            time.sleep(0.1)