import os
import base64
import tempfile
from io import BytesIO

try:
    from gtts import gTTS
except Exception:
    gTTS = None

from utils.translation import SUPPORTED_LANGUAGES


def create_tts_audio_html(text, language_name):
    """Generate an HTML audio element with TTS audio (for Streamlit web app)."""
    if not text or gTTS is None:
        return ""

    lang_code = SUPPORTED_LANGUAGES.get(language_name, "en")

    try:
        tts = gTTS(text=text, lang=lang_code)
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)

        audio_base64 = base64.b64encode(buffer.read()).decode()
        audio_html = f"""
        <audio autoplay controls>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        return audio_html
    except Exception:
        return ""


def speak_text(text, lang="en"):
    """
    Speak text aloud using gTTS (for the desktop / OpenCV app).
    Saves to a temp file and plays it via the OS default player.
    Works on Windows, macOS, and Linux.
    """
    if not text or gTTS is None:
        return

    try:
        import subprocess
        import sys

        tts = gTTS(text=text, lang=lang)
        temp_file = os.path.join(tempfile.gettempdir(), "gesture_tts.mp3")
        tts.save(temp_file)

        # Cross-platform audio playback
        if sys.platform == "win32":
            os.startfile(temp_file)
        elif sys.platform == "darwin":
            subprocess.Popen(["afplay", temp_file])
        else:
            # Linux — try common players, fail silently if none available
            subprocess.Popen(
                ["xdg-open", temp_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass