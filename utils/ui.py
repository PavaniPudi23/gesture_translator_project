import streamlit as st


def inject_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: white;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
        .title-text {
            font-size: 2.2rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.2rem;
        }
        .subtitle-text {
            font-size: 1rem;
            color: #cbd5e1;
            margin-bottom: 1rem;
        }
        .card {
            background: rgba(255,255,255,0.08);
            padding: 1rem;
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
            margin-bottom: 1rem;
        }
        .result-box {
            background: #111827;
            color: white;
            padding: 1rem;
            border-radius: 16px;
            font-size: 1.1rem;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown('<div class="title-text">Gesture Language Translator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle-text">Real-time gesture recognition, translation, and speech output</div>',
        unsafe_allow_html=True,
    )


def render_card(title):
    st.markdown(f'<div class="card"><b>{title}</b></div>', unsafe_allow_html=True)


def display_result_box(label, translated_text, confidence):
    st.markdown(
        f"""
        <div class="result-box">
            Prediction: {label}<br>
            Translation: {translated_text}<br>
            Confidence: {confidence:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )