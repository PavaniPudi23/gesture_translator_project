from deep_translator import GoogleTranslator

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Punjabi": "pa",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Kannada": "kn",
    "Malayalam": "ml",
}


def translate_text(text, target_language):
    """
    Translate text into the selected target language.
    Returns the translated text. If translation fails, returns original text.
    """
    if not text or target_language == "English":
        return text

    lang_code = SUPPORTED_LANGUAGES.get(target_language, "en")

    try:
        translated = GoogleTranslator(source="auto", target=lang_code).translate(text)
        return translated if translated else text
    except Exception:
        return text