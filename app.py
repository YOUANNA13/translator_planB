import streamlit as st
from translator_st2 import Translator
import numpy as np
import sounddevice as sd
import tempfile
import os

# Initialize the translator
# Streamlit Interface
@st.cache_resource
def get_translator():
    return Translator(device="cpu")


def main():
    st.title("Multi-Language Translator")

    translator = get_translator()

    # Language selection
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox("From:", list(translator.lang_codes.keys()), index=0)
    with col2:
        target_lang = st.selectbox("To:", list(translator.lang_codes.keys()), index=1)

    # Input method selection
    input_method = st.radio("Choose input method:", ["Text", "Voice"])

    if input_method == "Text":
        # Text input
        input_text = st.text_area("Enter text to translate:", height=100)

        if st.button("Translate"):
            if input_text:
                with st.spinner("Translating..."):
                    translated_text = translator.translate_text(input_text, source_lang, target_lang)
                st.success("Translation complete!")
                st.write("Translated text:")
                st.info(translated_text)

                # Text-to-speech option
                if st.button("Listen to translation"):
                    with st.spinner("Generating audio..."):
                        translator.text_to_speech(translated_text, target_lang)
            else:
                st.warning("Please enter some text to translate.")

    else:  # Voice input
        st.write("Click the button and speak:")
        if st.button("Start Recording"):
            with st.spinner("Recording..."):
                audio_data, sample_rate = translator.record_audio(duration=5)

            st.success("Recording complete!")

            # Convert speech to text
            with st.spinner("Converting speech to text..."):
                recognized_text = translator.speech_to_text(audio_data, sample_rate)

            st.write("Recognized text:")
            st.info(recognized_text)

            # Translate the recognized text
            with st.spinner("Translating..."):
                translated_text = translator.translate_text(recognized_text, source_lang, target_lang)

            st.write("Translated text:")
            st.info(translated_text)

            # Text-to-speech option
            if st.button("Listen to translation"):
                with st.spinner("Generating audio..."):
                    translator.text_to_speech(translated_text, target_lang)


if __name__ == "__main__":
    main()