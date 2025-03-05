import streamlit as st
import speech_recognition as sr
from transformers import pipeline
import pyttsx3
import io
import soundfile as sf

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize LLM
llm = pipeline('text-generation', model='gpt2')

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

st.title("Speech-to-Speech LLM Bot")

# Audio input
st.write("Click to record your voice:")
if st.button("Record"):
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        st.write("Processing...")

    try:
        # Convert speech to text
        text = recognizer.recognize_google(audio)
        st.write(f"You said: {text}")

        # Generate response using LLM
        response = llm(text, max_length=50, num_return_sequences=1)[0]['generated_text']
        st.write(f"Bot response: {response}")
        tts_engine.save_to_file(response, 'response.wav')
        tts_engine.runAndWait()
        audio_file = open('response.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

    except sr.UnknownValueError:
        st.write("Could not understand audio")
    except sr.RequestError as e:
        st.write(f"Could not request results; {e}")


user_input = st.text_input("Or type your message here:")
if user_input:
    response = llm(user_input, max_length=50, num_return_sequences=1)[0]['generated_text']
    st.write(f"Bot response: {response}")
    tts_engine.save_to_file(response, 'response.wav')
    tts_engine.runAndWait()
    audio_file = open('response.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')