import os
import time
import pyttsx3
import whisper
import speech_recognition as sr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Constants
TEMP_AUDIO_DIR = "./temp_audio"

# Initialize whisper model
whisper_model = whisper.load_model("small")

def speak_text(text):
    """
    Convert text to speech and save as audio file
    
    Args:
        text (str): Text to be converted to speech
        
    Returns:
        bytes: Audio file as bytes
    """
    # Ensure temp directory exists
    if not os.path.exists(TEMP_AUDIO_DIR):
        os.makedirs(TEMP_AUDIO_DIR)
    
    timestamp = int(time.time() * 1000)
    audio_filename = f"audio_{timestamp}.wav"
    audio_path = os.path.join(TEMP_AUDIO_DIR, audio_filename)
    
    tts_engine.save_to_file(text, audio_path)
    tts_engine.runAndWait()
    
    with open(audio_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()
    
    return audio_bytes

def recognize_speech():
    """
    Recognize speech from microphone
    
    Returns:
        str: Recognized text from speech
    """
    if not os.path.exists(TEMP_AUDIO_DIR):
        os.makedirs(TEMP_AUDIO_DIR)
    
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)
    
    try:
        temp_audio_path = os.path.join(TEMP_AUDIO_DIR, "temp_audio.wav")
        with open(temp_audio_path, "wb") as f:
            f.write(audio.get_wav_data())
        
        result = whisper_model.transcribe(temp_audio_path)
        return result["text"]
    
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    
    except sr.RequestError as e:
        return f"Could not process audio; {e}"