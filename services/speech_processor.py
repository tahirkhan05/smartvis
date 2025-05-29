import speech_recognition as sr
import pyttsx3
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
from utils.logger import logger
from config.settings import settings

class SpeechProcessor:
    def __init__(self):
        # Initialize Speech Recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize Text-to-Speech
        self.tts_engine = pyttsx3.init()
        self._setup_tts()
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        
        self._calibrate_microphone()
        logger.info("Speech processor initialized")
    
    def _setup_tts(self):
        try:
            voices = self.tts_engine.getProperty('voices')
            if voices and len(voices) > int(settings.TTS_VOICE):
                self.tts_engine.setProperty('voice', voices[int(settings.TTS_VOICE)].id)
            
            self.tts_engine.setProperty('rate', settings.TTS_RATE)
            self.tts_engine.setProperty('volume', 0.9)
        except Exception as e:
            logger.warning(f"TTS setup warning: {e}")
    
    def _calibrate_microphone(self):
        try:
            with self.microphone as source:
                logger.info("Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Microphone calibrated")
        except Exception as e:
            logger.error(f"Microphone calibration failed: {e}")
    
    def listen_for_speech(self, timeout=5):
        try:
            logger.info(f"Listening for {timeout} seconds...")
            print(f"🎤 Speak now... (listening for {timeout} seconds)")
            
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            print("🔄 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Recognized: {text}")
            print(f"🎙️ You said: {text}")
            return text
            
        except sr.WaitTimeoutError:
            logger.warning("Speech timeout")
            print("❌ No speech detected")
            return None
        except sr.UnknownValueError:
            logger.warning("Speech not understood")
            print("❌ Could not understand speech")
            return None
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            print(f"❌ Speech recognition failed: {e}")
            return None
    
    def speak(self, text):
        try:
            logger.info(f"Speaking: {text}")
            print(f"🔊 Speaking: {text}")
            
            # Run TTS in separate thread to avoid blocking
            def tts_worker():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            tts_thread = threading.Thread(target=tts_worker, daemon=True)
            tts_thread.start()
            tts_thread.join(timeout=10)  # Max 10 seconds for TTS
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            print(f"❌ Text-to-speech failed: {e}")