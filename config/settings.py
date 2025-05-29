import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # RTSP Stream Configuration
    RTSP_URL = os.getenv("RTSP_URL", "rtsp://192.168.1.100:8554/live.sdp")
    FPS = int(os.getenv("FPS", 60))
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 1280))
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 720))
    
    # Model Configuration
    OBJECT_DETECTION_MODEL = os.getenv("OBJECT_DETECTION_MODEL", "yolo11s.pt")
    VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "google/paligemma-3b-mix-224")
    HF_TOKEN = os.getenv("HF_TOKEN", "")  # Hugging Face token
    
    # OCR Configuration
    TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")  # Path to tesseract executable
    OCR_LANG = os.getenv("OCR_LANG", "eng")
    
    # Speech Configuration
    TTS_VOICE = os.getenv("TTS_VOICE", "0")  # Voice index for TTS
    TTS_RATE = int(os.getenv("TTS_RATE", 150))
    STT_TIMEOUT = int(os.getenv("STT_TIMEOUT", 5))
    
    # System Configuration
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 8))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOCAL_SAVE_PATH = os.getenv("LOCAL_SAVE_PATH", "data/captured_frames")
    
    # Performance Configuration
    TARGET_FPS = int(os.getenv("TARGET_FPS", 60))
    MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", 30))
    PROCESSING_INTERVAL = float(os.getenv("PROCESSING_INTERVAL", 0.016))  # ~60 FPS
    DETECTION_CONF_THRESHOLD = float(os.getenv("DETECTION_CONF_THRESHOLD", 0.25))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.45))

settings = Settings()