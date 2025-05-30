# 👀 SmartVis - AI Vision System

SmartVis is a real-time AI vision system that combines object detection, vision language models (VLM), optical character recognition (OCR), and speech processing for RTSP video streams.

## 🎞️ Demo Video

https://github.com/user-attachments/assets/a922c0af-a7cb-42bd-8891-dad62295a4ce

## Features

- **Real-time Object Detection** - YOLO11 for 60+ FPS performance
- **Vision Language Model** - PaliGemma 3B for visual question answering
- **Optical Character Recognition** - Tesseract OCR for text extraction
- **Speech Processing** - Voice commands and text-to-speech responses
- **RTSP Stream Support** - Process live video feeds
- **Performance Optimized** - Multi-threaded processing with adaptive frame skipping

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Tesseract OCR
- Microphone and speakers
- RTSP stream source

## Installation

### 1. Clone and Setup

```bash
git clone https://github.com/tahirkhan05/smartvis
cd smartvis
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install PyTorch with CUDA Support

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Tesseract OCR

**Windows:**
1. Download from [Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install the .exe file
3. Update `TESSERACT_PATH` in `.env`

**Linux:**
```bash
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
sudo yum install tesseract           # CentOS/RHEL
```

### 4. Configure Environment

Create `.env` in project root directory:

```env

# RTSP Stream (use IP Webcam app for testing)
RTSP_URL=rtsp://your_camera_ip:port/stream
FPS=60
FRAME_WIDTH=1280
FRAME_HEIGHT=720

# Model Configuration
OBJECT_DETECTION_MODEL=yolo11s.pt
VLM_MODEL_ID=google/paligemma-3b-mix-224
HF_TOKEN=your_huggingface_token

# OCR Configuration
TESSERACT_PATH="C:/Program Files/Tesseract-OCR/tesseract.exe"
OCR_LANG=eng

# Speech Configuration
TTS_VOICE=0
TTS_RATE=150
STT_TIMEOUT=5

# System Configuration
MAX_WORKERS=8
LOG_LEVEL=INFO
DEBUG=False
LOCAL_SAVE_PATH=data/captured_frames

# Performance Configuration
TARGET_FPS=60
MAX_QUEUE_SIZE=30
PROCESSING_INTERVAL=0.016
DETECTION_CONF_THRESHOLD=0.25
IOU_THRESHOLD=0.45
```

### 5. Hugging Face Token (Optional)

For PaliGemma VLM access:
1. Create account at [Hugging Face](https://huggingface.co/)
2. Generate access token
3. Add to `HF_TOKEN` in `.env`

## Usage

### Starting SmartVis

```bash
python main.py
```

### Controls

- **'S'** - Ask a question about the current frame
- **'O'** - Extract text from the current frame  
- **'C'** - Manual current frame capture
- **'Q'** - Quit application

### Testing RTSP Stream

For testing, use IP Webcam app on Android:
1. Install "IP Webcam" from Play Store
2. Start server and note the IP
3. Update `RTSP_URL` in `.env`

## Architecture

```
smartvis/
├── main.py                    # Main application entry
├── config/
│   └── settings.py           # Configuration management
├── models/
│   ├── object_detection.py   # YOLO11 implementation
│   ├── vlm_processor.py      # PaliGemma VLM
│   └── ocr_processor.py      # Tesseract OCR
├── services/
│   ├── stream_processor.py   # Stream processing orchestration
│   ├── image_processor.py    # Frame display and saving
│   └── speech_processor.py   # Speech recognition/synthesis
└── utils/
    └── logger.py             # Logging utilities
```

## Key Components

### Object Detection
- **Model**: YOLO11s (YOLOv11 Small)
- **Performance**: 60+ FPS on modern GPUs
- **Features**: Real-time bounding boxes, confidence scores, class labels

### Vision Language Model
- **Model**: [PaliGemma 3B](https://huggingface.co/google/paligemma-3b-mix-224)
- **Capabilities**: Visual question answering, scene description, object counting
- **Input**: Natural language questions about images

### OCR Processing
- **Engine**: Tesseract OCR
- **Features**: Text extraction with bounding boxes, confidence scores
- **Languages**: Configurable (default: English)

### Speech Processing
- **Recognition**: Google Speech Recognition
- **Synthesis**: pyttsx3 TTS engine
- **Workflow**: Voice question → VLM analysis → Spoken response

## Performance Optimization

- **Multi-threading**: Separate threads for capture, processing, and display
- **Frame Skipping**: Adaptive skipping based on processing load
- **Queue Management**: LIFO queues for latest frames
- **Mixed Precision**: FP16 inference on CUDA
- **Warm-up**: Model pre-loading for consistent performance

## Troubleshooting

### RTSP Stream Issues
- Verify camera IP and port
- Check network connectivity
- Try different stream URLs (h264_pcm.sdp, live.sdp)

### CUDA/GPU Issues
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Tesseract OCR Issues
- Ensure Tesseract is in PATH or TESSERACT_PATH is correct
- Install additional language packs if needed

### Audio Issues
- Check microphone permissions
- Verify audio device availability
- Test with system audio settings

## Configuration Options

### Performance Tuning
- `TARGET_FPS`: Desired frame rate (default: 60)
- `MAX_QUEUE_SIZE`: Frame buffer size (default: 30)
- `DETECTION_CONF_THRESHOLD`: Detection confidence (default: 0.25)
- `MAX_WORKERS`: Thread pool size (default: 8)

### Model Selection
- `OBJECT_DETECTION_MODEL`: YOLO model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt)
- `VLM_MODEL_ID`: Hugging Face model identifier
- `OCR_LANG`: Tesseract language codes (eng, spa, fra, etc.)

## Hardware Recommendations

### Minimum Requirements
- CPU: 4+ cores
- RAM: 8GB
- GPU: GTX 1060 / RTX 2060 (4GB VRAM)
- Storage: 5GB free space

### Recommended Setup
- CPU: 8+ cores (Intel i7/AMD Ryzen 7)
- RAM: 16GB+
- GPU: RTX 3070/4060 (8GB VRAM)
- SSD: For model storage and frame saving

## License

This project uses multiple open-source components:
- YOLO11: [Ultralytics](https://github.com/ultralytics/ultralytics)
- PaliGemma: [Google Research](https://huggingface.co/google/paligemma-3b-mix-224)
- Tesseract: Apache 2.0 License
