import cv2
import time
import sys
import threading
import queue
import os
from config.settings import settings
from services.stream_processor import StreamProcessor
from services.image_processor import ImageProcessor
from services.speech_processor import SpeechProcessor
from utils.logger import logger

class SmartVis:
    def __init__(self):
        self._print_welcome()
        logger.info("Initializing SmartVis...")
        
        # Ensure save directory exists
        os.makedirs(settings.LOCAL_SAVE_PATH, exist_ok=True)
        logger.info(f"Frame save directory: {settings.LOCAL_SAVE_PATH}")
        
        self.stream_processor = StreamProcessor()
        self.image_processor = ImageProcessor()
        self.speech_processor = SpeechProcessor()
        
        self.running = False
        self.info_text = "Real-time Object Detection Active"
        self.latest_frame = None
        
        # Frame saving
        self.last_save_time = 0
        self.save_interval = 1.0  # Save frame every 1 second

        # Non-blocking processing queues
        self.speech_queue = queue.Queue()
        self.ocr_queue = queue.Queue()
        self.processing_lock = threading.Lock()
        self.is_processing_speech = False
        self.is_processing_ocr = False
        
        # Start background workers
        self.speech_worker = threading.Thread(target=self._speech_worker, daemon=True)
        self.ocr_worker = threading.Thread(target=self._ocr_worker, daemon=True)

        logger.info("SmartVis initialized successfully")

    def _print_welcome(self):
        """Print welcome message with system info"""
        print("\n" + "="*60)
        print("🚀 SMARTVIS - AI VISION SYSTEM")
        print("="*60)
        print("🎯 Real-time Object Detection with YOLO11")
        print("🧠 Vision Language Model (VLM) Integration")
        print("📄 Optical Character Recognition (OCR)")
        print("🎤 Voice Commands & Text-to-Speech")
        print("📹 RTSP Stream Processing")
        print("-"*60)
        print("⚙️  SYSTEM CONFIGURATION:")
        print(f"   📊 Target FPS: {settings.TARGET_FPS}")
        print(f"   📺 Resolution: {settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT}")
        print(f"   🎥 RTSP URL: {settings.RTSP_URL}")
        print(f"   💾 Save Path: {settings.LOCAL_SAVE_PATH}")
        print(f"   🔧 Max Workers: {settings.MAX_WORKERS}")
        print("-"*60)
        print("🎮 CONTROLS:")
        print("   'S' - Ask a question about what you see")
        print("   'O' - Extract text from current frame")
        print("   'Q' - Quit application")
        print("="*60)
        print("🔄 Initializing components...")
    
    def start(self):
        logger.info("Starting SmartVis...")
        
        # Start background workers
        self.speech_worker.start()
        self.ocr_worker.start()
        
        # Start stream processing - detection is always active now
        self.stream_processor.start()
        self.running = True

        print("✅ SmartVis started successfully!")
        print("🎥 Live stream active - Object detection running...")
        logger.info("SmartVis started successfully")
        logger.info("Controls: 's'=ask question | 'o'=extract text | 'q'=quit")
        
        try:
            self.main_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def main_loop(self):
        while self.running:
            # Get latest processed frame
            result = self.stream_processor.get_latest_result()
            
            if result:
                frame = result["frame"]
                
                # Store latest frame (thread-safe)
                with self.processing_lock:
                    self.latest_frame = frame.copy()
                
                # Auto-save frames periodically
                current_time = time.time()
                if current_time - self.last_save_time > self.save_interval:
                    self._save_current_frame()
                    self.last_save_time = current_time
                
                # Show detection results count
                if result["type"] == "detection":
                    detection_count = len(result["data"])
                    if not self.is_processing_speech and not self.is_processing_ocr:
                        self.info_text = f"Detecting objects... Found: {detection_count} objects"
                
                # Display frame with UI - this continues smoothly
                key = self.image_processor.display_frame(
                    frame, 
                    self.stream_processor.current_fps,
                    self.info_text
                )
                
                # Handle key presses - non-blocking
                if key == ord('q'):
                    break
                elif key == ord('s') and not self.is_processing_speech:
                    self._queue_speech_processing()
                elif key == ord('o') and not self.is_processing_ocr:
                    self._queue_ocr_processing()
                elif key == ord('c'):  # Manual capture
                    self._save_current_frame(prefix="manual")
            
            time.sleep(0.001)  # Minimal delay for smooth streaming
    
    def _save_current_frame(self, prefix="auto"):
        """Save current frame to disk"""
        with self.processing_lock:
            if self.latest_frame is not None:
                filename = self.image_processor.save_frame(self.latest_frame, prefix)
                if filename and prefix == "manual":
                    self.info_text = f"📸 Frame saved: {os.path.basename(filename)}"
                    self._reset_info_delayed(2, "save")
    
    def _queue_speech_processing(self):
        """Queue speech processing without blocking main thread"""
        with self.processing_lock:
            if self.latest_frame is not None:
                frame_copy = self.latest_frame.copy()
                self.speech_queue.put(frame_copy)
                self.is_processing_speech = True
                self.info_text = "🎤 Ready to listen..."
    
    def _queue_ocr_processing(self):
        """Queue OCR processing without blocking main thread"""
        with self.processing_lock:
            if self.latest_frame is not None:
                frame_copy = self.latest_frame.copy()
                self.ocr_queue.put(frame_copy)
                self.is_processing_ocr = True
                self.info_text = "📄 Extracting text..."
    
    def _speech_worker(self):
        """Background worker for speech processing"""
        while True:
            try:
                # Wait for speech processing request
                frame = self.speech_queue.get(timeout=1)
                
                # Update UI
                self.info_text = "🎤 Listening for your question..."
                
                # Save frame for speech analysis
                self.image_processor.save_frame(frame, "speech_analysis")
                
                # Listen for speech (this blocks but runs in background)
                text = self.speech_processor.listen_for_speech(timeout=10)
                
                if text:
                    self.info_text = f"🤔 Processing: {text[:30]}..."
                    
                    # Use VLM to answer the question
                    response = self.stream_processor.analyze_with_vlm(frame, text)
                    
                    # Speak the response
                    self.speech_processor.speak(response)
                    self.info_text = f"💬 Answer: {response[:50]}..."
                    logger.info(f"Question: {text}")
                    logger.info(f"Answer: {response}")
                    
                    # Reset info after delay
                    self._reset_info_delayed(5, "speech")
                else:
                    self.info_text = "❌ No question detected"
                    self._reset_info_delayed(3, "speech")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speech worker error: {e}")
                self.info_text = f"❌ Speech error: {str(e)[:30]}"
                self._reset_info_delayed(3, "speech")
    
    def _ocr_worker(self):
        """Background worker for OCR processing"""
        while True:
            try:
                # Wait for OCR processing request
                frame = self.ocr_queue.get(timeout=1)
                
                # Save frame for OCR analysis
                self.image_processor.save_frame(frame, "ocr_analysis")
                
                # Extract text from frame
                text_results, _ = self.stream_processor.extract_text_from_frame(frame)
                
                if text_results:
                    # Combine all detected text
                    all_text = " ".join([result['text'] for result in text_results])
                    self.info_text = f"📄 Text found: {all_text[:50]}..."
                    
                    # Speak the extracted text
                    self.speech_processor.speak(f"I found the following text: {all_text}")
                    logger.info(f"Extracted text: {all_text}")
                else:
                    self.info_text = "📄 No text detected in image"
                    self.speech_processor.speak("No text was found in the image")
                
                # Reset info after delay
                self._reset_info_delayed(3, "ocr")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"OCR worker error: {e}")
                self.info_text = f"❌ OCR error: {str(e)[:30]}"
                self._reset_info_delayed(3, "ocr")
    
    def _reset_info_delayed(self, delay, task_type):
        """Reset info text after delay"""
        def reset():
            time.sleep(delay)
            with self.processing_lock:
                if task_type == "speech":
                    self.is_processing_speech = False
                elif task_type == "ocr":
                    self.is_processing_ocr = False
                
                # Only reset if no other processing is active
                if not self.is_processing_speech and not self.is_processing_ocr:
                    self.info_text = "Real-time Object Detection Active"
        
        threading.Thread(target=reset, daemon=True).start()
    
    def stop(self):
        print("\n🛑 Shutting down SmartVis...")
        logger.info("Stopping SmartVis...")
        self.running = False
        self.stream_processor.stop()
        cv2.destroyAllWindows()
        print("✅ SmartVis stopped successfully")
        print("👋 Thank you for using SmartVis!")
        logger.info("SmartVis stopped")

if __name__ == "__main__":
    # Check if running with proper Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    try:
        rover = SmartVis()
        rover.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)