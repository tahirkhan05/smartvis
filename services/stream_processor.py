import cv2
import threading
import time
import numpy as np
from queue import Queue, LifoQueue
from config.settings import settings
from utils.logger import logger

class StreamProcessor:
    def __init__(self):
        # Initialize processors
        from models.object_detection import ObjectDetector
        from models.vlm_processor import VLMProcessor
        from models.ocr_processor import OCRProcessor
        
        try:
            self.object_detector = ObjectDetector(settings.OBJECT_DETECTION_MODEL)
        except Exception as e:
            logger.warning(f"Object detection disabled: {e}")
            self.object_detector = None
        
        try:
            self.vlm_processor = VLMProcessor()
        except Exception as e:
            logger.warning(f"VLM disabled: {e}")
            self.vlm_processor = None
            
        try:
            self.ocr_processor = OCRProcessor()
        except Exception as e:
            logger.warning(f"OCR disabled: {e}")
            self.ocr_processor = None
        
        # Queues and threading
        self.frame_queue = Queue(maxsize=settings.MAX_QUEUE_SIZE)
        self.result_queue = LifoQueue(maxsize=10)
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        # Always keep detection active - removed mode switching
        
        # Optimize capture settings
        self.cap_params = {
            cv2.CAP_PROP_BUFFERSIZE: 1,
            cv2.CAP_PROP_FPS: settings.FPS,
            cv2.CAP_PROP_FRAME_WIDTH: settings.FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT: settings.FRAME_HEIGHT,
        }
    
    def capture_frames(self):
        cap = cv2.VideoCapture(settings.RTSP_URL)
        
        # Apply optimized settings
        for prop, value in self.cap_params.items():
            cap.set(prop, value)
        
        if not cap.isOpened():
            logger.error("Failed to open RTSP stream")
            self.stop_event.set()
            return
        
        logger.info("RTSP capture started")
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame capture failed")
                continue
            
            # Maintain queue size for optimal performance
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            
            self.frame_queue.put(frame)
            
            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.current_fps = self.frame_count
                self.frame_count = 0
                self.last_fps_time = current_time
        
        cap.release()
        logger.info("RTSP capture stopped")
    
    def process_frames(self):
        frame_skip = 0
        
        while not self.stop_event.is_set():
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue
            
            frame = self.frame_queue.get()
            
            # Dynamic frame skipping for performance
            if frame_skip > 0:
                frame_skip -= 1
                continue
            
            try:
                start_time = time.time()
                
                # Always run object detection if available
                if self.object_detector:
                    detections, annotated_frame = self.object_detector.detect_objects(frame)
                    result = {"type": "detection", "data": detections, "frame": annotated_frame}
                else:
                    result = {"type": "passthrough", "data": [], "frame": frame}
                
                # Adaptive frame skipping based on processing time
                process_time = time.time() - start_time
                target_time = 1.0 / settings.TARGET_FPS
                if process_time > target_time:
                    frame_skip = max(0, int(process_time / target_time))
                
                # Update result queue
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        pass
                self.result_queue.put(result)
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    def start(self):
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        process_thread = threading.Thread(target=self.process_frames, daemon=True)
        
        capture_thread.start()
        process_thread.start()
        
        logger.info(f"Stream processor started (target: {settings.TARGET_FPS} FPS)")
        return capture_thread, process_thread
    
    def stop(self):
        self.stop_event.set()
    
    def get_latest_result(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None
    
    # Removed set_processing_mode - always detect
    
    def analyze_with_vlm(self, frame, question):
        """Process frame with VLM while keeping detection running"""
        if self.vlm_processor:
            return self.vlm_processor.analyze_image(frame, question)
        return "VLM not available"
    
    def extract_text_from_frame(self, frame):
        """Extract text from frame while keeping detection running"""
        if self.ocr_processor:
            return self.ocr_processor.extract_text_with_boxes(frame)
        return [], frame