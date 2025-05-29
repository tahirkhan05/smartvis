import cv2
import numpy as np
import torch
from ultralytics import YOLO
from utils.logger import logger
import time
import random
from config.settings import settings

class ObjectDetector:
    def __init__(self, model_path):
        # Fix: Use lowercase model name
        if model_path == "yolo11s.pt":
            model_path = "yolo11s.pt"  # Ensure lowercase
        
        self.model = self._load_model(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Optimized settings for high FPS
        self.conf_threshold = settings.DETECTION_CONF_THRESHOLD
        self.iou_threshold = settings.IOU_THRESHOLD
        self.max_det = 100
        
        # Mixed precision for CUDA
        self.use_amp = self.device == 'cuda'
        
        # Color palette for different classes
        self.colors = self._generate_color_palette()
        
        self._warmup()
        logger.info(f"YOLO11s loaded on {self.device}")
    
    def _load_model(self, model_path):
        try:
            # Download if not exists
            model = YOLO(model_path)
            logger.info(f"Model {model_path} loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            # Fallback to nano model
            logger.info("Falling back to yolo11n.pt")
            return YOLO("yolo11n.pt")
    
    def _generate_color_palette(self):
        """Generate a fixed set of distinct colors for different object classes"""
        np.random.seed(42)  # Fixed seed for consistent colors
        colors = []
        for _ in range(100):  # Generate 100 colors
            color = tuple(np.random.randint(50, 255, 3).tolist())
            colors.append(color)
        return colors
    
    def _warmup(self):
        try:
            # Warm up with dummy image
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(3):
                _ = self.model(dummy_img, verbose=False)
            logger.info("Model warm-up completed")
        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")
    
    def detect_objects(self, frame):
        try:
            start_time = time.time()
            
            # Run inference
            results = self.model(
                frame, 
                verbose=False,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det
            )
            
            detected_objects = []
            annotated_frame = frame.copy()
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    for i, box in enumerate(boxes):
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        label = self.model.names[class_id] if class_id < len(self.model.names) else f"class_{class_id}"
                        
                        detected_objects.append({
                            'label': label,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        })
                        
                        # Get consistent color for this class
                        color = self.colors[class_id % len(self.colors)]
                        
                        # Draw bounding box with class-specific color
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Draw label with better readability
                        label_text = f"{label} {confidence:.2f}"
                        
                        # Calculate text size with larger font
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label_text, font, font_scale, thickness
                        )
                        
                        # Draw background rectangle for text
                        cv2.rectangle(
                            annotated_frame, 
                            (x1, y1 - text_height - baseline - 10), 
                            (x1 + text_width + 10, y1), 
                            color, 
                            -1
                        )
                        
                        # Draw text in white for better contrast
                        cv2.putText(
                            annotated_frame, 
                            label_text, 
                            (x1 + 5, y1 - 8), 
                            font, 
                            font_scale, 
                            (255, 255, 255),  # White text
                            thickness
                        )
            
            inference_time = (time.time() - start_time) * 1000
            if settings.DEBUG:
                logger.debug(f"Detection: {inference_time:.1f}ms, {len(detected_objects)} objects")
            
            return detected_objects, annotated_frame
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return [], frame