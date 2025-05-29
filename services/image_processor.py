import cv2
import os
import time
from config.settings import settings
from utils.logger import logger

class ImageProcessor:
    def __init__(self):
        os.makedirs(settings.LOCAL_SAVE_PATH, exist_ok=True)
    
    def save_frame(self, frame, prefix="frame"):
        try:
            timestamp = int(time.time() * 1000)
            filename = f"{settings.LOCAL_SAVE_PATH}/{prefix}_{timestamp}.jpg"
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            logger.info(f"Frame saved: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Save error: {e}")
            return None
    
    def display_frame(self, frame, fps=0, info_text=""):
        # Add UI elements
        display_frame = self._add_ui_overlay(frame, fps, info_text)
        
        cv2.namedWindow('SmartVis', cv2.WINDOW_NORMAL)
        cv2.imshow('SmartVis', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        return key
    
    def _add_ui_overlay(self, frame, fps, info_text):
        overlay_frame = frame.copy()
        h, w, _ = frame.shape
        
        # Status bar background - taller for better text
        cv2.rectangle(overlay_frame, (0, h-80), (w, h), (40, 40, 40), -1)
        
        # Header background
        cv2.rectangle(overlay_frame, (0, 0), (w, 60), (40, 40, 40), -1)
        
        # FPS counter with larger font
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(overlay_frame, fps_text, (15, h-45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Controls with better readability
        controls = "Controls: 'S' = Ask Question | 'O' = Extract Text | 'Q' = Quit"
        cv2.putText(overlay_frame, controls, (15, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Title with larger font
        cv2.putText(overlay_frame, "SMARTVIS - AI Vision System", (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Status/info text with better positioning
        if info_text:
            # Split long text into multiple lines if needed
            max_chars = 80
            if len(info_text) > max_chars:
                lines = [info_text[i:i+max_chars] for i in range(0, len(info_text), max_chars)]
                for i, line in enumerate(lines[:2]):  # Show max 2 lines
                    cv2.putText(overlay_frame, line, (15, 85 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(overlay_frame, info_text, (15, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add performance indicator
        if fps > 0:
            if fps >= 30:
                perf_color = (0, 255, 0)  # Green
                perf_text = "EXCELLENT"
            elif fps >= 20:
                perf_color = (0, 255, 255)  # Yellow
                perf_text = "GOOD"
            elif fps >= 10:
                perf_color = (0, 165, 255)  # Orange
                perf_text = "FAIR"
            else:
                perf_color = (0, 0, 255)  # Red
                perf_text = "POOR"
            
            cv2.putText(overlay_frame, perf_text, (w-150, h-45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, perf_color, 2)
        
        return overlay_frame