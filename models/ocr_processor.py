import cv2
import numpy as np
from PIL import Image
from utils.logger import logger
from config.settings import settings
import shutil
import subprocess
import sys
import os
import platform

class OCRProcessor:
    def __init__(self):
        self.tesseract_available = self._check_tesseract()
        if self.tesseract_available:
            import pytesseract
            self.pytesseract = pytesseract
            
            if settings.TESSERACT_PATH:
                pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH
            
            self.config = f'--oem 3 --psm 6 -l {settings.OCR_LANG}'
            logger.info("OCR processor initialized with Tesseract")
        else:
            self.pytesseract = None
            logger.warning("Tesseract not available - OCR disabled")
            self._print_installation_guide()
    
    def _check_tesseract(self):
        """Check if Tesseract is available with Windows-specific paths"""
        try:
            # First check if pytesseract can be imported
            import pytesseract
            
            # Windows-specific common installation paths
            if platform.system() == "Windows":
                common_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
                    settings.TESSERACT_PATH if settings.TESSERACT_PATH else ""
                ]
                
                for path in common_paths:
                    if path and os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        logger.info(f"Found Tesseract at: {path}")
                        return self._test_tesseract(path)
            
            # Check if tesseract executable exists in PATH
            tesseract_cmd = shutil.which("tesseract")
            if tesseract_cmd:
                return self._test_tesseract(tesseract_cmd)
            
            logger.warning("Tesseract executable not found")
            return False
                
        except ImportError:
            logger.warning("pytesseract not installed. Install with: pip install pytesseract")
            return False
        except Exception as e:
            logger.warning(f"Tesseract check failed: {e}")
            return False
    
    def _test_tesseract(self, tesseract_path):
        """Test if tesseract actually works"""
        try:
            result = subprocess.run([tesseract_path, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info(f"Tesseract test successful: {tesseract_path}")
                return True
            else:
                logger.warning(f"Tesseract test failed: {tesseract_path}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning("Tesseract test timed out")
            return False
        except Exception as e:
            logger.warning(f"Tesseract test error: {e}")
            return False
    
    def _print_installation_guide(self):
        """Print OS-specific installation instructions"""
        system = platform.system()
        
        print("\n" + "="*60)
        print("TESSERACT OCR INSTALLATION REQUIRED")
        print("="*60)
        
        if system == "Windows":
            print("Windows Installation:")
            print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("2. Install the .exe file")
            print("3. Add to PATH or update .env with TESSERACT_PATH")
            print("4. Install Python package: pip install pytesseract")
        elif system == "Darwin":  # macOS
            print("macOS Installation:")
            print("1. Install via Homebrew: brew install tesseract")
            print("2. Install Python package: pip install pytesseract")
        else:  # Linux
            print("Linux Installation:")
            print("1. Ubuntu/Debian: sudo apt-get install tesseract-ocr")
            print("2. CentOS/RHEL: sudo yum install tesseract")
            print("3. Install Python package: pip install pytesseract")
        
        print("\nAfter installation, restart the application.")
        print("="*60 + "\n")
    
    def extract_text(self, frame):
        if not self.tesseract_available:
            return "OCR not available - Tesseract not installed"
        
        try:
            # Preprocessing for better OCR
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Enhance text visibility
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Extract text
            text = self.pytesseract.image_to_string(gray, config=self.config)
            text = text.strip()
            
            if text:
                logger.debug(f"OCR extracted: {text[:100]}...")
            
            return text
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def extract_text_with_boxes(self, frame):
        if not self.tesseract_available:
            return [], frame
        
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Get text with bounding boxes
            data = self.pytesseract.image_to_data(gray, config=self.config, 
                                                output_type=self.pytesseract.Output.DICT)
            
            results = []
            annotated_frame = frame.copy()
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Confidence threshold
                    text = data['text'][i].strip()
                    if text:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        results.append({
                            'text': text,
                            'bbox': [x, y, x+w, y+h],
                            'confidence': data['conf'][i]
                        })
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, text, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            return results, annotated_frame
        except Exception as e:
            logger.error(f"OCR with boxes error: {e}")
            return [], frame