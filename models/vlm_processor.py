import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import cv2
import numpy as np
from utils.logger import logger
from config.settings import settings

class VLMProcessor:
    def __init__(self):
        self.model_id = settings.VLM_MODEL_ID
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Load processor and model using transformers
            self.processor = PaliGemmaProcessor.from_pretrained(
                self.model_id,
                token=settings.HF_TOKEN if settings.HF_TOKEN else None
            )
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map=self.device,
                token=settings.HF_TOKEN if settings.HF_TOKEN else None
            )
            
            logger.info(f"VLM processor initialized with {self.model_id} on {self.device}")
            
        except Exception as e:
            logger.error(f"VLM initialization failed: {e}")
            self.processor = None
            self.model = None
    
    def _format_question_for_paligemma(self, question):
        """Format questions with proper image tokens for PaliGemma"""
        question_lower = question.lower().strip()
        
        # Add image token at the beginning as recommended
        image_token = "<image>"
        
        # Convert common questions to PaliGemma-compatible prompts
        if any(word in question_lower for word in ['what do you see', 'describe', 'what is in', 'tell me about']):
            return f"{image_token} describe en"
        elif any(word in question_lower for word in ['count', 'how many']):
            return f"{image_token} count {question_lower.split()[-1]}" if question_lower.split() else f"{image_token} count objects"
        elif any(word in question_lower for word in ['detect', 'find', 'locate']):
            # Extract object to detect
            words = question_lower.split()
            if len(words) > 1:
                target = words[-1]
                return f"{image_token} detect {target}"
            return f"{image_token} detect objects"
        elif any(word in question_lower for word in ['caption', 'title']):
            return f"{image_token} caption en"
        elif any(word in question_lower for word in ['ocr', 'text', 'read']):
            return f"{image_token} ocr"
        else:
            # For other questions, try a general description approach
            return f"{image_token} describe en"
    
    def analyze_image(self, frame, question="What do you see in this image?"):
        if self.processor is None or self.model is None:
            return "VLM not available - check HF_TOKEN and model access"
        
        try:
            # Convert frame to PIL Image
            if isinstance(frame, np.ndarray):
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
                pil_image = Image.fromarray(frame_rgb)
            else:
                pil_image = frame
            
            # Resize to model's expected input size (PaliGemma uses 224x224)
            pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Format the question for PaliGemma with proper image tokens
            formatted_prompt = self._format_question_for_paligemma(question)
            
            logger.info(f"Using PaliGemma prompt: '{formatted_prompt}' for question: '{question}'")
            
            # Prepare inputs for PaliGemma - only pass text with image tokens
            inputs = self.processor(
                text=formatted_prompt,
                images=pil_image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate response with optimized parameters for PaliGemma
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Reduced for PaliGemma
                    do_sample=False,    # Deterministic for better results
                    temperature=1.0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens (skip input)
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.processor.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            
            # Handle empty or very short responses
            if not response or len(response) < 2:
                # Try alternative prompt for description
                if not formatted_prompt.endswith("describe en"):
                    logger.info("Retrying with description prompt")
                    return self._retry_with_description(pil_image)
                else:
                    response = "I can see the image but cannot provide a detailed description."
            
            # Enhance response based on original question context
            enhanced_response = self._enhance_response(response, question)
            
            logger.info(f"VLM Response: {enhanced_response}")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"VLM analysis error: {e}")
            return f"Vision analysis failed: {str(e)}"
    
    def _retry_with_description(self, pil_image):
        """Retry with a simple description prompt"""
        try:
            inputs = self.processor(
                text="<image> describe en",  # Add image token here too
                images=pil_image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.processor.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return response.strip() if response.strip() else "I can see the image but cannot describe it in detail."
            
        except Exception as e:
            logger.error(f"Retry failed: {e}")
            return "Unable to analyze the image."
    
    def _enhance_response(self, response, original_question):
        """Enhance the response based on the original question context"""
        if not response or len(response) < 3:
            return response
        
        question_lower = original_question.lower()
        
        # Add context based on question type
        if any(word in question_lower for word in ['what do you see', 'describe', 'what is in']):
            if not response.startswith(('I can see', 'The image shows', 'This image', 'There is', 'There are')):
                response = f"I can see {response}"
        elif any(word in question_lower for word in ['count', 'how many']):
            if not any(num_word in response.lower() for num_word in ['one', 'two', 'three', 'four', 'five', 'many', 'several', 'multiple']):
                response = f"I can see {response}"
        
        return response