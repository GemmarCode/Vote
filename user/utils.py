import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def process_webcam_image(base64_image):
    """Process webcam image from base64 string to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_image)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size (200x200)
        resized = cv2.resize(gray, (200, 200))
        
        return resized
    except Exception as e:
        print(f"Error processing webcam image: {str(e)}")
        return None

def calculate_face_similarity(face1, face2):
    """Calculate similarity between two face images using SSIM"""
    try:
        # Ensure both images are the same size
        if face1.shape != face2.shape:
            face2 = cv2.resize(face2, (face1.shape[1], face1.shape[0]))
        
        # Calculate SSIM (Structural Similarity Index)
        score = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)[0][0]
        
        # Convert score to range 0-1
        similarity = (score + 1) / 2
        
        return similarity
    except Exception as e:
        print(f"Error calculating face similarity: {str(e)}")
        return 0 