import os
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image

def process_webcam_image(base64_image):
    """
    Process the base64 encoded webcam image to a format usable for face detection
    """
    try:
        # Remove the data URL prefix if present
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
            
        # Decode the base64 string
        image_data = base64.b64decode(base64_image)
        
        # Convert to image
        image = Image.open(BytesIO(image_data))
        
        # Convert PIL image to OpenCV format (numpy array)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        print(f"Processed webcam image, shape: {opencv_image.shape}")
        return opencv_image
    except Exception as e:
        print(f"Error processing webcam image: {str(e)}")
        return None

def preprocess_face_image(image, target_size=(224, 224)):
    """
    Preprocess a face image for comparison:
    1. Convert to grayscale for more robust comparison
    2. Resize to a standard size
    3. Apply histogram equalization for lighting normalization
    4. Apply Gaussian blur to reduce noise
    """
    try:
        if image is None:
            print("Input image is None")
            return None
            
        # Make a copy to avoid modifying the original
        processed = image.copy()
        
        # Convert to grayscale
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
        # Resize to target size
        processed = cv2.resize(processed, target_size)
        
        # Apply histogram equalization
        processed = cv2.equalizeHist(processed)
        
        # Apply slight Gaussian blur to reduce noise
        processed = cv2.GaussianBlur(processed, (5, 5), 0)
        
        return processed
    except Exception as e:
        print(f"Error preprocessing face image: {str(e)}")
        return None

def verify_voter_face(captured_face, stored_face, threshold=0.45):
    """
    Verify if the captured face matches the stored face
    Returns dict with verification result and distance
    """
    try:
        # Use face_recognition for more accurate comparison if available
        try:
            import face_recognition
            
            # Convert images if needed (face_recognition expects RGB)
            if len(captured_face.shape) == 3 and captured_face.shape[2] == 3:
                captured_rgb = cv2.cvtColor(captured_face, cv2.COLOR_BGR2RGB)
            else:
                captured_rgb = captured_face
                
            if len(stored_face.shape) == 3 and stored_face.shape[2] == 3:
                stored_rgb = cv2.cvtColor(stored_face, cv2.COLOR_BGR2RGB)
            else:
                stored_rgb = stored_face
            
            # Get face encodings
            captured_encoding = face_recognition.face_encodings(captured_rgb)
            stored_encoding = face_recognition.face_encodings(stored_rgb)
            
            if captured_encoding and stored_encoding:
                # Calculate face distance
                distance = face_recognition.face_distance([stored_encoding[0]], captured_encoding[0])[0]
                verified = distance < threshold
                
                return {
                    'verified': verified,
                    'distance': float(distance),
                    'method': 'face_recognition'
                }
            else:
                print("Could not extract face encodings, using fallback method")
        except (ImportError, IndexError) as e:
            print(f"Face recognition not available: {str(e)}, using fallback")
            
        # Fallback to simple image comparison
        # Preprocess both images
        captured_processed = preprocess_face_image(captured_face)
        stored_processed = preprocess_face_image(stored_face)
        
        if captured_processed is None or stored_processed is None:
            return {'verified': False, 'distance': 1.0, 'method': 'fallback_failed'}
        
        # Try structural similarity if available
        try:
            from skimage.metrics import structural_similarity as ssim
            similarity = ssim(stored_processed, captured_processed)
            distance = 1.0 - similarity  # Convert similarity to distance
            
            return {
                'verified': distance < threshold,
                'distance': float(distance),
                'method': 'ssim'
            }
        except ImportError:
            # Use histogram comparison as last resort
            hist1 = cv2.calcHist([stored_processed], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([captured_processed], [0], None, [256], [0, 256])
            
            # Normalize histograms
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # Compare histograms (correlation method - higher is better)
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            distance = 1.0 - correlation  # Convert to distance
            
            return {
                'verified': distance < threshold,
                'distance': float(distance),
                'method': 'histogram'
            }
    except Exception as e:
        print(f"Error in face verification: {str(e)}")
        return {'verified': False, 'distance': 1.0, 'method': 'error'} 