import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.model_zoo import get_model
from typing import Tuple, Optional
import os

def process_webcam_image(image_data):
    """Process base64 image data from webcam."""
    try:
        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize to standard size
        image = cv2.resize(image, (640, 640))
        
        return image
        
    except Exception as e:
        print(f"Error processing webcam image: {str(e)}")
        return None

class FaceRecognition:
    def __init__(self):
        """Initialize InsightFace models."""
        # Initialize face analysis
        self.app = FaceAnalysis(
            name='buffalo_l',  # Using large model for better accuracy
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize face recognition model
        self.recognition_model = get_model('arcface_r100')
        
    def get_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding using InsightFace."""
        try:
            # Detect and get face embedding
            faces = self.app.get(image)
            
            if not faces:
                print("No face detected in image")
                return None
                
            # Get the first face (assuming single face)
            face = faces[0]
            
            # Get face embedding
            embedding = face.embedding
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error getting face embedding: {str(e)}")
            return None
    
    def compare_faces(self, face1: np.ndarray, face2: np.ndarray) -> Tuple[float, float]:
        """Compare two faces using InsightFace."""
        try:
            # Get face embeddings
            embedding1 = self.get_face_embedding(face1)
            embedding2 = self.get_face_embedding(face2)
            
            if embedding1 is None or embedding2 is None:
                return 0.0, 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            
            return similarity, 0.0
            
        except Exception as e:
            print(f"Error comparing faces: {str(e)}")
            return 0.0, 0.0

def verify_voter_face(uploaded_face_data, stored_face_data):
    """
    Verify voter's face by comparing uploaded face with stored face data.
    Returns a dictionary with verification results.
    """
    try:
        # Initialize face recognition
        face_recognition = FaceRecognition()
        
        # Convert stored face data to numpy array if it's bytes
        if isinstance(stored_face_data, bytes):
            stored_face = np.frombuffer(stored_face_data, dtype=np.uint8)
            stored_face = cv2.imdecode(stored_face, cv2.IMREAD_COLOR)
        else:
            stored_face = stored_face_data
        
        # Process uploaded face data
        if isinstance(uploaded_face_data, str):
            # If it's base64 string, decode it
            if ',' in uploaded_face_data:
                uploaded_face_data = uploaded_face_data.split(',')[1]
            uploaded_bytes = base64.b64decode(uploaded_face_data)
            uploaded_face = cv2.imdecode(np.frombuffer(uploaded_bytes, np.uint8), cv2.IMREAD_COLOR)
        else:
            uploaded_face = uploaded_face_data
        
        # Ensure images are valid
        if uploaded_face is None or stored_face is None:
            raise ValueError("Invalid image data")
        
        # Compare faces
        similarity, _ = face_recognition.compare_faces(uploaded_face, stored_face)
        
        # Set threshold for verification
        threshold = 0.6
        verified = similarity >= threshold
        
        # Print verification details
        print("\n=== Face Verification Results ===")
        print(f"✅ Verified: {verified}")
        print(f"📏 Similarity Score: {similarity:.4f}")
        print(f"🎯 Threshold: {threshold}")
        
        return {
            'verified': verified,
            'distance': 1 - similarity,  # Convert similarity to distance
            'max_threshold_to_verify': threshold,
            'model': 'InsightFace',
            'similarity_metric': 'cosine'
        }
        
    except Exception as e:
        print(f"Error in face verification: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'verified': False,
            'distance': 1.0,
            'max_threshold_to_verify': 0.6,
            'model': 'InsightFace',
            'similarity_metric': 'cosine',
            'error': str(e)
        }

def can_proceed_to_vote(verification_result):
    """
    Check if the voter can proceed to voting based on face verification results.
    """
    if verification_result.get('verified', False):
        print("\n✅ Face verification successful! You can proceed to voting.")
        return True
    else:
        print("\n❌ Face verification failed. Please try again.")
        print("\nTips to improve verification:")
        print("1. Ensure good lighting - your face should be clearly visible")
        print("2. Look directly at the camera")
        print("3. Remove any face coverings (masks, sunglasses, etc.)")
        print("4. Keep your face centered in the frame")
        print("5. Stay still while the photo is being taken")
        
        distance = verification_result.get('distance', 1.0)
        threshold = verification_result.get('max_threshold_to_verify', 0.6)
        
        print("\nTechnical Details:")
        print(f"Similarity Score: {(1 - distance):.2%}")
        print(f"Required Score: {(1 - threshold):.2%}")
        print(f"Gap to threshold: {(distance - threshold):.4f}")
        return False