import cv2
import numpy as np
import base64
import insightface
from insightface.app import FaceAnalysis
from django.conf import settings
import os
import traceback

class FaceRecognition:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
    def get_face_embedding(self, image):
        """
        Get face embedding from an image
        """
        try:
            faces = self.app.get(image)
            if len(faces) != 1:
                return None
            return faces[0].embedding
        except Exception as e:
            print(f"Error getting face embedding: {str(e)}")
            return None
            
    def verify_face(self, face1, face2):
        """
        Verify if two faces match
        """
        try:
            embedding1 = self.get_face_embedding(face1)
            embedding2 = self.get_face_embedding(face2)
            
            if embedding1 is None or embedding2 is None:
                return {
                    'verified': False,
                    'message': 'Could not detect face in one or both images',
                    'distance': float('inf')
                }
            
            # Calculate distance between embeddings
            distance = np.linalg.norm(embedding1 - embedding2)
            
            # Set threshold for verification (adjust as needed)
            threshold = 0.6
            
            return {
                'verified': distance < threshold,
                'message': 'Face verification successful' if distance < threshold else 'Face verification failed',
                'distance': distance
            }
            
        except Exception as e:
            print(f"Error during face verification: {str(e)}")
            return {
                'verified': False,
                'message': f'Error during face verification: {str(e)}',
                'distance': float('inf')
            }

def process_webcam_image(face_data):
    """
    Process the webcam image data and return a numpy array
    """
    try:
        # Remove data URL prefix if present
        if ',' in face_data:
            face_data = face_data.split(',')[1]
        
        # Decode base64 image
        image_data = base64.b64decode(face_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
            
        return img
        
    except Exception as e:
        print(f"Error processing webcam image: {str(e)}")
        return None

def verify_voter_face(captured_face, stored_face):
    """
    Verify if the captured face matches the stored face using InsightFace
    """
    try:
        # Initialize face analyzer with different parameters
        app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
        app.prepare(ctx_id=0, det_thresh=0.3, det_size=(320, 320))  # Lower detection threshold and size
        
        # Debug image properties
        print(f"Image shape: {stored_face.shape}")
        print(f"Image dtype: {stored_face.dtype}")
        print(f"Image min/max values: {stored_face.min()}, {stored_face.max()}")
        
        # Ensure image is in correct format (BGR)
        if len(stored_face.shape) == 3 and stored_face.shape[2] == 3:
            # Try different preprocessing steps
            preprocessing_steps = [
                lambda img: img,  # Original
                lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB),  # RGB
                lambda img: cv2.resize(img, (640, 640)),  # Resize
                lambda img: cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX),  # Normalize
                lambda img: cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))  # Enhance contrast
            ]
            
            stored_faces = None
            for i, step in enumerate(preprocessing_steps):
                try:
                    processed_image = step(stored_face.copy())
                    if len(processed_image.shape) == 2:  # If grayscale, convert back to BGR
                        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
                    faces = app.get(processed_image)
                    print(f"Preprocessing step {i}: Found {len(faces)} faces")
                    if len(faces) > 0:
                        stored_faces = faces
                        break
                except Exception as e:
                    print(f"Error in preprocessing step {i}: {str(e)}")
                    continue
        
            if stored_faces is None or len(stored_faces) == 0:
                print("No face detected in stored image after all preprocessing attempts")
                return {
                    'verified': False,
                    'message': 'No face detected in stored image',
                    'distance': float('inf')
                }
        
        # Process captured face
        captured_faces = app.get(captured_face)
        print(f"Captured faces detected: {len(captured_faces)}")
        
        if len(captured_faces) != 1:
            return {
                'verified': False,
                'message': 'Please ensure only one face is visible in the frame',
                'distance': float('inf')
            }
        
        # Get face embeddings
        captured_embedding = captured_faces[0].embedding
        stored_embedding = stored_faces[0].embedding
        
        # Calculate distance
        distance = np.linalg.norm(captured_embedding - stored_embedding)
        print(f"Calculated distance: {distance}")
        
        # Normalize the distance to a 0-1 scale
        normalized_distance = 1 / (1 + distance)  # This will convert large distances to small values
        print(f"Normalized distance: {normalized_distance}")
        
        # Now compare with threshold
        return {
            'verified': normalized_distance > 0.6,  # Note: changed to > because higher normalized value means better match
            'message': 'Face verification successful' if normalized_distance > 0.6 else 'Face verification failed',
            'distance': normalized_distance
        }
        
    except Exception as e:
        print(f"Error during face verification: {str(e)}")
        print(f"Full error: {traceback.format_exc()}")
        return {
            'verified': False,
            'message': f'Error during face verification: {str(e)}',
            'distance': float('inf')
        } 