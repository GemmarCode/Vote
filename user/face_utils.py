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
        self.face_analyzer = FaceAnalysis(name='buffalo_l')
        self.face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
        
    def detect_face(self, image):
        """Detect face in image"""
        try:
            faces = self.face_analyzer.get(image)
            if len(faces) == 0:
                return None
            return faces[0]
        except Exception as e:
            print(f"Error detecting face: {str(e)}")
            return None
        
    def get_face_landmarks(self, image):
        """Get face landmarks"""
        face = self.detect_face(image)
        if face is None:
            return None
        return face.landmarks_2d
        
    def check_face_visibility(self, landmarks):
        """Check if face is properly visible (not just eyes)"""
        try:
            if landmarks is None or len(landmarks) < 68:
                return False
                
            # Get key facial landmarks
            nose = landmarks[30]
            left_eye = landmarks[36]
            right_eye = landmarks[45]
            mouth_left = landmarks[48]
            mouth_right = landmarks[54]
            
            # Calculate face visibility metrics
            eye_distance = np.linalg.norm(right_eye - left_eye)
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            nose_to_mouth = np.linalg.norm(nose - (mouth_left + mouth_right) / 2)
            
            # Get image dimensions for relative measurements
            img_height = np.linalg.norm(landmarks[8] - landmarks[27])  # Face height
            img_width = np.linalg.norm(landmarks[16] - landmarks[0])   # Face width
            
            # Calculate relative measurements
            relative_eye_distance = eye_distance / img_width
            relative_mouth_width = mouth_width / img_width
            relative_nose_mouth = nose_to_mouth / img_height
            
            # Check if face is properly visible with relative measurements
            # 1. Eyes should be visible and at a reasonable distance
            if relative_eye_distance < 0.2:  # Eyes too close together
                return False
                
            # 2. Mouth should be visible and at a reasonable distance from nose
            if relative_mouth_width < 0.2 or relative_nose_mouth < 0.1:  # Mouth too small or too close to nose
                return False
                
            # 3. Check face angle (should be relatively straight)
            left_eye_center = (landmarks[36] + landmarks[39]) / 2
            right_eye_center = (landmarks[42] + landmarks[45]) / 2
            nose_tip = landmarks[30]
            
            # Calculate face angle
            eye_line = right_eye_center - left_eye_center
            nose_line = nose_tip - (left_eye_center + right_eye_center) / 2
            angle = np.arctan2(np.cross(eye_line, nose_line), np.dot(eye_line, nose_line))
            
            # Face should be relatively straight (angle close to 0)
            if abs(angle) > 0.3:  # More than about 17 degrees
                return False
                
            return True
            
        except Exception as e:
            print(f"Error checking face visibility: {str(e)}")
            return False
        
    def get_face_embedding(self, image):
        """Get face embedding vector"""
        face = self.detect_face(image)
        if face is None:
            return None
        return face.embedding
        
    def compare_faces(self, embedding1, embedding2):
        """Compare two face embeddings and return distance"""
        try:
            if embedding1 is None or embedding2 is None:
                return float('inf')
                
            # Normalize embeddings
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine distance
            distance = 1 - np.dot(embedding1, embedding2)
            return distance
            
        except Exception as e:
            print(f"Error comparing faces: {str(e)}")
            return float('inf')

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
            distance = self.compare_faces(embedding1, embedding2)
            
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
            print("Failed to decode image")
            return None
            
        return img
        
    except Exception as e:
        print(f"Error processing webcam image: {str(e)}")
        print(f"Full error: {traceback.format_exc()}")
        return None

def verify_voter_face(captured_face, stored_face):
    """
    Simplified face verification using multiple methods
    """
    try:
        # Make sure both images are valid
        if captured_face is None or stored_face is None:
            print("One or both images are None")
            return {
                'verified': False,
                'message': 'Invalid image data',
                'distance': float('inf')
            }
            
        # Resize images to the same dimensions
        height, width = 250, 250
        captured_face_resized = cv2.resize(captured_face, (width, height))
        stored_face_resized = cv2.resize(stored_face, (width, height))
        
        # Convert to grayscale
        captured_gray = cv2.cvtColor(captured_face_resized, cv2.COLOR_BGR2GRAY)
        stored_gray = cv2.cvtColor(stored_face_resized, cv2.COLOR_BGR2GRAY)
        
        # 1. Apply histogram equalization to normalize lighting
        captured_eq = cv2.equalizeHist(captured_gray)
        stored_eq = cv2.equalizeHist(stored_gray)
        
        # 2. Calculate histogram correlation
        hist1 = cv2.calcHist([captured_eq], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([stored_eq], [0], None, [256], [0, 256])
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        print(f"Histogram score: {hist_score}")
        
        # 3. Apply template matching
        result = cv2.matchTemplate(captured_eq, stored_eq, cv2.TM_CCOEFF_NORMED)
        _, template_score, _, _ = cv2.minMaxLoc(result)
        print(f"Template score: {template_score}")
        
        # 4. Try MSE (Mean Squared Error)
        mse = np.sum((captured_eq.astype("float") - stored_eq.astype("float")) ** 2)
        mse /= float(captured_eq.shape[0] * captured_eq.shape[1])
        # Convert MSE to similarity score (higher is better)
        mse_score = 1 / (1 + mse/100)
        print(f"MSE-based score: {mse_score}")
        
        # 5. Try Laplacian correlation
        lap1 = cv2.Laplacian(captured_eq, cv2.CV_64F)
        lap2 = cv2.Laplacian(stored_eq, cv2.CV_64F)
        lap_corr = np.corrcoef(lap1.flatten(), lap2.flatten())[0, 1]
        # Handle NaN values
        if np.isnan(lap_corr):
            lap_corr = 0
        print(f"Laplacian correlation: {lap_corr}")
            
        # Calculate combined score with weights
        # Emphasize template matching and histogram which are more robust
        combined_score = (
            0.35 * hist_score + 
            0.35 * template_score + 
            0.15 * mse_score + 
            0.15 * max(0, lap_corr)
        )
        
        # Use the standard threshold - 0.6
        threshold = 0.6
        
        print(f"Combined score: {combined_score}")
        print(f"Threshold: {threshold}")
        print(f"Verified: {combined_score > threshold}")
        
        return {
            'verified': combined_score > threshold,
            'message': 'Face verification successful' if combined_score > threshold else 'Face verification failed',
            'distance': 1 - combined_score  # Convert similarity to distance
        }
        
    except Exception as e:
        print(f"Error in verify_voter_face: {str(e)}")
        print(traceback.format_exc())
        return {
            'verified': False,
            'message': f'Error during face verification: {str(e)}',
            'distance': float('inf')
        } 