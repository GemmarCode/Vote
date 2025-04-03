import cv2
import numpy as np
import base64
import insightface
from insightface.app import FaceAnalysis
from django.conf import settings
import os
import traceback
import dlib
from PIL import Image
import io
import face_recognition
import logging
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the FaceRecognition class that's being imported by views.py
class FaceRecognition:
    def __init__(self):
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception as e:
            print(f"Error initializing FaceRecognition: {str(e)}")
            print(traceback.format_exc())
        
    def get_face_embedding(self, image):
        """Get face embedding from an image"""
        try:
            # First preprocess the image
            preprocessed_image = preprocess_face_image(image)
            if preprocessed_image is None:
                print("Failed to preprocess image")
                return None
                
            # Convert back to BGR for InsightFace (it expects color images)
            preprocessed_bgr = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
            
            # Get faces using InsightFace
            faces = self.app.get(preprocessed_bgr)
            if len(faces) != 1:
                print(f"Expected 1 face, got {len(faces)} faces")
                return None
            return faces[0].embedding
        except Exception as e:
            print(f"Error getting face embedding: {str(e)}")
            print(traceback.format_exc())
            return None
    
    def align_face(self, image):
        """
        Detect face in image and align it
        This is a legacy method that now uses our standardized preprocessing
        """
        try:
            # Use the standardized preprocessing function
            preprocessed_img = preprocess_face_image(image)
            if preprocessed_img is None:
                print("Failed to preprocess face image")
                return None
                
            # Convert back to BGR (color) for compatibility
            if len(preprocessed_img.shape) == 2:  # If grayscale
                preprocessed_color = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
            else:
                preprocessed_color = preprocessed_img
                
            return preprocessed_color
        except Exception as e:
            print(f"Error aligning face: {str(e)}")
            print(traceback.format_exc())
            return None
    
    def verify_face(self, face1, face2):
        """Verify if two faces match"""
        # Get face embeddings
        embedding1 = self.get_face_embedding(face1)
        embedding2 = self.get_face_embedding(face2)
        
        if embedding1 is None or embedding2 is None:
            print("Failed to get embeddings")
            return {
                'verified': False,
                'message': 'Failed to extract face features',
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
        
    def compare_faces(self, embedding1, embedding2):
        """Compare two face embeddings and return distance"""
        if embedding1 is None or embedding2 is None:
            return float('inf')
            
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine distance (1 - cosine similarity)
        distance = 1 - np.dot(embedding1, embedding2)
        
        # Convert to Python float if it's a numpy type
        if hasattr(distance, 'item'):
            distance = float(distance.item())
            
        return distance
# End of FaceRecognition class

class MultiFaceRecognition:
    """A class that combines multiple face recognition models for better accuracy"""
    
    def __init__(self):
        self.insightface_available = False
        self.dlib_available = False
        self.face_recognition_available = False
        
        # Try to initialize InsightFace
        try:
            self.insightface_model = FaceAnalysis(name='buffalo_l')
            self.insightface_model.prepare(ctx_id=-1, det_size=(640, 640))
            self.insightface_available = True
            logger.info("InsightFace initialized successfully")
        except ImportError:
            logger.warning("InsightFace not available")
        except Exception as e:
            logger.error(f"Error initializing InsightFace: {str(e)}")
        
        # Try to initialize dlib
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            models_dir = os.path.join(settings.BASE_DIR, 'models')
            shape_predictor_path = os.path.join(models_dir, 'shape_predictor_68_face_landmarks.dat')
            face_rec_model_path = os.path.join(models_dir, 'dlib_face_recognition_resnet_model_v1.dat')
            
            if os.path.exists(shape_predictor_path) and os.path.exists(face_rec_model_path):
                self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
                self.face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
                self.dlib_available = True
                logger.info("Dlib initialized successfully")
            else:
                logger.warning(f"Dlib model files not found at {models_dir}")
                self.dlib_available = False
        except ImportError:
            logger.warning("dlib not available")
        except Exception as e:
            logger.error(f"Error initializing dlib: {str(e)}")
            self.dlib_available = False
        
        # Try to initialize face_recognition
        try:
            import face_recognition
            self.face_recognition_available = True
            logger.info("face_recognition initialized successfully")
        except ImportError:
            logger.warning("face_recognition not available")
            
    def detect_faces(self, image):
        """Detect faces in image using all available models"""
        faces = {}
        
        # Preprocess the image first
        preprocessed_image = preprocess_face_image(image)
        if preprocessed_image is None:
            logger.error("Failed to preprocess image")
            return faces
            
        # Convert back to BGR for face detection
        preprocessed_bgr = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
        
        # InsightFace detection
        if self.insightface_available:
            try:
                insightface_result = self.insightface_model.get(preprocessed_bgr)
                if len(insightface_result) > 0:
                    faces['insightface'] = insightface_result[0]
                    logger.info("Face detected with InsightFace")
            except Exception as e:
                logger.error(f"Error in InsightFace detection: {str(e)}")
        
        # dlib detection
        if self.dlib_available:
            try:
                # Convert to RGB for dlib
                rgb_image = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)
                dlib_faces = self.face_detector(rgb_image)
                if len(dlib_faces) > 0:
                    faces['dlib'] = dlib_faces[0]
                    logger.info("Face detected with dlib")
            except Exception as e:
                logger.error(f"Error in dlib detection: {str(e)}")
        
        # face_recognition library (based on dlib)
        try:
            # Convert to RGB for face_recognition
            rgb_image = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            if len(face_locations) > 0:
                faces['face_recognition'] = face_locations[0]
                logger.info("Face detected with face_recognition")
        except Exception as e:
            logger.error(f"Error in face_recognition detection: {str(e)}")
            
        return faces
    
    def get_face_embeddings(self, image, faces):
        """Get face embeddings from all available models"""
        embeddings = {}
        
        # Preprocess the image first
        preprocessed_image = preprocess_face_image(image)
        if preprocessed_image is None:
            logger.error("Failed to preprocess image")
            return embeddings
            
        # Convert back to BGR for embedding extraction
        preprocessed_bgr = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
        
        # Store the preprocessed image for traditional comparison fallback
        embeddings['image'] = preprocessed_image
        
        # InsightFace embedding
        if self.insightface_available and 'insightface' in faces:
            try:
                insightface_embedding = faces['insightface'].embedding
                if insightface_embedding is not None:
                    embeddings['insightface'] = insightface_embedding
                    logger.info("Got InsightFace embedding")
            except Exception as e:
                logger.error(f"Error getting InsightFace embedding: {str(e)}")
        
        # dlib embedding
        if self.dlib_available and 'dlib' in faces:
            try:
                rgb_image = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)
                shape = self.shape_predictor(rgb_image, faces['dlib'])
                dlib_embedding = self.face_rec_model.compute_face_descriptor(rgb_image, shape)
                embeddings['dlib'] = np.array(dlib_embedding)
                logger.info("Got dlib embedding")
            except Exception as e:
                logger.error(f"Error getting dlib embedding: {str(e)}")
        
        # face_recognition embedding
        if 'face_recognition' in faces:
            try:
                rgb_image = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_image, [faces['face_recognition']])
                if len(face_encodings) > 0:
                    embeddings['face_recognition'] = face_encodings[0]
                    logger.info("Got face_recognition embedding")
            except Exception as e:
                logger.error(f"Error getting face_recognition embedding: {str(e)}")
                
        return embeddings
    
    def compare_embeddings(self, embeddings1, embeddings2):
        """Compare embeddings from multiple models and return a combined score"""
        scores = {}
        weights = {
            'insightface': 0.4,      # InsightFace is very accurate
            'dlib': 0.3,             # dlib is also quite accurate
            'face_recognition': 0.3   # face_recognition is based on dlib
        }
        
        # Compare InsightFace embeddings
        if 'insightface' in embeddings1 and 'insightface' in embeddings2:
            try:
                e1 = embeddings1['insightface']
                e2 = embeddings2['insightface']
                
                # Normalize embeddings
                e1 = e1 / np.linalg.norm(e1)
                e2 = e2 / np.linalg.norm(e2)
                
                # Calculate cosine similarity (1 = identical, 0 = completely different)
                similarity = np.dot(e1, e2)
                
                # Convert numpy array to float if needed
                if hasattr(similarity, 'item'):
                    similarity = float(similarity.item())
                    
                scores['insightface'] = similarity
                logger.info(f"InsightFace similarity: {similarity}")
            except Exception as e:
                logger.error(f"Error comparing InsightFace embeddings: {str(e)}")
        
        # Compare dlib embeddings
        if 'dlib' in embeddings1 and 'dlib' in embeddings2:
            try:
                e1 = embeddings1['dlib']
                e2 = embeddings2['dlib']
                
                # Calculate Euclidean distance and convert to similarity (0-1 range)
                distance = np.linalg.norm(e1 - e2)
                
                # Convert numpy array to float if needed
                if hasattr(distance, 'item'):
                    distance = float(distance.item())
                    
                similarity = 1.0 / (1.0 + distance)
                scores['dlib'] = similarity
                logger.info(f"dlib similarity: {similarity}")
            except Exception as e:
                logger.error(f"Error comparing dlib embeddings: {str(e)}")
        
        # Compare face_recognition embeddings
        if 'face_recognition' in embeddings1 and 'face_recognition' in embeddings2:
            try:
                e1 = embeddings1['face_recognition']
                e2 = embeddings2['face_recognition']
                
                # Calculate face_recognition distance and convert to similarity
                distance = np.linalg.norm(e1 - e2)
                
                # Convert numpy array to float if needed
                if hasattr(distance, 'item'):
                    distance = float(distance.item())
                    
                similarity = 1.0 / (1.0 + distance)
                scores['face_recognition'] = similarity
                logger.info(f"face_recognition similarity: {similarity}")
            except Exception as e:
                logger.error(f"Error comparing face_recognition embeddings: {str(e)}")
        
        # Calculate weighted average score
        total_weight = 0.0
        weighted_score = 0.0
        
        for model, score in scores.items():
            # Ensure score is a float
            if hasattr(score, 'item'):
                score = float(score.item())
                
            weighted_score += score * weights[model]
            total_weight += weights[model]
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            # Fallback to traditional method if no embeddings were compared
            final_score = compare_images_traditional(embeddings1.get('image'), embeddings2.get('image'))
            
        logger.info(f"Final combined similarity score: {final_score}")
        return final_score
        
def compare_images_traditional(image1, image2):
    """Traditional image comparison as a fallback"""
    if image1 is None or image2 is None:
        return 0.0
        
    # Try SSIM first
    try:
        from skimage.metrics import structural_similarity as ssim
        similarity = ssim(image1, image2)
        return float(similarity)
    except (ImportError, Exception):
        pass
    
    # Fallback to histogram comparison
    try:
        hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Compare histograms (correlation method - higher is better)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return float(similarity)
    except Exception:
        return 0.0

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
    Preprocess a face image for comparison with enhanced lighting normalization
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
        
        # Try to detect face first and crop to just the face area
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(processed, 1.1, 5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face
                
                # Add some margin (15%)
                margin_x = int(w * 0.15)
                margin_y = int(h * 0.15)
                
                # Calculate new coordinates with margins
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                w = min(processed.shape[1] - x, w + 2 * margin_x)
                h = min(processed.shape[0] - y, h + 2 * margin_y)
                
                # Crop to face region
                processed = processed[y:y+h, x:x+w]
                print(f"Face detected and cropped: {w}x{h}")
        except Exception as e:
            print(f"Face detection failed: {str(e)}, using full image")
            
        # Resize to target size
        processed = cv2.resize(processed, target_size)
        
        # Use CLAHE instead of simple histogram equalization for better lighting normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
        
        # Apply slight Gaussian blur to reduce noise
        processed = cv2.GaussianBlur(processed, (5, 5), 0)
        
        # Normalize pixel values to 0-255 range
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
        
        return processed
    except Exception as e:
        print(f"Error preprocessing face image: {str(e)}")
        print(traceback.format_exc())
        return None

def verify_voter_face(captured_face, stored_face, threshold=0.6):
    """
    Enhanced voter face verification with multiple methods and improved accuracy
    """
    try:
        # Make sure both images are valid
        if captured_face is None or stored_face is None:
            print("One or both images are None")
            return {
                'verified': False,
                'message': 'Invalid image data',
                'distance': 1.0,
                'method': 'error'
            }
            
        # Detect faces in captured image to ensure we have a face
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            captured_gray = cv2.cvtColor(captured_face, cv2.COLOR_BGR2GRAY) if len(captured_face.shape) == 3 else captured_face
            faces = face_cascade.detectMultiScale(captured_gray, 1.1, 5, minSize=(30, 30))
            
            if len(faces) == 0:
                print("No face detected in webcam image")
                return {
                    'verified': False, 
                    'distance': 1.0, 
                    'method': 'no_face_detected',
                    'message': 'No face detected in webcam image'
                }
        except Exception as e:
            print(f"Face detection error: {str(e)}, continuing with verification")
        
        # Use face_recognition for more accurate comparison if available
        try:
            import face_recognition
            
            # Convert images to RGB (face_recognition expects RGB)
            if len(captured_face.shape) == 3 and captured_face.shape[2] == 3:
                captured_rgb = cv2.cvtColor(captured_face, cv2.COLOR_BGR2RGB)
            else:
                # If grayscale, convert to RGB first
                captured_rgb = cv2.cvtColor(captured_face, cv2.COLOR_GRAY2RGB)
                
            if len(stored_face.shape) == 3 and stored_face.shape[2] == 3:
                stored_rgb = cv2.cvtColor(stored_face, cv2.COLOR_BGR2RGB)
            else:
                # If grayscale, convert to RGB first
                stored_rgb = cv2.cvtColor(stored_face, cv2.COLOR_GRAY2RGB)
            
            # Get face encodings
            captured_encoding = face_recognition.face_encodings(captured_rgb)
            stored_encoding = face_recognition.face_encodings(stored_rgb)
            
            if captured_encoding and stored_encoding:
                # Calculate face distance
                distance = face_recognition.face_distance([stored_encoding[0]], captured_encoding[0])[0]
                verified = distance < 0.6  # Standard threshold for face_recognition
                
                return {
                    'verified': verified,
                    'distance': float(distance),
                    'method': 'face_recognition',
                    'message': 'Face verification successful' if verified else 'Face verification failed'
                }
            else:
                print("Could not extract face encodings, using fallback methods")
        except (ImportError, IndexError, Exception) as e:
            print(f"Face recognition not available or error: {str(e)}, using fallback")
            
        # Apply identical preprocessing to both images for traditional comparison
        captured_processed = preprocess_face_image(captured_face)
        stored_processed = preprocess_face_image(stored_face)
        
        if captured_processed is None or stored_processed is None:
            return {
                'verified': False, 
                'distance': 1.0, 
                'method': 'preprocessing_failed',
                'message': 'Failed to preprocess images'
            }
        
        # Collect scores from multiple comparison methods
        scores = []
        methods = []
        
        # 1. Histogram Comparison - Correlation
        hist1 = cv2.calcHist([stored_processed], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([captured_processed], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Compare histograms (correlation method - higher is better)
        corr_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        scores.append(float(corr_score) if hasattr(corr_score, 'item') else corr_score)
        methods.append("Correlation")
        print(f"Histogram correlation score: {corr_score}")
        
        # 2. Histogram Comparison - Intersection
        inter_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
        # Normalize intersection score
        norm_inter_score = inter_score / float(np.sum(hist1)) if np.sum(hist1) > 0 else 0
        scores.append(float(norm_inter_score) if hasattr(norm_inter_score, 'item') else norm_inter_score)
        methods.append("Intersection")
        print(f"Histogram intersection score: {norm_inter_score}")
        
        # 3. Structural Similarity (SSIM) - if available
        try:
            from skimage.metrics import structural_similarity
            ssim_score = structural_similarity(stored_processed, captured_processed)
            scores.append(float(ssim_score) if hasattr(ssim_score, 'item') else ssim_score)
            methods.append("SSIM")
            print(f"SSIM score: {ssim_score}")
        except ImportError:
            print("skimage not available, skipping SSIM")
        
        # 4. Template Matching
        try:
            # Resize to same dimensions if needed
            if stored_processed.shape != captured_processed.shape:
                captured_processed_resized = cv2.resize(captured_processed, 
                                                      (stored_processed.shape[1], stored_processed.shape[0]))
            else:
                captured_processed_resized = captured_processed
                
            # Use normalized correlation coefficient for template matching
            result = cv2.matchTemplate(stored_processed, captured_processed_resized, cv2.TM_CCOEFF_NORMED)
            _, template_score, _, _ = cv2.minMaxLoc(result)
            scores.append(float(template_score) if hasattr(template_score, 'item') else template_score)
            methods.append("Template")
            print(f"Template matching score: {template_score}")
        except Exception as e:
            print(f"Template matching error: {str(e)}")
            
        # Find the best score and method
        if not scores:
            return {
                'verified': False,
                'distance': 1.0,
                'method': 'no_scores',
                'message': 'No comparison methods produced valid scores'
            }
            
        best_index = np.argmax(scores)
        best_score = scores[best_index]
        best_method = methods[best_index]
        
        # Convert to distance (1 - similarity)
        distance = 1.0 - best_score
        verified = best_score >= threshold
        
        print(f"Best score: {best_score} from method: {best_method}")
        print(f"Threshold: {threshold}")
        print(f"Verified: {verified}")
        
        return {
            'verified': verified,
            'distance': float(distance),
            'method': best_method,
            'score': float(best_score),
            'message': 'Face verification successful' if verified else 'Face verification failed'
        }
    except Exception as e:
        print(f"Error in verify_voter_face: {str(e)}")
        print(traceback.format_exc())
        return {
            'verified': False,
            'message': f'Error during face verification: {str(e)}',
            'distance': 1.0,
            'method': 'error'
        } 