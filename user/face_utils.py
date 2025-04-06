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

# Try to import face_recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition package not available")

# Try to import dlib
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logger.warning("dlib package not available")

# Try to import insightface
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("insightface package not available")

class FaceRecognition:
    def __init__(self):
        # Initialize face detector and embedding model
        try:
            from insightface.app import FaceAnalysis
            self.model = FaceAnalysis(name='buffalo_l')
            self.model.prepare(ctx_id=-1, det_size=(640, 640))
            self.model_loaded = True
            logger.info("InsightFace model loaded successfully")
        except Exception as e:
            self.model_loaded = False
            logger.error(f"Error loading InsightFace model: {str(e)}")
    
    def get_face_embedding(self, image):
        """Get face embedding from image"""
        try:
            if not self.model_loaded:
                logger.error("Model not loaded")
                return None
                
            # Preprocess image
            if image is None:
                logger.error("Input image is None")
                return None
                
            # Get faces
            faces = self.model.get(image)
            if len(faces) == 0:
                logger.warning("No face detected")
                return None
                
            # Get embedding from the first detected face
            embedding = faces[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Error getting face embedding: {str(e)}")
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
        threshold = 0.5
        
        return {
            'verified': distance < threshold,
            'message': 'Face verification successful' if distance < threshold else 'Face verification failed',
            'distance': distance
        }
    
    def compare_faces(self, embedding1, embedding2):
        """Calculate distance between two face embeddings"""
        try:
            if embedding1 is None or embedding2 is None:
                return float('inf')
                
            # Normalize the embeddings
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            
            # Convert to distance (0 means identical, 1 means completely different)
            distance = 1.0 - similarity
            
            return distance
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            return float('inf')

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

def verify_voter_face(captured_face, stored_face, threshold=0.5):
    """
    Verify identity using multiple face recognition methods.
    
    Args:
        captured_face: The captured face image from the webcam
        stored_face: The stored face image for the student
        threshold: Minimum score threshold for verification
        
    Returns:
        Dictionary with verification results
    """
    try:
        # Initialize methods and scores
        methods = []
        scores = []
        verification_results = []
        
        # 1. Try face_recognition (if available)
        try:
            from face_recognition import face_encodings, face_distance, face_locations
            
            # Get face encodings
            captured_encodings = face_encodings(captured_face)
            stored_encodings = face_encodings(stored_face)
            
            if captured_encodings and stored_encodings:
                # Calculate face distance
                distance = float(face_distance([stored_encodings[0]], captured_encodings[0])[0])
                score = 1.0 - distance
                
                methods.append('face_recognition')
                scores.append(score)
                # Face recognition has a lower threshold as it's more accurate
                verification_results.append({
                    'method': 'face_recognition',
                    'score': score,
                    'threshold': threshold,
                    'verified': score >= threshold
                })
                
                print(f"face_recognition distance: {distance}, score: {score}")
        except (ImportError, Exception) as e:
            print(f"face_recognition failed: {str(e)}")
            
        # 2. Try SSIM (structural similarity index)
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Preprocess images for SSIM
            captured_processed = preprocess_face_image(captured_face)
            stored_processed = preprocess_face_image(stored_face)
            
            if captured_processed is not None and stored_processed is not None:
                # Calculate similarity
                score = ssim(stored_processed, captured_processed)
                
                methods.append('ssim')
                scores.append(score)
                verification_results.append({
                    'method': 'ssim',
                    'score': score,
                    'threshold': threshold,
                    'verified': score >= threshold
                })
                
                print(f"SSIM score: {score}")
        except (ImportError, Exception) as e:
            print(f"SSIM failed: {str(e)}")
            
        # 3. Try histogram correlation
        try:
            # Preprocess images
            captured_processed = preprocess_face_image(captured_face)
            stored_processed = preprocess_face_image(stored_face)
            
            if captured_processed is not None and stored_processed is not None:
                # Calculate histogram for each image
                hist1 = cv2.calcHist([stored_processed], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([captured_processed], [0], None, [256], [0, 256])
                
                # Compare histograms using correlation method
                score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                
                methods.append('histogram_correlation')
                scores.append(score)
                verification_results.append({
                    'method': 'histogram_correlation',
                    'score': score,
                    'threshold': 0.7,  # Higher threshold for histogram
                    'verified': score >= 0.7
                })
                
                print(f"Histogram correlation score: {score}")
        except Exception as e:
            print(f"Histogram correlation failed: {str(e)}")
            
        # 4. Try histogram intersection
        try:
            # Preprocess images
            captured_processed = preprocess_face_image(captured_face)
            stored_processed = preprocess_face_image(stored_face)
            
            if captured_processed is not None and stored_processed is not None:
                # Calculate histogram for each image
                hist1 = cv2.calcHist([stored_processed], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([captured_processed], [0], None, [256], [0, 256])
                
                # Compare histograms using intersection method
                score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT) / np.sum(hist1)
                
                methods.append('histogram_intersection')
                scores.append(score)
                verification_results.append({
                    'method': 'histogram_intersection',
                    'score': score,
                    'threshold': 0.7,  # Higher threshold for histogram
                    'verified': score >= 0.7
                })
                
                print(f"Histogram intersection score: {score}")
        except Exception as e:
            print(f"Histogram intersection failed: {str(e)}")
            
        # 5. Try template matching
        try:
            # Preprocess images
            captured_processed = preprocess_face_image(captured_face, target_size=(64, 64))
            stored_processed = preprocess_face_image(stored_face, target_size=(64, 64))
            
            if captured_processed is not None and stored_processed is not None:
                # Convert to grayscale if needed
                if len(captured_processed.shape) > 2:
                    captured_processed = cv2.cvtColor(captured_processed, cv2.COLOR_BGR2GRAY)
                if len(stored_processed.shape) > 2:
                    stored_processed = cv2.cvtColor(stored_processed, cv2.COLOR_BGR2GRAY)
                
                # Apply template matching
                result = cv2.matchTemplate(captured_processed, stored_processed, cv2.TM_CCOEFF_NORMED)
                score = np.max(result)
                
                methods.append('template_matching')
                scores.append(score)
                verification_results.append({
                    'method': 'template_matching',
                    'score': score,
                    'threshold': 0.6,
                    'verified': score >= 0.6
                })
                
                print(f"Template matching score: {score}")
        except Exception as e:
            print(f"Template matching failed: {str(e)}")
            
        # If we have results, compute the average score and find the best score
        if scores:
            # Calculate average score
            average_score = sum(scores) / len(scores)
            print(f"Average score: {average_score}")
            
            # Find the best score and method
            best_score_index = np.argmax(scores)
            best_score = scores[best_score_index]
            best_method = methods[best_score_index]
            
            print(f"Best score: {best_score} ({best_method})")
            
            # Calculate distance based on best score
            distance = 1.0 - best_score
            
            # Count successful methods (methods that pass their respective thresholds)
            successful_methods = sum(1 for result in verification_results if result.get('verified', False))
            
            # Create verification summary
            verification_summary = f"Methods: {len(methods)}, Succeeded: {successful_methods}"
            print(f"Verification summary: {verification_summary}")
            
            # For quick verification, check if we have enough successful methods
            verified = (successful_methods >= 2)
            
            # Return enhanced result with detailed information
            return {
                'verified': verified,
                'distance': distance,
                'score': best_score,
                'method': best_method,
                'methods_tried': methods,
                'all_scores': scores,
                'verification_details': verification_results,
                'verification_summary': f"{verification_summary} (Need at least 2 to pass)",
                'successful_methods': successful_methods,
                'required_methods': 2,
                'passing_threshold': threshold
            }
        else:
            print("All verification methods failed")
            return {
                'verified': False,
                'message': 'All verification methods failed',
                'distance': 1.0,
                'method': 'all_failed'
            }
    except Exception as e:
        print(f"Error in verify_voter_face: {str(e)}")
        print(traceback.format_exc())
        return {
            'verified': False,
            'message': f'Error during verification: {str(e)}',
            'distance': 1.0,
            'method': 'error'
        } 