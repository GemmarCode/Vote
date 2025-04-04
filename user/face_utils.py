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
            faces = face_cascade.detectMultiScale(captured_face, 1.1, 5, minSize=(30, 30))
            
            if len(faces) == 0:
                print("No face detected in captured image")
                return {
                    'verified': False,
                    'message': 'No face detected in captured image',
                    'distance': 1.0,
                    'method': 'face_detection'
                }
        except Exception as e:
            print(f"Error detecting face: {str(e)}")
            # Continue anyway as preprocessing might still work
            
        # Preprocess both images for better comparison
        captured_processed = preprocess_face_image(captured_face)
        stored_processed = preprocess_face_image(stored_face)
        
        if captured_processed is None or stored_processed is None:
            print("Failed to preprocess one or both images")
            return {
                'verified': False,
                'message': 'Failed to preprocess images',
                'distance': 1.0,
                'method': 'preprocessing_failed'
            }
        
        # Use multiple comparison methods for more accurate results
        methods = []
        scores = []
        best_score = 0
        best_method = None
        
        # Method 1: face_recognition library (most accurate if available)
        try:
            import face_recognition
            
            # Convert grayscale to BGR if needed
            if len(captured_processed.shape) == 2:
                captured_rgb = cv2.cvtColor(captured_processed, cv2.COLOR_GRAY2RGB)
            else:
                captured_rgb = cv2.cvtColor(captured_processed, cv2.COLOR_BGR2RGB)
                
            if len(stored_processed.shape) == 2:
                stored_rgb = cv2.cvtColor(stored_processed, cv2.COLOR_GRAY2RGB)
            else:
                stored_rgb = cv2.cvtColor(stored_processed, cv2.COLOR_BGR2RGB)
            
            # Get face encodings
            captured_encoding = face_recognition.face_encodings(captured_rgb)
            stored_encoding = face_recognition.face_encodings(stored_rgb)
            
            if captured_encoding and stored_encoding:
                # Calculate face distance
                distance = face_recognition.face_distance([stored_encoding[0]], captured_encoding[0])[0]
                verified = distance < 0.5  # Standard threshold for face_recognition
                
                score = 1.0 - distance  # Convert distance to similarity score
                methods.append('face_recognition')
                scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_method = 'face_recognition'
                
                print(f"face_recognition distance: {distance}, score: {score}")
        except (ImportError, IndexError, Exception) as e:
            print(f"face_recognition comparison failed: {str(e)}")
            
        # Method 2: Structural Similarity Index (SSIM)
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # SSIM works best with grayscale images
            if len(captured_processed.shape) == 3:
                captured_gray = cv2.cvtColor(captured_processed, cv2.COLOR_BGR2GRAY)
            else:
                captured_gray = captured_processed
                
            if len(stored_processed.shape) == 3:
                stored_gray = cv2.cvtColor(stored_processed, cv2.COLOR_BGR2GRAY)
            else:
                stored_gray = stored_processed
            
            score = ssim(stored_gray, captured_gray)
            methods.append('ssim')
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_method = 'ssim'
                
            print(f"SSIM score: {score}")
        except (ImportError, Exception) as e:
            print(f"SSIM comparison failed: {str(e)}")
            
        # Method 3: Histogram Comparison
        try:
            # Prepare histograms
            if len(captured_processed.shape) == 3:
                hist1 = cv2.calcHist([captured_processed], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            else:
                hist1 = cv2.calcHist([captured_processed], [0], None, [256], [0, 256])
                
            if len(stored_processed.shape) == 3:
                hist2 = cv2.calcHist([stored_processed], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            else:
                hist2 = cv2.calcHist([stored_processed], [0], None, [256], [0, 256])
            
            # Normalize histograms
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # Compare histograms using correlation method
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            methods.append('histogram_correlation')
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_method = 'histogram_correlation'
                
            print(f"Histogram correlation score: {score}")
            
            # Also try intersection method
            score_intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
            # Normalize intersection score
            score_intersection = score_intersection / (min(np.sum(hist1), np.sum(hist2)))
            methods.append('histogram_intersection')
            scores.append(score_intersection)
            
            if score_intersection > best_score:
                best_score = score_intersection
                best_method = 'histogram_intersection'
                
            print(f"Histogram intersection score: {score_intersection}")
        except Exception as e:
            print(f"Histogram comparison failed: {str(e)}")
            
        # Method 4: Template Matching (basic but can help)
        try:
            # Resize to ensure same dimensions
            captured_resized = cv2.resize(captured_processed, (100, 100))
            stored_resized = cv2.resize(stored_processed, (100, 100))
            
            # Template matching
            result = cv2.matchTemplate(captured_resized, stored_resized, cv2.TM_CCOEFF_NORMED)
            score = result.max()
            methods.append('template_matching')
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_method = 'template_matching'
                
            print(f"Template matching score: {score}")
        except Exception as e:
            print(f"Template matching failed: {str(e)}")
        
        # Get average score if we have any methods
        if len(scores) > 0:
            avg_score = sum(scores) / len(scores)
            print(f"Average score: {avg_score}")
            print(f"Best score: {best_score} ({best_method})")
            
            # Calculate distance (lower is better)
            distance = 1.0 - best_score
            
            # Enhanced result with more details
            verification_results = []
            
            # Create detailed results for each method
            for i, method in enumerate(methods):
                # Use stricter threshold for histogram methods which can generate more false positives
                method_threshold = threshold
                if method == 'histogram_intersection' or method == 'histogram_correlation':
                    method_threshold = threshold + 0.2  # More strict threshold for histogram methods
                
                verification_results.append({
                    'method': method,
                    'score': scores[i],
                    'verified': scores[i] >= method_threshold
                })
            
            # Count successful methods
            successful_methods = sum(1 for r in verification_results if r['verified'])
            verification_summary = f"Methods: {len(methods)}, Succeeded: {successful_methods}"
            print(f"Verification summary: {verification_summary}")
            
            # Require at least 2 successful methods or a very high score on face_recognition
            fr_verified = False
            if 'face_recognition' in methods:
                fr_index = methods.index('face_recognition')
                fr_verified = scores[fr_index] >= (threshold + 0.3)  # Very high threshold for trusting face_recognition alone
                
            verified = (successful_methods >= 2) or fr_verified
            
            # Return enhanced result
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
                'required_methods': 2
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