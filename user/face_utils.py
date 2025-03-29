import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from deepface import DeepFace
import open3d as o3d
from scipy.spatial.distance import cosine
import mediapipe as mp
from typing import Tuple, Optional, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from tensorflow.keras.preprocessing import image
import imutils

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
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        gray = cv2.resize(gray, (200, 200))
        
        return gray
    except Exception as e:
        print(f"Error processing webcam image: {str(e)}")
        return None

def compare_faces(captured_face, stored_face_data):
    """Compare captured face with stored face data."""
    try:
        # Convert stored face data to numpy array
        stored_face = np.frombuffer(stored_face_data, dtype=np.uint8)
        stored_face = cv2.imdecode(stored_face, cv2.IMREAD_GRAYSCALE)
        stored_face = cv2.resize(stored_face, (200, 200))
        
        # Calculate similarity using structural similarity index
        score = cv2.compareSSIM(captured_face, stored_face)
        
        # Return True if similarity is above threshold
        return score >= 0.8
    except Exception as e:
        print(f"Error comparing faces: {str(e)}")
        return False

def verify_3d_face(image):
    """Verify 3D face landmarks and orientation."""
    try:
        # Initialize face detector
        face_mesh = cv2.FaceDetectorYN.create(
            "face_detection_yunet_2023mar.onnx",
            "",
            (320, 320),
            0.9,
            0.3,
            5000
        )
        
        # Detect 3D landmarks
        _, faces = face_mesh.detect(image)
        
        if faces is not None and len(faces) > 0:
            # Extract 3D landmarks
            landmarks = faces[0][4:].reshape(-1, 3)
            
            # Calculate face orientation
            nose_tip = landmarks[4]
            left_eye = landmarks[1]
            right_eye = landmarks[0]
            
            # Check if face is relatively straight
            eye_line = np.linalg.norm(right_eye - left_eye)
            nose_offset = abs(nose_tip[2] - (left_eye[2] + right_eye[2]) / 2)
            
            return nose_offset < eye_line * 0.1
        return False
    except Exception as e:
        print(f"Error in 3D face verification: {str(e)}")
        return False

def calculate_face_similarity(face1, face2):
    """
    Calculate similarity between two face images using multiple metrics.
    Both images should be grayscale and the same size.
    Returns a similarity score between 0 and 1.
    """
    try:
        print("\n=== Calculating Face Similarity ===")
        print(f"📊 Face1 shape: {face1.shape}")
        print(f"📊 Face2 shape: {face2.shape}")
        
        # Ensure both images are the same size
        if face1.shape != face2.shape:
            print("⚠️ Images have different shapes, resizing...")
            face2 = cv2.resize(face2, (face1.shape[1], face1.shape[0]))
        
        # Calculate histogram correlation (40% weight)
        print("📊 Calculating histogram correlation")
        hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        print(f"📈 Histogram correlation score: {hist_score:.2%}")
        
        # Calculate template matching score (40% weight)
        print("🎯 Calculating template matching")
        result = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)
        template_score = np.max(result)
        print(f"🎯 Template matching score: {template_score:.2%}")
        
        # Calculate MSE (20% weight)
        print("📉 Calculating mean squared error")
        mse = np.mean((face1 - face2) ** 2)
        mse_score = 1 / (1 + mse)  # Convert MSE to similarity score
        print(f"📉 MSE-based score: {mse_score:.2%}")
        
        # Calculate weighted average
        final_score = (0.4 * hist_score) + (0.4 * template_score) + (0.2 * mse_score)
        print(f"🏆 Final similarity score: {final_score:.2%}")
        
        return final_score
        
    except Exception as e:
        print(f"❌ Error calculating face similarity: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        return 0.0

def local_binary_pattern(image, n_points, radius, method='uniform'):
    """Calculate Local Binary Pattern for face texture analysis"""
    lbp = np.zeros_like(image)
    for i in range(radius, image.shape[0] - radius):
        for j in range(radius, image.shape[1] - radius):
            center = image[i, j]
            pattern = 0
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                x = i + radius * np.cos(angle)
                y = j + radius * np.sin(angle)
                x1 = int(np.floor(x))
                x2 = int(np.ceil(x))
                y1 = int(np.floor(y))
                y2 = int(np.ceil(y))
                
                # Bilinear interpolation
                fx = x - x1
                fy = y - y1
                w1 = (1 - fx) * (1 - fy)
                w2 = fx * (1 - fy)
                w3 = (1 - fx) * fy
                w4 = fx * fy
                neighbor = w1 * image[x1, y1] + w2 * image[x2, y1] + \
                          w3 * image[x1, y2] + w4 * image[x2, y2]
                
                pattern |= (neighbor > center) << k
            lbp[i, j] = pattern
    return lbp

class FaceRecognition:
    def __init__(self):
        """Initialize face recognition models."""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Initialize VGGFace model with modern TensorFlow
        self.vgg_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3))
        self.vgg_model.trainable = False
        
    def preprocess_face(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for better face recognition"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for noise reduction while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Apply adaptive thresholding
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Normalize
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        return gray
    
    def align_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Enhanced face alignment using MediaPipe"""
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        # Get the first face landmarks
        landmarks = results.multi_face_landmarks[0]
        
        # Get eye landmarks for alignment
        left_eye = np.array([
            landmarks.landmark[33].x * image.shape[1],
            landmarks.landmark[33].y * image.shape[0]
        ])
        right_eye = np.array([
            landmarks.landmark[263].x * image.shape[1],
            landmarks.landmark[263].y * image.shape[0]
        ])
        
        # Calculate angle for alignment
        eye_center = (left_eye + right_eye) / 2
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                    right_eye[0] - left_eye[0]))
        
        # Rotate image
        center = tuple(map(int, eye_center))
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        
        # Get face bounding box
        h, w = aligned.shape[:2]
        face_landmarks = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
        x, y, w, h = cv2.boundingRect(face_landmarks.astype(np.int32))
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(w + 2*padding, aligned.shape[1] - x)
        h = min(h + 2*padding, aligned.shape[0] - y)
        
        # Crop face region
        aligned = aligned[y:y+h, x:x+w]
        
        # Resize to standard size
        aligned = cv2.resize(aligned, (224, 224))
        
        return aligned
    
    def get_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding using multiple models"""
        try:
            # Preprocess and align face
            aligned = self.align_face(image)
            if aligned is None:
                return None
            
            # Get DeepFace embedding
            deepface_embedding = DeepFace.represent(
                aligned,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="retinaface"
            )[0]['embedding']
            
            # Get VGGFace embedding
            img_array = image.img_to_array(aligned)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = utils.preprocess_input(img_array)
            vgg_embedding = self.vgg_model.predict(img_array)[0]
            
            # Combine embeddings
            combined_embedding = np.concatenate([deepface_embedding, vgg_embedding])
            
            # Normalize the combined embedding
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            
            return combined_embedding
            
        except Exception as e:
            print(f"Error getting face embedding: {str(e)}")
            return None
    
    def create_3d_face(self, image: np.ndarray) -> Optional[o3d.geometry.TriangleMesh]:
        """Create 3D face model using MediaPipe landmarks"""
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to 3D points
        points = []
        for landmark in landmarks.landmark:
            points.append([landmark.x, landmark.y, landmark.z])
            
        points = np.array(points)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals
        pcd.estimate_normals()
        
        # Create mesh using Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
        
        # Clean up mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh
    
    def compare_faces(self, face1: np.ndarray, face2: np.ndarray) -> Tuple[float, float]:
        """Compare two faces using multiple methods"""
        # Get face embeddings
        embedding1 = self.get_face_embedding(face1)
        embedding2 = self.get_face_embedding(face2)
        
        if embedding1 is None or embedding2 is None:
            return 0.0, 0.0
            
        # Calculate cosine similarity between embeddings
        embedding_similarity = 1 - cosine(embedding1, embedding2)
        
        # Create 3D models
        mesh1 = self.create_3d_face(face1)
        mesh2 = self.create_3d_face(face2)
        
        if mesh1 is None or mesh2 is None:
            return embedding_similarity, 0.0
            
        # Calculate ICP score
        icp_score = self.calculate_icp_score(mesh1, mesh2)
        
        return embedding_similarity, icp_score
    
    def calculate_icp_score(self, mesh1: o3d.geometry.TriangleMesh, 
                          mesh2: o3d.geometry.TriangleMesh) -> float:
        """Calculate ICP score between two 3D meshes"""
        # Convert meshes to point clouds
        pcd1 = mesh1.sample_points_uniformly(number_of_points=1000)
        pcd2 = mesh2.sample_points_uniformly(number_of_points=1000)
        
        # Perform ICP with multiple iterations
        best_fitness = 0
        for i in range(5):  # Try 5 different initial alignments
            icp_result = o3d.pipelines.registration.registration_icp(
                pcd1, pcd2,
                max_correspondence_distance=0.05,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            best_fitness = max(best_fitness, icp_result.fitness)
        
        # Convert fitness score to percentage
        return best_fitness * 100