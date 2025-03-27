import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

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