import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def process_webcam_image(base64_image):
    """
    Process a base64 encoded image from webcam into a numpy array.
    Returns a grayscale face image of size 200x200.
    """
    try:
        print("\n=== Processing Webcam Image ===")
        # Remove data URL prefix if present
        if ',' in base64_image:
            print("📝 Removing data URL prefix")
            base64_image = base64_image.split(',')[1]
        
        # Decode base64 string
        print("🔄 Decoding base64 image")
        image_bytes = base64.b64decode(base64_image)
        
        # Convert to PIL Image
        print("🖼️ Converting to PIL Image")
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to numpy array
        print("🔢 Converting to numpy array")
        image_array = np.array(image)
        
        # Convert to grayscale if image is RGB
        if len(image_array.shape) == 3:
            print("⚫ Converting to grayscale")
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to standard size
        print("📐 Resizing to 200x200")
        image_array = cv2.resize(image_array, (200, 200))
        
        print(f"✅ Successfully processed image. Shape: {image_array.shape}")
        return image_array
        
    except Exception as e:
        print(f"❌ Error processing webcam image: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        return None

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

def compare_faces(face1, face2, threshold=0.8):
    """
    Compare two faces and return True if they match above the threshold.
    """
    try:
        print(f"\n=== Comparing Faces (threshold: {threshold:.2%}) ===")
        similarity = calculate_face_similarity(face1, face2)
        result = similarity >= threshold
        print(f"{'✅ Match found!' if result else '❌ No match'} (Score: {similarity:.2%})")
        return result
    except Exception as e:
        print(f"❌ Error comparing faces: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        return False

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