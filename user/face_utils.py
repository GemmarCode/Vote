import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
from torchvision import transforms
from user.models import UserProfile
import json
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import base64
import tempfile

# Initialize models
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=False, device='cpu')  # or 'cuda' if using GPU

def get_face_embedding(image_path):
    """Extract face embedding from an image file, only if a face is detected."""
    try:
        img = Image.open(image_path).convert('RGB')
        # Detect face
        face = mtcnn(img)
        if face is None:
            print("No face detected in the image.")
            return None
        # face is a tensor of shape (3, 160, 160)
        face = face.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embedding = model(face).numpy()[0]
        return embedding
    except Exception as e:
        print(f"Error extracting face embedding: {str(e)}")
        return None

def verify_face(image_path, threshold=0.6):
    """Verify if the face in the image matches any stored face embeddings"""
    # Get embedding for the input image
    input_embedding = get_face_embedding(image_path)
    if input_embedding is None:
        return None, "Failed to extract face embedding from input image"
    
    # Get all stored embeddings
    profiles = UserProfile.objects.filter(extracted_image__isnull=False)
    
    best_match = None
    best_distance = float('inf')
    
    for profile in profiles:
        stored_embedding = profile.get_face_embedding()
        if stored_embedding is None:
            continue
            
        # Calculate Euclidean distance between embeddings
        distance = np.linalg.norm(input_embedding - stored_embedding)
        
        if distance < best_distance:
            best_distance = distance
            best_match = profile
    
    if best_match is None:
        return None, "No matching face found in database"
    
    if best_distance > threshold:
        return None, "Face verification failed - distance exceeds threshold"
    
    return best_match, "Face verification successful"

def preprocess_face_image(image_path):
    """Preprocess face image for verification"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = transforms.Resize((160, 160))(img)
        return img
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def recognize_face(input_embedding, threshold=0.5):
    """
    Compare input_embedding to all UserProfile embeddings.
    Returns (student_number, student_name, score) if a match above threshold is found, else (None, None, None).
    """
    input_vec = np.array(input_embedding).reshape(1, -1)
    best_score = -1
    best_student = None
    for student in UserProfile.objects.exclude(face_image__isnull=True).exclude(face_image=""):
        try:
            db_vec = np.array(json.loads(student.face_image)).reshape(1, -1)
            similarity = cosine_similarity(input_vec, db_vec)[0][0]
            if similarity > best_score:
                best_score = similarity
                best_student = student
        except Exception as e:
            continue
    if best_student and best_score > threshold:
        return best_student.student_number, best_student.student_name, float(best_score)
    return None, None, None

def process_webcam_image(base64_image):
    """Decode a base64 image and save it as a temporary file, returning the file path."""
    try:
        header, encoded = base64_image.split(',', 1)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, img)
        return temp_file.name
    except Exception as e:
        print(f"Error processing webcam image: {str(e)}")
        return None 