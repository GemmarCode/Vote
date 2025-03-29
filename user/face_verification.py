import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
import base64
import os
from .models import UserProfile
import json
from .face_utils import FaceRecognition

@login_required
def verify_face(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            face_data = data.get('face_data')
            
            if not face_data:
                return JsonResponse({'verified': 'false', 'message': 'No face data provided'})
            
            # Remove the data URL prefix if present
            if ',' in face_data:
                face_data = face_data.split(',')[1]
            
            # Decode base64 image
            image_data = base64.b64decode(face_data)
            
            # Save the uploaded image temporarily
            temp_path = default_storage.save('temp/verify.jpg', ContentFile(image_data))
            temp_file = os.path.join(settings.MEDIA_ROOT, temp_path)
            
            # Read the image
            img = cv2.imread(temp_file)
            if img is None:
                return JsonResponse({'verified': 'false', 'message': 'Could not read image'})
            
            # Get the user's stored face data
            user_profile = UserProfile.objects.get(user=request.user)
            if not user_profile.face_data:
                return JsonResponse({'verified': 'false', 'message': 'No stored face data found'})
            
            # Convert stored face data back to numpy array
            stored_face = np.frombuffer(user_profile.face_data, dtype=np.uint8)
            stored_face = cv2.imdecode(stored_face, cv2.IMREAD_COLOR)
            
            # Initialize face recognition system
            face_recognition = FaceRecognition()
            
            # Compare faces using multiple methods
            embedding_similarity, icp_score = face_recognition.compare_faces(img, stored_face)
            
            # Calculate final score (weighted combination)
            final_score = (0.7 * embedding_similarity + 0.3 * icp_score) * 100
            
            # Clean up temporary file
            os.remove(temp_file)
            
            # More lenient threshold (60% instead of 80%)
            if final_score >= 60:
                return JsonResponse({
                    'verified': 'true',
                    'score': round(final_score, 2),
                    'embedding_similarity': round(embedding_similarity * 100, 2),
                    'icp_score': round(icp_score, 2)
                })
            else:
                return JsonResponse({
                    'verified': 'false',
                    'message': f'Face verification failed. Match score: {round(final_score, 2)}%',
                    'score': round(final_score, 2),
                    'embedding_similarity': round(embedding_similarity * 100, 2),
                    'icp_score': round(icp_score, 2)
                })
                
        except Exception as e:
            return JsonResponse({'verified': 'false', 'message': str(e)})
    
    return JsonResponse({'verified': 'false', 'message': 'Invalid request method'}) 