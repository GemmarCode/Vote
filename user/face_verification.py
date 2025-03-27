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
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Get the user's stored face data
            user_profile = UserProfile.objects.get(user=request.user)
            if not user_profile.face_data:
                return JsonResponse({'verified': 'false', 'message': 'No stored face data found'})
            
            # Convert stored face data back to numpy array
            stored_face = np.frombuffer(user_profile.face_data, dtype=np.uint8)
            
            # Resize the uploaded image to match stored face dimensions
            stored_face = stored_face.reshape(-1, gray.shape[1])
            gray = cv2.resize(gray, (stored_face.shape[1], stored_face.shape[0]))
            
            # Normalize both images
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            stored_face = cv2.normalize(stored_face, None, 0, 255, cv2.NORM_MINMAX)
            
            # Calculate similarity using multiple methods
            # 1. Structural Similarity Index (SSIM)
            ssim_score = cv2.compareSSIM(gray, stored_face)
            
            # 2. Histogram comparison
            hist1 = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([stored_face], [0], None, [256], [0, 256])
            hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # 3. Template matching
            template_score = cv2.matchTemplate(gray, stored_face, cv2.TM_CCOEFF_NORMED)[0][0]
            
            # Calculate weighted average of all scores
            # Give more weight to SSIM as it's generally more reliable
            final_score = (0.5 * ssim_score + 0.3 * hist_score + 0.2 * template_score) * 100
            
            # Clean up temporary file
            os.remove(temp_file)
            
            # More lenient threshold (60% instead of 80%)
            if final_score >= 60:
                return JsonResponse({'verified': 'true', 'score': round(final_score, 2)})
            else:
                return JsonResponse({
                    'verified': 'false',
                    'message': f'Face verification failed. Match score: {round(final_score, 2)}%',
                    'score': round(final_score, 2)
                })
                
        except Exception as e:
            return JsonResponse({'verified': 'false', 'message': str(e)})
    
    return JsonResponse({'verified': 'false', 'message': 'Invalid request method'}) 