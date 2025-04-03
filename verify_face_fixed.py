@csrf_exempt
def verify_face(request):
    """Basic face verification function"""
    try:
        # Get the captured face image from the request
        data = json.loads(request.body)
        face_data = data.get('face_data')
        
        if not face_data:
            print("No face data in request")
            return JsonResponse({
                'success': False,
                'message': 'No face data provided'
            }, status=400)
        
        # Process the webcam image
        captured_face = process_webcam_image(face_data)
        if captured_face is None:
            print("Failed to process captured image")
            return JsonResponse({
                'success': False,
                'message': 'Failed to process captured image'
            }, status=400)
        
        # For debugging: save the captured face
        try:
            debug_dir = os.path.join(settings.MEDIA_ROOT, 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f'captured_{int(time.time())}.jpg')
            cv2.imwrite(debug_path, captured_face)
            print(f"Saved debug captured face to {debug_path}")
        except Exception as e:
            print(f"Warning: Could not save debug image: {str(e)}")
            
        # Get all face images from the face_data directory
        face_dir = os.path.join(settings.MEDIA_ROOT, 'face_data')
        print(f"Checking faces in directory: {face_dir}")
        
        if not os.path.exists(face_dir):
            print("Face data directory does not exist")
            return JsonResponse({
                'success': False,
                'message': 'Face data directory not found'
            }, status=400)
            
        face_files = [f for f in os.listdir(face_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(face_files)} face files")
        
        best_match = None
        best_score = 0
        verification_method = 'advanced'  # Use 'advanced' for new method or 'simple' for old method
        
        if verification_method == 'advanced':
            # Using our enhanced multi-method approach
            print("Using advanced face verification")
            for filename in face_files:
                try:
                    student_number = os.path.splitext(filename)[0]
                    print(f"Processing file: {filename} (Student Number: {student_number})")
                    
                    stored_image_path = os.path.join(face_dir, filename)
                    stored_image = cv2.imread(stored_image_path)
                    
                    if stored_image is not None:
                        # Save debug preprocessed image for the first few comparisons
                        if best_match is None:
                            try:
                                preprocessed_captured = preprocess_face_image(captured_face)
                                preprocessed_stored = preprocess_face_image(stored_image)
                                
                                if preprocessed_captured is not None and preprocessed_stored is not None:
                                    debug_dir = os.path.join(settings.MEDIA_ROOT, 'debug')
                                    os.makedirs(debug_dir, exist_ok=True)
                                    
                                    cv2.imwrite(os.path.join(debug_dir, f'preprocessed_captured_{int(time.time())}.jpg'), 
                                                preprocessed_captured)
                                    cv2.imwrite(os.path.join(debug_dir, f'preprocessed_stored_{student_number}.jpg'),
                                                preprocessed_stored)
                                    print(f"Saved debug preprocessed images for {student_number}")
                            except Exception as e:
                                print(f"Warning: Could not save debug preprocessed images: {str(e)}")
                        
                        # Use the advanced verification method
                        verification_result = verify_voter_face(captured_face, stored_image)
                        print(f"Verification result for {student_number}: {verification_result}")
                        
                        if verification_result['verified']:
                            best_match = student_number
                            best_score = 1.0 - verification_result['distance']  # Convert distance to similarity
                            print(f"Found match: {student_number}, Score: {best_score}")
                            break  # Stop once we find a match
                    else:
                        print(f"Failed to load image: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
        else:
            # Using simple image comparison
            print("Using simple face verification")
            for filename in face_files:
                try:
                    student_number = os.path.splitext(filename)[0]
                    print(f"Processing file: {filename} (Student Number: {student_number})")
                    
                    stored_image_path = os.path.join(face_dir, filename)
                    stored_image = cv2.imread(stored_image_path)
                    
                    if stored_image is not None:
                        # Use a simpler face recognition approach
                        if stored_image.shape[0] > 0 and captured_face.shape[0] > 0:
                            # Use standardized preprocessing
                            captured_processed = preprocess_face_image(captured_face)
                            stored_processed = preprocess_face_image(stored_image)
                            
                            if captured_processed is None or stored_processed is None:
                                print(f"Failed to preprocess images for {student_number}")
                                continue
                                
                            # Calculate structural similarity index
                            try:
                                from skimage.metrics import structural_similarity as ssim
                                score = ssim(stored_processed, captured_processed)
                                print(f"Similarity score for {student_number}: {score}")
                                
                                if score > best_score:
                                    best_score = score
                                    best_match = student_number
                            except ImportError:
                                # Fallback to histogram comparison if skimage not available
                                hist1 = cv2.calcHist([stored_processed], [0], None, [256], [0, 256])
                                hist2 = cv2.calcHist([captured_processed], [0], None, [256], [0, 256])
                                score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                                print(f"Histogram similarity for {student_number}: {score}")
                                
                                if score > best_score:
                                    best_score = score
                                    best_match = student_number
                    else:
                        print(f"Failed to load image: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
        
        # Set threshold based on verification method
        threshold = 0.6 if verification_method == 'advanced' else 0.6
        print(f"Best match: {best_match}, Score: {best_score}, Threshold: {threshold}")
        
        if best_match and best_score > threshold:
            try:
                user_profile = UserProfile.objects.get(student_number=best_match)
                print(f"Found matching user profile for student number: {best_match}")
                
                # Store user profile ID in session
                request.session['user_profile_id'] = user_profile.id
                request.session['face_verified'] = True
                
                return JsonResponse({
                    'success': True,
                    'message': 'Face verification successful',
                    'user_profile_id': user_profile.id,
                    'student_number': user_profile.student_number
                })
            except UserProfile.DoesNotExist:
                print(f"No user profile found for student number: {best_match}")
                return JsonResponse({
                    'success': False,
                    'message': 'User profile not found'
                }, status=404)
        else:
            print("Face verification failed")
            return JsonResponse({
                'success': False,
                'message': 'Face verification failed. Please try again.'
            }, status=400)
            
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'message': f'Error during verification: {str(e)}'
        }, status=500) 