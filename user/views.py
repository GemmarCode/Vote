from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from .models import UserProfile, Candidate, Vote, VotingPhase, FaceData
from admin_panel.models import ElectionSettings
from django.core.exceptions import ValidationError
from .face_utils import process_webcam_image, verify_voter_face
import cv2
import numpy as np
from django.db import transaction
import json
import traceback
import base64
from django.db import models
from django.urls import reverse
from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta
import os
from django.conf import settings
from django.contrib.auth.decorators import login_required
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.http import require_http_methods
import insightface
from insightface.app import FaceAnalysis
from django.views.decorators.csrf import csrf_exempt
from .face_utils import FaceRecognition

LOGIN_URL = 'login'  # or whatever your login URL name is

def home_view(request):
    # Get election settings
    try:
        election_settings = ElectionSettings.objects.get(id=1)
    except ElectionSettings.DoesNotExist:
        election_settings = None
    
    context = {
        'election_settings': election_settings
    }
    return render(request, 'mainpage.html', context)

def candidates_view(request):
    # Get only approved candidates
    all_candidates = Candidate.objects.filter(approved=True)
    
    # Initialize dictionaries for national and college-wise candidates
    national_candidates = {}
    college_candidates = {}
    
    # Dictionary to store position display names
    position_names = dict(Candidate.POSITION_CHOICES)
    college_names = dict(UserProfile.COLLEGES)
    
    # First, handle national candidates
    national_positions = [pos[0] for pos in Candidate.NATIONAL_POSITIONS]
    for candidate in all_candidates.filter(position__in=national_positions):
            position_name = position_names.get(candidate.position, candidate.position)
            if position_name not in national_candidates:
                national_candidates[position_name] = []
            national_candidates[position_name].append(candidate)
    
    # Then, organize local candidates by college
    local_positions = [pos[0] for pos in Candidate.LOCAL_POSITIONS]
    for college_code, college_name in UserProfile.COLLEGES:
        college_candidates[college_name] = {}
        college_local_candidates = all_candidates.filter(
            college=college_code, 
            position__in=local_positions
        )
        
        # Group by position within each college
        for candidate in college_local_candidates:
            position_name = position_names.get(candidate.position, candidate.position)
            if position_name not in college_candidates[college_name]:
                college_candidates[college_name][position_name] = []
            college_candidates[college_name][position_name].append(candidate)
    
    # Sort candidates within each position
    for position in national_candidates:
        national_candidates[position].sort(key=lambda x: x.user.get_full_name())
    
    for college in college_candidates:
        for position in college_candidates[college]:
            college_candidates[college][position].sort(key=lambda x: x.user.get_full_name())
    
    # Remove empty college entries
    college_candidates = {k: v for k, v in college_candidates.items() if v}
    
    context = {
        'national_candidates': national_candidates,
        'college_candidates': college_candidates,
        'colleges': UserProfile.COLLEGES
    }
    
    return render(request, 'candidates.html', context)

def results_view(request):
    # Get all positions
    national_positions = ['PRESIDENT', 'VICE_PRESIDENT', 'SECRETARY', 'TREASURER', 'AUDITOR', 'PRO']
    
    # Get candidates and their vote counts
    national_results = {}
    local_results = {}
    
    # Process national positions
    for position in national_positions:
        candidates = Candidate.objects.filter(position=position)\
            .annotate(vote_count=models.Count('vote'))\
            .order_by('-vote_count')
        if candidates.exists():
            national_results[position] = candidates
    
    # Process local positions (grouped by college)
    colleges = UserProfile.COLLEGES
    local_positions = Candidate.objects.exclude(position__in=national_positions)\
        .values_list('position', flat=True).distinct()
    
    for college_code, college_name in colleges:
        local_results[college_name] = {}
        for position in local_positions:
            candidates = Candidate.objects.filter(
                position=position,
                college=college_code
            ).annotate(
                vote_count=models.Count('vote')
            ).order_by('-vote_count')
            if candidates.exists():
                local_results[college_name][position] = candidates
    
    context = {
        'national_results': national_results,
        'local_results': local_results,
        'total_voters': User.objects.count() - 1,  # Excluding admin
        'total_votes_cast': Vote.objects.count(),
    }
    
    return render(request, 'results.html', context)

def voting_view(request):
    if request.method == 'POST':
        try:
            # Get the face data from the request
            data = json.loads(request.body)
            face_data = data.get('face_data')
            
            if not face_data:
                return JsonResponse({
                    'verified': False,
                    'message': 'No face data provided'
                }, status=400)
            
            # Process the captured image
            captured_face = process_webcam_image(face_data)
            
            if captured_face is None:
                return JsonResponse({
                    'verified': False,
                    'message': 'Failed to process captured image'
                }, status=400)
            
            # Get all face data from database
            all_face_data = FaceData.objects.all()
            
            # Compare with stored embeddings
            best_match = None
            best_distance = float('inf')
            
            for stored_face in all_face_data:
                stored_embedding = stored_face.get_embedding()
                if stored_embedding is not None:
                    # Get face embedding from stored image
                    stored_image = cv2.imread(stored_face.face_image.path)
                    if stored_image is not None:
                        verification_result = verify_voter_face(captured_face, stored_image)
                        if verification_result['distance'] < best_distance:
                            best_distance = verification_result['distance']
                            best_match = stored_face
            
            # Set threshold for face matching (adjust as needed)
            threshold = 0.6
            
            if best_match and best_distance < threshold:
                # Face verified successfully
                login(request, best_match.user)
                request.session['face_verified'] = True
                
                # Get voting data
                national_candidates = get_national_candidates()
                local_candidates = get_local_candidates(best_match.user.userprofile)
                
                return JsonResponse({
                    'verified': True,
                    'message': 'Face verification successful',
                    'user_id': best_match.user.id,
                    'username': best_match.user.username
                })
            else:
                return JsonResponse({
                    'verified': False,
                    'message': 'Face verification failed. Please try again.'
                }, status=400)
                
        except Exception as e:
            print(f"Error during face verification: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({
                'verified': False,
                'message': f'Error during face verification: {str(e)}'
            }, status=500)
    
    # For GET requests, show the face verification modal
    return render(request, 'vote.html', {
        'show_verification_modal': True
    })

def get_national_candidates():
    """Get all national candidates"""
    candidates = Candidate.objects.filter(approved=True)
    national_positions = [pos[0] for pos in Candidate.NATIONAL_POSITIONS]
    return {c.position: c for c in candidates.filter(position__in=national_positions)}

def get_local_candidates(user_profile):
    """Get local candidates based on user's college and department"""
    candidates = Candidate.objects.filter(approved=True)
    local_positions = [pos[0] for pos in Candidate.LOCAL_POSITIONS]
    
    # Filter candidates based on user's college and department
    college_candidates = candidates.filter(
        position__in=local_positions,
        college=user_profile.college
    )
    
    department_candidates = candidates.filter(
        position__in=local_positions,
        college=user_profile.college,
        department=user_profile.department,
        year_level=user_profile.year_level
    )
    
    return {
        'college': {c.position: c for c in college_candidates},
        'department': {c.position: c for c in department_candidates}
    }

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
        
        # Loop through all images in the directory
        for filename in face_files:
            try:
                student_number = os.path.splitext(filename)[0]
                print(f"Processing file: {filename} (Student Number: {student_number})")
                
                stored_image_path = os.path.join(face_dir, filename)
                stored_image = cv2.imread(stored_image_path)
                
                if stored_image is not None:
                    # Use a simpler face recognition approach
                    if stored_image.shape[0] > 0 and captured_face.shape[0] > 0:
                        # Resize images to same size for comparison
                        stored_resized = cv2.resize(stored_image, (300, 300))
                        captured_resized = cv2.resize(captured_face, (300, 300))
                        
                        # Convert to grayscale for simpler comparison
                        stored_gray = cv2.cvtColor(stored_resized, cv2.COLOR_BGR2GRAY)
                        captured_gray = cv2.cvtColor(captured_resized, cv2.COLOR_BGR2GRAY)
                        
                        # Calculate structural similarity index
                        try:
                            from skimage.metrics import structural_similarity as ssim
                            score = ssim(stored_gray, captured_gray)
                            print(f"Similarity score for {student_number}: {score}")
                            
                            if score > best_score:
                                best_score = score
                                best_match = student_number
                        except ImportError:
                            # Fallback to histogram comparison if skimage not available
                            hist1 = cv2.calcHist([stored_gray], [0], None, [256], [0, 256])
                            hist2 = cv2.calcHist([captured_gray], [0], None, [256], [0, 256])
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
        
        # Simple threshold for matching
        threshold = 0.6  # Standard threshold
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

def cast_vote(request, candidate_id):
    # Check if face is verified
    if not request.session.get('face_verified'):
        return JsonResponse({'success': False, 'message': 'Face verification required'})
    
    if request.method == 'POST':
        try:
            # Check if voting phase is active
            voting_phase = VotingPhase.objects.first()
            if not voting_phase or not voting_phase.is_active():
                return JsonResponse({'success': False, 'message': 'Voting is not currently active'})
            
            # Get candidate
            candidate = Candidate.objects.get(id=candidate_id)
            
            # Get the verified user
            user = request.user
            
            # Check if user has already voted for this position
            if Vote.objects.filter(user=user, candidate__position=candidate.position).exists():
                return JsonResponse({'success': False, 'message': 'You have already voted for this position'})
            
            # Create vote
            Vote.objects.create(user=user, candidate=candidate)
            
            return JsonResponse({'success': True, 'message': 'Vote cast successfully'})
            
        except Candidate.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'Candidate not found'})
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': 'Invalid request method'})

@staff_member_required
def admin_dashboard(request):
    # Get counts for dashboard stats
    total_users = User.objects.count()
    total_candidates = Candidate.objects.count()
    total_votes = Vote.objects.count()
    
    # Calculate voter turnout
    eligible_voters = User.objects.filter(is_active=True).count()
    voters = Vote.objects.values('user').distinct().count()
    voter_turnout = round((voters / eligible_voters * 100) if eligible_voters > 0 else 0, 1)
    
    context = {
        'total_users': total_users,
        'total_candidates': total_candidates,
        'total_votes': total_votes,
        'voter_turnout': voter_turnout,
    }
    
    return render(request, 'admin/admin_dashboard.html', context)

def admin_panel_login(request):
    # Always logout any existing session when accessing admin login
    logout(request)
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None and user.is_superuser:
            login(request, user)
            messages.success(request, 'Welcome to Admin Panel!')
            return redirect('admin_panel:dashboard')
        else:
            messages.error(request, 'Invalid admin credentials.')
            return redirect('admin_panel_login')
            
    return render(request, 'admin_panel_login.html')

@api_view(['GET'])
def check_voting_availability(request):
    """Check if voting is currently available"""
    try:
        voting_phase = VotingPhase.objects.first()
        if not voting_phase:
            return Response({
                'available': False,
                'message': 'Voting period has not been set up yet.'
            })
        
        now = timezone.now()
        if now < voting_phase.start_time:
            return Response({
                'available': False,
                'message': 'Voting period has not started yet.'
            })
        elif now > voting_phase.end_time:
            return Response({
                'available': False,
                'message': 'Voting period has ended.'
            })
        
        # Check if there are any candidates
        if not Candidate.objects.filter(approved=True).exists():
            return Response({
                'available': False,
                'message': 'No candidates available for voting.'
            })
        
        return Response({
            'available': True,
            'message': 'Voting is available.'
        })
    except Exception as e:
        return Response({
            'available': False,
            'message': str(e)
        }, status=500)

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, 'Successfully logged in!')
            return redirect('user:mainpage')
        else:
            messages.error(request, 'Invalid username or password.')
            return redirect('user:login')
    
    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    messages.success(request, 'Successfully logged out!')
    return redirect('user:mainpage')
