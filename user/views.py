from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from .models import UserProfile, Candidate, Vote, VotingPhase, FaceData
from admin_panel.models import ElectionSettings
from django.core.exceptions import ValidationError
from .face_utils import process_webcam_image, verify_voter_face, preprocess_face_image
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
import time

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
    # Get all positions and candidates for voting
    national_positions = {}
    for pos_code, pos_name in Candidate.NATIONAL_POSITIONS:
        candidates = Candidate.objects.filter(position=pos_code, approved=True)
        if candidates.exists():
            national_positions[pos_name] = json.dumps([
                {
                    'id': c.id,
                    'user_name': c.user.get_full_name(),
                    'college': c.college,
                    'department': c.department,
                    'year_level': c.year_level,
                    'photo': c.photo.url if c.photo and c.photo.url else None,
                    'position': c.position
                } for c in candidates
            ])
    
    # Create local_positions dictionary
    local_positions = {}
    for pos_code, pos_name in Candidate.LOCAL_POSITIONS:
        candidates = Candidate.objects.filter(position=pos_code, approved=True)
        if candidates.exists():
            local_positions[pos_name] = json.dumps([
                {
                    'id': c.id,
                    'user_name': c.user.get_full_name(),
                    'college': c.college,
                    'department': c.department,
                    'year_level': c.year_level,
                    'photo': c.photo.url if c.photo and c.photo.url else None,
                    'position': c.position
                } for c in candidates
            ])
    
    # Always require face verification
    # Clear any existing verification
    if 'face_verified' in request.session:
        del request.session['face_verified']
    if 'user_profile_id' in request.session:
        del request.session['user_profile_id']
    
    context = {
        'national_candidates': national_positions,
        'local_candidates': local_positions,
        'show_verification_modal': True  # Always show verification modal
    }
    
    return render(request, 'vote.html', context)

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
        department=user_profile.course,
        year_level=user_profile.year_level
    )
    
    return {
        'college': {c.position: c for c in college_candidates},
        'department': {c.position: c for c in department_candidates}
    }

@csrf_exempt
def verify_face(request):
    """Enhanced face verification function with multiple attempts and better logging"""
    # Initialize variables that might be referenced in exception handlers
    best_match = None
    best_score = 0
    best_method = None
    best_verification_result = None
    threshold = 0.6
    
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
        
        # Create debug directory if it doesn't exist
        debug_dir = os.path.join(settings.MEDIA_ROOT, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        # For debugging: save the captured face
        try:
            timestamp = int(time.time())
            debug_path = os.path.join(debug_dir, f'captured_{timestamp}.jpg')
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
        
        best_distance = float('inf')
        verification_method = 'advanced'  # Use 'advanced' for new method or 'simple' for old method
        verification_attempts = 1  # Perform only 1 verification attempt to reduce processing time
        
        # Initialize verification log file
        log_file_path = os.path.join(debug_dir, 'verification_log.txt')
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"\n--- New Verification Session: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        # Collection of all valid candidates (those with ≥2 successful methods)
        valid_candidates = []
        
        # Perform only a single verification attempt
        print(f"Verification attempt 1/1")
            
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
                                    debug_captured_path = os.path.join(debug_dir, f'preprocessed_captured_{timestamp}.jpg')
                                    debug_stored_path = os.path.join(debug_dir, f'preprocessed_stored_{student_number}.jpg')
                                    
                                    cv2.imwrite(debug_captured_path, preprocessed_captured)
                                    cv2.imwrite(debug_stored_path, preprocessed_stored)
                                    print(f"Saved debug preprocessed images for {student_number}")
                            except Exception as e:
                                print(f"Warning: Could not save debug preprocessed images: {str(e)}")
                        
                        # Use the advanced verification method
                        verification_result = verify_voter_face(captured_face, stored_image)
                        print(f"Verification result for {student_number}: {verification_result}")
                        
                        # Log verification result
                        try:
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(f"Student: {student_number}\n")
                                log_file.write(f"Method: {verification_result.get('method', 'unknown')}\n")
                                log_file.write(f"Distance: {verification_result.get('distance', 'unknown')}\n")
                                log_file.write(f"Score: {verification_result.get('score', 1.0 - verification_result.get('distance', 1.0))}\n")
                                log_file.write(f"Verified: {verification_result.get('verified', False)}\n\n")
                        except Exception as e:
                            print(f"Error writing to log file: {str(e)}")
                        
                        # Store the verification result for the best match
                        if best_match is None or student_number == best_match:
                            best_verification_result = verification_result
                        
                        # Get successful methods count
                        successful_methods = verification_result.get('successful_methods', 0)
                        score = 1.0 - verification_result.get('distance', 1.0)
                        
                        # IMPORTANT: First, prioritize candidates with at least 2 successful methods
                        if successful_methods >= 2:
                            valid_candidates.append({
                                'student_number': student_number,
                                'score': score,
                                'method': verification_result.get('method', 'unknown'),
                                'distance': verification_result.get('distance', 1.0),
                                'successful_methods': successful_methods,
                                'verification_result': verification_result
                            })
                            
                        # Track best overall match regardless of methods
                        if score > best_score:
                            best_match = student_number
                            best_score = score
                            best_distance = verification_result.get('distance', 1.0)
                            best_method = verification_result.get('method', 'unknown')
                            print(f"Found better match ({score}) for {student_number}")
                    else:
                        print(f"Failed to load image: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
        
        # Select the best match from candidates with at least 2 successful methods
        if valid_candidates:
            # Sort by score (highest first)
            valid_candidates.sort(key=lambda x: x['score'], reverse=True)
            best_candidate = valid_candidates[0]
            
            # Update best match info with the best valid candidate
            best_match = best_candidate['student_number']
            best_score = best_candidate['score']
            best_method = best_candidate['method']
            best_distance = best_candidate['distance']
            best_verification_result = best_candidate['verification_result']
            print(f"Selected best valid candidate: {best_match} (score: {best_score}, methods: {best_candidate['successful_methods']})")
        
        # Set threshold based on verification method
        threshold = 0.6 if verification_method == 'advanced' else 0.6
        print(f"Best match: {best_match}, Score: {best_score}, Method: {best_method}, Threshold: {threshold}")
        
        # Additional verification check - require minimum number of successful methods
        min_required_success = 2  # Require at least 2 successful methods
        successful_methods_count = 0
        methods_details = []
        
        # If we have verification_details in the result, check how many methods succeeded
        if hasattr(best_verification_result, 'get') and best_verification_result.get('verification_details'):
            verification_details = best_verification_result.get('verification_details', [])
            successful_methods_count = sum(1 for detail in verification_details if detail.get('verified', False))
            methods_details = [
                f"{detail.get('method')}: {detail.get('score', 0):.2f}/{detail.get('threshold', threshold)}"
                for detail in verification_details
            ]
            print(f"Successful verification methods: {successful_methods_count}/{len(verification_details)}")
            print(f"Method details: {', '.join(methods_details)}")
        
        # Final verification result
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"FINAL RESULT:\n")
            log_file.write(f"Face files checked: {len(face_files)}\n")
            log_file.write(f"Best match: {best_match}\n")
            log_file.write(f"Best score: {best_score}\n")
            log_file.write(f"Best method: {best_method}\n")
            log_file.write(f"Threshold: {threshold}\n")
            log_file.write(f"Successful methods: {successful_methods_count}/{min_required_success} required\n")
            log_file.write(f"Methods details: {', '.join(methods_details)}\n")
            log_file.write(f"Verified: {best_match is not None and best_score > threshold and successful_methods_count >= min_required_success}\n")
            log_file.write("-------------------------------------\n")
        
        # IMPORTANT: Verify BOTH conditions: score threshold AND minimum number of successful methods
        verification_passed = (best_match and 
                              best_score > threshold and 
                              successful_methods_count >= min_required_success)
        
        if verification_passed:
            try:
                user_profile = UserProfile.objects.get(student_number=best_match)
                print(f"Found matching user profile for student number: {best_match}")
                
                # Don't store in session - make verification temporary
                # Just return the user profile data to the client
                
                return JsonResponse({
                    'success': True,
                    'message': 'Face verification successful',
                    'user_profile_id': user_profile.id,
                    'student_number': user_profile.student_number,
                    'college': user_profile.college,
                    'department': user_profile.course,
                    'year_level': user_profile.year_level,
                    'score': best_score,
                    'method': best_method,
                    'successful_methods': successful_methods_count,
                    'required_methods': min_required_success,
                    'method_details': methods_details,
                    'verification_summary': f"Checked {len(face_files)} face images, best match: {best_match} (score: {best_score:.2f}, methods: {successful_methods_count}/{min_required_success})",
                    'redirect_url': reverse('user:vote')  # Add URL for redirection
                })
            except UserProfile.DoesNotExist:
                print(f"No user profile found for student number: {best_match}")
                return JsonResponse({
                    'success': False,
                    'message': 'User profile not found'
                }, status=404)
        else:
            print("Face verification failed")
            
            # Generate detailed error message explaining why verification failed
            failure_reason = "Verification failed"
            if not best_match:
                failure_reason = "No face match found in our records"
            elif best_score <= threshold:
                failure_reason = f"Confidence score too low ({best_score:.2f}, needed >{threshold})"
            elif successful_methods_count < min_required_success:
                failure_reason = f"Not enough verification methods succeeded ({successful_methods_count}/{min_required_success} required)"
            
            # Include details about which methods passed/failed
            method_results = ""
            if methods_details:
                method_results = " - Methods: " + ", ".join(methods_details)
            
            return JsonResponse({
                'success': False,
                'message': 'Face verification failed. Please try again with better lighting and positioning.',
                'best_score': best_score,
                'threshold': threshold,
                'method': best_method,
                'successful_methods': successful_methods_count,
                'required_methods': min_required_success,
                'reason': failure_reason,
                'method_details': methods_details,
                'verification_summary': f"Checked {len(face_files)} face images. Best match: {best_match or 'None'} with score {best_score:.2f}. {method_results}"
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
    # We no longer check session for face verification
    # Instead, rely on the data sent in the POST request
    if request.method == 'POST':
        try:
            # Parse the request data
            data = json.loads(request.body.decode('utf-8'))
            user_profile_id = data.get('user_profile_id')
            
            if not user_profile_id:
                return JsonResponse({
                    'success': False,
                    'message': 'User profile ID required. Please verify your face first.'
                }, status=400)
            
            # Check if voting phase is active
            voting_phase = VotingPhase.objects.first()
            if not voting_phase or not voting_phase.is_active():
                return JsonResponse({'success': False, 'message': 'Voting is not currently active'})
            
            # Get candidate
            candidate = Candidate.objects.get(id=candidate_id)
            
            # Get the user profile from the request data
            try:
                user_profile = UserProfile.objects.get(id=user_profile_id)
                user = user_profile.user
            except UserProfile.DoesNotExist:
                return JsonResponse({'success': False, 'message': 'User profile not found in database'})
            
            # Verify eligibility based on position type and user profile
            if not is_eligible_to_vote(user_profile, candidate):
                return JsonResponse({
                    'success': False, 
                    'message': 'You are not eligible to vote for this candidate'
                })
            
            # Check if user has already voted for this position
            if Vote.objects.filter(user=user, candidate__position=candidate.position).exists():
                return JsonResponse({'success': False, 'message': 'You have already voted for this position'})
            
            # Create vote
            Vote.objects.create(user=user, candidate=candidate)
            
            # Log the vote (for audit purposes)
            print(f"Vote recorded - User: {user.username}, Candidate: {candidate.user.username}, Position: {candidate.position}")
            
            return JsonResponse({'success': True, 'message': 'Vote cast successfully'})
            
        except Candidate.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'Candidate not found'})
        except Exception as e:
            print(f"Error casting vote: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': 'Invalid request method'})

def is_eligible_to_vote(user_profile, candidate):
    """Check if a user is eligible to vote for a specific candidate based on position type"""
    # Get position type (national, college, department)
    position = candidate.position
    
    # National positions - everyone can vote
    national_positions = [pos[0] for pos in Candidate.NATIONAL_POSITIONS if pos[0] not in 
                         ['GOVERNOR', 'VICE_GOVERNOR', 'SECRETARY_COLLEGE', 'TREASURER_COLLEGE', 
                          'AUDITOR_COLLEGE', 'PRO_COLLEGE']]
    
    if position in national_positions:
        return True
    
    # College positions - only users from the same college can vote
    college_positions = ['GOVERNOR', 'VICE_GOVERNOR', 'SECRETARY_COLLEGE', 'TREASURER_COLLEGE', 
                         'AUDITOR_COLLEGE', 'PRO_COLLEGE']
    
    if position in college_positions:
        return user_profile.college == candidate.college
    
    # Department positions - users from same department, college, and year level
    return (user_profile.college == candidate.college and
            user_profile.course == candidate.department and
            user_profile.year_level == candidate.year_level)

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
