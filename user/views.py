from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from .models import UserProfile, Candidate, Vote, VotingPhase, FaceData, VerificationCode
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
    # Get candidates based on student profile
    # Get only approved candidates
    all_candidates = Candidate.objects.all().select_related('user_profile')
    
    # Initialize dictionaries for national and college-wise candidates
    national_candidates = {}
    college_candidates = {}
    
    # Dictionary to store position display names
    position_names = dict(Candidate.POSITION_CHOICES)
    college_names = dict(UserProfile.COLLEGES)
    
    # First, handle national candidates
    # Keep positions in the order defined in the model
    for position_code, position_display in Candidate.NATIONAL_POSITIONS:
        candidates_for_position = all_candidates.filter(position=position_code)
        if candidates_for_position.exists():
            national_candidates[position_display] = list(candidates_for_position)
    
    # Then, organize local candidates by college
    # Keep positions in the order defined in the model
    local_positions = [pos[0] for pos in Candidate.LOCAL_POSITIONS]
    for college_code, college_name in UserProfile.COLLEGES:
        college_candidates[college_name] = {}
        
        # Process each local position in order
        for position_code, position_display in Candidate.LOCAL_POSITIONS:
            # Filter candidates for this college and position
            college_position_candidates = all_candidates.filter(
                position=position_code,
                user_profile__college=college_code
            )
            
            # Add to dictionary if candidates exist
            if college_position_candidates.exists():
                college_candidates[college_name][position_display] = list(college_position_candidates)
    
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
                user_profile__college=college_code
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
        candidates = Candidate.objects.filter(position=pos_code)
        if candidates.exists():
            national_positions[pos_name] = json.dumps([
                {
                    'id': c.id,
                    'user_profile': {
                        'student_name': c.user_profile.student_name,
                        'student_number': c.user_profile.student_number
                    },
                    'college': c.user_profile.college,
                    'department': c.user_profile.course,
                    'year_level': c.user_profile.year_level,
                    'photo': c.photo.url if c.photo and hasattr(c.photo, 'url') else None,
                    'position': c.position
                } for c in candidates
            ])
    
    # Create local_positions dictionary
    local_positions = {}
    for pos_code, pos_name in Candidate.LOCAL_POSITIONS:
        candidates = Candidate.objects.filter(position=pos_code)
        if candidates.exists():
            local_positions[pos_name] = json.dumps([
                {
                    'id': c.id,
                    'user_profile': {
                        'student_name': c.user_profile.student_name,
                        'student_number': c.user_profile.student_number
                    },
                    'college': c.user_profile.college,
                    'department': c.user_profile.course,
                    'year_level': c.user_profile.year_level,
                    'photo': c.photo.url if c.photo and hasattr(c.photo, 'url') else None,
                    'position': c.position
                } for c in candidates
            ])
    
    # Always require face verification
    # Clear any existing verification
    if 'face_verified' in request.session:
        del request.session['face_verified']
    if 'user_profile_id' in request.session:
        del request.session['user_profile_id']
    
    # Log how many candidates are available 
    total_candidates = Candidate.objects.count()
    print(f"Loading voting view with {total_candidates} total candidates.")
    print(f"National positions: {list(national_positions.keys())}")
    print(f"Local positions: {list(local_positions.keys())}")
    
    context = {
        'national_candidates': national_positions,
        'local_candidates': local_positions,
        'show_verification_modal': True  # Always show verification modal
    }
    
    return render(request, 'vote.html', context)

def get_national_candidates():
    """Get all national candidates"""
    candidates = Candidate.objects.all()
    national_positions = [pos[0] for pos in Candidate.NATIONAL_POSITIONS]
    return {c.position: c for c in candidates.filter(position__in=national_positions)}

def get_local_candidates(user_profile):
    """Get local candidates based on user's college and department"""
    candidates = Candidate.objects.all()
    local_positions = [pos[0] for pos in Candidate.LOCAL_POSITIONS]
    
    # Filter candidates based on user's college and department
    college_candidates = []
    for candidate in candidates:
        if (candidate.position in local_positions and 
            candidate.user_profile.college == user_profile.college):
            college_candidates.append(candidate)
    
    department_candidates = []
    for candidate in candidates:
        if (candidate.position in local_positions and
            candidate.user_profile.college == user_profile.college and
            candidate.user_profile.course == user_profile.course and
            candidate.user_profile.year_level == user_profile.year_level):
            department_candidates.append(candidate)
    
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
                        # Use the advanced verification method
                        verification_result = verify_voter_face(captured_face, stored_image)
                        print(f"Verification result for {student_number}: {verification_result}")
                        
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
                            best_verification_result = verification_result
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
            except UserProfile.DoesNotExist:
                return JsonResponse({'success': False, 'message': 'User profile not found in database'})
            
            # Verify eligibility based on position type and user profile
            if not is_eligible_to_vote(user_profile, candidate):
                return JsonResponse({
                    'success': False, 
                    'message': 'You are not eligible to vote for this candidate'
                })
            
            # Check if user has already voted for this position
            if Vote.objects.filter(user_profile=user_profile, candidate__position=candidate.position).exists():
                return JsonResponse({'success': False, 'message': 'You have already voted for this position'})
            
            # Create vote
            Vote.objects.create(user_profile=user_profile, candidate=candidate)
            
            # Log the vote (for audit purposes)
            print(f"Vote recorded - User Profile: {user_profile.student_number}, Candidate: {candidate.user_profile.student_number}, Position: {candidate.position}")
            
            return JsonResponse({'success': True, 'message': 'Vote cast successfully'})
            
        except Candidate.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'Candidate not found'})
        except Exception as e:
            print(f"Error casting vote: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': 'Invalid request method'})

def is_eligible_to_vote(user_profile, candidate):
    """
    Check if a user is eligible to vote for a specific candidate.
    Rules:
    - All users can vote for national positions
    - Users can only vote for candidates from their own college for college positions
    - Users can only vote for candidates from their own department and year level for department positions
    """
    # Get position type
    national_positions = [pos[0] for pos in Candidate.NATIONAL_POSITIONS]
    college_positions = [pos[0] for pos in Candidate.COLLEGE_POSITIONS]
    department_positions = [pos[0] for pos in Candidate.LOCAL_POSITIONS]
    
    position = candidate.position
    
    # National positions - everyone can vote
    if position in national_positions:
        return True
    
    # College positions - only users from the same college
    if position in college_positions:
        return user_profile.college == candidate.user_profile.college
    
    # Department positions - only users from the same department and year level
    if position in department_positions:
        return (user_profile.college == candidate.user_profile.college and
                user_profile.course == candidate.user_profile.course and
                user_profile.year_level == candidate.user_profile.year_level)
    
    # Unknown position type
    return False

@staff_member_required
def admin_dashboard(request):
    # Get counts for dashboard stats
    total_users = UserProfile.objects.count()
    total_candidates = Candidate.objects.count()
    total_votes = Vote.objects.count()
    
    # Calculate voter turnout
    eligible_voters = UserProfile.objects.count()
    voters = Vote.objects.values('user_profile').distinct().count()
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
        if not Candidate.objects.exists():
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

# Add this new API endpoint that will check if voting is currently available
@require_http_methods(["GET"])
def check_voting_status(request):
    """
    API endpoint to check if voting is currently available
    """
    try:
        # Get election settings
        settings = ElectionSettings.objects.first()
        
        if not settings:
            return JsonResponse({
                'is_voting_open': False,
                'message': 'Voting schedule has not been set up yet.'
            })
        
        now = timezone.now()
        
        # Check if voting is active based on the start and end dates
        if not settings.voting_start or not settings.voting_end:
            return JsonResponse({
                'is_voting_open': False,
                'message': 'Voting schedule has not been set up yet.'
            })
        
        # If voting hasn't started yet
        if now < settings.voting_start:
            hours_remaining = int((settings.voting_start - now).total_seconds() / 3600)
            if hours_remaining > 24:
                days = int(hours_remaining / 24)
                return JsonResponse({
                    'is_voting_open': False,
                    'message': f'Voting will start in {days} day{"s" if days > 1 else ""}.'
                })
            else:
                return JsonResponse({
                    'is_voting_open': False,
                    'message': f'Voting will start in {hours_remaining} hour{"s" if hours_remaining > 1 else ""}.'
                })
                
        # If voting has ended
        if now > settings.voting_end:
            return JsonResponse({
                'is_voting_open': False,
                'message': 'Voting period has ended.'
            })
        
        # If we get here, voting is active
        hours_left = int((settings.voting_end - now).total_seconds() / 3600)
        return JsonResponse({
            'is_voting_open': True,
            'message': f'Voting is currently open! {hours_left} hour{"s" if hours_left > 1 else ""} remaining.',
            'voting_end': settings.voting_end.isoformat()
        })
    
    except Exception as e:
        print(f"Error checking voting status: {str(e)}")
        return JsonResponse({
            'is_voting_open': False,
            'message': 'Error checking voting status.'
        }, status=500)

@csrf_exempt
def verify_code(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            code = data.get('code')
            
            if not code:
                return JsonResponse({
                    'success': False,
                    'message': 'Verification code is required'
                })
            
            # Find the verification code
            try:
                verification_code = VerificationCode.objects.get(code=code, is_used=False)
            except VerificationCode.DoesNotExist:
                return JsonResponse({
                    'success': False,
                    'message': 'Invalid or expired verification code'
                })
            
            # Check if code is valid
            if not verification_code.is_valid():
                return JsonResponse({
                    'success': False,
                    'message': 'Verification code has expired'
                })
            
            # Get the user profile
            try:
                user_profile = UserProfile.objects.get(student_number=verification_code.student_number)
            except UserProfile.DoesNotExist:
                return JsonResponse({
                    'success': False,
                    'message': 'User profile not found'
                })
            
            # Store student number before deleting the code
            student_number = verification_code.student_number
            
            # Delete the code instead of marking it as used
            verification_code.delete()
            
            # Check if voting is available
            voting_phase = VotingPhase.objects.first()
            if not voting_phase or not voting_phase.is_active:
                return JsonResponse({
                    'success': True,
                    'user_profile_id': user_profile.id,
                    'student_number': user_profile.student_number,
                    'college': user_profile.college,
                    'department': user_profile.course,
                    'year_level': user_profile.year_level,
                    'redirect_url': reverse('user:mainpage')
                })
            
            # Return success with user profile data
            return JsonResponse({
                'success': True,
                'user_profile_id': user_profile.id,
                'student_number': user_profile.student_number,
                'college': user_profile.college,
                'department': user_profile.course,
                'year_level': user_profile.year_level,
                'redirect_url': reverse('user:vote')
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'message': 'Invalid request format'
            })
        except Exception as e:
            print(f"Error during code verification: {str(e)}")
            return JsonResponse({
                'success': False,
                'message': f'Error during verification: {str(e)}'
            })
    
    return JsonResponse({
        'success': False,
        'message': 'Invalid request method'
    })

@csrf_exempt
def submit_all_votes(request):
    """
    API endpoint to submit all votes at once.
    Receives a list of votes, validates each one, then saves them all in a single transaction.
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'message': 'Invalid request method'
        }, status=405)
    
    try:
        # Parse the request data
        data = json.loads(request.body.decode('utf-8'))
        user_profile_id = data.get('user_profile_id')
        votes = data.get('votes', [])
        
        print(f"[VOTE DEBUG] Processing {len(votes)} votes for user profile ID: {user_profile_id}")
        print(f"[VOTE DEBUG] Votes data: {votes}")
        
        # Validation checks
        if not user_profile_id:
            print("[VOTE ERROR] No user profile ID provided")
            return JsonResponse({
                'success': False,
                'message': 'User profile ID required. Please verify your identity first.'
            }, status=400)
        
        if not votes or not isinstance(votes, list) or len(votes) == 0:
            print("[VOTE ERROR] No votes provided")
            return JsonResponse({
                'success': False,
                'message': 'No votes provided.'
            }, status=400)
        
        # Check if voting phase is active
        settings = ElectionSettings.objects.first()
        if not settings or not settings.is_voting_open():
            print("[VOTE ERROR] Voting is not currently active")
            return JsonResponse({
                'success': False, 
                'message': 'Voting is not currently active'
            }, status=403)
        
        # Get the user profile
        try:
            user_profile = UserProfile.objects.get(id=user_profile_id)
            print(f"[VOTE DEBUG] Found user profile: {user_profile.student_number} ({user_profile.student_name})")
        except UserProfile.DoesNotExist:
            print(f"[VOTE ERROR] User profile not found for ID: {user_profile_id}")
            return JsonResponse({
                'success': False, 
                'message': 'User profile not found in database'
            }, status=404)
        
        # Prepare storage for vote objects and errors
        vote_objects = []
        positions_voted = set()
        errors = []
        
        # Validate each vote
        for i, vote_data in enumerate(votes):
            candidate_id = vote_data.get('candidateId')
            position = vote_data.get('position')
            
            print(f"[VOTE DEBUG] Processing vote {i+1}: candidate ID {candidate_id}, position {position}")
            
            # Skip if already validated a vote for this position
            if position in positions_voted:
                print(f"[VOTE WARNING] Multiple votes for position {position}")
                errors.append(f"Multiple votes for position {position} - only the first will be counted")
                continue
                
            try:
                # Get candidate
                candidate = Candidate.objects.get(id=candidate_id)
                print(f"[VOTE DEBUG] Found candidate: {candidate.user_profile.student_name} for position {candidate.position}")
                
                # Verify position matches
                if candidate.position != position:
                    print(f"[VOTE ERROR] Position mismatch: expected {position}, got {candidate.position}")
                    errors.append(f"Position mismatch for candidate {candidate_id}")
                    continue
                
                # Verify eligibility based on position type and user profile
                if not is_eligible_to_vote(user_profile, candidate):
                    print(f"[VOTE ERROR] User {user_profile.student_number} not eligible to vote for {candidate.position}")
                    errors.append(f"Not eligible to vote for {candidate.position}")
                    continue
                
                # Create vote object (don't save yet)
                vote_objects.append(Vote(user_profile=user_profile, candidate=candidate))
                positions_voted.add(position)
                print(f"[VOTE DEBUG] Vote validated for {candidate.position}")
                
            except Candidate.DoesNotExist:
                print(f"[VOTE ERROR] Candidate {candidate_id} not found")
                errors.append(f"Candidate {candidate_id} not found")
                continue
            except Exception as e:
                print(f"[VOTE ERROR] Unexpected error processing vote for candidate {candidate_id}: {str(e)}")
                errors.append(f"Error processing vote: {str(e)}")
                continue
        
        # If no valid votes, return error
        if not vote_objects:
            print("[VOTE ERROR] No valid votes to submit")
            return JsonResponse({
                'success': False,
                'message': 'No valid votes to submit.',
                'errors': errors
            }, status=400)
        
        # Check if user has already voted for any positions
        existing_votes = Vote.objects.filter(user_profile=user_profile)
        existing_positions = existing_votes.values_list('candidate__position', flat=True)
        
        if existing_positions:
            print(f"[VOTE ERROR] User already voted for positions: {list(existing_positions)}")
            return JsonResponse({
                'success': False,
                'message': 'You have already voted for some positions.',
                'positions': list(existing_positions)
            }, status=400)
        
        # Save all votes in a single transaction
        try:
            with transaction.atomic():
                for vote in vote_objects:
                    vote.save()
                    print(f"[VOTE SUCCESS] Vote recorded - User: {user_profile.student_number}, Candidate: {vote.candidate.user_profile.student_number}, Position: {vote.candidate.position}")
        except Exception as e:
            print(f"[VOTE ERROR] Failed to save votes: {str(e)}")
            return JsonResponse({
                'success': False,
                'message': f'Error saving votes to database: {str(e)}'
            }, status=500)
        
        print(f"[VOTE SUCCESS] Successfully submitted {len(vote_objects)} votes for user {user_profile.student_number}")
        return JsonResponse({
            'success': True,
            'message': f'Successfully submitted {len(vote_objects)} votes.',
            'positions_voted': list(positions_voted),
            'warnings': errors if errors else None
        })
        
    except json.JSONDecodeError:
        print("[VOTE ERROR] Invalid JSON data")
        return JsonResponse({
            'success': False,
            'message': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        print(f"[VOTE ERROR] Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'message': f'Error submitting votes: {str(e)}'
        }, status=500)
