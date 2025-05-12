from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout as auth_logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from .models import UserProfile, Candidate, Vote, VotingPhase, VerificationCode, Photo
from admin_panel.models import ElectionSettings, ChairmanAccount
from django.core.exceptions import ValidationError
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
from django.views.decorators.http import require_http_methods, require_GET
import insightface
from insightface.app import FaceAnalysis
from django.views.decorators.csrf import csrf_exempt
import time
from PIL import Image
import io
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from .face_utils import process_webcam_image, get_face_embedding, recognize_face
import datetime
from collections import OrderedDict

LOGIN_URL = 'login'  # or whatever your login URL name is

def face_upload_view(request):
    """Render the face upload template."""
    return render(request, 'face_upload.html')

@csrf_exempt
def verify_student_id(request):
    """Verify if the student ID exists in the database."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            student_id = data.get('student_id')
            
            # Check if student number exists
            user_profile = UserProfile.objects.filter(student_number=student_id).first()
            
            if user_profile:
                return JsonResponse({
                    'valid': True,
                    'message': 'Student ID verified successfully'
                })
            else:
                return JsonResponse({
                    'valid': False,
                    'message': 'Invalid Student ID'
                })
        except Exception as e:
            return JsonResponse({
                'valid': False,
                'message': str(e)
            }, status=400)
    
    return JsonResponse({
        'valid': False,
        'message': 'Invalid request method'
    }, status=405)

@csrf_exempt
def submit_face_images(request):
    """Process and store multiple face images for a user profile."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            student_id = data.get('student_id')
            images = data.get('images', [])
            angles = data.get('angles', [])  # List of angles corresponding to each image
            
            if not images:
                return JsonResponse({
                    'success': False,
                    'message': 'No images provided'
                }, status=400)
            
            if len(images) > 5:
                return JsonResponse({
                    'success': False,
                    'message': 'Maximum 5 photos allowed'
                }, status=400)
            
            # Get user profile
            try:
                user_profile = UserProfile.objects.get(student_number=student_id)
            except UserProfile.DoesNotExist:
                return JsonResponse({
                    'success': False,
                    'message': 'Invalid Student ID'
                }, status=400)
            
            # Create face_data directory if it doesn't exist
            face_data_dir = os.path.join(settings.MEDIA_ROOT, 'face_data')
            os.makedirs(face_data_dir, exist_ok=True)
            
            # Delete existing photos for this user
            Photo.objects.filter(user_profile=user_profile).delete()
            
            # Process each image
            embeddings = []
            saved_photos = []
            
            with transaction.atomic():
                for idx, image_data in enumerate(images):
                    try:
                        # Remove the data URL prefix if present
                        if ',' in image_data:
                            image_data = image_data.split(',')[1]
                        
                        # Decode base64 image
                        image_bytes = base64.b64decode(image_data)
                        
                        # Generate unique filename
                        timestamp = int(time.time() * 1000)
                        filename = f"{student_id}_{timestamp}_{idx}.jpg"
                        
                        # Create Photo instance
                        photo = Photo(
                            user_profile=user_profile
                        )
                        
                        # Save the image file to the model's ImageField
                        from django.core.files.base import ContentFile
                        photo.photo.save(filename, ContentFile(image_bytes), save=True)
                        saved_photos.append(photo)
                        
                        # Extract face embedding
                        if user_profile.extract_and_save_face_embedding(photo.photo.path):
                            embedding = user_profile.get_face_embedding()
                            if embedding is not None:
                                embeddings.append(embedding)
                        
                    except Exception as e:
                        # Log the error
                        print(f"Error processing image {idx}: {str(e)}")
                        # Delete any saved photos
                        for photo in saved_photos:
                            photo.delete()
                        return JsonResponse({
                            'success': False,
                            'message': f'Error processing image {idx + 1}: {str(e)}'
                        }, status=500)
            
            if not embeddings:
                return JsonResponse({
                    'success': False,
                    'message': 'Failed to process any of the face images.'
                }, status=400)
            
            # Calculate and save average embedding to UserProfile
            avg_embedding = np.mean(embeddings, axis=0)
            user_profile.face_image = json.dumps(avg_embedding.tolist())
            user_profile.save()
            
            return JsonResponse({
                'success': True,
                'message': f'Successfully uploaded and processed {len(saved_photos)} face images'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'message': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            print(f"Error: {str(e)}")
            return JsonResponse({
                'success': False,
                'message': f'Error processing face images: {str(e)}'
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'message': 'Invalid request method'
    }, status=405)

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
    # Get active election settings
    active_settings = ElectionSettings.objects.filter(is_active=True).first()
    voting_ongoing = False
    if not active_settings or active_settings.is_voting_open:
        voting_ongoing = True
        context = {
            'voting_ongoing': True,
            'total_voters': UserProfile.objects.count(),
            'total_votes_cast': Vote.objects.count(),
        }
        return render(request, 'results.html', context)
    # ... existing results logic ...
    # (copy the rest of the function as is, but add 'voting_ongoing': False to context)
    national_positions = [pos[0] for pos in Candidate.NATIONAL_POSITIONS]
    college_positions = Candidate.COLLEGE_POSITIONS
    local_positions = Candidate.LOCAL_POSITIONS
    candidates = Candidate.objects.all().select_related('user_profile')
    votes = Vote.objects.all()
    total_voters = UserProfile.objects.count()
    total_votes_cast = votes.count()
    national_results = OrderedDict()
    for pos_code, pos_name in Candidate.NATIONAL_POSITIONS:
        pos_candidates = candidates.filter(position=pos_code)
        ranked_candidates = []
        for c in pos_candidates:
            vote_count = votes.filter(candidate=c).count()
            percentage = (vote_count / total_votes_cast * 100) if total_votes_cast > 0 else 0
            ranked_candidates.append({
                'candidate': c,
                'vote_count': vote_count,
                'percentage': percentage
            })
        ranked_candidates.sort(key=lambda x: x['vote_count'], reverse=True)
        if ranked_candidates:
            national_results[pos_name] = ranked_candidates
    college_results = OrderedDict()
    colleges = [c[0] for c in UserProfile.COLLEGES]
    for pos_code, pos_name in Candidate.COLLEGE_POSITIONS:
        college_results[pos_name] = OrderedDict()
        for college in colleges:
            pos_candidates = candidates.filter(position=pos_code, user_profile__college=college)
            ranked_candidates = []
            for c in pos_candidates:
                vote_count = votes.filter(candidate=c).count()
                percentage = (vote_count / total_votes_cast * 100) if total_votes_cast > 0 else 0
                ranked_candidates.append({
                    'candidate': c,
                    'votes': vote_count,
                    'percentage': percentage
                })
            ranked_candidates.sort(key=lambda x: x['votes'], reverse=True)
            college_results[pos_name][college] = ranked_candidates
        colleges_list = list(college_results[pos_name].keys())
        max_candidates = max((len(candidates_list) for candidates_list in college_results[pos_name].values()), default=0)
        rows = []
        for i in range(max_candidates):
            row = []
            for college in colleges_list:
                candidates_list = college_results[pos_name][college]
                row.append(candidates_list[i] if i < len(candidates_list) else None)
            rows.append(row)
        college_results[pos_name] = {
            'colleges': colleges_list,
            'rows': rows,
        }
    COLLEGE_COURSES = {
        'CAS': ['BSINT', 'BSCS'],
        'CBA': ['BSBA', 'BSOA', 'BSHM'],
        'CCJE': ['BSCRIM'],
        'CAF': ['BSA', 'BSF', 'BS ANSCI'],
        'CIT': ['BSIT'],
        'CTED': ['BEED', 'BSED major in english', 'BSED major in math', 'BSED major in science'],
    }
    department_results = OrderedDict()
    departments = [c[0] for c in UserProfile.COLLEGES]
    years = [y[0] for y in UserProfile.YEAR_LEVEL_CHOICES if y[0] in ['1', '2', '3']]
    for dept in departments:
        department_results[dept] = OrderedDict()
        valid_courses = COLLEGE_COURSES.get(dept, [])
        for course in valid_courses:
            department_results[dept][course] = OrderedDict()
            for year in years:
                department_results[dept][course][year] = OrderedDict()
                for pos_code, pos_name in Candidate.LOCAL_POSITIONS:
                    pos_candidates = candidates.filter(
                        position=pos_code,
                        user_profile__college=dept,
                        user_profile__course=course,
                        user_profile__year_level=year
                    )
                    ranked_candidates = []
                    for c in pos_candidates:
                        vote_count = votes.filter(candidate=c).count()
                        percentage = (vote_count / total_votes_cast * 100) if total_votes_cast > 0 else 0
                        ranked_candidates.append({
                            'candidate': c,
                            'votes': vote_count,
                            'percentage': percentage
                        })
                    ranked_candidates.sort(key=lambda x: x['votes'], reverse=True)
                    if ranked_candidates:
                        department_results[dept][course][year][pos_name] = ranked_candidates
    context = {
        'national_results': national_results,
        'college_results': college_results,
        'department_results': department_results,
        'total_voters': total_voters,
        'total_votes_cast': total_votes_cast,
        'voting_ongoing': False,
    }
    return render(request, 'results.html', context)

def voting_view(request):
    # Require supervisor authentication for voting facility
    if not request.session.get('facility_supervisor_authenticated'):
        return redirect('user:supervisor_login')
    # Get all positions and candidates for voting
    national_positions = {}
    for pos_code, pos_name in Candidate.NATIONAL_POSITIONS:
        candidates = Candidate.objects.filter(position=pos_code)
        if candidates.exists():
            national_positions[pos_name] = [
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
            ]
    # College positions
    college_positions = {}
    for pos_code, pos_name in Candidate.COLLEGE_POSITIONS:
        candidates = Candidate.objects.filter(position=pos_code)
        if candidates.exists():
            college_positions[pos_name] = [
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
            ]
    # Create local_positions dictionary
    local_positions = {}
    for pos_code, pos_name in Candidate.LOCAL_POSITIONS:
        candidates = Candidate.objects.filter(position=pos_code)
        if candidates.exists():
            local_positions[pos_name] = [
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
            ]
    # Always require face verification
    if 'face_verified' in request.session:
        del request.session['face_verified']
    if 'user_profile_id' in request.session:
        del request.session['user_profile_id']
    
    # Log how many candidates are available 
    total_candidates = Candidate.objects.count()
    print(f"Loading voting view with {total_candidates} total candidates.")
    print(f"National positions: {list(national_positions.keys())}")
    print(f"College positions: {list(college_positions.keys())}")
    print(f"Local positions: {list(local_positions.keys())}")
    
    context = {
        'national_candidates': json.dumps(national_positions) if national_positions else '{}',
        'college_candidates': json.dumps(college_positions) if college_positions else '{}',
        'local_candidates': json.dumps(local_positions) if local_positions else '{}',
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
            
            # Check if voting is active using ElectionSettings
            active_settings = ElectionSettings.objects.filter(is_active=True).first()
            if not active_settings or not active_settings.is_voting_open:
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
            if Vote.objects.filter(user_profile=user_profile, candidate__position=candidate.position, school_year=active_settings.school_year).exists():
                return JsonResponse({'success': False, 'message': 'You have already voted for this position'})
            
            # Create vote with active school year
            Vote.objects.create(
                user_profile=user_profile,
                candidate=candidate,
                school_year=active_settings.school_year
            )
            
            # Log the vote (for audit purposes)
            print(f"Vote recorded - User Profile: {user_profile.student_number}, Candidate: {candidate.user_profile.student_number}, Position: {candidate.position}, School Year: {active_settings.school_year}")
            
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
    auth_logout(request)
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            if user.is_superuser:
                login(request, user)
                messages.success(request, 'Welcome to Admin Panel!')
                return redirect('admin_panel:dashboard')
            # Chairman login
            if ChairmanAccount.objects.filter(user=user).exists():
                login(request, user)
                messages.success(request, 'Welcome Chairman!')
                return redirect('admin_panel:dashboard')
            else:
                messages.error(request, 'You do not have permission to access this panel.')
                return redirect('admin_panel_login')
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
    auth_logout(request)
    messages.success(request, 'Successfully logged out!')
    return redirect('user:mainpage')

@require_http_methods(["GET"])
def check_voting_status(request):
    """API endpoint to check if voting is currently open."""
    try:
        print("\n=== Checking Voting Status ===")
        # Get active election settings
        active_settings = ElectionSettings.objects.filter(is_active=True).first()
        print(f"Active Settings: {active_settings}")
        if not active_settings:
            print("No active election settings found")
            return JsonResponse({'is_voting_open': False, 'message': 'No active election found.'})
        now = datetime.datetime.now()
        print(f"Current time: {now}")
        # Check if all required fields are set
        print(f"Voting Date: {active_settings.voting_date}")
        print(f"Voting Time Start: {active_settings.voting_time_start}")
        print(f"Voting Time End: {active_settings.voting_time_end}")
        if not (active_settings.voting_date and active_settings.voting_time_start and active_settings.voting_time_end):
            print("Missing required voting schedule fields")
            return JsonResponse({'is_voting_open': False, 'message': 'Voting schedule has not been set up yet.'})
        # Create naive datetime objects for start and end times
        voting_start = datetime.datetime.combine(active_settings.voting_date, active_settings.voting_time_start)
        voting_end = datetime.datetime.combine(active_settings.voting_date, active_settings.voting_time_end)
        print(f"Voting Start: {voting_start}")
        print(f"Voting End: {voting_end}")
        # If voting hasn't started yet
        if now < voting_start:
            time_to_start = voting_start - now
            hours = time_to_start.total_seconds() / 3600
            print(f"Voting hasn't started yet. Time until start: {hours} hours")
            if hours > 24:
                days = int(hours / 24)
                return JsonResponse({
                    'is_voting_open': False,
                    'message': f'Voting will start in {days} day{"s" if days > 1 else ""}.'
                })
            else:
                hours = int(hours)
                return JsonResponse({
                    'is_voting_open': False,
                    'message': f'Voting will start in {hours} hour{"s" if hours > 1 else ""}.'
                })
        # If voting has ended
        if now >= voting_end:
            print("Voting period has ended")
            return JsonResponse({
                'is_voting_open': False,
                'message': 'Voting period has ended.'
            })
        # If we get here, voting is active
        time_remaining = voting_end - now
        hours_left = int(time_remaining.total_seconds() / 3600)
        print(f"Voting is active! {hours_left} hours remaining")
        return JsonResponse({
            'is_voting_open': True,
            'message': f'Voting is currently open! {hours_left} hour{"s" if hours_left > 1 else ""} remaining.',
            'voting_end': voting_end.isoformat()
        })
    except Exception as e:
        print(f"\nError in check_voting_status: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return JsonResponse({'is_voting_open': False, 'message': 'An error occurred while checking the voting status.'}, status=500)

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
            
            # Check if already voted in the current school year
            active_settings = ElectionSettings.objects.filter(is_active=True).first()
            already_voted = False
            if active_settings:
                vote_count = Vote.objects.filter(
                    user_profile=user_profile,
                    school_year=active_settings.school_year
                ).count()
                already_voted = vote_count > 0
            
            # Store student number before deleting the code
            student_number = verification_code.student_number
            
            # Do NOT delete or mark the code as used here
            # verification_code.delete()
            
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
                    'redirect_url': reverse('user:mainpage'),
                    'already_voted': already_voted
                })
            
            # Return success with user profile data
            return JsonResponse({
                'success': True,
                'user_profile_id': user_profile.id,
                'student_number': user_profile.student_number,
                'college': user_profile.college,
                'department': user_profile.course,
                'year_level': user_profile.year_level,
                'redirect_url': reverse('user:vote'),
                'already_voted': already_voted
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
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            user_profile_id = data.get('user_profile_id')
            votes = data.get('votes', [])
            verification_code = data.get('verification_code')  # <-- Get code from request
            print(f"[VOTE DEBUG] User profile ID: {user_profile_id}")
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
            # Check if voting is active using ElectionSettings
            active_settings = ElectionSettings.objects.filter(is_active=True).first()
            if not active_settings or not active_settings.is_voting_open:
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
            # EXTRA CHECK: Prevent voting if user has already voted in the current active school year
            already_voted = False
            if active_settings:
                vote_count = Vote.objects.filter(
                    user_profile=user_profile,
                    school_year=active_settings.school_year
                ).count()
                already_voted = vote_count > 0
            if already_voted:
                print(f"[VOTE ERROR] User {user_profile.student_number} already voted in school year {active_settings.school_year}")
                return JsonResponse({
                    'success': False,
                    'message': 'You have already voted in this election.'
                }, status=400)
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
                    vote_objects.append(Vote(user_profile=user_profile, candidate=candidate, school_year=active_settings.school_year))
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
                    # Mark verification code as used if provided
                    if verification_code:
                        from user.models import VerificationCode
                        try:
                            code_obj = VerificationCode.objects.get(code=verification_code, is_used=False)
                            code_obj.is_used = True
                            code_obj.save()
                            print(f"[VOTE SUCCESS] Verification code {verification_code} marked as used.")
                        except VerificationCode.DoesNotExist:
                            print(f"[VOTE WARNING] Verification code {verification_code} not found or already used.")
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
    return JsonResponse({
        'success': False,
        'message': 'Invalid request method'
    }, status=405)

@csrf_exempt
def verify_face(request):
    try:
        print("\n=== Starting Face Verification ===")
        data = json.loads(request.body)
        face_data = data.get('face_data')
        if not face_data:
            print("No face data provided")
            return JsonResponse({'success': False, 'message': 'No face data provided'}, status=400)

        # Decode and save the webcam image
        temp_image_path = process_webcam_image(face_data)
        if not temp_image_path:
            print("Failed to process captured image")
            return JsonResponse({'success': False, 'message': 'Failed to process captured image'}, status=400)

        # Extract embedding from the image
        input_embedding = get_face_embedding(temp_image_path)
        if input_embedding is None:
            print("Failed to extract face embedding")
            return JsonResponse({'success': False, 'message': 'Failed to extract face embedding'}, status=400)

        # Compare to all UserProfile embeddings with a stricter threshold
        student_number, student_name, score = recognize_face(input_embedding, threshold=0.65)
        print(f"Face recognition result - Student Number: {student_number}, Score: {score}")
        
        if student_number:
            try:
                user_profile = UserProfile.objects.get(student_number=student_number)
                print(f"Found user profile for student: {student_number}")
                
                # Check if already voted in the current school year
                active_settings = ElectionSettings.objects.filter(is_active=True).first()
                already_voted = False
                
                if active_settings:
                    print(f"Checking votes for school year: {active_settings.school_year}")
                    # Check if user has any votes in the current school year
                    vote_count = Vote.objects.filter(
                        user_profile=user_profile,
                        school_year=active_settings.school_year
                    ).count()
                    already_voted = vote_count > 0
                    print(f"Found {vote_count} votes for this user. Already voted: {already_voted}")
                else:
                    print("No active election settings found")
                
                response_data = {
                    'success': True,
                    'message': f'Welcome {user_profile.student_name}!',
                    'user_profile_id': user_profile.id,
                    'student_number': user_profile.student_number,
                    'college': user_profile.college,
                    'department': user_profile.course,
                    'year_level': user_profile.year_level,
                    'score': score,
                    'already_voted': already_voted
                }
                print(f"Sending response: {response_data}")
                return JsonResponse(response_data)
                
            except UserProfile.DoesNotExist:
                print(f"No user profile found for student number: {student_number}")
                return JsonResponse({'success': False, 'message': 'User profile not found'}, status=404)
        else:
            print("Face not recognized")
            return JsonResponse({'success': False, 'message': 'Face not recognized'}, status=400)

    except Exception as e:
        print(f"Error in verify_face: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return JsonResponse({'success': False, 'message': f'Error during verification: {str(e)}'}, status=500)

@require_GET
@csrf_exempt
def check_already_voted(request):
    """
    API endpoint to check if a user has already voted in the current active school year.
    Expects ?user_profile_id=... as a GET parameter.
    """
    user_profile_id = request.GET.get('user_profile_id')
    if not user_profile_id:
        return JsonResponse({'already_voted': False, 'error': 'user_profile_id required'}, status=400)
    try:
        user_profile = UserProfile.objects.get(id=user_profile_id)
    except UserProfile.DoesNotExist:
        return JsonResponse({'already_voted': False, 'error': 'User profile not found'}, status=404)
    active_settings = ElectionSettings.objects.filter(is_active=True).first()
    already_voted = False
    if active_settings:
        vote_count = Vote.objects.filter(
            user_profile=user_profile,
            school_year=active_settings.school_year
        ).count()
        already_voted = vote_count > 0
    return JsonResponse({'already_voted': already_voted})

def supervisor_login(request):
    """Login for chairman, admin, or committee to unlock voting page."""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user and (user.is_superuser or getattr(user, 'chairman_profile', None) or getattr(user, 'committee_profile', None)):
            login(request, user)
            request.session['facility_supervisor_authenticated'] = True
            return redirect('user:vote')
        else:
            messages.error(request, 'Only chairman, admin, or committee can unlock voting.')
    return render(request, 'login.html')

def supervisor_logout(request):
    """Logout supervisor and lock voting page."""
    request.session.pop('facility_supervisor_authenticated', None)
    auth_logout(request)
    return redirect('user:supervisor_login')
