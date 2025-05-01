from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from .models import UserProfile, Candidate, Vote, VotingPhase, FaceData
from admin_panel.models import ElectionSettings
from django.core.exceptions import ValidationError
from .face_utils import process_webcam_image, preprocess_face_image, get_face_embedding, recognize_face
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
        # Filter candidates for this college
        college_local_candidates = all_candidates.filter(
            position__in=local_positions,
            user_profile__college=college_code
        )
        
        # Group by position within each college
        for candidate in college_local_candidates:
            position_name = position_names.get(candidate.position, candidate.position)
            if position_name not in college_candidates[college_name]:
                college_candidates[college_name][position_name] = []
            college_candidates[college_name][position_name].append(candidate)
    
    # Sort candidates within each position
    for position in national_candidates:
        national_candidates[position].sort(key=lambda x: x.user_profile.student_name)
    
    for college in college_candidates:
        for position in college_candidates[college]:
            college_candidates[college][position].sort(key=lambda x: x.user_profile.student_name)
    
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
        candidates = Candidate.objects.filter(position=pos_code)
        if candidates.exists():
            national_positions[pos_name] = json.dumps([
                {
                    'id': c.id,
                    'user_name': c.user_profile.student_name,
                    'college': c.user_profile.college,
                    'department': c.user_profile.course,
                    'year_level': c.user_profile.year_level,
                    'photo': c.photo.url if c.photo and c.photo.url else None,
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
                    'user_name': c.user_profile.student_name,
                    'college': c.user_profile.college,
                    'department': c.user_profile.course,
                    'year_level': c.user_profile.year_level,
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
    try:
        data = json.loads(request.body)
        face_data = data.get('face_data')
        if not face_data:
            return JsonResponse({'success': False, 'message': 'No face data provided'}, status=400)

        # Decode and save the webcam image
        temp_image_path = process_webcam_image(face_data)
        if not temp_image_path:
            return JsonResponse({'success': False, 'message': 'Failed to process captured image'}, status=400)

        # Extract embedding from the image
        input_embedding = get_face_embedding(temp_image_path)
        if input_embedding is None:
            return JsonResponse({'success': False, 'message': 'Failed to extract face embedding'}, status=400)

        # Compare to all UserProfile embeddings
        student_number, student_name, score = recognize_face(input_embedding, threshold=0.5)
        if student_number:
            user_profile = UserProfile.objects.get(student_number=student_number)
            return JsonResponse({
                'success': True,
                'message': f'Welcome {user_profile.student_name}!',
                'user_profile_id': user_profile.id,
                'student_number': user_profile.student_number,
                'college': user_profile.college,
                'department': user_profile.course,
                'year_level': user_profile.year_level,
                'score': score
            })
        else:
            return JsonResponse({'success': False, 'message': 'Face not recognized'}, status=400)

    except Exception as e:
        print(f"Error in verify_face: {str(e)}")
        return JsonResponse({'success': False, 'message': f'Error during verification: {str(e)}'}, status=500)

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
        return user_profile.college == candidate.user_profile.college
    
    # Department positions - users from same department, college, and year level
    return (user_profile.college == candidate.user_profile.college and
            user_profile.course == candidate.user_profile.course and
            user_profile.year_level == candidate.user_profile.year_level)

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
