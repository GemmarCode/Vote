from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required, user_passes_test
from user.models import Candidate, Vote, UserProfile, VerificationCode
from .decorators import superuser_required
from django.contrib import messages
from django.db.models import Count, Q, F, ExpressionWrapper, FloatField
from django.utils import timezone
from .models import AdminActivity
from django.contrib.auth import get_user_model, authenticate, login, logout, update_session_auth_hash
from django.db.models.functions import TruncHour
from datetime import timedelta, datetime
import json
from .models import ElectionSettings, CommitteeAccount
import pandas as pd
from django.db import transaction
import os
import cv2
import numpy as np
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings as django_settings
import zipfile
import tempfile
import shutil
from pathlib import Path
from django.db.utils import IntegrityError
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
from django.template.loader import render_to_string
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from django.views.decorators.csrf import csrf_exempt
import random
import string
from django.urls import reverse
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch
from torchvision import transforms

model = InceptionResnetV1(pretrained='vggface2').eval()

def preprocess_face_image(img):
    # Resize to 160x160 as required by facenet-pytorch
    return cv2.resize(img, (160, 160))

def get_face_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transforms.Resize((160,160))(img)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor).numpy()[0]
    return embedding.tolist()

# Create your views here.

def is_admin(user):
    return user.is_superuser

def is_committee(user):
    return CommitteeAccount.objects.filter(user=user).exists()

def committee_required(function):
    actual_decorator = user_passes_test(is_committee)
    if function:
        return actual_decorator(function)
    return actual_decorator

def get_current_school_year():
    from .models import ElectionSettings
    active_settings = ElectionSettings.objects.filter(is_active=True).first()
    return active_settings.school_year if active_settings else None

@login_required
@user_passes_test(is_admin)
def dashboard(request):
    # Get active election settings
    active_settings = ElectionSettings.objects.filter(is_active=True).first()
    
    if not active_settings:
        context = {
            'no_active_year': True,
            'current_school_year': None
        }
        return render(request, 'dashboard.html', context)
    
    # Basic Statistics - Filter by active school year
    total_users = UserProfile.objects.filter(school_year=active_settings.school_year).count()
    total_candidates = Candidate.objects.filter(school_year=active_settings.school_year).count()
    total_votes = Vote.objects.filter(school_year=active_settings.school_year).count()
    voter_turnout = round((total_votes / total_users * 100) if total_users > 0 else 0, 1)
    
    # Initialize vote trends and national candidates
    vote_trends = {}
    national_candidates = Candidate.objects.filter(
        position='National',
        school_year=active_settings.school_year
    )

    # Vote Trends for National Candidates (last 24 hours)
    if active_settings.is_voting_open:
        twenty_four_hours_ago = timezone.now() - timedelta(hours=24)

        for candidate in national_candidates:
            votes = Vote.objects.filter(
                candidate=candidate,
                created_at__gte=twenty_four_hours_ago,
                school_year=active_settings.school_year
            ).annotate(
                hour=TruncHour('created_at')
            ).values('hour').annotate(
                count=Count('id')
            ).order_by('hour')
            
            vote_trends[candidate.user_profile.student_name] = [
                {'timestamp': vote['hour'].isoformat(), 'count': vote['count']}
                for vote in votes
            ]
    
    # School Year Statistics
    school_years = ElectionSettings.objects.all().order_by('-school_year')
    school_year_stats = {}
    
    for year in school_years:
        year_users = UserProfile.objects.filter(school_year=year.school_year).count()
        year_candidates = Candidate.objects.filter(school_year=year.school_year).count()
        year_votes = Vote.objects.filter(school_year=year.school_year).count()
        year_turnout = round((year_votes / year_users * 100) if year_users > 0 else 0, 1)
        
        school_year_stats[year.school_year] = {
            'total_students': year_users,
            'total_candidates': year_candidates,
            'total_votes': year_votes,
            'voter_turnout': year_turnout
        }
    
    # Election Status
    election_status = {
        'current_phase': 'Voting Period' if active_settings and active_settings.is_voting_open else 'Setup Period',
        'next_milestone': 'Voting Period Ends' if active_settings and active_settings.is_voting_open else 'Voting Period Starts',
        'time_to_next': None,
        'is_voting_open': active_settings.is_voting_open if active_settings else False
    }
    
    # Calculate time to next milestone only if we have valid dates
    if active_settings:
        if active_settings.is_voting_open and active_settings.voting_time_end:
            voting_end = timezone.make_aware(timezone.datetime.combine(
                active_settings.voting_date,
                active_settings.voting_time_end
            ))
            time_remaining = voting_end - timezone.now()
            if time_remaining.total_seconds() > 0:
                election_status['time_to_next'] = {
                    'total_seconds': int(time_remaining.total_seconds())
                }
        elif not active_settings.is_voting_open and active_settings.voting_time_start:
            voting_start = timezone.make_aware(timezone.datetime.combine(
                active_settings.voting_date,
                active_settings.voting_time_start
            ))
            time_remaining = voting_start - timezone.now()
            if time_remaining.total_seconds() > 0:
                election_status['time_to_next'] = {
                    'total_seconds': int(time_remaining.total_seconds())
                }
    
    # Predictive Analytics
    if active_settings and active_settings.is_voting_open:
        # Calculate average votes per hour
        votes_per_hour = Vote.objects.filter(
            created_at__gte=timezone.now() - timedelta(hours=24)
        ).count() / 24
        
        # Predict additional votes until end
        if active_settings.voting_time_end:
            voting_end = timezone.make_aware(timezone.datetime.combine(
                active_settings.voting_date,
                active_settings.voting_time_end
            ))
            hours_remaining = (voting_end - timezone.now()).total_seconds() / 3600
            predicted_votes = int(votes_per_hour * hours_remaining)
        else:
            predicted_votes = 0
        
        # Predict final turnout
        current_turnout = (total_votes / total_users * 100) if total_users > 0 else 0
        predicted_turnout = min(100, round(current_turnout + (current_turnout * 0.2), 1))  # Assume 20% increase
    else:
        predicted_votes = 0
        predicted_turnout = 0
    
    # Geographic Distribution (simulated data)
    regions = ['North', 'South', 'East', 'West', 'Central']
    voter_counts = [random.randint(100, 500) for _ in range(5)]
    turnout_percentages = [random.randint(60, 95) for _ in range(5)]
    
    geographic_data = {
        'regions': regions,
        'voter_counts': voter_counts,
        'turnout_percentages': turnout_percentages
    }
    
    # Voting Activity by Hour
    voting_by_hour = [0] * 24
    if active_settings and active_settings.is_voting_open:
        hourly_votes = Vote.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=1)
        ).annotate(
            hour=TruncHour('created_at')
        ).values('hour').annotate(
            count=Count('id')
        ).order_by('hour')
        
        for vote in hourly_votes:
            hour = vote['hour'].hour
            voting_by_hour[hour] = vote['count']
    
    # System Health (simulated data)
    system_health = {
        'server_status': 'Healthy',
        'database_status': 'Healthy',
        'face_verification_success_rate': 95.5,
        'error_rate': 0.5,
        'average_response_time': 150
    }
    
    # Candidate Performance
    candidate_performance = []
    if active_settings and active_settings.is_voting_open:
        # Get candidates with vote counts
        candidates = Candidate.objects.annotate(
            vote_count=Count('vote')
        ).order_by('-vote_count')[:5]
        
        # Calculate percentages manually
        for candidate in candidates:
            percentage = 0
            if total_votes > 0:
                percentage = (candidate.vote_count / total_votes) * 100
                
            candidate_performance.append({
                'name': candidate.user_profile.student_name,
                'position': candidate.position,
                'votes': candidate.vote_count,
                'percentage': round(percentage, 1)
            })
    
    # Anomalies and Warnings
    anomalies = []
    if active_settings and active_settings.is_voting_open:
        # Check for unusual voting patterns
        recent_votes = Vote.objects.filter(
            created_at__gte=timezone.now() - timedelta(hours=1)
        ).count()
        
        if recent_votes > 100:  # Threshold for high activity
            anomalies.append({
                'type': 'High Voting Activity',
                'description': f'Unusually high number of votes ({recent_votes}) in the last hour',
                'severity': 'Medium'
            })
        
        # Check for system performance
        if system_health['error_rate'] > 1:
            anomalies.append({
                'type': 'System Performance',
                'description': 'Error rate above normal threshold',
                'severity': 'High'
            })
    
    # Device Usage (simulated data)
    device_usage = {
        'mobile': 65,
        'desktop': 30,
        'tablet': 5
    }
    
    # Accessibility Metrics (simulated data)
    accessibility_metrics = {
        'face_verification_success_rate': 95.5,
        'average_verification_time': 2.5,
        'failed_attempts': 45
    }
    
    # Recent Activity
    recent_activities = []
    if active_settings:
        recent_votes = Vote.objects.select_related(
            'user_profile',
            'candidate__user_profile'
        ).order_by('-created_at')[:10]
        
        for vote in recent_votes:
            recent_activities.append({
                'created_at': vote.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'action': 'Vote Cast',
                'details': f"{vote.user_profile.student_name} voted for {vote.candidate.user_profile.student_name}"
            })
    
    context = {
        'total_users': total_users,
        'total_candidates': total_candidates,
        'total_votes': total_votes,
        'voter_turnout': voter_turnout,
        'vote_trends': json.dumps(vote_trends),
        'school_years': school_years,
        'school_year_stats': school_year_stats,
        'election_status': election_status,
        'predicted_votes': predicted_votes,
        'predicted_turnout': predicted_turnout,
        'geographic_data': json.dumps(geographic_data),
        'voting_by_hour': json.dumps(voting_by_hour),
        'system_health': system_health,
        'candidate_performance': candidate_performance,
        'anomalies': anomalies,
        'device_usage': device_usage,
        'accessibility_metrics': accessibility_metrics,
        'recent_activities': recent_activities,
        'current_school_year': get_current_school_year()
    }
    
    return render(request, 'dashboard.html', context)

@login_required
@user_passes_test(is_admin)
def manage_users(request):
    active_settings = ElectionSettings.objects.filter(is_active=True).first()
    if not active_settings:
        context = {
            'all_users': [],
            'current_school_year': None,
            'no_active_year': True,
            'no_active_year_message': 'No active school year yet.'
        }
        return render(request, 'manage_users.html', context)
    current_school_year = active_settings.school_year
    all_users = UserProfile.objects.filter(school_year=current_school_year).order_by('-id')
    context = {
        'all_users': all_users,
        'current_school_year': current_school_year
    }
    return render(request, 'manage_users.html', context)

def process_face_photo(photo_file, student_number):
    """Process uploaded photo and convert it to face data"""
    try:
        # Save the uploaded file temporarily
        temp_path = default_storage.save(f'temp/{photo_file.name}', ContentFile(photo_file.read()))
        temp_file = os.path.join(django_settings.MEDIA_ROOT, temp_path)
        
        # Read the image with high quality
        img = cv2.imread(temp_file, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not read image file")
        
        # Create face_data directory if it doesn't exist
        face_data_dir = os.path.join(django_settings.MEDIA_ROOT, 'face_data')
        os.makedirs(face_data_dir, exist_ok=True)
        
        # Use the standardized preprocessing function
        preprocessed_img = preprocess_face_image(img)
        if preprocessed_img is None:
            raise ValueError("Failed to preprocess face image")
            
        # Convert back to BGR (color) for saving if it's grayscale
        if len(preprocessed_img.shape) == 2:  # If grayscale
            preprocessed_color = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
        else:
            preprocessed_color = preprocessed_img
        
        # Save processed image to face_data directory
        face_path = os.path.join('face_data', f'{student_number}.jpg')
        cv2.imwrite(os.path.join(django_settings.MEDIA_ROOT, face_path), preprocessed_color, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Store the image in binary format for the face_data field
        _, buffer = cv2.imencode('.jpg', preprocessed_color)
        face_bytes = buffer.tobytes()
        
        # Clean up temp file
        os.remove(temp_file)
        
        # Save a debug copy to help with troubleshooting
        try:
            debug_dir = os.path.join(django_settings.MEDIA_ROOT, 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f'register_{student_number}.jpg'), 
                        preprocessed_color, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as e:
            print(f"Warning: Could not save debug image: {str(e)}")
        
        return face_bytes
    except Exception as e:
        raise ValueError(f"Error processing photo: {str(e)}")

def process_photo_from_path(photo_path, student_number):
    """Process photo from a file path and convert it to face data"""
    try:
        # Read the image with high quality
        img = cv2.imread(photo_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not read image file")
        
        # Create face_data directory if it doesn't exist
        face_data_dir = os.path.join(django_settings.MEDIA_ROOT, 'face_data')
        os.makedirs(face_data_dir, exist_ok=True)
        
        # Use the standardized preprocessing function
        preprocessed_img = preprocess_face_image(img)
        if preprocessed_img is None:
            raise ValueError("Failed to preprocess face image")
            
        # Convert back to BGR (color) for saving if it's grayscale
        if len(preprocessed_img.shape) == 2:  # If grayscale
            preprocessed_color = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
        else:
            preprocessed_color = preprocessed_img
        
        # Save processed image to face_data directory
        face_path = os.path.join('face_data', f'{student_number}.jpg')
        cv2.imwrite(os.path.join(django_settings.MEDIA_ROOT, face_path), preprocessed_color, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Store the image in binary format for the face_data field
        _, buffer = cv2.imencode('.jpg', preprocessed_color)
        face_bytes = buffer.tobytes()
        
        # Save a debug copy to help with troubleshooting
        try:
            debug_dir = os.path.join(django_settings.MEDIA_ROOT, 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f'import_{student_number}.jpg'), 
                        preprocessed_color, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as e:
            print(f"Warning: Could not save debug image: {str(e)}")
        
        return face_bytes
    except Exception as e:
        raise ValueError(f"Error processing photo: {str(e)}")

@login_required
@user_passes_test(is_admin)
def register_user(request):
    if request.method == 'POST':
        try:
            # --- Get current active school year ---
            try:
                active_settings = ElectionSettings.objects.get(is_active=True)
                current_school_year = active_settings.school_year
            except ElectionSettings.DoesNotExist:
                return JsonResponse({'error': "No active school year is set. Please activate a school year in Election Settings first."}, status=400)
            except ElectionSettings.MultipleObjectsReturned:
                return JsonResponse({'error': "Multiple active school years found. Please ensure only one school year is active."}, status=400)
            # --------------------------------------

            # Get form data from POST
            student_number = request.POST.get('student_number')
            student_name = request.POST.get('student_name')
            sex = request.POST.get('sex')
            year_level = request.POST.get('year_level')
            course = request.POST.get('course')
            college = request.POST.get('college')
            
            # Simple validation
            if not all([student_number, student_name, sex, college, year_level, course]):
                return JsonResponse({'error': 'All fields are required'}, status=400)
            
            # Check if UserProfile already exists for this student number
            if UserProfile.objects.filter(student_number=student_number).exists():
                return JsonResponse({'error': 'Student number already exists in User Profiles.'}, status=400)
            
            # Normalize year_level to string if it's an integer
            if isinstance(year_level, int):
                year_level = str(year_level)
            
            # Create UserProfile ONLY, assigning the active school year
            UserProfile.objects.create(
                student_number=student_number,
                student_name=student_name,
                sex=sex,
                year_level=year_level,
                course=course,
                college=college,
                school_year=current_school_year # Assign active school year
            )
            
            # Log activity
            AdminActivity.objects.create(
                admin_user=request.user,
                action='REGISTER_USER',
                details=f"Registered new student: {student_name} ({student_number})"
            )
            
            return JsonResponse({'success': True, 'message': f'Student {student_name} added successfully for school year {current_school_year}.'})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    # If GET request, redirect to manage_users view
    return redirect('admin_panel:manage_users')

@login_required
@user_passes_test(is_admin)
def manage_candidates(request):
    active_settings = ElectionSettings.objects.filter(is_active=True).first()
    if not active_settings:
        context = {
            'candidates': [],
            'national_positions': Candidate.NATIONAL_POSITIONS,
            'college_positions': Candidate.COLLEGE_POSITIONS,
            'local_positions': Candidate.LOCAL_POSITIONS,
            'users_json': '[]',
            'current_school_year': None,
            'no_active_year': True,
            'no_active_year_message': 'No active school year yet.'
        }
        return render(request, 'manage_candidates.html', context)
    current_school_year = active_settings.school_year
    candidates = Candidate.objects.filter(school_year=current_school_year).select_related('user_profile')
    if request.method == 'POST':
        action = request.POST.get('action')
        
        # Handle deleting a candidate
        if action == 'delete':
            candidate_id = request.POST.get('candidate_id')
            try:
                candidate = Candidate.objects.get(id=candidate_id)
                candidate.delete()
                messages.success(request, 'Candidate deleted successfully.')
            except Candidate.DoesNotExist:
                messages.error(request, 'Candidate not found.')
            return redirect('admin_panel:manage_candidates')
        
        # Handle editing a candidate
        elif action == 'edit':
            candidate_id = request.POST.get('candidate_id')
            try:
                candidate = Candidate.objects.get(id=candidate_id)
                position = request.POST.get('position')
                platform = request.POST.get('platform', '')
                achievements = request.POST.get('achievements', '')
                photo = request.FILES.get('photo')

                if position:
                    candidate.position = position
                candidate.platform = platform
                candidate.achievements = achievements
                if photo:
                    candidate.photo = photo
                candidate.save()
                # Log activity
                AdminActivity.objects.create(
                    admin_user=request.user,
                    action='EDIT_CANDIDATE',
                    details=f"Edited candidate: {candidate.user_profile.student_name} ({candidate.user_profile.student_number}) for position {position}",
                )
                messages.success(request, f'Candidate with student number {candidate.user_profile.student_number} has been updated successfully.')
                return redirect('admin_panel:manage_candidates')
            except Candidate.DoesNotExist:
                messages.error(request, 'Candidate not found.')
            return redirect('admin_panel:manage_candidates')
        
        # Handle adding a new candidate
        else:
            try:
                # Get essential form data
                user_profile_id = request.POST.get('user_profile_id')
                position = request.POST.get('position')
                platform = request.POST.get('platform', '')
                achievements = request.POST.get('achievements', '')
                photo = request.FILES.get('photo')
                
                # Validate data
                if not user_profile_id:
                    messages.error(request, 'No user selected. Please search and select a user.')
                    return redirect('admin_panel:manage_candidates')
                
                if not position:
                    messages.error(request, 'Position is required.')
                    return redirect('admin_panel:manage_candidates')
                
                # Get the user profile
                try:
                    user_profile = UserProfile.objects.get(id=user_profile_id)
                except UserProfile.DoesNotExist:
                    messages.error(request, 'Selected user not found.')
                    return redirect('admin_panel:manage_candidates')
                
                # Check if user is already a candidate
                if Candidate.objects.filter(user_profile=user_profile).exists():
                    messages.error(request, 'This student is already registered as a candidate.')
                    return redirect('admin_panel:manage_candidates')
                
                # Create candidate with essential fields
                try:
                    candidate = Candidate(
                        user_profile=user_profile,
                        position=position,
                        platform=platform,
                        achievements=achievements,
                        school_year=current_school_year  # Automatically set active school year
                    )
                    # Save photo if provided
                    if photo:
                        candidate.photo = photo
                    candidate.save()
                    # Log activity
                    AdminActivity.objects.create(
                        admin_user=request.user,
                        action='ADD_CANDIDATE',
                        details=f"Added candidate: {user_profile.student_name} ({user_profile.student_number}) for position {position}"
                    )
                    messages.success(request, f'Candidate with student number {user_profile.student_number} has been added successfully.')
                    return redirect('admin_panel:manage_candidates')
                    
                except IntegrityError as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    print(f"Integrity Error adding candidate: {str(e)}")
                    print(f"Detailed error traceback: {error_trace}")
                    
                    messages.error(request, f'Database error: This student is already registered as a candidate. Each student can only be a candidate once.')
                    return redirect('admin_panel:manage_candidates')
                
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"Error adding candidate: {str(e)}")
                print(f"Detailed error traceback: {error_trace}")
                
                messages.error(request, f'Error adding candidate: {str(e)}')
                return redirect('admin_panel:manage_candidates')
        return redirect('admin_panel:manage_candidates')

    # Get position lists
    national_positions = Candidate.NATIONAL_POSITIONS
    college_positions = Candidate.COLLEGE_POSITIONS
    local_positions = Candidate.LOCAL_POSITIONS
    
    # Get all users who are not already candidates for the active school year
    existing_candidate_users = Candidate.objects.filter(school_year=current_school_year).values_list('user_profile_id', flat=True)
    available_users = UserProfile.objects.filter(school_year=current_school_year).exclude(id__in=existing_candidate_users)
    users_json = json.dumps([{
        'id': user.id,
        'student_number': user.student_number,
        'name': user.student_name
    } for user in available_users])
    
    context = {
        'candidates': candidates,
        'national_positions': national_positions,
        'college_positions': college_positions,
        'local_positions': local_positions,
        'users_json': users_json,
        'current_school_year': current_school_year
    }
    
    return render(request, 'manage_candidates.html', context)

@login_required
@user_passes_test(is_admin)
def manage_elections(request):
    # Get all school years
    all_settings = ElectionSettings.objects.all().order_by('-school_year')
    active_settings = ElectionSettings.objects.filter(is_active=True).first()
    
    if request.method == 'POST':
        action = request.POST.get('action')
        try:
            if action == 'create_school_year':
                # Create new school year
                school_year = request.POST.get('school_year')
                if not school_year:
                    raise ValueError("School year is required")
                
                # Validate school year format
                try:
                    start_year, end_year = map(int, school_year.split('-'))
                    if end_year != start_year + 1:
                        raise ValueError("End year must be start year + 1")
                except ValueError:
                    raise ValueError("School year must be in YYYY-YYYY format")
                
                # Check if school year already exists
                if ElectionSettings.objects.filter(school_year=school_year).exists():
                    raise ValueError("This school year already exists")
                
                # Create new settings for the school year
                new_settings = ElectionSettings.objects.create(
                    school_year=school_year,
                    is_active=True  # This will automatically deactivate other years
                )
                # Inactivate all committee accounts for the new school year
                CommitteeAccount.objects.all().update(is_active=False)
                # Log activity
                AdminActivity.objects.create(
                    admin_user=request.user,
                    action='CREATE_SCHOOL_YEAR',
                    details=f'Created new school year: {school_year}'
                )
                
                messages.success(request, f'New school year {school_year} created successfully.')
                return redirect('admin_panel:manage_elections')
            
            elif action == 'update_settings':
                # Get the active settings
                settings = ElectionSettings.objects.get(is_active=True)
                
                # Parse datetime strings from the form
                voting_date = request.POST.get('voting_date')
                voting_time_start = request.POST.get('voting_time_start')
                voting_time_end = request.POST.get('voting_time_end')
                
                # Helper to parse time string
                def parse_time_string(time_str):
                    try:
                        return datetime.strptime(time_str, '%H:%M').time()
                    except ValueError:
                        return datetime.strptime(time_str, '%H:%M:%S').time()
                
                # Update settings with parsed datetime objects
                if voting_date:
                    settings.voting_date = timezone.make_aware(datetime.strptime(voting_date, '%Y-%m-%d')).date()
                if voting_time_start:
                    settings.voting_time_start = parse_time_string(voting_time_start)
                if voting_time_end:
                    settings.voting_time_end = parse_time_string(voting_time_end)
                
                settings.save()
                # Log activity
                AdminActivity.objects.create(
                    admin_user=request.user,
                    action='UPDATE_ELECTION_TIMELINE',
                    details=f'Updated election timeline for school year {settings.school_year}: Date={settings.voting_date}, Start={settings.voting_time_start}, End={settings.voting_time_end}'
                )
                messages.success(request, 'Election settings updated successfully.')
                return redirect('admin_panel:manage_elections')
            
            elif action == 'activate_school_year':
                school_year_id = request.POST.get('school_year_id')
                if not school_year_id:
                    raise ValueError("School year ID is required")
                
                settings = ElectionSettings.objects.get(id=school_year_id)
                settings.is_active = True
                settings.save()  # This will automatically deactivate other years
                
                messages.success(request, f'School year {settings.school_year} activated successfully.')
                return redirect('admin_panel:manage_elections')
            
        except Exception as e:
            messages.error(request, f'Error updating settings: {str(e)}')
            return redirect('admin_panel:manage_elections')
    
    # Get statistics for each school year
    school_year_stats = {}
    for setting in all_settings:
        stats = {
            'total_students': UserProfile.objects.filter(school_year=setting.school_year).count(),
            'total_candidates': Candidate.objects.filter(school_year=setting.school_year).count(),
            'total_votes': Vote.objects.filter(school_year=setting.school_year).count(),
        }
        school_year_stats[setting.school_year] = stats
    
    context = {
        'all_settings': all_settings,
        'active_settings': active_settings,
        'school_year_stats': school_year_stats,
        'current_school_year': get_current_school_year()
    }
    return render(request, 'manage_elections.html', context)

@login_required
@user_passes_test(is_admin)
def results(request):
    # Get active election settings
    active_settings = ElectionSettings.objects.filter(is_active=True).first()
    
    if not active_settings:
        context = {
            'no_active_year': True,
            'current_school_year': None
        }
        return render(request, 'results.html', context)
    
    # Get election results data - Filter by active school year
    candidates = Candidate.objects.filter(
        school_year=active_settings.school_year
    ).select_related('user_profile')
    
    results_data = []
    for candidate in candidates:
        votes_count = Vote.objects.filter(
            candidate=candidate,
            school_year=active_settings.school_year
        ).count()
        results_data.append({
            'candidate': candidate,
            'votes': votes_count
        })
    
    context = {
        'results_data': results_data,
        'current_school_year': active_settings.school_year
    }
    return render(request, 'results.html', context)

@login_required
@user_passes_test(is_admin)
def import_users(request):
    if request.method == 'POST' and request.FILES.get('excel_file'):
        file = request.FILES['excel_file']
        
        # Column name mappings with possible variations
        COLUMN_MAPPINGS = {
            'student_number': [
                'student_number', 'studentnumber', 'student no', 'student no.', 
                'student_no', 'id_number', 'id number', 'id'
            ],
            'student_name': [
                'student_name', 'studentname', 'name', 'full name', 'fullname',
                'complete name', 'completename'
            ],
            'sex': [
                'sex', 'gender', 'sex/gender'
            ],
            'year_level': [
                'year_level', 'yearlevel', 'year', 'level', 'grade_level',
                'grade level', 'year standing'
            ],
            'course': [
                'course', 'program', 'degree', 'degree program', 'course_program'
            ],
            'college': [
                'college', 'school', 'department', 'dept', 'faculty'
            ]
        }

        try:
            # Read file based on extension
            file_ext = os.path.splitext(file.name)[1].lower()
            if file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file)

            # --- Get current active school year ---
            try:
                active_settings = ElectionSettings.objects.get(is_active=True)
                current_school_year = active_settings.school_year
            except ElectionSettings.DoesNotExist:
                messages.error(request, "No active school year is set. Please activate a school year in Election Settings before importing users.")
                return redirect('admin_panel:manage_users')
            except ElectionSettings.MultipleObjectsReturned:
                messages.error(request, "Multiple active school years found. Please ensure only one school year is active in Election Settings.")
                return redirect('admin_panel:manage_users')
            # --------------------------------------

            # Clean column names (remove whitespace, lowercase, etc)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

            # Create clean_row dictionary to store matched columns
            column_matches = {}

            # Find matching columns for each required field
            for required_col, possible_names in COLUMN_MAPPINGS.items():
                # Convert possible names to lowercase and clean format
                possible_names = [name.lower().strip().replace(' ', '_') for name in possible_names]
                
                # Find the first matching column
                matched_col = None
                for col in df.columns:
                    if col in possible_names:
                        matched_col = col
                        break
                
                if matched_col:
                    column_matches[required_col] = matched_col
                else:
                    messages.error(request, f"Could not find a column matching '{required_col}'. "
                                         f"Possible names are: {', '.join(possible_names)}")
                    return redirect('admin_panel:manage_users')
            
            # Process each row with matched columns
            success_count = 0
            error_count = 0
            error_logs = []
            
            for index, row in df.iterrows():
                try:
                    # Clean and get data using matched columns
                    student_number = str(row[column_matches['student_number']]).strip()
                    student_name = str(row[column_matches['student_name']]).strip()
                    sex = str(row[column_matches['sex']]).strip().upper()
                    year_level = str(row[column_matches['year_level']]).strip()
                    course = str(row[column_matches['course']]).strip()
                    college = str(row[column_matches['college']]).strip().upper()

                    # Validate student number
                    if not student_number or pd.isna(student_number):
                        error_logs.append(f"Row {index + 2}: Empty student number")
                        error_count += 1
                        continue
                    
                    # Check for existing student number in UserProfile
                    if UserProfile.objects.filter(student_number=student_number).exists():
                        error_logs.append(f"Row {index + 2}: Student number {student_number} already exists in UserProfile")
                        error_count += 1
                        continue
                    
                    # Validate sex (handle common variations)
                    sex_mapping = {
                        'M': ['M', 'MALE', 'MAN'],
                        'F': ['F', 'FEMALE', 'WOMAN']
                    }
                    sex_normalized = None
                    for key, values in sex_mapping.items():
                        if sex in values:
                            sex_normalized = key
                            break
                    
                    if not sex_normalized:
                        error_logs.append(f"Row {index + 2}: Invalid sex value '{sex}' for student {student_number}")
                        error_count += 1
                        continue
                    
                    # Validate year level
                    if year_level not in ['1', '2', '3', '4', '5']:
                        error_logs.append(f"Row {index + 2}: Invalid year level '{year_level}' for student {student_number}")
                        error_count += 1
                        continue
                    
                    # Validate and normalize college code
                    college_mapping = {
                        'CAS': ['CAS', 'ARTS AND SCIENCES', 'COLLEGE OF ARTS AND SCIENCES'],
                        'CAF': ['CAF', 'AGRICULTURE AND FORESTRY', 'COLLEGE OF AGRICULTURE AND FORESTRY'],
                        'CCJE': ['CCJE', 'CRIMINAL JUSTICE', 'COLLEGE OF CRIMINAL JUSTICE EDUCATION'],
                        'CBA': ['CBA', 'BUSINESS', 'COLLEGE OF BUSINESS ADMINISTRATION'],
                        'CTED': ['CTED', 'EDUCATION', 'COLLEGE OF TEACHER EDUCATION'],
                        'CIT': ['CIT', 'TECHNOLOGY', 'COLLEGE OF INDUSTRIAL TECHNOLOGY']
                    }
                    
                    college_normalized = None
                    for key, values in college_mapping.items():
                        if college in values:
                            college_normalized = key
                            break

                    if not college_normalized:
                        error_logs.append(f"Row {index + 2}: Invalid college code '{college}' for student {student_number}")
                        error_count += 1
                        continue

                    # Create new UserProfile ONLY, assigning the active school year
                    UserProfile.objects.create(
                        student_number=student_number,
                        student_name=student_name,
                        sex=sex_normalized,
                        year_level=year_level,
                        course=course,
                        college=college_normalized,
                        school_year=current_school_year  # Assign active school year
                    )
                    success_count += 1

                except Exception as e:
                    error_logs.append(f"Row {index + 2}: Error processing record - {str(e)}")
                    error_count += 1
                    continue

            # Log activity
            AdminActivity.objects.create(
                admin_user=request.user,
                action='IMPORT_STUDENTS',
                details=f"Imported {success_count} student records from Excel file. {error_count} records failed to import."
            )

            # Show import results
            if success_count > 0:
                messages.success(request, f"Successfully imported {success_count} student records for school year {current_school_year}.")
            if error_count > 0:
                messages.warning(request, f"Failed to import {error_count} records. Check the error log below:")
                for error in error_logs:
                    messages.warning(request, error)
            
        except Exception as e:
            messages.error(request, f"Error processing file: {str(e)}")
        
        return redirect('admin_panel:manage_users')
    
    messages.error(request, "No file was uploaded.")
    return redirect('admin_panel:manage_users')

@login_required
@user_passes_test(is_admin)
def import_photos(request):
    """Import student photos from ZIP file"""
    if request.method == 'POST' and request.FILES.get('photos_zip'):
        zip_file = request.FILES['photos_zip']
        
        # Create directory if it doesn't exist
        face_data_dir = os.path.join(django_settings.MEDIA_ROOT, 'face_data')
        os.makedirs(face_data_dir, exist_ok=True)
        
        def generate_progress():
            success_count = 0
            error_count = 0
            errors = []
            
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    total_files = len(zip_ref.namelist())
                    processed_files = 0
                    
                    for filename in zip_ref.namelist():
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            try:
                                # Extract student number from filename
                                student_number = os.path.splitext(os.path.basename(filename))[0]
                                
                                # Extract and process the image
                                with zip_ref.open(filename) as file:
                                    img_data = file.read()
                                    img_array = np.frombuffer(img_data, np.uint8)
                                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                                    
                                    if img is None:
                                        errors.append(f"Could not read image for {filename}")
                                        error_count += 1
                                        continue
                                        
                                    # Process the image
                                    try:
                                        # Use the standardized preprocessing function
                                        preprocessed_img = preprocess_face_image(img)
                                        if preprocessed_img is None:
                                            errors.append(f"Failed to preprocess face image for {filename}")
                                            error_count += 1
                                            continue
                                            
                                        # Convert back to BGR (color) for saving if it's grayscale
                                        if len(preprocessed_img.shape) == 2:  # If grayscale
                                            preprocessed_color = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
                                        else:
                                            preprocessed_color = preprocessed_img
                                        
                                        # Save processed image to face_data directory
                                        face_path = os.path.join(face_data_dir, f"{student_number}.jpg")
                                        cv2.imwrite(face_path, preprocessed_color, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                        
                                        # Save a debug copy
                                        debug_dir = os.path.join(django_settings.MEDIA_ROOT, 'debug')
                                        os.makedirs(debug_dir, exist_ok=True)
                                        cv2.imwrite(os.path.join(debug_dir, f'import_{student_number}.jpg'), 
                                                preprocessed_color, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                        
                                        # Try to update the student profile if it exists
                                        try:
                                            user_profile = UserProfile.objects.get(student_number=student_number)
                                            # Extract embedding using facenet-pytorch
                                            embedding = get_face_embedding(os.path.join(django_settings.MEDIA_ROOT, face_path))
                                            # Store embedding as JSON string in face_image field
                                            user_profile.face_image = json.dumps(embedding)
                                            user_profile.save()
                                        except UserProfile.DoesNotExist:
                                            # Just continue - we've already saved the file
                                            pass
                                        
                                        success_count += 1
                                    except Exception as e:
                                        errors.append(f"Error processing image {filename}: {str(e)}")
                                        error_count += 1
                                        continue
                                    
                            except Exception as e:
                                errors.append(f"Error processing {filename}: {str(e)}")
                                error_count += 1
                            
                            processed_files += 1
                            progress = (processed_files / total_files) * 100
                            
                            yield json.dumps({
                                'status': 'processing',
                                'progress': progress,
                                'current_file': filename,
                                'success_count': success_count,
                                'error_count': error_count
                            }) + '\n'
                    
                    # Log activity after completion
                    AdminActivity.objects.create(
                        admin_user=request.user,
                        action='IMPORT_PHOTOS',
                        details=f"Imported {success_count} student photos. {error_count} photos failed to import."
                    )
                    
                    # Final status
                    yield json.dumps({
                        'status': 'completed',
                        'success_count': success_count,
                        'error_count': error_count,
                        'errors': errors[:10],  # Only include first 10 errors
                        'additional_errors': max(0, len(errors) - 10)
                    }) + '\n'
                    
            except Exception as e:
                yield json.dumps({
                    'status': 'error',
                    'message': str(e)
                }) + '\n'
        
        return StreamingHttpResponse(generate_progress(), content_type='text/event-stream')
    
    return redirect('admin_panel:manage_users')

def admin_panel_login(request):
    # Always logout any existing session when accessing admin login
    logout(request)
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            if user.is_superuser:
                login(request, user)
                messages.success(request, 'Welcome to Admin Panel!')
                return redirect('admin_panel:dashboard')
            elif CommitteeAccount.objects.filter(user=user).exists():
                login(request, user)
                messages.success(request, 'Welcome to Committee Panel!')
                return redirect('admin_panel:verification_codes')
            else:
                messages.error(request, 'You do not have permission to access this panel.')
                return redirect('admin_panel:admin_panel_login')
        else:
            messages.error(request, 'Invalid credentials.')
            return redirect('admin_panel:admin_panel_login')
            
    return render(request, 'admin_panel_login.html')

@login_required
def generate_report(request):
    active_settings = ElectionSettings.objects.filter(is_active=True).first()
    if not active_settings:
        context = {
            'total_voters': 0,
            'total_votes_cast': 0,
            'voter_turnout': 0,
            'total_candidates': 0,
            'votes_by_position': {},
            'non_voters_by_college': {},
            'current_school_year': None,
            'no_active_year': True,
            'no_active_year_message': 'No active school year yet.'
        }
        return render(request, 'admin_panel/reports.html', context)
    current_school_year = active_settings.school_year
    # Get all candidates and their votes for the active school year
    candidates = Candidate.objects.filter(school_year=current_school_year).select_related('user_profile')
    votes = Vote.objects.filter(school_year=current_school_year)
    # Get all users who haven't voted for the active school year
    non_voters = UserProfile.objects.filter(school_year=current_school_year).exclude(
        id__in=votes.values('user_profile')
    ).order_by('college', 'student_name')
    # Calculate statistics
    total_voters = UserProfile.objects.filter(school_year=current_school_year).count()
    total_votes_cast = votes.count()
    total_candidates = candidates.count()
    voter_turnout = (total_votes_cast / total_voters * 100) if total_voters > 0 else 0
    # Group non-voters by college
    non_voters_by_college = {}
    for voter in non_voters:
        if voter.college not in non_voters_by_college:
            non_voters_by_college[voter.college] = []
        non_voters_by_college[voter.college].append(voter)
    # Group votes by position
    votes_by_position = {}
    for candidate in candidates:
        position = candidate.position
        if position not in votes_by_position:
            votes_by_position[position] = []
        vote_count = votes.filter(candidate=candidate).count()
        votes_by_position[position].append({
            'candidate': candidate,
            'votes': vote_count,
            'percentage': (vote_count / total_votes_cast * 100) if total_votes_cast > 0 else 0
        })
        # Sort by vote count
        votes_by_position[position].sort(key=lambda x: x['votes'], reverse=True)
    context = {
        'total_voters': total_voters,
        'total_votes_cast': total_votes_cast,
        'voter_turnout': voter_turnout,
        'total_candidates': total_candidates,
        'votes_by_position': votes_by_position,
        'non_voters_by_college': non_voters_by_college,
        'current_school_year': current_school_year
    }
    if request.GET.get('format') == 'pdf':
        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        elements.append(Paragraph("Election Report", title_style))
        
        # Summary Statistics
        elements.append(Paragraph("Summary Statistics", styles['Heading2']))
        summary_data = [
            ["Total Voters", str(total_voters)],
            ["Total Votes Cast", str(total_votes_cast)],
            ["Voter Turnout", f"{voter_turnout:.1f}%"],
            ["Total Candidates", str(total_candidates)]
        ]
        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        
        # Results by Position
        for position, candidates in votes_by_position.items():
            elements.append(Paragraph(f"Results for {position}", styles['Heading2']))
            position_data = [["Candidate", "Student Number", "College", "Votes", "Percentage"]]
            for candidate_info in candidates:
                position_data.append([
                    candidate_info['candidate'].user_profile.student_name,
                    candidate_info['candidate'].user_profile.student_number,
                    candidate_info['candidate'].user_profile.college,
                    str(candidate_info['votes']),
                    f"{candidate_info['percentage']:.1f}%"
                ])
            position_table = Table(position_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 0.75*inch, 0.75*inch])
            position_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(position_table)
            elements.append(Spacer(1, 20))
        
        # Non-voters by College
        elements.append(Paragraph("Non-Voters by College", styles['Heading2']))
        for college, voters in non_voters_by_college.items():
            elements.append(Paragraph(college, styles['Heading3']))
            non_voters_data = [["Student Number", "Name", "Course", "Year Level"]]
            for voter in voters:
                non_voters_data.append([
                    voter.student_number,
                    voter.student_name,
                    voter.course,
                    voter.year_level
                ])
            non_voters_table = Table(non_voters_data, colWidths=[1.25*inch, 2*inch, 1.25*inch, 0.75*inch])
            non_voters_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(non_voters_table)
            elements.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(elements)
        
        # Get the value of the BytesIO buffer and return the response
        pdf = buffer.getvalue()
        buffer.close()
        
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="election_report.pdf"'
        response.write(pdf)
        return response
    
    return render(request, 'admin_panel/reports.html', context)

def admin_results(request):
    """
    View for displaying comprehensive election results in the admin panel.
    This is separate from the user-side results page and provides more detailed analytics.
    Only shows results after voting has ended.
    """
    # Get active election settings
    active_settings = ElectionSettings.objects.filter(is_active=True).first()
    
    if not active_settings:
        context = {
            'no_active_year': True,
            'current_school_year': None
        }
        return render(request, 'admin_panel/results.html', context)
    
    # Check if voting has ended
    current_time = timezone.now()
    voting_ended = not active_settings.is_voting_open and active_settings.voting_time_end and current_time > timezone.make_aware(timezone.datetime.combine(active_settings.voting_date, active_settings.voting_time_end))
    voting_status = "Active" if active_settings.is_voting_open else "Not Started" if active_settings.voting_time_start and current_time < timezone.make_aware(timezone.datetime.combine(active_settings.voting_date, active_settings.voting_time_start)) else "Ended"
    
    # If voting hasn't ended, just show the waiting message
    if not voting_ended:
        context = {
            'voting_ended': False,
            'voting_status': voting_status,
            'current_school_year': active_settings.school_year
        }
        return render(request, 'admin_panel/results.html', context)
    
    # Get all candidates and their votes - Filter by active school year
    candidates = Candidate.objects.filter(
        school_year=active_settings.school_year
    ).select_related('user_profile')
    votes = Vote.objects.filter(school_year=active_settings.school_year)
    
    # Calculate statistics - Filter by active school year
    total_voters = UserProfile.objects.filter(school_year=active_settings.school_year).count()
    total_votes_cast = votes.count()
    total_candidates = candidates.count()
    voter_turnout = (total_votes_cast / total_voters * 100) if total_voters > 0 else 0
    
    # Get all users who haven't voted - Filter by active school year
    non_voters = UserProfile.objects.filter(
        school_year=active_settings.school_year
    ).exclude(
        id__in=Vote.objects.filter(school_year=active_settings.school_year).values('user_profile')
    ).order_by('college', 'student_name')
    
    total_non_voters = non_voters.count()
    non_voter_percentage = (total_non_voters / total_voters * 100) if total_voters > 0 else 0
    
    # Group votes by position
    votes_by_position = {}
    for candidate in candidates:
        position = candidate.position
        if position not in votes_by_position:
            votes_by_position[position] = []
        
        # Count votes for this candidate
        vote_count = votes.filter(candidate=candidate).count()
        
        # Calculate percentage
        percentage = (vote_count / total_votes_cast * 100) if total_votes_cast > 0 else 0
        
        # Determine if this candidate is a winner (highest votes for their position)
        is_winner = True
        for other_candidate in votes_by_position.get(position, []):
            if other_candidate['votes'] > vote_count:
                is_winner = False
                other_candidate['is_winner'] = False
                break
        
        votes_by_position[position].append({
            'candidate': candidate,
            'votes': vote_count,
            'percentage': percentage,
            'is_winner': is_winner
        })
    
    # Sort candidates by votes (descending) within each position
    for position in votes_by_position:
        votes_by_position[position] = sorted(
            votes_by_position[position], 
            key=lambda x: x['votes'], 
            reverse=True
        )
    
    # Create an ordered dictionary for position results based on hierarchy
    ordered_votes_by_position = {}
    
    # First add national positions in order
    for position_code, position_display in Candidate.NATIONAL_POSITIONS:
        if position_code in votes_by_position:
            ordered_votes_by_position[position_display] = votes_by_position[position_code]
    
    # Then add college positions in order
    for position_code, position_display in Candidate.COLLEGE_POSITIONS:
        if position_code in votes_by_position:
            ordered_votes_by_position[position_display] = votes_by_position[position_code]
    
    # Finally add local positions in order
    for position_code, position_display in Candidate.LOCAL_POSITIONS:
        if position_code in votes_by_position:
            ordered_votes_by_position[position_display] = votes_by_position[position_code]
    
    # Calculate voter turnout by college - Filter by active school year
    college_turnout = {}
    colleges = UserProfile.objects.filter(
        school_year=active_settings.school_year
    ).values_list('college', flat=True).distinct()
    
    for college in colleges:
        # Count total voters in this college
        total_college_voters = UserProfile.objects.filter(
            college=college,
            school_year=active_settings.school_year
        ).count()
        
        # Count voters who have voted
        college_voters = UserProfile.objects.filter(
            college=college,
            school_year=active_settings.school_year,
            id__in=Vote.objects.filter(school_year=active_settings.school_year).values('user_profile')
        ).count()
        
        # Calculate percentage
        percentage = (college_voters / total_college_voters * 100) if total_college_voters > 0 else 0
        
        college_turnout[college] = {
            'total': total_college_voters,
            'voted': college_voters,
            'percentage': percentage
        }
    
    context = {
        'voting_ended': True,
        'voting_status': voting_status,
        'total_voters': total_voters,
        'total_votes_cast': total_votes_cast,
        'total_candidates': total_candidates,
        'voter_turnout': voter_turnout,
        'votes_by_position': ordered_votes_by_position,
        'college_turnout': college_turnout,
        'total_non_voters': total_non_voters,
        'non_voter_percentage': non_voter_percentage,
        'current_school_year': active_settings.school_year
    }
    
    return render(request, 'admin_panel/results.html', context)

@login_required
def verification_codes_view(request):
    """View for managing verification codes - accessible by both admin and committee members"""
    if not (request.user.is_superuser or CommitteeAccount.objects.filter(user=request.user).exists()):
        messages.error(request, 'You do not have permission to access this page.')
        return redirect('admin_panel:admin_panel_login')
        
    verification_codes = VerificationCode.objects.all().order_by('-created_at')
    return render(request, 'admin_panel/verification_codes.html', {
        'verification_codes': verification_codes,
        'is_committee': CommitteeAccount.objects.filter(user=request.user).exists(),
        'current_school_year': get_current_school_year()
    })

@login_required
def generate_code(request):
    """Generate a new verification code for a student - accessible by both admin and committee members"""
    if not (request.user.is_superuser or CommitteeAccount.objects.filter(user=request.user).exists()):
        return JsonResponse({
            'success': False,
            'message': 'You do not have permission to perform this action'
        }, status=403)
        
    try:
        data = json.loads(request.body)
        student_number = data.get('student_number')

        if not student_number:
            return JsonResponse({
                'success': False,
                'message': 'Student number is required'
            }, status=400)

        # Check if student exists
        try:
            UserProfile.objects.get(student_number=student_number)
        except UserProfile.DoesNotExist:
            return JsonResponse({
                'success': False,
                'message': 'Student not found'
            }, status=404)

        # Generate a unique 6-digit code
        from user.models import VerificationCode
        import random
        while True:
            code = ''.join(random.choices(string.digits, k=6))
            if not VerificationCode.objects.filter(code=code).exists():
                break
        expires_at = timezone.now() + timedelta(hours=1)

        # Create new verification code
        verification_code = VerificationCode.objects.create(
            student_number=student_number,
            code=code,
            expires_at=expires_at
        )

        return JsonResponse({
            'success': True,
            'student_number': student_number,
            'code': code,
            'expires_at': expires_at.strftime('%Y-%m-%d %H:%M:%S'),
            'current_school_year': get_current_school_year()
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'message': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)

@login_required
def regenerate_code(request, code_id):
    """Regenerate a verification code - accessible by both admin and committee members"""
    if not (request.user.is_superuser or CommitteeAccount.objects.filter(user=request.user).exists()):
        return JsonResponse({
            'success': False,
            'message': 'You do not have permission to perform this action'
        }, status=403)
        
    if request.method == 'POST':
        try:
            code = VerificationCode.objects.get(id=code_id)
            # Generate a new code
            new_code = ''.join(random.choices(string.digits, k=6))
            code.code = new_code
            code.created_at = timezone.now()
            code.expires_at = timezone.now() + timedelta(hours=1)
            code.is_used = False
            code.save()
            
            return JsonResponse({
                'success': True,
                'code': new_code,
                'student_number': code.student_number,
                'expires_at': code.expires_at.strftime('%Y-%m-%d %H:%M:%S'),
                'current_school_year': get_current_school_year()
            })
        except VerificationCode.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'Code not found'}, status=404)
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=500)
    return JsonResponse({'success': False, 'message': 'Invalid request method'}, status=400)

@login_required
def settings(request):
    # Get all users who are not already committee members
    committee_usernames = CommitteeAccount.objects.values_list('user__username', flat=True)
    available_users = UserProfile.objects.exclude(student_number__in=committee_usernames)
    
    # Create JSON representation of available users
    users_json = json.dumps([{
        'id': user.id,
        'student_number': user.student_number,
        'name': user.student_name
    } for user in available_users])
    
    # Get existing committee accounts
    committee_accounts = CommitteeAccount.objects.all().select_related('user')
    
    # Get all admin and committee users for activity logs
    admin_users = User.objects.filter(is_superuser=True)
    committee_users = User.objects.filter(committee_profile__isnull=False)
    users = admin_users | committee_users
    
    context = {
        'users_json': users_json,
        'committee_accounts': committee_accounts,
        'users': users,
        'current_school_year': get_current_school_year()
    }
    return render(request, 'settings.html', context)

@login_required
def create_committee(request):
    if request.method == 'POST':
        user_profile_id = request.POST.get('user_profile_id')
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        try:
            # Check if username already exists
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists.')
                return redirect('admin_panel:settings')
            
            # Get the user profile
            user_profile = UserProfile.objects.get(id=user_profile_id)
            
            # Check if user is already a committee member
            if CommitteeAccount.objects.filter(user__username=user_profile.student_number).exists():
                messages.error(request, 'This student is already a committee member.')
                return redirect('admin_panel:settings')
            
            # Create new user
            user = User.objects.create_user(
                username=username,
                password=password,
                first_name=user_profile.student_name
            )
            
            # Create committee account
            CommitteeAccount.objects.create(user=user)

            # Log the activity
            AdminActivity.objects.create(
                admin_user=request.user,
                action='CREATE_COMMITTEE_ACCOUNT',
                details=f'Created committee account for {user_profile.student_name} ({user_profile.student_number})'
            )
            
            # Redirect with success parameters
            return redirect(f"{reverse('admin_panel:settings')}?success=true&student_name={user_profile.student_name}")
            
        except UserProfile.DoesNotExist:
            messages.error(request, 'Selected student not found.')
        except Exception as e:
            messages.error(request, f'Error creating committee account: {str(e)}')
        
        return redirect('admin_panel:settings')
    
    return redirect('admin_panel:settings')

@login_required
def delete_committee(request, committee_id):
    if request.method == 'POST':
        try:
            committee = CommitteeAccount.objects.get(id=committee_id)
            user = committee.user
            committee.delete()
            user.delete()
            messages.success(request, 'Committee account deleted successfully.')
        except CommitteeAccount.DoesNotExist:
            messages.error(request, 'Committee account not found.')
        except Exception as e:
            messages.error(request, f'Error deleting committee account: {str(e)}')
    
    return redirect('admin_panel:settings')

@login_required
@user_passes_test(is_admin)
def change_password(request):
    """Change the admin's password"""
    if request.method == 'POST':
        try:
            current_password = request.POST.get('current_password')
            new_password = request.POST.get('new_password')
            confirm_password = request.POST.get('confirm_password')
            
            # Check if current password is correct
            if not request.user.check_password(current_password):
                return JsonResponse({
                    'success': False,
                    'error': 'Current password is incorrect.'
                })
            
            # Check if new passwords match
            if new_password != confirm_password:
                return JsonResponse({
                    'success': False,
                    'error': 'New passwords do not match.'
                })
            
            # Check password strength
            if len(new_password) < 8:
                return JsonResponse({
                    'success': False,
                    'error': 'Password must be at least 8 characters long.'
                })
            
            # Change the password
            request.user.set_password(new_password)
            request.user.save()
            
            # Log the activity
            AdminActivity.objects.create(
                admin_user=request.user,
                action='CHANGE_PASSWORD',
                details='Admin password was changed'
            )
            
            # Update the session to prevent logout
            update_session_auth_hash(request, request.user)
            
            return JsonResponse({
                'success': True,
                'message': 'Password changed successfully.',
                'current_school_year': get_current_school_year()
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method.'
    })

@login_required
def committee_change_password(request):
    """Change the committee member's password"""
    if not CommitteeAccount.objects.filter(user=request.user).exists():
        return JsonResponse({
            'success': False,
            'error': 'You do not have permission to perform this action'
        }, status=403)
        
    if request.method == 'POST':
        try:
            current_password = request.POST.get('current_password')
            new_password = request.POST.get('new_password')
            confirm_password = request.POST.get('confirm_password')
            
            # Check if current password is correct
            if not request.user.check_password(current_password):
                return JsonResponse({
                    'success': False,
                    'error': 'Current password is incorrect.'
                })
            
            # Check if new passwords match
            if new_password != confirm_password:
                return JsonResponse({
                    'success': False,
                    'error': 'New passwords do not match.'
                })
            
            # Check password strength
            if len(new_password) < 8:
                return JsonResponse({
                    'success': False,
                    'error': 'Password must be at least 8 characters long.'
                })
            
            # Change the password
            request.user.set_password(new_password)
            request.user.save()
            
            # Update the session to prevent logout
            update_session_auth_hash(request, request.user)
            
            return JsonResponse({
                'success': True,
                'message': 'Password changed successfully.',
                'current_school_year': get_current_school_year()
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return render(request, 'admin_panel/committee_settings.html')

@login_required
@user_passes_test(is_admin)
def activity_logs(request):
    from .models import AdminActivity
    from django.contrib.auth.models import User
    # Get all admin and committee users
    admin_users = User.objects.filter(is_superuser=True)
    committee_users = User.objects.filter(committee_profile__isnull=False)
    users = admin_users | committee_users
    users = users.distinct().order_by('username')
    logs = AdminActivity.objects.all().order_by('-timestamp')
    context = {
        'users': users,
        'logs': logs,
        'current_school_year': get_current_school_year()
    }
    return render(request, 'activity_logs.html', context)

@login_required
@user_passes_test(is_admin)
def user_activity_logs(request, user_id):
    user = get_object_or_404(User, id=user_id)
    activities = AdminActivity.objects.filter(admin_user=user).order_by('-created_at')
    
    # Handle search
    search_query = request.GET.get('search', '')
    if search_query:
        activities = activities.filter(
            Q(action__icontains=search_query) |
            Q(details__icontains=search_query)
        )
    
    return render(request, 'admin_panel/user_activity_logs.html', {
        'activities': activities,
        'user': user,
        'current_school_year': get_current_school_year()
    })

@login_required
@user_passes_test(is_admin)
def toggle_committee_status(request, committee_id):
    if request.method == 'POST':
        try:
            committee = CommitteeAccount.objects.get(id=committee_id)
            committee.is_active = not committee.is_active
            committee.save()
            
            # Log the activity
            AdminActivity.objects.create(
                admin_user=request.user,
                action='TOGGLE_COMMITTEE_STATUS',
                details=f'Committee account status changed to {"active" if committee.is_active else "inactive"} for {committee.user.username}'
            )
            
            return JsonResponse({
                'success': True,
                'is_active': committee.is_active,
                'message': f'Committee account {"activated" if committee.is_active else "deactivated"} successfully.',
                'current_school_year': get_current_school_year()
            })
        except CommitteeAccount.DoesNotExist:
            return JsonResponse({
                'success': False,
                'message': 'Committee account not found.'
            }, status=404)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': str(e)
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'message': 'Invalid request method.'
    }, status=400)

