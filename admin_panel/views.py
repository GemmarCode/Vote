from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required, user_passes_test
from user.models import Candidate, Vote, UserProfile
from .decorators import superuser_required
from django.contrib import messages
from django.db.models import Count
from django.utils import timezone
from .models import AdminActivity
from django.contrib.auth import get_user_model, authenticate, login, logout
from django.db.models.functions import TruncHour
from datetime import timedelta, datetime
import json
from .models import ElectionSettings
import pandas as pd
from django.db import transaction
import os
import cv2
import numpy as np
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import zipfile
import tempfile
import shutil
from pathlib import Path

# Create your views here.

def is_admin(user):
    return user.is_superuser

@login_required
@user_passes_test(is_admin)
def dashboard(request):
    User = get_user_model()
    total_users = User.objects.count()
    total_candidates = Candidate.objects.count()
    total_votes = Vote.objects.count()
    voter_turnout = (total_votes / total_users * 100) if total_users > 0 else 0
    
    # Get national candidates and their vote trends
    national_positions = [pos[0] for pos in Candidate.NATIONAL_POSITIONS]
    national_candidates = Candidate.objects.filter(
        position__in=national_positions, 
        approved=True
    ).select_related('user')
    
    # Get votes from the last 24 hours grouped by hour
    time_threshold = timezone.now() - timedelta(hours=24)
    vote_trends = {}
    
    for candidate in national_candidates:
        # Get all votes for this candidate in the last 24 hours
        votes = Vote.objects.filter(
            candidate=candidate,
            created_at__gte=time_threshold
        ).annotate(
            hour=TruncHour('created_at')
        ).values('hour').annotate(
            count=Count('id')
        ).order_by('hour')
        
        # Convert QuerySet to dictionary for easier lookup
        vote_counts = {
            v['hour'].replace(tzinfo=None): v['count'] 
            for v in votes
        }
        
        # Create hourly data points
        trend_data = []
        current_time = time_threshold
        while current_time <= timezone.now():
            hour_key = current_time.replace(minute=0, second=0, microsecond=0, tzinfo=None)
            trend_data.append({
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M'),
                'count': vote_counts.get(hour_key, 0)
            })
            current_time += timedelta(hours=1)
        
        # Only add candidates who have votes
        if any(point['count'] > 0 for point in trend_data):
            candidate_name = f"{candidate.user.get_full_name()} ({candidate.position})"
            vote_trends[candidate_name] = trend_data
    
    context = {
        'total_users': total_users,
        'total_candidates': total_candidates,
        'total_votes': total_votes,
        'voter_turnout': round(voter_turnout, 1),
        'vote_trends': json.dumps(vote_trends)
    }
    return render(request, 'dashboard.html', context)

@login_required
@user_passes_test(is_admin)
def manage_users(request):
    User = get_user_model()
    
    if request.method == 'POST':
        action = request.POST.get('action')
        user_id = request.POST.get('user_id')
        
        try:
            user = User.objects.get(id=user_id)
            
            if action == 'activate':
                user.is_active = True
                user.save()
                messages.success(request, f'User {user.username} has been activated.')
            
            elif action == 'deactivate':
                # Prevent deactivating superusers
                if user.is_superuser:
                    messages.error(request, 'Cannot deactivate superuser accounts.')
                else:
                    user.is_active = False
                    user.save()
                    messages.success(request, f'User {user.username} has been deactivated.')
            
        except User.DoesNotExist:
            messages.error(request, 'User not found.')
        
        return redirect('admin_panel:manage_users')
    
    # Get all users except superusers, ordered by registration date
    all_users = User.objects.filter(is_superuser=False).select_related('userprofile').order_by('-date_joined')
    print("DEBUG - Total users:", User.objects.count())
    print("DEBUG - Non-superuser users:", all_users.count())
    print("DEBUG - User usernames:", [u.username for u in all_users])
    
    # Get only new (inactive) users
    new_users = all_users.filter(is_active=False)
    print("DEBUG - New users:", new_users.count())
    print("DEBUG - New user usernames:", [u.username for u in new_users])   
    context = {
        'all_users': all_users,
        'new_users': new_users
    }
    return render(request, 'manage_users.html', context)

def process_face_photo(photo_file, student_id):
    """Process uploaded photo and convert it to face data"""
    try:
        # Save the uploaded file temporarily
        temp_path = default_storage.save(f'temp/{photo_file.name}', ContentFile(photo_file.read()))
        temp_file = os.path.join(settings.MEDIA_ROOT, temp_path)
        
        # Read the image with high quality
        img = cv2.imread(temp_file, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not read image file")
        
        # Save original photo with high quality
        face_path = os.path.join('face_data', f'{student_id}.jpg')
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT, face_path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Clean up temp file
        os.remove(temp_file)
        
        return gray.tobytes()
    except Exception as e:
        raise ValueError(f"Error processing photo: {str(e)}")

def process_photo_from_path(photo_path, student_id):
    """Process photo from a file path and convert it to face data"""
    try:
        # Read the image with high quality
        img = cv2.imread(photo_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not read image file")
        
        # Save original photo with high quality
        face_path = os.path.join('face_data', f'{student_id}.jpg')
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT, face_path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        return gray.tobytes()
    except Exception as e:
        raise ValueError(f"Error processing photo: {str(e)}")

@login_required
@user_passes_test(is_admin)
def register_user(request):
    if request.method == 'POST':
        student_id = request.POST.get('student_id')
        student_name = request.POST.get('student_name')
        sex = request.POST.get('sex')
        year_level = request.POST.get('year_level')
        course = request.POST.get('course')
        college = request.POST.get('college')
        photo = request.FILES.get('photo')
        
        try:
            # Check if student ID already exists
            if User.objects.filter(username=student_id).exists():
                messages.error(request, 'Student ID already exists.')
                return redirect('admin_panel:register_user')
            
            # Validate college
            valid_colleges = ['CAS', 'CAF', 'CCJE', 'CBA', 'CTED', 'CIT']
            if college not in valid_colleges:
                messages.error(request, 'Invalid college selected.')
                return redirect('admin_panel:register_user')
            
            # Validate sex
            valid_sex = ['M', 'F']
            if sex not in valid_sex:
                messages.error(request, 'Invalid sex selected.')
                return redirect('admin_panel:register_user')
            
            # Validate year level
            try:
                year_level = int(year_level)
                if year_level < 1 or year_level > 5:
                    messages.error(request, 'Year level must be between 1 and 5.')
                    return redirect('admin_panel:register_user')
            except ValueError:
                messages.error(request, 'Invalid year level.')
                return redirect('admin_panel:register_user')
            
            # Process photo if provided
            face_data = None
            if photo:
                try:
                    face_data = process_face_photo(photo, student_id)
                except ValueError as e:
                    messages.error(request, str(e))
                    return redirect('admin_panel:register_user')
            
            # Create new user with student ID as username
            user = User.objects.create_user(
                username=student_id,
                password=student_id,  # Default password is the student ID
                first_name=student_name,
                is_active=False
            )
            
            # Create user profile with additional fields
            profile = UserProfile.objects.create(
                user=user,
                student_id=student_id,
                college=college,
                department=course,  # Using course as department
                year_level=str(year_level),  # Convert to string since the model expects string
                gender=sex,  # Using sex as gender
                age=18,  # Default age
                contact_number='N/A',  # Default contact number
                face_data=face_data
            )
            
            # Log the admin activity
            AdminActivity.objects.create(
                admin_user=request.user,
                action='CREATE',
                action_model='User',
                description=f"Registered new user: {student_id}"
            )
            
            messages.success(request, f'User {student_id} has been successfully registered.')
            return redirect('admin_panel:manage_users')
            
        except Exception as e:
            messages.error(request, f'Error creating user: {str(e)}')
            return redirect('admin_panel:register_user')
    
    return render(request, 'register_user.html')

@login_required
@user_passes_test(is_admin)
def manage_candidates(request):
    # Get all candidates, including unapproved ones
    candidates = Candidate.objects.all().select_related('user')
    
    # Handle approve/reject actions
    if request.method == 'POST':
        candidate_id = request.POST.get('candidate_id')
        action = request.POST.get('action')
        
        try:
            candidate = Candidate.objects.get(id=candidate_id)
            if action == 'approve':
                candidate.approved = True
                candidate.save()
                messages.success(request, f'Candidate {candidate.user.get_full_name()} has been approved.')
            elif action == 'reject':
                candidate.delete()
                messages.success(request, f'Candidate {candidate.user.get_full_name()} has been rejected.')
        except Candidate.DoesNotExist:
            messages.error(request, 'Candidate not found.')
        
        return redirect('admin_panel:manage_candidates')

    # Get position lists
    national_positions = [pos[0] for pos in Candidate.NATIONAL_POSITIONS]
    local_positions = [pos[0] for pos in Candidate.LOCAL_POSITIONS]

    # Group pending candidates
    pending_national = candidates.filter(approved=False, position__in=national_positions)
    pending_local = candidates.filter(approved=False, position__in=local_positions)

    # Group approved candidates
    approved_national = candidates.filter(approved=True, position__in=national_positions)
    approved_local = candidates.filter(approved=True, position__in=local_positions)

    context = {
        'pending_candidates': {
            'national': pending_national,
            'local': pending_local
        },
        'approved_candidates': {
            'national': approved_national,
            'local': approved_local
        }
    }
    return render(request, 'manage_candidates.html', context)

@login_required
@user_passes_test(is_admin)
def manage_elections(request):
    # Get or create election settings
    settings, created = ElectionSettings.objects.get_or_create(id=1)
    
    if request.method == 'POST':
        try:
            # Parse datetime strings from the form
            candidacy_deadline = request.POST.get('candidacy_deadline')
            voting_start = request.POST.get('voting_start')
            voting_end = request.POST.get('voting_end')
            
            # Update settings with parsed datetime objects
            if candidacy_deadline:
                settings.candidacy_deadline = timezone.make_aware(datetime.strptime(candidacy_deadline, '%Y-%m-%dT%H:%M'))
            if voting_start:
                settings.voting_start = timezone.make_aware(datetime.strptime(voting_start, '%Y-%m-%dT%H:%M'))
            if voting_end:
                settings.voting_end = timezone.make_aware(datetime.strptime(voting_end, '%Y-%m-%dT%H:%M'))
            
            settings.save()
            messages.success(request, 'Election settings updated successfully.')
            
        except Exception as e:
            messages.error(request, f'Error updating settings: {str(e)}')
    
    context = {
        'settings': settings,
        'current_time': timezone.now()
    }
    return render(request, 'manage_elections.html', context)

@login_required
@user_passes_test(is_admin)
def results(request):
    # Get election results data
    candidates = Candidate.objects.all()
    results_data = []
    for candidate in candidates:
        votes_count = Vote.objects.filter(candidate=candidate).count()
        results_data.append({
            'candidate': candidate,
            'votes': votes_count
        })
    
    context = {
        'results_data': results_data
    }
    return render(request, 'results.html', context)

@login_required
@user_passes_test(is_admin)
def import_users(request):
    if request.method == 'POST' and request.FILES.get('excel_file'):
        excel_file = request.FILES['excel_file']
        
        try:
            # Read the Excel file
            df = pd.read_excel(excel_file)
            
            # Validate required columns
            required_columns = ['student_id', 'student_name', 'sex', 'year_level', 'course', 'college']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                messages.error(request, f'Missing required columns: {", ".join(missing_columns)}')
                return redirect('admin_panel:register_user')
            
            # Initialize counters
            created_count = 0
            error_count = 0
            errors = []
            
            # Process each row in the DataFrame
            with transaction.atomic():
                for index, row in df.iterrows():
                    try:
                        # Check if student ID already exists
                        if User.objects.filter(username=row['student_id']).exists():
                            errors.append(f"Row {index + 2}: Student ID '{row['student_id']}' already exists")
                            error_count += 1
                            continue
                        
                        # Validate college
                        valid_colleges = ['CAS', 'CAF', 'CCJE', 'CBA', 'CTED', 'CIT']
                        if row['college'] not in valid_colleges:
                            errors.append(f"Row {index + 2}: Invalid college '{row['college']}'")
                            error_count += 1
                            continue
                        
                        # Validate sex
                        if row['sex'] not in ['M', 'F']:
                            errors.append(f"Row {index + 2}: Invalid sex '{row['sex']}'. Must be 'M' or 'F'")
                            error_count += 1
                            continue
                        
                        # Validate year level
                        try:
                            year_level = int(row['year_level'])
                            if year_level < 1 or year_level > 5:
                                errors.append(f"Row {index + 2}: Year level must be between 1 and 5")
                                error_count += 1
                                continue
                        except ValueError:
                            errors.append(f"Row {index + 2}: Invalid year level")
                            error_count += 1
                            continue
                        
                        # Create user
                        user = User.objects.create_user(
                            username=str(row['student_id']),
                            password=str(row['student_id']),  # Default password is the student ID
                            first_name=row['student_name'],
                            is_active=False
                        )
                        
                        # Create user profile
                        UserProfile.objects.create(
                            user=user,
                            student_id=str(row['student_id']),
                            college=row['college'],
                            department=row['course'],  # Using course as department
                            year_level=str(year_level),  # Convert to string since the model expects string
                            gender=row['sex'],  # Using sex as gender
                            age=18,  # Default age
                            contact_number='N/A',  # Default contact number
                            face_data=None  # Will be updated later with photo import
                        )
                        
                        created_count += 1
                        
                    except Exception as e:
                        errors.append(f"Row {index + 2}: {str(e)}")
                        error_count += 1
            
            # Log the admin activity
            AdminActivity.objects.create(
                admin_user=request.user,
                action='CREATE',
                action_model='User',
                description=f"Imported {created_count} users from Excel file"
            )
            
            # Show success/error messages
            if created_count > 0:
                messages.success(request, f'Successfully created {created_count} users')
            
            if error_count > 0:
                messages.warning(request, f'Failed to create {error_count} users')
                for error in errors:
                    messages.error(request, error)
            
        except Exception as e:
            messages.error(request, f'Error processing Excel file: {str(e)}')
        
        return redirect('admin_panel:manage_users')
    
    return redirect('admin_panel:register_user')

@login_required
@user_passes_test(is_admin)
def import_photos(request):
    if request.method == 'POST' and request.FILES.get('photos_zip'):
        photos_zip = request.FILES['photos_zip']
        temp_dir = None
        
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Extract ZIP file
            with zipfile.ZipFile(photos_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Initialize counters
            processed_count = 0
            error_count = 0
            errors = []
            
            # Process each photo in the extracted directory
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            student_id = Path(file).stem  # Get filename without extension
                            photo_path = os.path.join(root, file)
                            
                            # Check if user exists
                            try:
                                user = User.objects.get(username=student_id)
                                user_profile = UserProfile.objects.get(user=user)
                                
                                # Process and save the photo
                                face_data = process_photo_from_path(photo_path, student_id)
                                user_profile.face_data = face_data
                                user_profile.save()
                                
                                processed_count += 1
                                
                            except User.DoesNotExist:
                                errors.append(f"Photo '{file}': User with ID '{student_id}' not found")
                                error_count += 1
                                continue
                            except UserProfile.DoesNotExist:
                                errors.append(f"Photo '{file}': User profile for ID '{student_id}' not found")
                                error_count += 1
                                continue
                            except ValueError as e:
                                errors.append(f"Photo '{file}': {str(e)}")
                                error_count += 1
                                continue
                            
                        except Exception as e:
                            errors.append(f"Error processing photo '{file}': {str(e)}")
                            error_count += 1
            
            # Log the admin activity
            AdminActivity.objects.create(
                admin_user=request.user,
                action='UPDATE',
                action_model='UserProfile',
                description=f"Imported photos for {processed_count} users"
            )
            
            # Show success/error messages
            if processed_count > 0:
                messages.success(request, f'Successfully processed photos for {processed_count} users')
            
            if error_count > 0:
                messages.warning(request, f'Failed to process {error_count} photos')
                for error in errors:
                    messages.error(request, error)
            
        except Exception as e:
            messages.error(request, f'Error processing ZIP file: {str(e)}')
        
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        return redirect('admin_panel:manage_users')
    
    return redirect('admin_panel:register_user')

def admin_panel_login(request):
    # Always logout any existing session when accessing admin login
    logout(request)
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None and user.is_staff:
            login(request, user)
            messages.success(request, 'Welcome to Admin Panel!')
            return redirect('admin_panel:dashboard')
        else:
            messages.error(request, 'Invalid admin credentials.')
            return redirect('admin_panel:admin_panel_login')
            
    return render(request, 'admin_panel_login.html')
