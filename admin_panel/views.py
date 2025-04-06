from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required, user_passes_test
from user.models import Candidate, Vote, UserProfile, VerificationCode
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
from user.face_utils import FaceRecognition, preprocess_face_image
from django.db.utils import IntegrityError
from django.http import JsonResponse, HttpResponse
from django.template.loader import render_to_string
from django.db.models import Q
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
        position__in=national_positions
    ).select_related('user_profile')
    
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
            candidate_name = f"{candidate.user_profile.student_name} ({candidate.position})"
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
    # Get all users ordered by id (newer users will have higher IDs)
    all_users = UserProfile.objects.all().order_by('-id')
    print("DEBUG - Total users:", UserProfile.objects.count())
    print("DEBUG - User student numbers:", [u.student_number for u in all_users])
    
    context = {
        'all_users': all_users
    }
    return render(request, 'manage_users.html', context)

def process_face_photo(photo_file, student_number):
    """Process uploaded photo and convert it to face data"""
    try:
        # Save the uploaded file temporarily
        temp_path = default_storage.save(f'temp/{photo_file.name}', ContentFile(photo_file.read()))
        temp_file = os.path.join(settings.MEDIA_ROOT, temp_path)
        
        # Read the image with high quality
        img = cv2.imread(temp_file, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not read image file")
        
        # Create face_data directory if it doesn't exist
        face_data_dir = os.path.join(settings.MEDIA_ROOT, 'face_data')
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
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT, face_path), preprocessed_color, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Store the image in binary format for the face_data field
        _, buffer = cv2.imencode('.jpg', preprocessed_color)
        face_bytes = buffer.tobytes()
        
        # Clean up temp file
        os.remove(temp_file)
        
        # Save a debug copy to help with troubleshooting
        try:
            debug_dir = os.path.join(settings.MEDIA_ROOT, 'debug')
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
        face_data_dir = os.path.join(settings.MEDIA_ROOT, 'face_data')
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
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT, face_path), preprocessed_color, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Store the image in binary format for the face_data field
        _, buffer = cv2.imencode('.jpg', preprocessed_color)
        face_bytes = buffer.tobytes()
        
        # Save a debug copy to help with troubleshooting
        try:
            debug_dir = os.path.join(settings.MEDIA_ROOT, 'debug')
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
            data = json.loads(request.body)
            student_number = data.get('student_number')
            student_name = data.get('student_name')
            sex = data.get('sex')
            year_level = data.get('year_level')
            course = data.get('course')
            college = data.get('college')
            
            # Simple validation
            if not all([student_number, student_name, sex, college, year_level, course]):
                return JsonResponse({'error': 'All fields are required'}, status=400)
            
            # Check if UserProfile already exists
            if UserProfile.objects.filter(student_number=student_number).exists():
                return JsonResponse({'error': 'Student number already exists'}, status=400)
            
            # Normalize year_level to string if it's an integer
            if isinstance(year_level, int):
                year_level = str(year_level)
            
            # Check if User already exists
            existing_user = User.objects.filter(username=student_number).first()
            
            if existing_user:
                # Use existing user
                user = existing_user
            else:
                # Create User
                user = User.objects.create_user(
                    username=student_number,
                    password=student_number,  # Default password is their student number
                    first_name=student_name,
                    is_active=True
                )
            
            # Create UserProfile
            UserProfile.objects.create(
                student_number=student_number,
                student_name=student_name,
                sex=sex,
                year_level=year_level,
                course=course,
                college=college
            )
            
            return JsonResponse({'success': True})
            
        except ValueError as e:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    # If GET request, redirect to manage_users view
    return redirect('admin_panel:manage_users')

@login_required
@user_passes_test(is_admin)
def manage_candidates(request):
    # Get all candidates
    candidates = Candidate.objects.all().select_related('user_profile')
    
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
                        achievements=achievements
                    )
                    
                    # Save photo if provided
                    if photo:
                        candidate.photo = photo
                    
                    candidate.save()
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
    
    context = {
        'candidates': candidates,
        'national_positions': national_positions,
        'college_positions': college_positions,
        'local_positions': local_positions,
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
    candidates = Candidate.objects.all().select_related('user_profile')
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
                    return redirect('admin_panel:register_user')
            
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
                    
                    # Check for existing student number
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

                    # Check if a User with this username already exists
                    existing_user = User.objects.filter(username=student_number).first()
                    
                    if existing_user:
                        # Use existing user
                        user = existing_user
                    else:
                        # Create new User if one doesn't exist
                        user = User.objects.create_user(
                            username=student_number,
                            password=student_number,  # Default password is student number
                            first_name=student_name,
                            is_active=True
                        )

                    # Create new UserProfile
                    UserProfile.objects.create(
                        user=user,
                        student_number=student_number,
                        student_name=student_name,
                        sex=sex_normalized,
                        year_level=year_level,
                        course=course,
                        college=college_normalized
                    )
                    success_count += 1

                except Exception as e:
                    error_logs.append(f"Row {index + 2}: Error processing record - {str(e)}")
                    error_count += 1
                    continue

            # Show import results
            if success_count > 0:
                messages.success(request, f"Successfully imported {success_count} student records.")
            if error_count > 0:
                messages.warning(request, f"Failed to import {error_count} records. Check the error log below:")
                for error in error_logs:
                    messages.warning(request, error)
            
        except Exception as e:
            messages.error(request, f"Error processing file: {str(e)}")
        
        return redirect('admin_panel:register_user')
    
    messages.error(request, "No file was uploaded.")
    return redirect('admin_panel:register_user')

@login_required
@user_passes_test(is_admin)
def import_photos(request):
    """Import student photos from ZIP file"""
    if request.method == 'POST' and request.FILES.get('photos_zip'):
        zip_file = request.FILES['photos_zip']
        
        # Create directory if it doesn't exist
        face_data_dir = os.path.join(settings.MEDIA_ROOT, 'face_data')
        os.makedirs(face_data_dir, exist_ok=True)
        
        import zipfile
        from io import BytesIO
        import traceback
        
        success_count = 0
        error_count = 0
        errors = []
            
        try:
            with zipfile.ZipFile(BytesIO(zip_file.read())) as z:
                for filename in z.namelist():
                    # Skip directories and non-image files
                    if filename.endswith('/') or not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    
                    # Extract student number from filename (remove extension)
                    base_filename = os.path.basename(filename)
                    student_number = os.path.splitext(base_filename)[0]
                    
                    try:
                        # Extract file from ZIP
                        image_data = z.read(filename)
                        img_array = np.frombuffer(image_data, np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                        if img is None:
                            errors.append(f"Error processing photo '{base_filename}': Could not decode image")
                            error_count += 1
                            continue
                            
                        # Preprocess the image using the same standardized function used for webcam images
                        preprocessed_img = preprocess_face_image(img)
                        
                        if preprocessed_img is None:
                            errors.append(f"Error processing photo '{base_filename}': Preprocessing failed")
                            error_count += 1
                            continue
                            
                        # Convert back to BGR (color) for saving
                        if len(preprocessed_img.shape) == 2:  # If grayscale
                            preprocessed_color = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
                        else:
                            preprocessed_color = preprocessed_img
                        
                        # Save processed image to face_data directory
                        output_path = os.path.join(face_data_dir, f"{student_number}.jpg")
                        cv2.imwrite(output_path, preprocessed_color)
                        success_count += 1
                        
                        # Log success
                        print(f"Successfully processed and saved photo for student: {student_number}")
                    
                    except Exception as e:
                        error_message = f"Error processing photo '{base_filename}': {str(e)}"
                        print(error_message)
                        print(traceback.format_exc())
                        errors.append(error_message)
                        error_count += 1
        
        except zipfile.BadZipFile:
            messages.error(request, "Invalid ZIP file. Please upload a valid ZIP file.")
            return redirect('admin_panel:register_user')
        
        # Show summary message
        if success_count > 0:
            messages.success(request, f"Successfully imported {success_count} photos.")
        
        if error_count > 0:
            error_message = f"Failed to import {error_count} photos. "
            if len(errors) <= 5:
                error_message += "Errors: " + "; ".join(errors)
            else:
                error_message += "Errors: " + "; ".join(errors[:5]) + f"; and {len(errors)-5} more errors."
            messages.warning(request, error_message)
        
        return redirect('admin_panel:register_user')
    
    messages.error(request, "No file uploaded or invalid request method.")
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

@login_required
def generate_report(request):
    # Get all candidates and their votes
    candidates = Candidate.objects.all().select_related('user_profile')
    votes = Vote.objects.all()
    
    # Get all users who haven't voted
    non_voters = UserProfile.objects.filter(
        ~Q(id__in=Vote.objects.values('user_profile'))
    ).order_by('college', 'student_name')
    
    # Calculate statistics
    total_voters = UserProfile.objects.count()
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
    # Check if voting has ended
    try:
        election_settings = ElectionSettings.objects.get(id=1)
        current_time = timezone.now()
        voting_ended = not election_settings.is_voting_open() and election_settings.voting_end and current_time > election_settings.voting_end
        voting_status = "Active" if election_settings.is_voting_open() else "Not Started" if election_settings.voting_start and current_time < election_settings.voting_start else "Ended"
    except ElectionSettings.DoesNotExist:
        voting_ended = False
        voting_status = "Not Configured"
    
    # If voting hasn't ended, just show the waiting message
    if not voting_ended:
        context = {
            'voting_ended': False,
            'voting_status': voting_status
        }
        return render(request, 'admin_panel/results.html', context)
    
    # Get all candidates and their votes
    candidates = Candidate.objects.all().select_related('user_profile')
    votes = Vote.objects.all()
    
    # Calculate statistics
    total_voters = UserProfile.objects.count()
    total_votes_cast = votes.count()
    total_candidates = candidates.count()
    voter_turnout = (total_votes_cast / total_voters * 100) if total_voters > 0 else 0
    
    # Get all users who haven't voted
    non_voters = UserProfile.objects.filter(
        ~Q(id__in=Vote.objects.values('user_profile'))
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
    
    # Calculate voter turnout by college
    college_turnout = {}
    colleges = UserProfile.objects.values_list('college', flat=True).distinct()
    
    for college in colleges:
        # Count total voters in this college
        total_college_voters = UserProfile.objects.filter(college=college).count()
        
        # Count voters who have voted
        college_voters = UserProfile.objects.filter(
            college=college,
            id__in=Vote.objects.values('user_profile')
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
        'votes_by_position': votes_by_position,
        'college_turnout': college_turnout,
        'total_non_voters': total_non_voters,
        'non_voter_percentage': non_voter_percentage,
    }
    
    return render(request, 'admin_panel/results.html', context)

def verification_codes_view(request):
    """View for managing verification codes"""
    verification_codes = VerificationCode.objects.all().order_by('-created_at')
    return render(request, 'admin_panel/verification_codes.html', {
        'verification_codes': verification_codes
    })

@csrf_exempt
def generate_code(request):
    """Generate a new verification code for a student"""
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

        # Generate a random 6-digit code
        code = ''.join(random.choices(string.digits, k=6))
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
            'expires_at': expires_at.strftime('%Y-%m-%d %H:%M:%S')
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

@csrf_exempt
def regenerate_code(request, code_id):
    if request.method == 'POST':
        try:
            code = VerificationCode.objects.get(id=code_id)
            # Generate a new code
            new_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            code.code = new_code
            code.created_at = timezone.now()
            code.expires_at = timezone.now() + timedelta(hours=24)
            code.is_used = False
            code.save()
            
            return JsonResponse({
                'success': True,
                'code': new_code,
                'student_number': code.student_number,
                'expires_at': code.expires_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        except VerificationCode.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'Code not found'}, status=404)
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=500)
    return JsonResponse({'success': False, 'message': 'Invalid request method'}, status=400)
