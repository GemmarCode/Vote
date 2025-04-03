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
from user.face_utils import FaceRecognition, preprocess_face_image

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
        student_number = request.POST.get('student_number')
        student_name = request.POST.get('student_name')
        sex = request.POST.get('sex')
        year_level = request.POST.get('year_level')
        course = request.POST.get('course')
        college = request.POST.get('college')
        photo = request.FILES.get('photo')
        
        try:
            # Check if student number already exists
            if User.objects.filter(username=student_number).exists():
                messages.error(request, 'Student number already exists.')
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
                    face_data = process_face_photo(photo, student_number)
                except ValueError as e:
                    messages.error(request, str(e))
                    return redirect('admin_panel:register_user')
            
            # Create new user with student number as username
            user = User.objects.create_user(
                username=student_number,
                password=student_number,  # Default password is the student number
                first_name=student_name,
                is_active=False
            )
            
            # Create user profile with additional fields
            profile = UserProfile.objects.create(
                user=user,
                student_number=student_number,
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
                description=f"Registered new user: {student_number}"
            )
            
            messages.success(request, f'User {student_number} has been successfully registered.')
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

    # Get all users from UserProfile for search
    users = UserProfile.objects.all().order_by('student_name')

    # Get unique colleges and departments for dropdowns
    colleges = UserProfile.objects.values_list('college', flat=True).distinct()
    departments = UserProfile.objects.values_list('course', flat=True).distinct()

    context = {
        'candidates': candidates,
        'users': users,
        'national_positions': national_positions,
        'local_positions': local_positions,
        'colleges': colleges,
        'departments': departments
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
                        error_logs.append(f"Row {index + 2}: Student number {student_number} already exists")
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

                    # Create new UserProfile
                    UserProfile.objects.create(
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
