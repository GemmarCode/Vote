from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from .models import UserProfile, Candidate, Vote, VotingPhase
from admin_panel.models import ElectionSettings
from django.core.exceptions import ValidationError
from .face_utils import process_webcam_image, calculate_face_similarity, compare_faces
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

def login_view(request):
    if request.method == 'POST':
        # Only handle face login
        face_data = request.POST.get('face_data')
        if not face_data:
            messages.error(request, 'No face data received. Please try again.')
            return redirect('login')

        try:
            print("\n=== Starting Face Login Process ===")
            
            # Process the captured image
            captured_face = process_webcam_image(face_data)
            
            if captured_face is not None:
                print("Successfully processed login image")
                print(f"Captured face shape: {captured_face.shape}")
                
                # Get all users with face data
                user_profiles = UserProfile.objects.filter(face_data__isnull=False).all()
                print(f"Found {len(list(user_profiles))} users with face data")
                
                # Try to match with stored face data
                matched_user = None
                highest_similarity = 0
                
                for profile in user_profiles:
                    stored_face = profile.get_face_data()
                    if stored_face is not None:
                        # Calculate similarity between captured face and stored face
                        similarity = calculate_face_similarity(captured_face, stored_face)
                        print(f"Similarity with {profile.user.username}: {similarity}")
                        
                        # Update if this is the best match so far
                        if similarity > highest_similarity and similarity > 0.8:  # 80% similarity threshold
                            highest_similarity = similarity
                            matched_user = profile.user
                
                if matched_user:
                    print(f"Matched user: {matched_user.username}")
                    
                    # Check if user is active
                    if not matched_user.is_active:
                        messages.error(request, 'Your account is not active. Please contact the administrator.')
                        return redirect('login')
                    
                    # Log the user in
                    login(request, matched_user)
                    messages.success(request, f'Welcome back, {matched_user.first_name}!')
                    
                    # Redirect based on user type
                    if matched_user.is_superuser:
                        return redirect('admin_panel:dashboard')
                    else:
                        return redirect('home')
                else:
                    messages.error(request, 'Face not recognized. Please try again.')
            else:
                messages.error(request, 'Could not process face image. Please try again.')
        
        except Exception as e:
            print(f"Login error: {str(e)}")
            print(traceback.format_exc())  # Full error traceback
            messages.error(request, 'An error occurred during face recognition. Please try again.')
    
    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    return redirect('login')

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

def file_candidacy(request):
    try:
        user_profile = UserProfile.objects.get(user=request.user)
        
        initial_data = {
            'first_name': request.user.first_name,
            'last_name': request.user.last_name,
            'student_id': user_profile.student_id,
            'college': user_profile.college,
            'department': user_profile.department,
            'year_level': user_profile.year_level,
            'gender': user_profile.gender,
            'age': user_profile.age,
            'contact_number': user_profile.contact_number,
            'get_college_display': dict(UserProfile.COLLEGES).get(user_profile.college, user_profile.college)
        }
        
        if request.method == 'POST':
            # Get form data
            position = request.POST.get('position')
            platform = request.POST.get('platform')
            achievements = request.POST.get('achievements')
            photo = request.FILES.get('photo')

            # Validate required fields
            if not all([position, platform]):
                messages.error(request, "Please fill all required fields")
                return render(request, 'file_candidacy.html', {'initial_data': initial_data})

            # Create new candidate (not approved by default)
            candidate = Candidate.objects.create(
                user=request.user,
                position=position.upper(),  # Store the position code in uppercase
                college=user_profile.college,
                department=user_profile.department,
                year_level=user_profile.year_level,
                platform=platform,
                photo=photo,
                achievements=achievements if achievements else None,
                approved=False  # Set to False by default
            )

            messages.success(request, "Your candidacy has been filed successfully! Please wait for admin approval.")
            return redirect('candidates')

        return render(request, 'file_candidacy.html', {'initial_data': initial_data})
        
    except UserProfile.DoesNotExist:
        messages.error(request, 'User profile not found. Please complete registration first.')
        return redirect('home')
    except Exception as e:
        messages.error(request, f'Error: {str(e)}')
        return redirect('home')

def profile_settings(request):
    user_profile = UserProfile.objects.get(user=request.user)
    try:
        candidate = Candidate.objects.get(user=request.user)
        is_candidate = True
    except Candidate.DoesNotExist:
        candidate = None
        is_candidate = False
    
    if request.method == 'POST':
        if 'update_profile' in request.POST:
            try:
                # Update User model fields
                request.user.first_name = request.POST.get('first_name')
                request.user.last_name = request.POST.get('last_name')
                request.user.email = request.POST.get('email')
                request.user.save()
                
                # Update UserProfile fields
                user_profile.student_id = request.POST.get('student_id')
                user_profile.college = request.POST.get('college')
                user_profile.department = request.POST.get('department')
                user_profile.year_level = request.POST.get('year_level')
                user_profile.gender = request.POST.get('gender')
                
                # Convert age to integer and validate
                try:
                    age = int(request.POST.get('age', '0'))
                    if 16 <= age <= 99:  # Validate age range
                        user_profile.age = age
                    else:
                        raise ValueError("Age must be between 16 and 99")
                except ValueError as e:
                    raise ValueError("Invalid age value. Please enter a number between 16 and 99.")
                
                user_profile.contact_number = request.POST.get('contact_number')
                
                # Handle profile picture upload
                if 'profile_picture' in request.FILES:
                    user_profile.profile_picture = request.FILES['profile_picture']
                
                user_profile.save()
                messages.success(request, 'Profile updated successfully!')
                
            except ValueError as e:
                messages.error(request, str(e))
            except Exception as e:
                messages.error(request, f'Error updating profile: {str(e)}')
                
        elif 'withdraw_candidacy' in request.POST:
            try:
                if candidate:
                    candidate.delete()
                    messages.success(request, 'Your candidacy has been withdrawn successfully.')
                    is_candidate = False
            except Exception as e:
                messages.error(request, f'Error withdrawing candidacy: {str(e)}')
                
        return redirect('profile_settings')
    
    context = {
        'user_profile': user_profile,
        'is_candidate': is_candidate,
        'candidate': candidate
    }
    return render(request, 'profile_settings.html', context)

def voting_view(request):
    if request.method == 'POST':
        print("\n=== Starting Face Recognition Process ===")
        # Handle face verification for voting
        face_data = request.POST.get('face_data')
        if not face_data:
            print("❌ Error: No face data received")
            messages.error(request, 'No face data received. Please try again.')
            return redirect('vote')

        try:
            print("📸 Processing captured webcam image...")
            # Process the captured image
            captured_face = process_webcam_image(face_data)
            
            if captured_face is not None:
                print("✅ Successfully processed captured image")
                print(f"📊 Captured image shape: {captured_face.shape}")
                
                # Variables to track best match
                best_match_score = 0
                best_match_student_id = None
                
                print(f"\n🔍 Searching for matches in: {settings.FACE_DATA_DIR}")
                # Loop through all images in the face_data directory
                face_files = os.listdir(settings.FACE_DATA_DIR)
                print(f"📁 Found {len(face_files)} files in face_data directory")
                
                for filename in face_files:
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        # Get student ID from filename
                        student_id = os.path.splitext(filename)[0]
                        print(f"\n⏳ Processing {filename} (Student ID: {student_id})")
                        
                        # Load and process stored face image
                        stored_image_path = os.path.join(settings.FACE_DATA_DIR, filename)
                        stored_image = cv2.imread(stored_image_path, cv2.IMREAD_GRAYSCALE)
                        
                        if stored_image is not None:
                            stored_image = cv2.resize(stored_image, (200, 200))
                            print(f"📊 Stored image shape: {stored_image.shape}")
                            
                            # Calculate similarity
                            similarity = calculate_face_similarity(captured_face, stored_image)
                            print(f"📊 Similarity score with {student_id}: {similarity:.2%}")
                            
                            # Update best match if this is better
                            if similarity > best_match_score:
                                best_match_score = similarity
                                best_match_student_id = student_id
                                print(f"✨ New best match! Student ID: {student_id} (Score: {similarity:.2%})")
                        else:
                            print(f"❌ Failed to load image: {filename}")
                
                print(f"\n=== Face Recognition Results ===")
                print(f"🏆 Best match score: {best_match_score:.2%}")
                print(f"🎓 Best match student ID: {best_match_student_id}")
                
                # If we found a match above threshold
                if best_match_score >= 0.8:  # 80% similarity threshold
                    print("✅ Match found above threshold (80%)")
                    try:
                        # Find user with matching student ID
                        user_profile = UserProfile.objects.get(student_id=best_match_student_id)
                        matched_user = user_profile.user
                        print(f"👤 Found matching user: {matched_user.username}")
                        
                        # Check if user is active
                        if not matched_user.is_active:
                            print("❌ User account is not active")
                            messages.error(request, 'Your account is not active. Please contact the administrator.')
                            return redirect('vote')
                        
                        # Log the user in
                        login(request, matched_user)
                        print(f"✅ Successfully logged in user: {matched_user.username}")
                        messages.success(request, f'Welcome back, {matched_user.first_name}!')
                        
                        # Redirect based on user type
                        if matched_user.is_superuser:
                            print("👑 Redirecting to admin panel (superuser)")
                            return redirect('admin_panel:dashboard')
                        else:
                            print("🏠 Redirecting to home")
                            return redirect('home')
                    except UserProfile.DoesNotExist:
                        print("❌ No UserProfile found for student ID:", best_match_student_id)
                        messages.error(request, 'No matching user account found.')
                        return redirect('vote')
                else:
                    print("❌ No match found above threshold (80%)")
                    messages.error(request, 'Face not recognized. Please try again.')
            else:
                print("❌ Failed to process captured image")
                messages.error(request, 'Could not process face image. Please try again.')
        
        except Exception as e:
            print(f"❌ Error during face recognition: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            messages.error(request, f'An error occurred during face recognition: {str(e)}')
    
    # If GET request or verification failed, show the face verification form
    return render(request, 'face_verification.html')

def verify_face(request):
    if request.method == 'POST':
        try:
            # Get the captured image data
            image_data = request.POST.get('imageData')
            captured_face = process_webcam_image(image_data)
            
            # Get all face images from the face_data directory
            best_match_score = 0
            best_match_student_id = None
            
            # Loop through all images in the face_data directory
            for filename in os.listdir(settings.FACE_DATA_DIR):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    # Extract student ID from filename (e.g., "2021001.jpg" -> "2021001")
                    student_id = os.path.splitext(filename)[0]
                    
                    # Load and process the stored face image
                    stored_face_path = os.path.join(settings.FACE_DATA_DIR, filename)
                    stored_face = cv2.imread(stored_face_path, cv2.IMREAD_GRAYSCALE)
                    stored_face = cv2.resize(stored_face, (200, 200))
                    
                    # Calculate similarity
                    similarity = calculate_face_similarity(captured_face, stored_face)
                    
                    # Update best match if this is the highest similarity so far
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_student_id = student_id
            
            # If we found a match above the threshold
            if best_match_score >= 0.8:  # 80% similarity threshold
                try:
                    # Get the user profile with matching student ID
                    user_profile = UserProfile.objects.get(student_id=best_match_student_id)
                    user = user_profile.user
                    
                    # Check if user is active
                    if user.is_active:
                        # Log the user in
                        login(request, user)
                        # Store verification status in session
                        request.session['face_verified'] = True
                        return JsonResponse({'success': True})
                    else:
                        return JsonResponse({
                            'success': False,
                            'error': 'User account is not active'
                        })
                except UserProfile.DoesNotExist:
                    return JsonResponse({
                        'success': False,
                        'error': 'No matching user found'
                    })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'Face verification failed - no match found'
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'An error occurred during face recognition: {str(e)}'
            })
    
    return render(request, 'face_verification.html')

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