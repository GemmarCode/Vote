import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Vote.settings')
django.setup()

from user.models import Candidate, CandidateProfile
from django.db import transaction
from django.contrib.auth.models import User

def migrate_data():
    """
    Migrate data from CandidateProfile to Candidate for existing records
    """
    print("Starting data migration from CandidateProfile to Candidate...")
    
    # Get all candidate profiles
    candidate_profiles = CandidateProfile.objects.all()
    print(f"Found {candidate_profiles.count()} candidate profiles to migrate")
    
    # Keep track of successes and failures
    success_count = 0
    error_count = 0
    errors = []
    
    with transaction.atomic():
        for profile in candidate_profiles:
            try:
                user = profile.user
                
                # Check if candidate already exists
                candidate = Candidate.objects.filter(user=user).first()
                
                if candidate:
                    # Update existing candidate with profile data
                    candidate.platform = profile.platform
                    candidate.achievements = profile.achievements
                    
                    # If photo is present in profile but not in candidate, copy it
                    if profile.photo and not candidate.photo:
                        candidate.photo = profile.photo
                        
                    candidate.save()
                    print(f"Updated existing candidate for user: {user.username}")
                else:
                    # Create new candidate from profile
                    new_candidate = Candidate(
                        user=user,
                        position=profile.position,
                        platform=profile.platform,
                        achievements=profile.achievements,
                        photo=profile.photo,
                        approved=profile.is_active
                    )
                    
                    # Get college, department, year_level from UserProfile if available
                    try:
                        user_profile = user.userprofile
                        new_candidate.college = user_profile.college
                        new_candidate.department = user_profile.course
                        try:
                            new_candidate.year_level = int(user_profile.year_level)
                        except (ValueError, TypeError):
                            pass
                    except Exception as e:
                        print(f"Warning: Could not get user profile data: {str(e)}")
                    
                    new_candidate.save()
                    print(f"Created new candidate for user: {user.username}")
                
                success_count += 1
            
            except Exception as e:
                error_count += 1
                error_message = f"Error migrating profile for user {profile.user.username}: {str(e)}"
                errors.append(error_message)
                print(error_message)
    
    # Print summary
    print("\nMigration Summary:")
    print(f"Total candidate profiles: {candidate_profiles.count()}")
    print(f"Successfully migrated: {success_count}")
    print(f"Failed migrations: {error_count}")
    
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")

if __name__ == "__main__":
    migrate_data() 