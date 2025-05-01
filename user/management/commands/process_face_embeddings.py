from django.core.management.base import BaseCommand
from user.models import UserProfile
from tqdm import tqdm

class Command(BaseCommand):
    help = 'Process face images and extract embeddings using facenet-pytorch'

    def handle(self, *args, **options):
        # Get all user profiles with face images but no embeddings
        profiles = UserProfile.objects.filter(face_image__isnull=False, extracted_image__isnull=True)
        
        self.stdout.write(f"Found {profiles.count()} profiles with face images to process")
        
        success_count = 0
        for profile in tqdm(profiles, desc="Processing face images"):
            if profile.extract_and_save_face_embedding():
                success_count += 1
            else:
                self.stdout.write(self.style.WARNING(f"Failed to process face image for {profile.student_number}"))
        
        self.stdout.write(self.style.SUCCESS(f"Successfully processed {success_count} face images")) 