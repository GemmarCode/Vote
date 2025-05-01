import os
import json
from django.core.management.base import BaseCommand
from user.models import UserProfile
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Add preprocessing function
def preprocess_face_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transforms.Resize((160, 160))(img)
    return img

class Command(BaseCommand):
    help = 'Import students with face images and extract face embeddings'

    def add_arguments(self, parser):
        parser.add_argument('--data-file', type=str, help='Path to JSON file containing student data')
        parser.add_argument('--images-dir', type=str, help='Directory containing student face images')

    def handle(self, *args, **options):
        data_file = options['data_file']
        images_dir = options['images_dir']

        if not data_file or not images_dir:
            self.stdout.write(self.style.ERROR('Please provide both --data-file and --images-dir arguments'))
            return

        if not os.path.exists(data_file):
            self.stdout.write(self.style.ERROR(f'Data file {data_file} does not exist'))
            return

        if not os.path.exists(images_dir):
            self.stdout.write(self.style.ERROR(f'Images directory {images_dir} does not exist'))
            return

        # Load student data
        with open(data_file, 'r') as f:
            students_data = json.load(f)

        success_count = 0
        error_count = 0

        for student in tqdm(students_data, desc="Importing students"):
            try:
                student_number = student['student_number']
                image_filename = f"{student_number}.jpg"  # Assuming images are named with student numbers
                image_path = os.path.join(images_dir, image_filename)

                if not os.path.exists(image_path):
                    self.stdout.write(self.style.WARNING(f'Image not found for student {student_number}'))
                    error_count += 1
                    continue

                # Preprocess the image
                preprocessed_img = preprocess_face_image(image_path)
                temp_path = os.path.join(images_dir, f"preprocessed_{student_number}.jpg")
                preprocessed_img.save(temp_path)

                # Create or update user profile
                user_profile, created = UserProfile.objects.get_or_create(
                    student_number=student_number,
                    defaults={
                        'student_name': student.get('student_name', ''),
                        'sex': student.get('sex', 'M'),
                        'year_level': student.get('year_level', '1'),
                        'course': student.get('course', ''),
                        'college': student.get('college', 'CAS'),
                        'school_year': student.get('school_year', '2023-2024')
                    }
                )

                # Extract and save face embedding using the preprocessed image
                if user_profile.extract_and_save_face_embedding(temp_path):
                    success_count += 1
                else:
                    error_count += 1
                    self.stdout.write(self.style.WARNING(f'Failed to extract face embedding for {student_number}'))

                # Clean up temp file
                os.remove(temp_path)

            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error processing student {student_number}: {str(e)}'))
                error_count += 1

        self.stdout.write(self.style.SUCCESS(f'Successfully imported {success_count} students'))
        if error_count > 0:
            self.stdout.write(self.style.WARNING(f'Failed to import {error_count} students')) 