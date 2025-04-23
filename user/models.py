from django.db import models
from django.contrib.auth.models import User
import cv2
import numpy as np
import pickle
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
import insightface
from insightface.app import FaceAnalysis
import os

# Create your models here.

class UserProfile(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
    ]

    YEAR_LEVEL_CHOICES = [
        ('1', 'First Year'),
        ('2', 'Second Year'),
        ('3', 'Third Year'),
        ('4', 'Fourth Year'),
        ('5', 'Fifth Year'),
    ]

    COLLEGES = [
        ('CAS', 'College of Arts and Sciences'),
        ('CAF', 'College of Agriculture and Forestry'),
        ('CCJE', 'College of Criminal Justice Education'),
        ('CBA', 'College of Business Administration'),
        ('CTED', 'College of Teacher Education'),
        ('CIT', 'College of Industrial Technology'),
    ]

    student_number = models.CharField(max_length=20, unique=True)
    student_name = models.CharField(max_length=100, null=True, blank=True)
    sex = models.CharField(max_length=1, choices=GENDER_CHOICES)
    year_level = models.CharField(max_length=20, choices=YEAR_LEVEL_CHOICES)
    course = models.CharField(max_length=100)
    college = models.CharField(max_length=10, choices=COLLEGES)
    school_year = models.CharField(max_length=9, help_text="Format: YYYY-YYYY")  # e.g., "2023-2024"

    class Meta:
        db_table = 'user_profile'
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
        unique_together = ['student_number', 'school_year']

    def __str__(self):
        return f"{self.student_name} - {self.student_number}"

    def get_college_display(self):
        return dict(self.COLLEGES).get(self.college, self.college)

class Candidate(models.Model):
    # Position Constants
    NATIONAL_POSITIONS = [
        ('PRESIDENT', 'President'),
        ('VICE_PRESIDENT', 'Vice President'),
        ('SECRETARY', 'Secretary'),
        ('TREASURER', 'Treasurer'),
        ('AUDITOR', 'Auditor'),
        ('PIO', 'Public Information Officer'),
        ('REPRESENTATIVE', 'Representative'),
    ]
    
    COLLEGE_POSITIONS = [
        ('GOVERNOR', 'Governor'),
        ('VICE_GOVERNOR', 'Vice Governor'),
        ('SECRETARY_COLLEGE', 'Secretary (College)'),
        ('TREASURER_COLLEGE', 'Treasurer (College)'),
        ('AUDITOR_COLLEGE', 'Auditor (College)'),
        ('PRO_COLLEGE', 'Public Information Officer (College)'),
    ]
    
    LOCAL_POSITIONS = [
        ('MAYOR', 'Mayor'),
        ('VICE_MAYOR', 'Vice Mayor'),
        ('SECRETARY_DEPT', 'Secretary (Department)'),
        ('TREASURER_DEPT', 'Treasurer (Department)'),
        ('AUDITOR_DEPT', 'Auditor (Department)'),
        ('PRO_DEPT', 'Public Information Officer (Department)'),
        ('REPRESENTATIVE_DEPT', 'Department Representative'),
    ]
    
    POSITION_CHOICES = NATIONAL_POSITIONS + COLLEGE_POSITIONS + LOCAL_POSITIONS
    
    # Essential fields
    user_profile = models.ForeignKey('UserProfile', on_delete=models.DO_NOTHING)
    position = models.CharField(max_length=50, choices=POSITION_CHOICES)
    platform = models.TextField(help_text="Describe the candidate's platform and goals", blank=True, default="")
    achievements = models.TextField(help_text="List the candidate's achievements, one per line", blank=True, default="")
    photo = models.ImageField(upload_to='candidate_photos/', blank=True, null=True)
    school_year = models.CharField(max_length=9, help_text="Format: YYYY-YYYY")
    
    # System fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['position']
        db_table = 'user_candidate'
        unique_together = ['user_profile', 'school_year']

    def __str__(self):
        return f"{self.user_profile.student_name} - {self.get_position_display()}"
        
    def get_student_number(self):
        """Helper method to get student number from related UserProfile"""
        return self.user_profile.student_number
            
    def get_college(self):
        """Helper method to get college from related UserProfile"""
        return self.user_profile.college
            
    def get_department(self):
        """Helper method to get department/course from related UserProfile"""
        return self.user_profile.course
            
    def get_year_level(self):
        """Helper method to get year level from related UserProfile"""
        return self.user_profile.year_level

def capture_face():
    """Capture face and return encoding"""
    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_encoding = None
    
    while True:
        ret, frame = video_capture.read()
        
        # Find face locations in current frame
        face_locations = face_recognition.face_locations(frame)
        
        # Draw rectangle around face
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Video', frame)
        
        # If face is detected, get encoding
        if len(face_locations) == 1:
            face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
            if face_encoding is not None:
                video_capture.release()
                cv2.destroyAllWindows()
                return pickle.dumps(face_encoding)
        
        # Press 'q' to quit without capturing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    return None

def verify_face(stored_encoding):
    """Verify face against stored encoding"""
    if stored_encoding is None:
        return False
        
    video_capture = cv2.VideoCapture(0)
    stored_encoding = pickle.loads(stored_encoding)
    
    while True:
        ret, frame = video_capture.read()
        
        face_locations = face_recognition.face_locations(frame)
        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Get encoding of detected face
            face_encoding = face_recognition.face_encodings(frame, [face_locations[0]])[0]
            
            # Compare with stored encoding
            matches = face_recognition.compare_faces([stored_encoding], face_encoding)
            
            if matches[0]:
                video_capture.release()
                cv2.destroyAllWindows()
                return True
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    return False

class Vote(models.Model):
    user_profile = models.ForeignKey('UserProfile', on_delete=models.CASCADE)
    candidate = models.ForeignKey('Candidate', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    school_year = models.CharField(max_length=9, help_text="Format: YYYY-YYYY")

    class Meta:
        db_table = 'user_vote'
        unique_together = ['user_profile', 'candidate', 'school_year']

class VotingPhase(models.Model):
    PHASE_CHOICES = [
        ('NOT_STARTED', 'Not Started'),
        ('ONGOING', 'Ongoing'),
        ('ENDED', 'Ended')
    ]
    
    phase = models.CharField(max_length=20, choices=PHASE_CHOICES, default='NOT_STARTED')
    start_date = models.DateTimeField(null=True, blank=True)
    end_date = models.DateTimeField(null=True, blank=True)
    
    def is_active(self):
        if self.phase != 'ONGOING':
            return False
        now = timezone.now()
        return self.start_date <= now <= self.end_date

class FaceData(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='face_data')
    face_embedding = models.BinaryField()  # Store face embedding as binary data
    face_image = models.ImageField(upload_to='face_data/')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if not self.face_embedding and self.face_image:
            # Initialize face analyzer
            app = FaceAnalysis(name='buffalo_l')
            app.prepare(ctx_id=-1, det_size=(640, 640))
            
            # Read and process the image
            img = insightface.data.load_image(self.face_image.path)
            faces = app.get(img)
            
            if len(faces) == 1:
                # Get the face embedding
                face = faces[0]
                self.face_embedding = face.embedding.tobytes()
            else:
                raise ValueError("Image must contain exactly one face")
        
        super().save(*args, **kwargs)

    def get_embedding(self):
        if self.face_embedding:
            return np.frombuffer(self.face_embedding, dtype=np.float32)
        return None

    def __str__(self):
        return f"Face data for {self.user.username}"

# Deprecated - Use Candidate model instead
# This model is kept for backward compatibility but should not be used for new code
class CandidateProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='candidate_profile')
    position = models.CharField(max_length=100)
    achievements = models.TextField(help_text="List the candidate's achievements, one per line")
    platform = models.TextField(help_text="Describe the candidate's platform and goals")
    photo = models.ImageField(upload_to='candidate_photos/', null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.get_full_name()} - {self.position}"

    class Meta:
        ordering = ['-created_at']

class VerificationCode(models.Model):
    student_number = models.CharField(max_length=20)
    code = models.CharField(max_length=6, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_used = models.BooleanField(default=False)

    def __str__(self):
        return f"Code for {self.student_number}"

    def is_valid(self):
        return not self.is_used and timezone.now() <= self.expires_at

