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

    class Meta:
        db_table = 'user_profile'
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'

    def __str__(self):
        return f"{self.student_name} - {self.student_number}"

    def get_college_display(self):
        return dict(self.COLLEGES).get(self.college, self.college)

    def set_face_data(self, data):
        """Helper method to properly set face data"""
        if isinstance(data, str):
            self.face_data = data.encode('utf-8')
        elif isinstance(data, bytes):
            self.face_data = data
        elif isinstance(data, np.ndarray):
            self.face_data = data.tobytes()
    
    def get_face_data(self):
        """Helper method to get face data as numpy array"""
        if not self.face_data:
            return None
        try:
            return np.frombuffer(self.face_data, dtype=np.uint8).reshape((200, 200))
        except:
            return None

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
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    position = models.CharField(max_length=50, choices=POSITION_CHOICES)
    college = models.CharField(max_length=100, blank=True, null=True)
    department = models.CharField(max_length=100, blank=True, null=True)
    year_level = models.IntegerField(blank=True, null=True)
    photo = models.ImageField(upload_to='candidate_photos/', blank=True, null=True)
    platform = models.TextField(help_text="Describe the candidate's platform and goals", blank=True, default="")
    achievements = models.TextField(help_text="List the candidate's achievements, one per line", blank=True, default="")
    approved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['position', 'college', 'department', 'year_level']
        unique_together = [
            ('position', 'college', 'department', 'year_level')
        ]

    def __str__(self):
        name = f"{self.user.get_full_name()} - {self.position}"
        if self.college:
            name += f" ({self.college}"
            if self.department:
                name += f", {self.department}"
            if self.year_level:
                name += f", Year {self.year_level}"
            name += ")"
        return name

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
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'candidate')  # Prevent double voting

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

