from django.db import models
from django.contrib.auth.models import User
import cv2
import numpy as np
import pickle
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone

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

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    student_id = models.CharField(max_length=20, unique=True)
    college = models.CharField(max_length=10, choices=COLLEGES)
    department = models.CharField(max_length=100)
    year_level = models.CharField(max_length=20, choices=YEAR_LEVEL_CHOICES)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    age = models.IntegerField(validators=[
        MinValueValidator(16),
        MaxValueValidator(99)
    ])
    contact_number = models.CharField(max_length=15)
    face_data = models.BinaryField(null=True, blank=True, editable=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    profile_picture = models.ImageField(upload_to='profile_pictures/', null=True, blank=True)

    class Meta:
        db_table = 'user_profile'
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'

    def __str__(self):
        return f"{self.user.username}'s Profile"

    def save(self, *args, **kwargs):
        # Validate age
        if self.age < 16 or self.age > 99:
            raise ValueError("Age must be between 16 and 99")
        
        # Validate student ID format if needed
        # Add any other custom validation here
        
        super().save(*args, **kwargs)

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
    # National Positions
    NATIONAL_POSITIONS = [
        ('PRESIDENT', 'President'),
        ('VICE_PRESIDENT', 'Vice President'),
        ('SECRETARY', 'Secretary'),
        ('TREASURER', 'Treasurer'),
        ('AUDITOR', 'Auditor'),
        ('PIO', 'Public Information Officer'),
        ('REPRESENTATIVE', 'Representative'),
        ('GOVERNOR', 'Governor'),
        ('VICE_GOVERNOR', 'Vice Governor'),
        ('SEC_GOVERNOR', 'Secretary'),
        ('TRES_GOVERNOR', 'Treasurer'),
        ('AUD_GOVERNOR', 'Auditor'),
        ('PIO_GOVERNOR', 'Public Information Officer')
    ]

    # Local Positions
    LOCAL_POSITIONS = [
        ('MAYOR', 'Mayor'),
        ('VICE_MAYOR', 'Vice Mayor'),
        ('SEC_LOCAL', 'Secretary'),
        ('TRES_LOCAL', 'Treasurer'),
        ('AUD_LOCAL', 'Auditor'),
        ('PIO_LOCAL', 'Public Information Officer'),
        ('REP_LOCAL', 'Representative')
    ]

    # All positions combined
    POSITION_CHOICES = NATIONAL_POSITIONS + LOCAL_POSITIONS

    COLLEGE_CHOICES = [
        ('CAS', 'College of Arts and Sciences'),
        ('CAF', 'College of Agriculture and Forestry'),
        ('CCJE', 'College of Criminal Justice Education'),
        ('CBA', 'College of Business Administration'),
        ('CTED', 'College of Teacher Education'),
        ('CIT', 'College of Industrial Technology'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    position = models.CharField(max_length=50, choices=POSITION_CHOICES)
    college = models.CharField(max_length=100, choices=COLLEGE_CHOICES)
    platform = models.TextField()
    achievements = models.TextField(null=True, blank=True)
    photo = models.ImageField(upload_to='candidate_photos/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    department = models.CharField(max_length=100)
    year_level = models.CharField(max_length=20)
    approved = models.BooleanField(default=False)

    @property
    def is_national(self):
        """Check if the candidate is running for a national position"""
        return self.position in [pos[0] for pos in self.NATIONAL_POSITIONS]

    @property
    def is_local(self):
        """Check if the candidate is running for a local position"""
        return any(position[0] == self.position for position in self.LOCAL_POSITIONS)

    def __str__(self):
        return f"{self.user.get_full_name()} - {self.get_position_display()}"

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

