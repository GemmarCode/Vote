from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# Create your models here.

class AdminActivity(models.Model):
    ACTION_CHOICES = [
        ('CREATE', 'Created'),
        ('UPDATE', 'Updated'),
        ('DELETE', 'Deleted'),
        ('LOGIN', 'Logged In'),
        ('OTHER', 'Other Action'),
    ]
    
    timestamp = models.DateTimeField(auto_now_add=True)
    admin_user = models.ForeignKey(User, on_delete=models.CASCADE)
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    action_model = models.CharField(max_length=50)  # The model being acted upon
    description = models.TextField()
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'Admin Activities'
    
    def __str__(self):
        return f"{self.admin_user.username} {self.get_action_display()} {self.action_model} at {self.timestamp}"

class ElectionSettings(models.Model):
    candidacy_deadline = models.DateTimeField(null=True, blank=True)
    voting_start = models.DateTimeField(null=True, blank=True)
    voting_end = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Election Settings'
        verbose_name_plural = 'Election Settings'

    def is_candidacy_open(self):
        if not self.candidacy_deadline:
            return False
        return timezone.now() <= self.candidacy_deadline

    def is_voting_open(self):
        if not self.voting_start or not self.voting_end:
            return False
        now = timezone.now()
        return self.voting_start <= now <= self.voting_end
