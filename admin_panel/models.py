from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.exceptions import ValidationError
from django.db.models.signals import post_save
from django.dispatch import receiver

# Create your models here.

class AdminActivity(models.Model):
    admin_user = models.ForeignKey(User, on_delete=models.CASCADE)
    action = models.CharField(max_length=255)
    details = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Admin Activities'

    def __str__(self):
        return f"{self.admin_user.username} - {self.action} - {self.created_at}"

class ElectionSettings(models.Model):
    school_year = models.CharField(max_length=9, help_text="Format: YYYY-YYYY")  # e.g., "2023-2024"
    is_active = models.BooleanField(default=True)
    candidacy_deadline = models.DateTimeField(null=True, blank=True)
    voting_start = models.DateTimeField(null=True, blank=True)
    voting_end = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Election Settings'
        verbose_name_plural = 'Election Settings'

    def is_candidacy_open(self):
        if not self.candidacy_deadline or not self.is_active:
            return False
        return timezone.now() <= self.candidacy_deadline

    def is_voting_open(self):
        if not self.voting_start or not self.voting_end or not self.is_active:
            return False
        now = timezone.now()
        return self.voting_start <= now <= self.voting_end

    def clean(self):
        # Validate school year format
        if self.school_year:
            try:
                start_year, end_year = map(int, self.school_year.split('-'))
                if end_year != start_year + 1:
                    raise ValidationError({'school_year': 'End year must be start year + 1'})
                if len(str(start_year)) != 4 or len(str(end_year)) != 4:
                    raise ValidationError({'school_year': 'Years must be in YYYY format'})
            except ValueError:
                raise ValidationError({'school_year': 'School year must be in YYYY-YYYY format'})

    def save(self, *args, **kwargs):
        if self.is_active:
            # Deactivate all other school years
            ElectionSettings.objects.exclude(id=self.id).update(is_active=False)
        super().save(*args, **kwargs)

class CommitteeAccount(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='committee_profile')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.get_full_name()} ({self.user.username})"
    
    class Meta:
        verbose_name = "Committee Account"
        verbose_name_plural = "Committee Accounts"

@receiver(post_save, sender=CommitteeAccount)
def set_committee_permissions(sender, instance, created, **kwargs):
    """Set up the committee user with limited permissions"""
    if created:
        # Make sure the user is staff but not superuser
        instance.user.is_staff = True
        instance.user.is_superuser = False
        instance.user.save()
