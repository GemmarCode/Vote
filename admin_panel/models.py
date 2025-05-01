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
    timestamp = models.DateTimeField(auto_now_add=True)
    details = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.admin_user.username} - {self.action} at {self.timestamp}"

class ElectionSettings(models.Model):
    school_year = models.CharField(max_length=9, unique=True, help_text="Format: YYYY-YYYY")
    voting_date = models.DateField(null=True, blank=True)
    voting_time_start = models.TimeField(null=True, blank=True)
    voting_time_end = models.TimeField(null=True, blank=True)
    is_active = models.BooleanField(default=False)

    def __str__(self):
        return f"Settings for {self.school_year} (Active: {self.is_active})"

    def clean(self):
        # Ensure voting end time is after voting start time on the same date
        if self.voting_time_start and self.voting_time_end and self.voting_time_end <= self.voting_time_start:
            raise ValidationError("Voting end time must be after voting start time.")
        # Check school year format
        if self.school_year:
            try:
                start_year, end_year = map(int, self.school_year.split('-'))
                if end_year != start_year + 1:
                    raise ValidationError("School year format must be YYYY-YYYY with consecutive years.")
            except (ValueError, IndexError):
                raise ValidationError("Invalid school year format. Use YYYY-YYYY.")

    def save(self, *args, **kwargs):
        # If this is being set to active, deactivate all others
        if self.is_active:
            ElectionSettings.objects.filter(is_active=True).exclude(pk=self.pk).update(is_active=False)
        # Ensure at least one setting is active if this one is being deactivated
        elif not self.is_active and ElectionSettings.objects.filter(is_active=True).count() == 1 and ElectionSettings.objects.get(is_active=True).pk == self.pk:
            pass # Or raise ValidationError("At least one school year must be active.")
        super().save(*args, **kwargs)

    @property
    def is_voting_open(self):
        now = timezone.localtime(timezone.now())  # Convert to local time
        if not (self.voting_date and self.voting_time_start and self.voting_time_end):
            return False
        voting_start = timezone.make_aware(timezone.datetime.combine(self.voting_date, self.voting_time_start))
        voting_end = timezone.make_aware(timezone.datetime.combine(self.voting_date, self.voting_time_end))
        voting_start = timezone.localtime(voting_start)
        voting_end = timezone.localtime(voting_end)
        return voting_start <= now < voting_end

class CommitteeAccount(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='committee_profile')
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Committee Account for {self.user.username}"

@receiver(post_save, sender=CommitteeAccount)
def set_committee_permissions(sender, instance, created, **kwargs):
    """Set up the committee user with limited permissions"""
    if created:
        # Make sure the user is staff but not superuser
        instance.user.is_staff = True
        instance.user.is_superuser = False
        instance.user.save()
