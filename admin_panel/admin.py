from django.contrib import admin
from user.models import VerificationCode
import random
import string
from datetime import timedelta
from django.utils import timezone

@admin.register(VerificationCode)
class VerificationCodeAdmin(admin.ModelAdmin):
    list_display = ('student_number', 'code', 'created_at', 'expires_at', 'is_used', 'is_valid')
    list_filter = ('is_used', 'created_at')
    search_fields = ('student_number', 'code')
    readonly_fields = ('code', 'created_at')
    actions = ['generate_new_codes']

    def generate_new_codes(self, request, queryset):
        for code_obj in queryset:
            # Generate a new 6-digit code
            new_code = ''.join(random.choices(string.digits, k=6))
            # Set expiration to 1 hour from now
            expires_at = timezone.now() + timedelta(hours=1)
            
            code_obj.code = new_code
            code_obj.expires_at = expires_at
            code_obj.is_used = False
            code_obj.save()
        
        self.message_user(request, f"Generated new codes for {queryset.count()} students.")
    generate_new_codes.short_description = "Generate new codes for selected students"

    def save_model(self, request, obj, form, change):
        if not change:  # Only for new objects
            # Generate a random 6-digit code
            obj.code = ''.join(random.choices(string.digits, k=6))
            # Set expiration to 1 hour from now
            obj.expires_at = timezone.now() + timedelta(hours=1)
        super().save_model(request, obj, form, change)
