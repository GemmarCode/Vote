from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.html import format_html
from django.db.models import Q
from .models import UserProfile, Vote, Candidate, VotingPhase
from django import forms

class CandidateAdminForm(forms.ModelForm):
    class Meta:
        model = Candidate
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super(CandidateAdminForm, self).__init__(*args, **kwargs)
        # Add autocomplete for user_profile field
        if 'user_profile' in self.fields:
            self.fields['user_profile'].widget.attrs['class'] = 'select2'

@admin.register(Candidate)
class CandidateAdmin(admin.ModelAdmin):
    form = CandidateAdminForm
    list_display = ('user_profile', 'position', 'display_college', 'display_department', 'display_year_level', 'created_at')
    list_filter = ('position',)
    search_fields = ('user_profile__student_number', 'user_profile__student_name', 'position')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user_profile', 'position', 'photo')
        }),
        ('Candidate Details', {
            'fields': ('platform', 'achievements'),
            'description': 'Candidate platform and achievements'
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def display_college(self, obj):
        return obj.get_college()
    display_college.short_description = 'College'
    
    def display_department(self, obj):
        return obj.get_department()
    display_department.short_description = 'Department'
    
    def display_year_level(self, obj):
        return obj.get_year_level()
    display_year_level.short_description = 'Year Level'
    
    def save_model(self, request, obj, form, change):
        if not change:  # If this is a new candidate
            obj.created_by = request.user
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)
