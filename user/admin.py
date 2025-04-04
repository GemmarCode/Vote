from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.html import format_html
from django.db.models import Q
from .models import UserProfile, Vote, Candidate, VotingPhase, FaceData
from django import forms

@admin.register(FaceData)
class FaceDataAdmin(admin.ModelAdmin):
    list_display = ('user', 'created_at', 'updated_at')
    search_fields = ('user__username', 'user__first_name', 'user__last_name')
    readonly_fields = ('created_at', 'updated_at')
    fieldsets = (
        (None, {
            'fields': ('user', 'face_image')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

class CandidateAdminForm(forms.ModelForm):
    class Meta:
        model = Candidate
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super(CandidateAdminForm, self).__init__(*args, **kwargs)
        # Add autocomplete for user field
        if 'user' in self.fields:
            self.fields['user'].widget.attrs['class'] = 'select2'

@admin.register(Candidate)
class CandidateAdmin(admin.ModelAdmin):
    form = CandidateAdminForm
    list_display = ('user', 'position', 'college', 'department', 'year_level', 'approved', 'created_at')
    list_filter = ('position', 'college', 'department', 'year_level', 'approved')
    search_fields = ('user__username', 'user__first_name', 'user__last_name', 'position', 'college', 'department')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'position', 'photo')
        }),
        ('Academic Details', {
            'fields': ('college', 'department', 'year_level'),
            'description': 'Fill these based on position requirements'
        }),
        ('Candidate Details', {
            'fields': ('platform', 'achievements'),
            'description': 'Candidate platform and achievements'
        }),
        ('Status', {
            'fields': ('approved',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def save_model(self, request, obj, form, change):
        if not change:  # If this is a new candidate
            obj.created_by = request.user
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)
