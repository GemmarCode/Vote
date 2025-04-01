from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.html import format_html
from django.db.models import Q
from .models import UserProfile, Vote, Candidate, VotingPhase, CandidateProfile, FaceData
from django import forms

@admin.register(CandidateProfile)
class CandidateProfileAdmin(admin.ModelAdmin):
    list_display = ('get_name', 'position', 'college', 'department', 'year_level', 'is_active', 'created_at')
    list_filter = ('position', 'is_active', 'created_at')
    search_fields = ('user__first_name', 'user__last_name', 'position', 'user__userprofile__college')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'position', 'photo', 'is_active')
        }),
        ('Candidate Details', {
            'fields': ('achievements', 'platform')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_name(self, obj):
        return obj.user.get_full_name()
    get_name.short_description = 'Name'
    
    def college(self, obj):
        return obj.user.userprofile.college
    college.short_description = 'College'
    
    def department(self, obj):
        return obj.user.userprofile.department
    department.short_description = 'Department'
    
    def year_level(self, obj):
        return obj.user.userprofile.year_level
    year_level.short_description = 'Year Level'
    
    def get_template(self, request, obj=None, **kwargs):
        if obj is None:
            return 'admin/user/candidateprofile/change_form.html'
        return 'admin/user/candidateprofile/change_form.html'
    
    def changelist_view(self, request, extra_context=None):
        # Get filter parameters
        position = request.GET.get('position', '')
        status = request.GET.get('status', '')
        
        # Get all candidates
        queryset = self.get_queryset(request)
        
        # Apply filters
        if position:
            queryset = queryset.filter(position=position)
        if status:
            queryset = queryset.filter(is_active=(status == 'active'))
        
        # Get unique positions for filter dropdown
        positions = CandidateProfile.objects.values_list('position', flat=True).distinct()
        
        context = {
            'candidates': queryset,
            'positions': positions,
            'selected_position': position,
            'selected_status': status,
        }
        context.update(extra_context or {})
        
        return super().changelist_view(request, context=context)
    
    def get_list_template(self):
        return 'admin/user/candidateprofile/change_list.html'

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
        
    def clean(self):
        cleaned_data = super().clean()
        position = cleaned_data.get('position')
        college = cleaned_data.get('college')
        department = cleaned_data.get('department')
        year_level = cleaned_data.get('year_level')
        
        # Validate based on position type
        if position in ['President', 'Vice President', 'Secretary', 'Treasurer', 'Auditor', 'Public Information Officer', 'Representative']:
            # National positions don't need department/year validation
            if department or year_level:
                raise forms.ValidationError("National positions should not have department or year level.")
                
        elif position in ['Governor', 'Vice Governor', 'Secretary (College)', 'Treasurer (College)', 'Auditor (College)', 'Public Information Officer (College)']:
            # College positions need college but not department/year
            if not college:
                raise forms.ValidationError("College positions must have a college specified.")
            if department or year_level:
                raise forms.ValidationError("College positions should not have department or year level.")
                
        else:
            # Department positions need all fields
            if not all([college, department, year_level]):
                raise forms.ValidationError("Department positions must have college, department, and year level specified.")
        
        return cleaned_data

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
