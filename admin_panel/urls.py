from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView

app_name = 'admin_panel'  # This is important for namespacing

urlpatterns = [
    path('', views.admin_panel_login, name='admin_panel_login'),
    path('logout/', LogoutView.as_view(next_page='admin_panel:admin_panel_login'), name='logout'),
    path('home', views.dashboard, name='dashboard'),
    path('users/', views.manage_users, name='manage_users'),
    path('users/register/', views.register_user, name='register_user'),
    path('users/import/', views.import_users, name='import_users'),
    path('users/import-photos/', views.import_photos, name='import_photos'),
    path('candidates/', views.manage_candidates, name='manage_candidates'),
    path('elections/', views.manage_elections, name='manage_elections'),
    path('results/', views.admin_results, name='admin_results'),
    path('reports/', views.generate_report, name='generate_report'),
    path('verification-codes/', views.verification_codes_view, name='verification_codes'),
    path('generate-code/', views.generate_code, name='generate_code'),
    path('regenerate-code/<int:code_id>/', views.regenerate_code, name='regenerate_code'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('settings/', views.settings, name='settings'),
    path('create-committee/', views.create_committee, name='create_committee'),
    path('delete-committee/<int:user_id>/', views.delete_committee, name='delete_committee'),
    path('change-password/', views.change_password, name='change_password'),
    path('committee-settings/', views.committee_change_password, name='committee_change_password'),
    path('user-activity-logs/<int:user_id>/', views.user_activity_logs, name='user_activity_logs'),
    path('toggle-committee-status/<int:committee_id>/', views.toggle_committee_status, name='toggle_committee_status'),
    path('activity-logs/', views.activity_logs, name='activity_logs'),
]