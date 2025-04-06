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
    path('results/', views.results, name='results'),
    path('reports/', views.generate_report, name='generate_report'),
]