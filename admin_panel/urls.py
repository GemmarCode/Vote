from django.urls import path
from . import views

app_name = 'admin_panel'  # This is important for namespacing

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('users/', views.manage_users, name='manage_users'),
    path('users/register/', views.register_user, name='register_user'),
    path('users/import/', views.import_users, name='import_users'),
    path('candidates/', views.manage_candidates, name='manage_candidates'),
    path('elections/', views.manage_elections, name='manage_elections'),
    path('results/', views.results, name='results'),
]