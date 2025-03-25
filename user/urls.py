from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('', views.home_view, name='home'),
    path('candidates/', views.candidates_view, name='candidates'),
    path('results/', views.results_view, name='results'),
    path('admin-panel/login/', views.admin_panel_login, name='admin_panel_login'),
    path('logout/', LogoutView.as_view(next_page='home'), name='logout'),
    path('file-candidacy/', views.file_candidacy, name='file_candidacy'),
    path('vote/', views.voting_view, name='vote'),
    path('api/verify-face/', views.verify_face, name='verify_face'),
    path('api/vote/<int:candidate_id>/', views.cast_vote, name='cast_vote'),
    path('profile/', views.profile_settings, name='profile_settings'),
]
