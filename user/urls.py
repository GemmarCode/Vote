from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView
from .views_backup import verify_face

app_name = 'user'

urlpatterns = [
    path('', views.home_view, name='mainpage'),
    path('vote/', views.voting_view, name='vote'),
    path('results/', views.results_view, name='results'),
    path('candidates/', views.candidates_view, name='candidates'),
    path('api/verify-code/', views.verify_code, name='verify_code'),
    path('api/vote/<int:candidate_id>/', views.cast_vote, name='cast_vote'),
    path('api/check-voting-status/', views.check_voting_status, name='check_voting_status'),
    path('api/submit-all-votes/', views.submit_all_votes, name='submit_all_votes'),
    path('api/verify-face/', verify_face, name='verify_face'),
]
