from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView
from .views import verify_face

app_name = 'user'

urlpatterns = [
    path('', views.home_view, name='mainpage'),
    path('vote/', views.voting_view, name='vote'),
    path('results/', views.results_view, name='results'),
    path('candidates/', views.candidates_view, name='candidates'),
    path('face-upload/', views.face_upload_view, name='face_upload'),
    path('api/verify-student-id/', views.verify_student_id, name='verify_student_id'),
    path('api/submit-face-images/', views.submit_face_images, name='submit_face_images'),
    path('api/verify-code/', views.verify_code, name='verify_code'),
    path('api/vote/<int:candidate_id>/', views.cast_vote, name='cast_vote'),
    path('api/check-voting-status/', views.check_voting_status, name='check_voting_status'),
    path('api/submit-all-votes/', views.submit_all_votes, name='submit_all_votes'),
    path('api/verify-face/', verify_face, name='verify_face'),
    path('api/check-already-voted/', views.check_already_voted, name='check_already_voted'),
    path('supervisor-login/', views.supervisor_login, name='supervisor_login'),
    path('supervisor-logout/', views.supervisor_logout, name='supervisor_logout'),
]
