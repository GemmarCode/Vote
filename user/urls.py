from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView

app_name = 'user'

urlpatterns = [
    path('', views.home_view, name='mainpage'),
    path('vote/', views.voting_view, name='vote'),
    path('results/', views.results_view, name='results'),
    path('candidates/', views.candidates_view, name='candidates'),
    path('api/verify-face/', views.verify_face, name='verify_face'),
    path('api/vote/<int:candidate_id>/', views.cast_vote, name='cast_vote'),
    path('api/check-voting-status/', views.check_voting_status, name='check_voting_status'),
]
