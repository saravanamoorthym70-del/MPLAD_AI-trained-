from django.urls import path
from analytics import views

urlpatterns = [
    path('dashboard/', views.DashboardAnalyticsAPIView.as_view(), name='api_dashboard'),
    path('states/', views.StateAnalyticsAPIView.as_view(), name='api_states'),
    path('categories/', views.CategoryAnalyticsAPIView.as_view(), name='api_categories'),
]
