"""Root URL configuration for MPLAD_AI."""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from django.contrib.auth import views as auth_django_views
from accounts import views as account_views
from projects import views as project_views
from analytics import views as analytics_views

urlpatterns = [
    path('admin/', admin.site.urls),

    # ---- Frontend pages ----
    path('', RedirectView.as_view(url='/dashboard/', permanent=False)),
    path('login/', account_views.login_page, name='login_page'),
    path('logout/', account_views.logout_page, name='logout_page'),
    path('dashboard/', analytics_views.dashboard_page, name='dashboard_page'),
    path('projects/', project_views.projects_page, name='projects_page'),
    path('projects/<str:project_id>/', project_views.project_detail_page, name='project_detail_page'),
    path('anomalies/', project_views.anomalies_page, name='anomalies_page'),
    path('fraud-risk/', project_views.fraud_risk_page, name='fraud_risk_page'),
    path('analytics/', analytics_views.analytics_page, name='analytics_page'),
    path('reports/', analytics_views.reports_page, name='reports_page'),
    path('upload/', project_views.upload_page, name='upload_page'),

    # ---- REST APIs ----
    path('api/auth/', include('accounts.urls')),
    path('api/projects/', include('projects.urls')),
    path('api/analytics/', include('analytics.urls')),

    path('api/anomalies/', project_views.AnomalyListAPIView.as_view(), name='api_anomaly_list'),
    path('api/anomalies/<str:anomaly_id>/', project_views.AnomalyDetailAPIView.as_view(), name='api_anomaly_detail'),

    path('api/ml/predict/<str:project_id>/', project_views.MLPredictAPIView.as_view(), name='api_ml_predict'),
    path('api/ml/analyze/', project_views.MLAnalyzeAPIView.as_view(), name='api_ml_analyze'),

    path('api/reports/', analytics_views.ReportsAPIView.as_view(), name='api_reports'),
]
