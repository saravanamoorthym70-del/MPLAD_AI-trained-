from django.urls import path
from projects import views

urlpatterns = [
    path('', views.ProjectListCreateAPIView.as_view(), name='api_project_list_create'),
    path('<str:project_id>/', views.ProjectDetailAPIView.as_view(), name='api_project_detail'),
    path('upload-csv/', views.CSVUploadAPIView.as_view(), name='api_csv_upload'),
]
