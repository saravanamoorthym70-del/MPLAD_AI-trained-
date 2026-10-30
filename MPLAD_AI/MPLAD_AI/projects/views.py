"""
Projects app views.

REST API (Stage 20/21):
    GET/POST   /api/projects/
    GET/PUT/DELETE /api/projects/<id>/
    GET        /api/anomalies/
    GET        /api/anomalies/<id>/
    PATCH      /api/anomalies/<id>/review/
    GET        /api/ml/predict/<project_id>/
    POST       /api/ml/analyze/
    POST       /api/projects/upload-csv/

Page views: server-rendered dashboard pages (Stages 13-19).
"""
import io
import datetime
import pandas as pd
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.paginator import Paginator
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions

from projects.models import Project, Anomaly, MLPrediction, PROJECT_CATEGORIES, PROJECT_STATUSES
from projects.serializers import ProjectWriteSerializer, AnomalyReviewSerializer
from projects.services import import_csv_to_mongo, run_ml_analysis_and_save, analyze_single_project
from accounts.decorators import session_login_required


# ==================================================================
# REST API — Projects CRUD (Stage 20)
# ==================================================================

class ProjectListCreateAPIView(APIView):
    def get_permissions(self):
        # GET is used by the session-authenticated dashboard's map/JS (Leaflet
        # fetch calls); POST (creating a project) still requires a valid JWT.
        if self.request.method == 'GET':
            return [permissions.AllowAny()]
        return [permissions.IsAuthenticated()]

    def get(self, request):
        qs = Project.objects()
        state = request.query_params.get('state')
        category = request.query_params.get('category')
        risk = request.query_params.get('risk')
        status_filter = request.query_params.get('status')
        search = request.query_params.get('search')

        if state:
            qs = qs.filter(state=state)
        if category:
            qs = qs.filter(project_category=category)
        if status_filter:
            qs = qs.filter(project_status=status_filter)
        if search:
            qs = qs.filter(project_name__icontains=search)

        projects = list(qs)
        data = [p.to_dict() for p in projects]

        if risk:
            preds = {pr.project_id: pr for pr in MLPrediction.objects(project_id__in=[p['project_id'] for p in data])}
            data = [d for d in data if preds.get(d['project_id']) and preds[d['project_id']].risk_level == risk]

        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 25))
        paginator = Paginator(data, page_size)
        page_obj = paginator.get_page(page)

        return Response({
            'count': paginator.count,
            'total_pages': paginator.num_pages,
            'page': page,
            'results': list(page_obj),
        })

    def post(self, request):
        serializer = ProjectWriteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        if Project.objects(project_id=data['project_id']).first():
            return Response({'detail': 'project_id already exists'}, status=status.HTTP_400_BAD_REQUEST)

        project = Project(**data)
        project.save()
        return Response(project.to_dict(), status=status.HTTP_201_CREATED)


class ProjectDetailAPIView(APIView):
    def get_object(self, project_id):
        return Project.objects(project_id=project_id).first()

    def get(self, request, project_id):
        project = self.get_object(project_id)
        if not project:
            return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)
        prediction = MLPrediction.objects(project_id=project_id).first()
        anomalies = Anomaly.objects(project_id=project_id)
        return Response({
            'project': project.to_dict(),
            'ml_prediction': prediction.to_dict() if prediction else None,
            'anomalies': [a.to_dict() for a in anomalies],
        })

    def put(self, request, project_id):
        project = self.get_object(project_id)
        if not project:
            return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)
        serializer = ProjectWriteSerializer(data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        for k, v in serializer.validated_data.items():
            setattr(project, k, v)
        project.updated_at = datetime.datetime.utcnow()
        project.save()
        return Response(project.to_dict())

    def delete(self, request, project_id):
        project = self.get_object(project_id)
        if not project:
            return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)
        if getattr(request.user, 'role', None) != 'Admin':
            return Response({'detail': 'Only Admin can delete projects'}, status=status.HTTP_403_FORBIDDEN)
        project.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# ==================================================================
# REST API — Anomalies (Stage 20)
# ==================================================================

class AnomalyListAPIView(APIView):
    def get(self, request):
        qs = Anomaly.objects()
        severity = request.query_params.get('severity')
        anomaly_type = request.query_params.get('type')
        review_status = request.query_params.get('review_status')
        if severity:
            qs = qs.filter(severity=severity)
        if anomaly_type:
            qs = qs.filter(anomaly_type=anomaly_type)
        if review_status:
            qs = qs.filter(review_status=review_status)
        qs = qs.order_by('-anomaly_score')
        return Response([a.to_dict() for a in qs])


class AnomalyDetailAPIView(APIView):
    def get(self, request, anomaly_id):
        anomaly = Anomaly.objects(anomaly_id=anomaly_id).first()
        if not anomaly:
            return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)
        return Response(anomaly.to_dict())

    def patch(self, request, anomaly_id):
        anomaly = Anomaly.objects(anomaly_id=anomaly_id).first()
        if not anomaly:
            return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)
        serializer = AnomalyReviewSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        anomaly.review_status = serializer.validated_data['review_status']
        anomaly.save()
        return Response(anomaly.to_dict())


# ==================================================================
# REST API — ML (Stage 21)
# ==================================================================

class MLPredictAPIView(APIView):
    def get(self, request, project_id):
        prediction = MLPrediction.objects(project_id=project_id).first()
        if not prediction:
            prediction = analyze_single_project(project_id)
            if not prediction:
                return Response({'detail': 'Project not found or analysis failed'}, status=status.HTTP_404_NOT_FOUND)
        return Response(prediction.to_dict())


class MLAnalyzeAPIView(APIView):
    def post(self, request):
        project_id = request.data.get('project_id')
        result = run_ml_analysis_and_save(project_ids=[project_id] if project_id else None)
        if not result.get('success'):
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
        if project_id:
            prediction = MLPrediction.objects(project_id=project_id).first()
            if prediction:
                payload = prediction.to_dict()
                payload['reasons'] = prediction.reasons
                return Response(payload)
        return Response(result)


class CSVUploadAPIView(APIView):
    def post(self, request):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({'detail': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            df = pd.read_csv(file_obj)
        except Exception as e:  # noqa: BLE001
            return Response({'detail': f'Could not parse CSV: {e}'}, status=status.HTTP_400_BAD_REQUEST)

        result = import_csv_to_mongo(df)
        if result.get('success') and result.get('inserted', 0) + result.get('updated', 0) > 0:
            run_ml_analysis_and_save()
        return Response(result)


# ==================================================================
# Server-rendered pages (Stages 13, 15-19)
# ==================================================================

@session_login_required
def projects_page(request):
    qs = Project.objects()
    context_filters = {
        'state': request.GET.get('state', ''),
        'category': request.GET.get('category', ''),
        'status': request.GET.get('status', ''),
        'search': request.GET.get('search', ''),
    }
    if context_filters['state']:
        qs = qs.filter(state=context_filters['state'])
    if context_filters['category']:
        qs = qs.filter(project_category=context_filters['category'])
    if context_filters['status']:
        qs = qs.filter(project_status=context_filters['status'])
    if context_filters['search']:
        qs = qs.filter(project_name__icontains=context_filters['search'])

    projects = list(qs)
    predictions = {p.project_id: p for p in MLPrediction.objects(project_id__in=[p.project_id for p in projects])}

    rows = []
    for p in projects:
        pred = predictions.get(p.project_id)
        rows.append({
            'p': p,
            'risk_score': pred.overall_risk_score if pred else None,
            'risk_level': pred.risk_level if pred else 'Not Analyzed',
        })

    risk_filter = request.GET.get('risk', '')
    if risk_filter:
        rows = [r for r in rows if r['risk_level'] == risk_filter]

    paginator = Paginator(rows, 20)
    page_obj = paginator.get_page(request.GET.get('page', 1))

    states = sorted(set(p.state for p in Project.objects.only('state')))
    return render(request, 'projects.html', {
        'page_obj': page_obj,
        'filters': context_filters,
        'states': states,
        'categories': PROJECT_CATEGORIES,
        'statuses': PROJECT_STATUSES,
        'risk_filter': risk_filter,
    })


@session_login_required
def project_detail_page(request, project_id):
    project = Project.objects(project_id=project_id).first()
    if not project:
        return redirect('projects_page')
    prediction = MLPrediction.objects(project_id=project_id).first()
    anomalies = Anomaly.objects(project_id=project_id).order_by('-detected_at')
    return render(request, 'project_detail.html', {
        'project': project,
        'prediction': prediction,
        'anomalies': anomalies,
    })


@session_login_required
def anomalies_page(request):
    qs = Anomaly.objects()
    severity = request.GET.get('severity', '')
    anomaly_type = request.GET.get('type', '')
    review_status = request.GET.get('review_status', '')
    if severity:
        qs = qs.filter(severity=severity)
    if anomaly_type:
        qs = qs.filter(anomaly_type=anomaly_type)
    if review_status:
        qs = qs.filter(review_status=review_status)
    qs = qs.order_by('-anomaly_score')

    if request.method == 'POST' and request.session.get('role') in ('Admin', 'Auditor'):
        anomaly_id = request.POST.get('anomaly_id')
        new_status = request.POST.get('review_status')
        anomaly = Anomaly.objects(anomaly_id=anomaly_id).first()
        if anomaly and new_status:
            anomaly.review_status = new_status
            anomaly.save()
        return redirect('anomalies_page')

    paginator = Paginator(list(qs), 25)
    page_obj = paginator.get_page(request.GET.get('page', 1))

    return render(request, 'anomalies.html', {
        'page_obj': page_obj,
        'severity': severity,
        'anomaly_type': anomaly_type,
        'review_status': review_status,
    })


@session_login_required
def fraud_risk_page(request):
    predictions = MLPrediction.objects().order_by('-fraud_risk_score')
    projects_by_id = {p.project_id: p for p in Project.objects(project_id__in=[pr.project_id for pr in predictions])}

    def bucket(pred):
        if pred.fraud_risk_score >= 60:
            return 'High Risk'
        if pred.fraud_risk_score >= 30:
            return 'Medium Risk'
        return 'Low Risk'

    high, medium, low = [], [], []
    for pred in predictions:
        proj = projects_by_id.get(pred.project_id)
        if not proj:
            continue
        entry = {'project': proj, 'prediction': pred}
        b = bucket(pred)
        if b == 'High Risk':
            high.append(entry)
        elif b == 'Medium Risk':
            medium.append(entry)
        else:
            low.append(entry)

    return render(request, 'fraud_detection.html', {
        'high': high[:100], 'medium': medium[:100], 'low': low[:100],
    })


@session_login_required
def upload_page(request):
    result = None
    if request.method == 'POST' and request.FILES.get('file'):
        if request.session.get('role') not in ('Admin', 'Analyst'):
            return render(request, 'upload.html', {'error': 'You do not have permission to upload data.'})
        df = pd.read_csv(request.FILES['file'])
        result = import_csv_to_mongo(df)
        if result.get('success') and result.get('inserted', 0) + result.get('updated', 0) > 0:
            ml_result = run_ml_analysis_and_save()
            result['ml_analysis'] = ml_result
    return render(request, 'upload.html', {'result': result})
