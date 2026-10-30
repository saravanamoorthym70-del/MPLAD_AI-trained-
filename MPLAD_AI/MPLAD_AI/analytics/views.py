"""Analytics app: dashboard/API endpoints and server-rendered pages
(Stages 13, 14, 19)."""
import json
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response

from accounts.decorators import session_login_required
from analytics.services import (
    dashboard_summary, chart_datasets, state_level_analytics,
    category_level_analytics, generate_csv_report, generate_pdf_report,
)


class DashboardAnalyticsAPIView(APIView):
    def get(self, request):
        return Response({
            'summary': dashboard_summary(),
            'charts': chart_datasets(),
        })


class StateAnalyticsAPIView(APIView):
    def get(self, request):
        return Response(state_level_analytics())


class CategoryAnalyticsAPIView(APIView):
    def get(self, request):
        return Response(category_level_analytics())


class ReportsAPIView(APIView):
    def get(self, request):
        report_type = request.query_params.get('type', 'project')
        # NOTE: intentionally NOT named "format" — that query param is
        # reserved by DRF for content-negotiation and collides with it.
        fmt = request.query_params.get('file_format', request.query_params.get('format', 'csv'))
        if fmt == 'pdf':
            pdf_bytes = generate_pdf_report(report_type)
            response = HttpResponse(pdf_bytes, content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{report_type}_report.pdf"'
            return response
        csv_text = generate_csv_report(report_type)
        response = HttpResponse(csv_text, content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{report_type}_report.csv"'
        return response


# ---------------------------- Pages ----------------------------

@session_login_required
def dashboard_page(request):
    summary = dashboard_summary()
    charts = chart_datasets()
    return render(request, 'dashboard.html', {
        'summary': summary,
        'charts_json': json.dumps(charts),
    })


@session_login_required
def analytics_page(request):
    states = state_level_analytics()
    categories = category_level_analytics()
    return render(request, 'analytics.html', {
        'states': states,
        'categories': categories,
        'charts_json': json.dumps(chart_datasets()),
    })


@session_login_required
def reports_page(request):
    return render(request, 'reports.html', {})
