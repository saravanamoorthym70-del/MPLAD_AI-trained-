"""
Aggregation helpers for the dashboard cards (Stage 13) and Chart.js
datasets (Stage 14), plus state/district/MP level analytics
(Stage 12 items 10-11) and CSV/PDF report generation (Stage 19).
"""
import io
import csv
import datetime
from collections import Counter, defaultdict

from projects.models import Project, Anomaly, MLPrediction


def dashboard_summary():
    projects = list(Project.objects())
    predictions = {p.project_id: p for p in MLPrediction.objects()}
    anomalies = list(Anomaly.objects())

    total_projects = len(projects)
    total_sanctioned = sum(p.sanctioned_amount for p in projects)
    total_expenditure = sum(p.expenditure_amount for p in projects)

    under_review = sum(1 for a in anomalies if a.review_status in ('Pending', 'Under Review'))
    high_risk = sum(1 for pr in predictions.values() if pr.risk_level in ('High', 'Critical'))
    delayed = sum(1 for p in projects if p.delay_days and p.delay_days > 0)
    potential_anomalies = len(anomalies)

    efficiency_scores = [100 - pr.inefficiency_score for pr in predictions.values()]
    avg_efficiency = round(sum(efficiency_scores) / len(efficiency_scores), 1) if efficiency_scores else 0

    return {
        'total_projects': total_projects,
        'total_sanctioned': round(total_sanctioned, 2),
        'total_expenditure': round(total_expenditure, 2),
        'projects_under_review': under_review,
        'high_risk_projects': high_risk,
        'delayed_projects': delayed,
        'potential_anomalies': potential_anomalies,
        'average_efficiency': avg_efficiency,
    }


def chart_datasets():
    projects = list(Project.objects())
    predictions = {p.project_id: p for p in MLPrediction.objects()}

    by_state = Counter(p.state for p in projects)
    by_category = Counter(p.project_category for p in projects)
    expenditure_by_state = defaultdict(float)
    for p in projects:
        expenditure_by_state[p.state] += p.expenditure_amount

    monthly_expenditure = defaultdict(float)
    for p in projects:
        if p.start_date:
            key = p.start_date.strftime('%Y-%m')
            monthly_expenditure[key] += p.expenditure_amount
    monthly_sorted = dict(sorted(monthly_expenditure.items()))

    risk_distribution = Counter(pr.risk_level for pr in predictions.values())
    status_distribution = Counter(p.project_status for p in projects)

    cost_overrun_buckets = Counter()
    for p in projects:
        co = p.cost_overrun or 0
        if co < 0:
            bucket = 'Under Budget'
        elif co < 0.1:
            bucket = '0-10%'
        elif co < 0.25:
            bucket = '10-25%'
        elif co < 0.5:
            bucket = '25-50%'
        else:
            bucket = '50%+'
        cost_overrun_buckets[bucket] += 1

    delay_buckets = Counter()
    for p in projects:
        d = p.delay_days or 0
        if d <= 0:
            bucket = 'On Time'
        elif d <= 30:
            bucket = '1-30 days'
        elif d <= 90:
            bucket = '31-90 days'
        else:
            bucket = '90+ days'
        delay_buckets[bucket] += 1

    contractor_risk = defaultdict(list)
    for p in projects:
        pred = predictions.get(p.project_id)
        if pred:
            contractor_risk[p.contractor].append(pred.overall_risk_score)
    contractor_avg_risk = {c: round(sum(v) / len(v), 1) for c, v in contractor_risk.items() if v}
    top_contractor_risk = dict(sorted(contractor_avg_risk.items(), key=lambda kv: kv[1], reverse=True)[:10])

    anomalies_by_month = defaultdict(int)
    for a in Anomaly.objects():
        if a.detected_at:
            anomalies_by_month[a.detected_at.strftime('%Y-%m')] += 1
    anomaly_trend = dict(sorted(anomalies_by_month.items()))

    return {
        'projects_by_state': dict(by_state),
        'projects_by_category': dict(by_category),
        'expenditure_by_state': {k: round(v, 2) for k, v in expenditure_by_state.items()},
        'monthly_expenditure': {k: round(v, 2) for k, v in monthly_sorted.items()},
        'risk_distribution': dict(risk_distribution),
        'status_distribution': dict(status_distribution),
        'cost_overrun_distribution': dict(cost_overrun_buckets),
        'delay_distribution': dict(delay_buckets),
        'contractor_risk_distribution': top_contractor_risk,
        'anomaly_trend': anomaly_trend,
    }


def state_level_analytics():
    projects = list(Project.objects())
    predictions = {p.project_id: p for p in MLPrediction.objects()}
    by_state = defaultdict(lambda: {'projects': 0, 'sanctioned': 0.0, 'expenditure': 0.0, 'high_risk': 0})
    for p in projects:
        s = by_state[p.state]
        s['projects'] += 1
        s['sanctioned'] += p.sanctioned_amount
        s['expenditure'] += p.expenditure_amount
        pred = predictions.get(p.project_id)
        if pred and pred.risk_level in ('High', 'Critical'):
            s['high_risk'] += 1
    return {k: v for k, v in by_state.items()}


def category_level_analytics():
    projects = list(Project.objects())
    by_cat = defaultdict(lambda: {'projects': 0, 'sanctioned': 0.0, 'expenditure': 0.0})
    for p in projects:
        c = by_cat[p.project_category]
        c['projects'] += 1
        c['sanctioned'] += p.sanctioned_amount
        c['expenditure'] += p.expenditure_amount
    return {k: v for k, v in by_cat.items()}


# ---------------------------- Stage 19: Reports ----------------------------

def generate_csv_report(report_type):
    output = io.StringIO()
    writer = csv.writer(output)

    if report_type == 'project':
        writer.writerow(['Project ID', 'Name', 'State', 'District', 'Category', 'Sanctioned',
                          'Expenditure', 'Completion %', 'Status'])
        for p in Project.objects():
            writer.writerow([p.project_id, p.project_name, p.state, p.district, p.project_category,
                              p.sanctioned_amount, p.expenditure_amount, p.completion_percentage, p.project_status])

    elif report_type == 'anomaly':
        writer.writerow(['Anomaly ID', 'Project ID', 'Type', 'Score', 'Severity', 'Review Status', 'Explanation'])
        for a in Anomaly.objects():
            writer.writerow([a.anomaly_id, a.project_id, a.anomaly_type, a.anomaly_score,
                              a.severity, a.review_status, a.explanation])

    elif report_type == 'financial':
        writer.writerow(['Project ID', 'Sanctioned', 'Released', 'Expenditure', 'Estimated Cost',
                          'Actual Cost', 'Cost Overrun %'])
        for p in Project.objects():
            writer.writerow([p.project_id, p.sanctioned_amount, p.released_amount, p.expenditure_amount,
                              p.estimated_cost, p.actual_cost, round((p.cost_overrun or 0) * 100, 2)])

    elif report_type == 'risk':
        writer.writerow(['Project ID', 'Anomaly Score', 'Fraud Risk', 'Inefficiency', 'Delay Risk',
                          'Overall Risk', 'Risk Level'])
        for pr in MLPrediction.objects():
            writer.writerow([pr.project_id, pr.anomaly_probability, pr.fraud_risk_score,
                              pr.inefficiency_score, pr.delay_risk_score, pr.overall_risk_score, pr.risk_level])

    elif report_type == 'state':
        writer.writerow(['State', 'Projects', 'Sanctioned', 'Expenditure', 'High Risk Projects'])
        for state, s in state_level_analytics().items():
            writer.writerow([state, s['projects'], round(s['sanctioned'], 2), round(s['expenditure'], 2), s['high_risk']])

    elif report_type == 'constituency':
        writer.writerow(['Constituency', 'MP Name', 'Projects', 'Sanctioned', 'Expenditure'])
        by_const = defaultdict(lambda: {'mp': '', 'projects': 0, 'sanctioned': 0.0, 'expenditure': 0.0})
        for p in Project.objects():
            c = by_const[p.constituency]
            c['mp'] = p.mp_name
            c['projects'] += 1
            c['sanctioned'] += p.sanctioned_amount
            c['expenditure'] += p.expenditure_amount
        for const, c in by_const.items():
            writer.writerow([const, c['mp'], c['projects'], round(c['sanctioned'], 2), round(c['expenditure'], 2)])

    return output.getvalue()


def generate_pdf_report(report_type):
    """Generate a simple tabular PDF report using reportlab (Stage 19)."""
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    csv_text = generate_csv_report(report_type)
    rows = list(csv.reader(io.StringIO(csv_text)))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    elements = [
        Paragraph(f'MPLAD AI — {report_type.title()} Report', styles['Title']),
        Paragraph(f'Generated: {datetime.datetime.utcnow().isoformat()} UTC', styles['Normal']),
        Spacer(1, 12),
    ]
    table = Table(rows[:500], repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
    ]))
    elements.append(table)
    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()
