"""Small shared helper functions used across apps."""
from datetime import datetime, date


def to_date(value):
    """Best-effort conversion of a value into a datetime.date."""
    if value is None or value == '':
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        for fmt in ('%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d'):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
    return None


def safe_div(numerator, denominator, default=0.0):
    """Division that never raises, returns `default` when denominator is 0."""
    try:
        if denominator in (None, 0):
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def risk_level_from_score(score):
    """Map a 0-100 score to the standard risk category labels (Stage 12)."""
    if score is None:
        return 'Unknown'
    if score >= 75:
        return 'Critical'
    if score >= 50:
        return 'High'
    if score >= 25:
        return 'Moderate'
    return 'Low'


def clamp(value, low=0, high=100):
    return max(low, min(high, value))


def paginate(queryset_list, page, page_size=20):
    """Simple in-memory pagination helper for a Python list."""
    page = max(1, int(page or 1))
    page_size = max(1, int(page_size or 20))
    total = len(queryset_list)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        'items': queryset_list[start:end],
        'page': page,
        'page_size': page_size,
        'total': total,
        'total_pages': max(1, (total + page_size - 1) // page_size),
    }
