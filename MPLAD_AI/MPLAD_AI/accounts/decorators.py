"""Simple session-based login-required decorator for server-rendered pages
(the JWT auth in accounts/authentication.py is used for the REST API)."""
from functools import wraps
from django.shortcuts import redirect


def session_login_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.session.get('user_id'):
            return redirect('login_page')
        return view_func(request, *args, **kwargs)
    return wrapper


def roles_required(*roles):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            if not request.session.get('user_id'):
                return redirect('login_page')
            if request.session.get('role') not in roles:
                return redirect('dashboard_page')
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator
