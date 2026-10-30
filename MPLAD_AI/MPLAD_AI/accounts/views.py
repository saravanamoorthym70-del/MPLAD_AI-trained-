"""
Authentication views.

API endpoints (Stage 20/22):
    POST /api/auth/register/
    POST /api/auth/login/
    GET  /api/auth/me/

Page views: server-rendered login/logout pages backed by Django sessions,
used by the HTML/Bootstrap frontend (separate from the JWT API, which is
meant for programmatic/API clients).
"""
from django.shortcuts import render, redirect
from django.contrib import messages
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions

from accounts.models import User
from accounts.serializers import RegisterSerializer, LoginSerializer
from accounts.authentication import tokens_for_user


class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        if User.objects(username=data['username']).first():
            return Response({'detail': 'Username already exists'}, status=status.HTTP_400_BAD_REQUEST)
        if User.objects(email=data['email']).first():
            return Response({'detail': 'Email already exists'}, status=status.HTTP_400_BAD_REQUEST)

        user = User(
            username=data['username'],
            email=data['email'],
            full_name=data.get('full_name', ''),
            role=data.get('role', 'Viewer'),
        )
        user.set_password(data['password'])
        user.save()

        tokens = tokens_for_user(user)
        return Response({'user': user.to_dict(), **tokens}, status=status.HTTP_201_CREATED)


class LoginView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        user = User.objects(username=data['username']).first()
        if not user or not user.check_password(data['password']):
            return Response({'detail': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
        if not user.is_active:
            return Response({'detail': 'Account is inactive'}, status=status.HTTP_403_FORBIDDEN)

        tokens = tokens_for_user(user)
        return Response({'user': user.to_dict(), **tokens})


class MeView(APIView):
    def get(self, request):
        return Response(request.user.to_dict())


# ---------------- Server-rendered pages (session-based, for the dashboard UI) ----------------

def login_page(request):
    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        user = User.objects(username=username).first()
        if user and user.check_password(password) and user.is_active:
            request.session['user_id'] = str(user.id)
            request.session['username'] = user.username
            request.session['role'] = user.role
            request.session['full_name'] = user.full_name
            return redirect('dashboard_page')
        error = 'Invalid username or password.'
        messages.error(request, error)
    return render(request, 'login.html', {'error': error})


def logout_page(request):
    request.session.flush()
    return redirect('login_page')
