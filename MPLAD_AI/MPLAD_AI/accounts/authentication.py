"""
Custom JWT authentication that issues/validates tokens for MongoEngine
`User` documents instead of Django's built-in ORM auth.User.
"""
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.exceptions import AuthenticationFailed
from accounts.models import User


def tokens_for_user(user):
    """Generate an access/refresh token pair for a MongoEngine User."""
    refresh = RefreshToken()
    refresh['user_id'] = str(user.id)
    refresh['username'] = user.username
    refresh['role'] = user.role
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }


class MongoJWTAuthentication(JWTAuthentication):
    """Resolves the `user_id` claim in a validated token to a Mongo User."""

    def get_user(self, validated_token):
        user_id = validated_token.get('user_id')
        if not user_id:
            raise AuthenticationFailed('Token missing user_id claim')
        try:
            user = User.objects.get(id=user_id)
        except Exception:
            raise AuthenticationFailed('User not found')
        if not user.is_active:
            raise AuthenticationFailed('User is inactive')
        return user
