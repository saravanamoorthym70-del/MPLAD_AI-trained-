"""No Django-ORM admin registration is possible for MongoEngine documents
via django.contrib.admin directly. User management for MongoEngine `User`
is instead exposed through the Django REST Framework API
(/api/auth/register/) and the 'User Management' dashboard page, which is
simpler and avoids needing a MongoEngine-Django-admin adapter package."""
