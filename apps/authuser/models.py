from django.contrib.auth.models import AbstractUser
from django.db import models


# Create your models here.
class AuthUser(AbstractUser):
    profile_image = models.ImageField(upload_to="users/profiles/")
