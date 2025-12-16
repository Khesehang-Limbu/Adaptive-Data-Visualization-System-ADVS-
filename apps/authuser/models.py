from django.contrib.auth.models import AbstractUser
from django.db import models


# Create your models here.
class AuthUser(AbstractUser):
    profile_image = models.ImageField(upload_to="users/profiles/")

    @property
    def latest_upload(self):
        from apps.pages.models import DatasetUploadModel

        return (
            DatasetUploadModel.objects.filter(user=self)
            .order_by("-uploaded_at")
            .first()
        )

    @property
    def get_all_datasets(self):
        from apps.pages.models import DatasetUploadModel

        return (
            DatasetUploadModel.objects.filter(user=self)
            .prefetch_related("upload_contexts")
            .all()
        )
