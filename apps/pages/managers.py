from django.db import models

from apps.pages.models import DatasetUploadModel


class DatasetManager(models.Manager):
    def create_with_analysis(self, **kwargs):
        dataset = self.create(**kwargs)

        from .tasks import analyze_dataset_task

        analyze_dataset_task.delay(dataset.id)

        return dataset


DatasetUploadModel.add_to_class("objects", DatasetManager())
