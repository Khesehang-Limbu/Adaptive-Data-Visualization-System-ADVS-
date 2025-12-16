from celery import shared_task

from apps.pages.models import DatasetUploadModel, StatusChoices
from apps.visualization.ml.utils import convert_numpy
from apps.visualization.services.visualizer_service import VisualizationService


@shared_task
def analyze_dataset_task(dataset_id):
    dataset = DatasetUploadModel.objects.get(id=dataset_id)
    service = VisualizationService()

    try:
        dataset.status = StatusChoices.PENDING
        dataset.save(update_fields=["status"])

        recommendations = service.generate_recommendations(dataset)

        dataset.top_candidate_charts = convert_numpy(recommendations)
        dataset.status = StatusChoices.COMPLETED
        dataset.save(update_fields=["top_candidate_charts", "status"])

    except Exception as e:
        dataset.status = StatusChoices.FAILED
        dataset.save(update_fields=["status"])
        raise
