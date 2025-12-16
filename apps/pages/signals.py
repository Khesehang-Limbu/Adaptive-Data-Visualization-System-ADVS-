import threading

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.db.models.signals import post_save
from django.dispatch import receiver

from apps.pages.models import StatusChoices, UserContext
from apps.visualization.ml.utils import ask_llm_for_viz


def get_llm_recommendation(instance):
    llm_response = ask_llm_for_viz(
        instance.text, instance.dataset.metadata, instance.dataset.get_df
    )

    if llm_response is None:
        llm_response = instance.fallback_recommendations()

    instance.recommendations = llm_response
    instance.status = StatusChoices.COMPLETED
    instance.save()

    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"ctx_{instance.pk}",
        {
            "type": "chart.ready",
            "charts": instance.recommendations,
            "ctx_id": instance.pk,
        },
    )


@receiver(post_save, sender=UserContext)
def get_recommendations(sender, instance, created, **kwargs):
    if created:
        if not instance.recommendations:
            threading.Thread(
                target=get_llm_recommendation, args=(instance,), daemon=True
            ).start()
