from functools import cached_property

import pandas as pd
from django.contrib.auth import get_user_model
from django.core.validators import FileExtensionValidator
from django.db import models

from apps.pages.utils import (
    generate_styled_table,
    prepare_chart_data,
    prepare_plotly_charts_from_recommendations,
)
from apps.visualization.ml.utils import convert_numpy
from apps.visualization.services.profiler import DatasetProfiler

User = get_user_model()


# Create your models here.
class TimestampedModel(models.Model):
    uploaded_at = models.DateTimeField(auto_now_add=True)
    edited_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class StatusChoices(models.TextChoices):
    PENDING = "pending", "Pending"
    COMPLETED = "completed", "Completed"
    CANCELED = "canceled", "Canceled"
    FAILED = "failed", "Failed"


class DatasetUploadModel(TimestampedModel):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="uploads")
    name = models.CharField(max_length=255, null=True, blank=True)
    file = models.FileField(
        upload_to="uploads/datasets/",
        validators=[FileExtensionValidator(allowed_extensions=["csv", "xlsx"])],
    )

    metadata = models.JSONField(null=True, blank=True)

    top_candidate_charts = models.JSONField(null=True, blank=True)

    status = models.CharField(
        choices=StatusChoices.choices,
        default=StatusChoices.PENDING,
        max_length=255,
        null=True,
        blank=True,
    )
    summary = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]
        indexes = [
            models.Index(fields=["user", "-uploaded_at"]),
        ]

    def save(self, *args, **kwargs):
        if self.file and not self.name:
            self.name = self.file.name
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

    @cached_property
    def get_df(self):
        return pd.read_csv(self.file.path, sep=None, engine="python")

    @cached_property
    def dataset_profiler(self):
        return DatasetProfiler()

    @cached_property
    def get_metadata(self):
        if self.metadata:
            return self.metadata

        self.metadata = convert_numpy(self.dataset_profiler.profile())
        self.save()
        return self.metadata

    @property
    def get_top_candidate_charts(self):
        if self.top_candidate_charts:
            return self.top_candidate_charts
        self.top_candidate_charts = None

    @property
    def get_initial_recommendation_charts_plotly(self):
        return prepare_plotly_charts_from_recommendations(self.top_candidate_charts)

    @property
    def generate_styled_table(self):
        return generate_styled_table(self.get_df.head(5))

    @property
    def get_all_contexts(self):
        return UserContext.objects.filter(dataset=self)


class UserContext(TimestampedModel):
    dataset = models.ForeignKey(
        DatasetUploadModel, on_delete=models.CASCADE, related_name="upload_contexts"
    )
    text = models.TextField(null=True, blank=True)
    recommendations = models.JSONField(null=True, blank=True)
    status = models.CharField(
        choices=StatusChoices.choices,
        max_length=255,
        null=True,
        blank=True,
        default=StatusChoices.PENDING,
    )

    def __str__(self):
        return f"{self.dataset} - context - {self.pk}"

    class Meta:
        ordering = ["-uploaded_at"]

    @property
    def get_chart_config(self):
        if self.recommendations:
            visualizations = self.recommendations.get("visualizations")

            charts_data = []
            for viz in visualizations:
                labels, data = prepare_chart_data(self.dataset.get_df, viz)
                chart_type = viz["chart_type"]
                title = viz.get("title", chart_type.title())
                x_axis = viz.get("axes").get("x") or viz.get("axes").get("category")
                y_axis = viz.get("axes").get("y") or viz.get("axes").get("value")
                chart_config = {
                    "type": chart_type,
                    "data": {
                        "labels": labels,
                        "datasets": [
                            {
                                "label": title,
                                "data": data,
                                "backgroundColor": "rgba(54, 162, 235, 0.5)",
                            }
                        ],
                    },
                    "options": {
                        "plugins": {"title": {"display": "true", "text": title}},
                        "scales": {
                            "x": {
                                "title": {
                                    "display": "true",
                                    "text": x_axis.title() if x_axis else "",
                                }
                            },
                            "y": {
                                "title": {
                                    "display": "true",
                                    "text": y_axis.title() if y_axis else "",
                                }
                            },
                        },
                    },
                }
                charts_data.append(
                    {
                        "context_id": self.pk,
                        "chart_config": chart_config,
                        "summary": viz.get("summary", ""),
                    }
                )

            return charts_data
        return None

    @property
    def get_chart_config_plotly(self):
        if not self.recommendations:
            return None

        charts_data = []
        visualizations = self.recommendations.get("visualizations", [])
        df = self.dataset.get_df

        for viz in visualizations:
            chart_type = viz.get("chart_type")
            title = viz.get("title", chart_type.title())
            x_axis = viz.get("axes", {}).get("x") or viz.get("axes", {}).get("category")
            y_axis = viz.get("axes", {}).get("y") or viz.get("axes", {}).get("value")

            traces = prepare_chart_data(df, viz)

            layout = {
                "title": {"text": title, "x": 0.5},
                "xaxis": {"title": x_axis or ""},
                "yaxis": {"title": y_axis or ""},
                "margin": {"t": 40, "b": 40, "l": 40, "r": 40},
            }

            charts_data.append(
                {
                    "context_id": self.pk,
                    "plotly_data": traces,
                    "plotly_layout": layout,
                    "summary": viz.get("summary", ""),
                }
            )

        return charts_data
