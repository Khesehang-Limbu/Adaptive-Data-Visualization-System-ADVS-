from django.urls import include, path

from . import views

dashboard_urlpatterns = [
    path("", views.DashboardPageView.as_view(), name="home"),
    path("datasets/", views.DatasetListView.as_view(), name="datasets"),
    path("datasets/upload/", views.DatasetUploadView.as_view(), name="dataset-upload"),
    path(
        "datasets/<int:pk>/", views.DatasetDetailView.as_view(), name="dataset-detail"
    ),
    path(
        "datasets/<int:pk>/contexts/",
        views.ContextListView.as_view(),
        name="dataset-contexts",
    ),
    path(
        "datasets/<int:pk>/charts/",
        views.ContextChartsView.as_view(),
        name="dataset-charts",
    ),
    path(
        "datasets/<int:pk>/summary/", views.stream_summary, name="dataset_summary_api"
    ),
    path(
        "datasets/<int:pk>/poll-analysis/",
        views.context_analysis_poll,
        name="dataset-analysis-poll",
    ),
    path("settings/", views.DashboardSettingsPageView.as_view(), name="settings"),
]

urlpatterns = [
    path("", views.HomePageView.as_view(), name="home"),
    path(
        "dashboard/",
        include((dashboard_urlpatterns, "dashboard"), namespace="dashboard"),
    ),
    path("upload/", views.DashboardUploadPageView.as_view(), name="upload"),
]
