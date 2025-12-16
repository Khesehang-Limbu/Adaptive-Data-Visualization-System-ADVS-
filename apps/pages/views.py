import json
import time

import numpy as np
import requests
from django.contrib.auth import get_user_model
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Count, Sum
from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, render
from django.template.loader import render_to_string
from django.urls import reverse, reverse_lazy
from django.views.generic import (
    CreateView,
    DetailView,
    ListView,
    TemplateView,
    UpdateView,
    View,
)

from apps.pages.forms import DatasetUploadForm, DatasetUploadFormset, UserProfileForm
from apps.pages.models import DatasetUploadModel, StatusChoices, UserContext
from apps.visualization.ml.constants import (
    LLM_API_ENDPOINT,
    LLM_DATASET_SUMMARY_PROMPT_STREAM,
    LLM_QWEN,
)
from apps.visualization.ml.utils import ask_llm_for_viz
from apps.visualization.services.visualizer_service import VisualizationService

# Create your views here.

User = get_user_model()


class HomePageView(TemplateView):
    template_name = "pages/index.html"


class DashboardPageView(LoginRequiredMixin, TemplateView):
    template_name = "dashboard/home.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        last_login_date = self.request.user.last_login
        total_datasets = self.request.user.get_all_datasets.count()
        total_contexts = (
            self.request.user.get_all_datasets.annotate(
                total=Count("upload_contexts"),
            )
            .aggregate(total_contexts=Sum("total"))
            .get("total_contexts")
        )
        completed_analysis = self.request.user.get_all_datasets.filter(
            status=StatusChoices.COMPLETED
        ).count()

        context.update(
            {
                "last_login_date": last_login_date,
                "total_datasets": total_datasets,
                "total_contexts": total_contexts or 0,
                "completed_analysis": completed_analysis,
            }
        )
        return context


class DashboardUploadPageView(LoginRequiredMixin, TemplateView):
    template_name = "dashboard/upload.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        latest_upload = (
            DatasetUploadModel.objects.filter(user=self.request.user)
            .order_by("-uploaded_at")
            .first()
        )

        if (
            latest_upload
            and latest_upload.status == StatusChoices.PENDING
            and not latest_upload.top_candidate_charts
        ):
            latest_upload.top_candidate_charts = latest_upload.get_top_candidate_charts
            latest_upload.save()

        context["form"] = DatasetUploadForm()
        context["latest_upload"] = latest_upload

        if latest_upload:
            context["formset"] = DatasetUploadFormset(
                instance=latest_upload, queryset=UserContext.objects.none()
            )
            context["previous_contexts"] = latest_upload.get_all_contexts
        else:
            context["formset"] = None

        return context

    def post(self, request, *args, **kwargs):
        data = request.POST
        files = request.FILES

        action = request.GET.get("action")
        custom_type = request.GET.get("type")

        if custom_type == "custom_chart":
            x_col = request.POST.get("x_column")
            y_col = request.POST.get("y_column")
            chart_type = request.POST.get("chart_type")
            upload_id = request.GET.get("id")

            instance = DatasetUploadModel.objects.get(user=request.user, pk=upload_id)
            df = instance.get_df.replace({np.nan: None})

            filtered_df = df[[x_col, y_col]]

            custom_chart = render_to_string(
                "dashboard/partials/single_chart.html",
                {
                    "chart_type": chart_type,
                    "x_column": x_col,
                    "y_column": y_col,
                    "json_data": json.dumps(filtered_df.to_dict(orient="records")),
                },
                request=request,
            )
            return HttpResponse(custom_chart)

        if action == "process-context":
            upload_id = request.GET.get("upload_id")
            instance = get_object_or_404(DatasetUploadModel, pk=upload_id)

            formset = DatasetUploadFormset(request.POST, instance=instance)

            if not formset.is_valid():
                return HttpResponse("Invalid context formset", status=400)

            saved_contexts = formset.save()

            context_obj = saved_contexts[-1]

            user_context_text = context_obj.text or ""

            llm_response = ask_llm_for_viz(
                user_context_text, instance.top_candidate_charts, instance.get_df
            )

            context_obj.recommendations = llm_response
            context_obj.status = StatusChoices.COMPLETED
            context_obj.save()

            chart_html = render_to_string(
                "dashboard/partials/_context_partial.html",
                {"charts_data": context_obj.get_chart_config},
            )

            return HttpResponse(chart_html)
        form = DatasetUploadForm(data, files)

        if form.is_valid():
            instance = form.save(commit=False)
            instance.user = request.user

            instance.top_candidate_charts = instance.get_top_candidate_charts
            instance.save()

            return self.render_to_response(self.get_context_data(**kwargs))

        return self.render_to_response(self.get_context_data(**kwargs))


class DashboardSettingsPageView(LoginRequiredMixin, UpdateView):
    template_name = "dashboard/settings.html"
    form_class = UserProfileForm
    success_url = reverse_lazy("settings")

    def get_object(self, queryset=None):
        return self.request.user


class DatasetListView(LoginRequiredMixin, ListView):
    model = DatasetUploadModel
    template_name = "dashboard/datasets_list.html"

    def get_queryset(self):
        return self.request.user.uploads.order_by("-uploaded_at")


class DatasetDetailView(LoginRequiredMixin, DetailView):
    model = DatasetUploadModel
    template_name = "dashboard/dataset_detail.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        dataset = self.get_object()

        service = VisualizationService()

        if dataset.top_candidate_charts:
            recommendations = dataset.top_candidate_charts
        else:
            recommendations = service.generate_recommendations(dataset)
            dataset.top_candidate_charts = recommendations
            dataset.save()
            print(recommendations)

        ctx["recommendations"] = recommendations
        ctx["metadata"] = dataset.metadata
        ctx["head_html"] = dataset.generate_styled_table
        ctx["contexts"] = dataset.get_all_contexts
        ctx["formset"] = DatasetUploadFormset(
            instance=dataset, queryset=UserContext.objects.none()
        )

        return ctx

    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        formset = DatasetUploadFormset(request.POST, instance=self.object)
        if formset.is_valid():
            contexts = formset.save()
            return render(
                request,
                "dashboard/partials/_context_partial.html",
                {"ctx": contexts[-1]},
            )

        return render(
            request, self.template_name, {"formset": formset, "dataset": self.object}
        )


class DatasetUploadView(LoginRequiredMixin, CreateView):
    model = DatasetUploadModel
    form_class = DatasetUploadForm
    template_name = "dashboard/partials/_upload_modal.html"

    def form_valid(self, form):
        form.instance.user = self.request.user
        response = super().form_valid(form)

        from .tasks import analyze_dataset_task

        analyze_dataset_task.delay(self.object.id)

        return response

    def get_success_url(self):
        return reverse("dashboard:dataset-detail", kwargs={"pk": self.object.pk})


class ContextListView(LoginRequiredMixin, ListView):
    model = UserContext
    template_name = "dashboard/partials/_context_partial.html"

    def get_queryset(self):
        dataset = get_object_or_404(
            DatasetUploadModel, pk=self.kwargs["pk"], user=self.request.user
        )
        return dataset.get_all_contexts.order_by("-uploaded_at")


class ContextChartsView(LoginRequiredMixin, View):
    def get(self, request, pk):
        dataset = get_object_or_404(DatasetUploadModel, pk=pk, user=request.user)
        contexts = dataset.get_all_contexts

        charts_data = []
        for ctx in contexts:
            if ctx.get_chart_config:
                charts_data.extend(ctx.get_chart_config)

        return render(
            request,
            "dashboard/partials/_context_partial.html",
            {"charts_data": charts_data},
        )


def stream_summary(request, pk):
    dataset = DatasetUploadModel.objects.get(pk=pk)

    payload = {
        "model": LLM_QWEN,
        "prompt": LLM_DATASET_SUMMARY_PROMPT_STREAM.format(
            data_head=dataset.get_df.head(20).to_csv(index=False)
        ),
        "stream": True,
    }

    summary = []

    def token_stream():
        nonlocal summary
        try:
            with requests.post(
                LLM_API_ENDPOINT,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                stream=True,
            ) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        response = data.get("response")
                        summary.append(response)
                        yield f"data: {response}\n\n"
                        time.sleep(0.01)
                    except:
                        pass

            full_summary = "".join(summary)
            dataset.summary = full_summary
            dataset.save(update_fields=["summary"])
        except GeneratorExit:
            return
        except Exception as e:
            print("Streaming error:", e)
            return

    return StreamingHttpResponse(token_stream(), content_type="text/event-stream")


def context_analysis_poll(request, pk):
    ctx = get_object_or_404(UserContext, pk=pk, dataset__user=request.user)
    return render(request, "dashboard/partials/_context_partial.html", {"ctx": ctx})
