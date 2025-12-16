from django.contrib import admin
from django.contrib.admin import StackedInline, register

from .models import DatasetUploadModel, UserContext


# Register your models here.
class UserContextInline(StackedInline):
    model = UserContext
    extra = 1


@register(DatasetUploadModel)
class DatasetUploadAdmin(admin.ModelAdmin):
    list_filter = ["name", "user", "status"]
    inlines = [
        UserContextInline,
    ]
