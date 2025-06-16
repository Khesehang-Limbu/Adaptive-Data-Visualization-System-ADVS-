from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from apps.authuser.models import AuthUser


# Register your models here.
class AuthUserAdmin(UserAdmin):
    model = AuthUser
    list_display = ("username", "email", "is_staff", "is_superuser")
    list_filter = ("is_staff", "is_superuser")
    search_fields = (
        "username",
        "email",
        "first_name",
        "last_name",
    )
    ordering = ("email",)
    fieldsets = UserAdmin.fieldsets + (
        (
            "Profile",
            {
                "fields": [
                    "profile_image",
                ],
            },
        ),
    )
    add_fieldsets = UserAdmin.fieldsets + (
        (
            "Profile",
            {
                "fields": [
                    "profile_image",
                ],
            },
        ),
    )


admin.site.register(AuthUser, AuthUserAdmin)
