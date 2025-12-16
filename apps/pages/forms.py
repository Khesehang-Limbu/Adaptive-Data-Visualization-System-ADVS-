from django import forms
from django.contrib.auth import get_user_model
from django.forms import inlineformset_factory

from .models import DatasetUploadModel, UserContext

User = get_user_model()


class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = DatasetUploadModel
        fields = ["file"]

    def __init__(self, *args, **kwargs):
        super(DatasetUploadForm, self).__init__(*args, **kwargs)
        self.fields["file"].widget.attrs[
            "class"
        ] = "block w-full text-sm border border-gray-300 p-2 rounded mb-4"
        self.fields["file"].label = "Choose File"


class UserContextForm(forms.ModelForm):
    class Meta:
        model = UserContext
        fields = ["text"]

    def __init__(self, *args, **kwargs):
        super(UserContextForm, self).__init__(*args, **kwargs)
        self.fields["text"].widget.attrs[
            "class"
        ] = "block w-full text-sm border border-gray-300 p-2 rounded mb-4"
        self.fields["text"].label = "What story do you want to tell with this data?"


DatasetUploadFormset = inlineformset_factory(
    DatasetUploadModel, UserContext, form=UserContextForm, extra=1, can_delete=True
)


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["first_name", "last_name", "profile_image"]

    def __init__(self, *args, **kwargs):
        super(UserProfileForm, self).__init__(*args, **kwargs)
        self.fields["profile_image"].widget.attrs[
            "class"
        ] = "mt-2 block w-full text-sm text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-600 file:text-white hover:file:bg-blue-700 cursor-pointer"
