from allauth.account.forms import ChangePasswordForm, LoginForm, SignupForm
from django import forms


class AuthUserLoginForm(LoginForm):
    def __init__(self, *args, **kwargs):
        super(AuthUserLoginForm, self).__init__(*args, **kwargs)
        self.fields["login"].widget = forms.TextInput(
            attrs={
                "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-sm",
                "placeholder": "Email or Username",
            }
        )

        self.fields["password"].widget = forms.PasswordInput(
            attrs={
                "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-sm",
                "placeholder": "Password",
            }
        )


class AuthUserRegisterForm(SignupForm):
    def __init__(self, *args, **kwargs):
        super(AuthUserRegisterForm, self).__init__(*args, **kwargs)
        self.fields["email"].widget = forms.EmailInput(
            attrs={
                "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-sm",
                "placeholder": "Email",
            }
        )

        self.fields["password1"].widget = forms.PasswordInput(
            attrs={
                "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-sm",
                "placeholder": "Password",
            }
        )

        self.fields["password2"].widget = forms.PasswordInput(
            attrs={
                "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-sm",
                "placeholder": "Confirm Password",
            }
        )


class AuthUserChangePasswordForm(ChangePasswordForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget = forms.PasswordInput(
                attrs={
                    "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-sm"
                }
            )
