from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r"ws/summary/(?P<pk>\d+)/$", consumers.SummaryConsumer.as_asgi()),
    re_path(r"ws/ctx/(?P<pk>\d+)/$", consumers.UserContextConsumer.as_asgi()),
]
