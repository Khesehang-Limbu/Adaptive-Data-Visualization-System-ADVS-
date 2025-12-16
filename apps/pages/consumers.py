import json

import httpx
from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from ..visualization.ml.constants import (
    LLM_API_ENDPOINT,
    LLM_DATASET_SUMMARY_PROMPT_STREAM,
    LLM_LLAMA,
)
from .models import DatasetUploadModel


class SummaryConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        pk = data.get("pk")
        await self.stream_llm(pk)

    async def stream_llm(self, pk):
        dataset = await sync_to_async(DatasetUploadModel.objects.get)(pk=pk)

        payload = {
            "model": LLM_LLAMA,
            "prompt": LLM_DATASET_SUMMARY_PROMPT_STREAM.format(
                dataset_summary=dataset.get_metadata
            ),
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                LLM_API_ENDPOINT,
                json=payload,
            ) as response:
                full = []

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                        token = chunk.get("thinking") or chunk.get("response") or ""
                    except:
                        token = line

                    full.append(token)

                    await self.send(json.dumps({"token": token}))

        summary = "".join(full)
        dataset.summary = summary
        await sync_to_async(dataset.save)()

        await self.send(json.dumps({"done": True}))


class UserContextConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.ctx_id = self.scope["url_route"]["kwargs"]["pk"]
        self.group_name = f"ctx_{self.ctx_id}"

        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def chart_ready(self, event):
        await self.send(
            text_data=json.dumps(
                {
                    "type": "chart_ready",
                    "ctx_id": event["ctx_id"],
                    "charts": event["charts"],
                }
            )
        )
