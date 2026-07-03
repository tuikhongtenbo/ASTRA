"""OpenAI-compatible vision client for DashScope-hosted Qwen VL models."""
from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import Iterable

from PIL import Image

from config.config import (
    DASHSCOPE_BASE_URL,
    VLM_API_ENABLE_THINKING,
    VLM_API_KEY,
    VLM_API_MAX_RETRIES,
)


_API_BACKEND_NAMES = {"api", "dashscope", "openai-compatible", "openai_compatible"}


def is_api_backend(name: str | None) -> bool:
    return (name or "").strip().lower() in _API_BACKEND_NAMES


def image_to_data_url(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    """Convert a PIL image to a data URL accepted by OpenAI-compatible APIs."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format=fmt, quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


class DashScopeVisionClient:
    """Small wrapper around the OpenAI SDK for DashScope MaaS VL inference."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = VLM_API_MAX_RETRIES,
        enable_thinking: bool = VLM_API_ENABLE_THINKING,
    ):
        self.model_name = model_name
        self.api_key = api_key or VLM_API_KEY
        self.base_url = base_url or DASHSCOPE_BASE_URL
        self.max_retries = max(1, int(max_retries or 1))
        self.enable_thinking = enable_thinking
        self._client = None

        if not self.api_key:
            raise RuntimeError(
                "DashScope API key is missing. Set DASHSCOPE_API_KEY, QWEN_API_KEY, "
                "VLM_API_KEY, or EXTRACTOR_API_KEY in your terminal or .env file."
            )
        if not self.base_url:
            raise RuntimeError(
                "DashScope base URL is missing. Set DASHSCOPE_BASE_URL or "
                "DASHSCOPE_WORKSPACE_ID in your terminal or .env file."
            )

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate(
        self,
        images: Iterable[Image.Image],
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        content = []
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})
        content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": "You are a helpful vision-language assistant."},
            {"role": "user", "content": content},
        ]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if not self.enable_thinking:
            kwargs["extra_body"] = {"enable_thinking": False}

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message.content
                return (message or "").strip()
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(2 ** attempt)

        raise RuntimeError(f"DashScope VLM request failed: {last_error}")