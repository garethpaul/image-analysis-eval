"""
$ pip install openai
Env:
  POE_API_KEY=your_api_key
"""

from openai import OpenAI
from .base_model import BaseVisionModel
from typing import Dict, Any
import os
import re


class PoeModel(BaseVisionModel):
    """Poe model implementation via OpenAI-compatible API."""

    def __init__(self, model_name: str, api_key: str | None):
        super().__init__(model_name)
        if not api_key:
            api_key = os.environ.get("POE_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://api.poe.com/v1")

    def _postprocess_text(self, content: str) -> str:
        """Remove markdown image tags and bare URLs from content."""
        if not content:
            return content
        # Strip markdown image tags
        cleaned = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", content)
        # Strip bare URLs
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        # Collapse excessive whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or content

    def generate_response(self, example: Dict[str, Any]) -> str:
        """Generate a response for the given example.

        Expects example to contain keys: 'media_url' and 'prompt'.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": example["media_url"]}},
                    {"type": "text", "text": example["prompt"]},
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        content = response.choices[0].message.content
        return self._postprocess_text(content)


