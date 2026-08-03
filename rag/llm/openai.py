"""Thin wrapper around the OpenAI chat completions API for generation."""
from typing import Dict, List

from rag.config import settings


class OpenAILLM:
    def __init__(self, model: str = None, api_key: str = None, temperature: float = None):
        from openai import OpenAI  # lazy import

        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def stream(self, messages: List[Dict[str, str]]):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
