"""Ollama provider using the ollama Python library."""

from __future__ import annotations

import uuid
from typing import Any

import json_repair

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_STRIP_PREFIXES = ("ollama/", "ollama_chat/")


class OllamaProvider(LLMProvider):
    """Ollama provider using ollama.AsyncClient."""

    def __init__(
        self,
        api_base: str = "http://localhost:11434",
        default_model: str = "llama3.2",
    ):
        super().__init__(api_key=None, api_base=api_base)
        self.default_model = default_model
        self._api_base = api_base.rstrip("/")

    def _get_client(self) -> Any:
        from ollama import AsyncClient
        return AsyncClient(host=self._api_base)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        resolved_model = model or self.default_model
        for prefix in _STRIP_PREFIXES:
            if resolved_model.startswith(prefix):
                resolved_model = resolved_model[len(prefix):]

        try:
            client = self._get_client()
            response = await client.chat(
                model=resolved_model,
                messages=self._sanitize_empty_content(messages),
                tools=tools or [],
                stream=False,
                options={"temperature": temperature, "num_predict": max(1, max_tokens)},
            )
            return self._parse(response)
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _parse(self, response: Any) -> LLMResponse:
        msg = response.message
        tool_calls: list[ToolCallRequest] = []
        for tc in msg.tool_calls or []:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json_repair.loads(args)
            tool_calls.append(
                ToolCallRequest(
                    id=uuid.uuid4().hex[:9],
                    name=tc.function.name,
                    arguments=args if isinstance(args, dict) else {},
                )
            )

        usage: dict[str, int] = {}
        if (n := getattr(response, "prompt_eval_count", None)) is not None:
            usage["prompt_tokens"] = n
        if (n := getattr(response, "eval_count", None)) is not None:
            usage["completion_tokens"] = n
        if len(usage) == 2:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return LLMResponse(
            content=msg.content or None,
            tool_calls=tool_calls,
            finish_reason=getattr(response, "done_reason", None) or "stop",
            usage=usage,
        )

    def get_default_model(self) -> str:
        return self.default_model
