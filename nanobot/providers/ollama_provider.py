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
        api_key: str | None = None,
    ):
        import os
        resolved_key = api_key or os.environ.get("OLLAMA_API_KEY")
        super().__init__(api_key=resolved_key, api_base=api_base)
        self.default_model = default_model
        self._api_base = api_base.rstrip("/")

    def _get_client(self) -> Any:
        from ollama import AsyncClient
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return AsyncClient(host=self._api_base, headers=headers)

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
                messages=self._convert_messages(self._sanitize_empty_content(messages)),
                tools=self._convert_tools(tools) if tools else [],
                stream=False,
                options={"temperature": temperature, "num_predict": max(1, max_tokens)},
            )
            return self._parse(response)
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[Any]:
        """Convert OpenAI-format tools to ollama Tool objects.

        The ollama Property type only supports type/description/enum/items —
        extra JSON Schema fields (minimum, maximum, minLength, etc.) must be dropped.
        """
        from ollama import Tool

        result = []
        for t in tools:
            fn = t.get("function", {})
            params = fn.get("parameters", {})
            properties = {
                name: Tool.Function.Parameters.Property(
                    type=prop.get("type"),
                    description=prop.get("description"),
                    enum=prop.get("enum"),
                    items=prop.get("items"),
                )
                for name, prop in params.get("properties", {}).items()
            }
            result.append(Tool(
                type="function",
                function=Tool.Function(
                    name=fn.get("name", ""),
                    description=fn.get("description", ""),
                    parameters=Tool.Function.Parameters(
                        type="object",
                        properties=properties,
                        required=params.get("required", []),
                    ),
                ),
            ))
        return result

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages for the ollama client.

        The ollama Pydantic Message model expects tool_calls[].function.arguments
        to be a dict, not a JSON string (which is what OpenAI wire format uses).
        """
        import json
        result = []
        for msg in messages:
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                result.append(msg)
                continue
            converted_tcs = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                converted_tcs.append({
                    **tc,
                    "function": {**fn, "arguments": args},
                })
            result.append({**msg, "tool_calls": converted_tcs})
        return result

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
