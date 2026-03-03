"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

import secrets
import string
from typing import Any

import httpx
import json_repair
from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_ALNUM = string.ascii_letters + string.digits


def _short_tool_id() -> str:
    """Generate a 9-char alphanumeric ID compatible with all providers."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


class CustomProvider(LLMProvider):

    def __init__(self, api_key: str = "no-key", api_base: str = "http://localhost:8000/v1", default_model: str = "default"):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=httpx.Timeout(600.0, connect=30.0),
        )

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
                   reasoning_effort: str | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._sanitize_empty_content(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
            "stream": True,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        try:
            stream = await self._client.chat.completions.create(**kwargs)
            return await self._consume_stream(stream)
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    async def _consume_stream(self, stream: Any) -> LLMResponse:
        """Buffer a streaming response into a single LLMResponse."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}

        async for chunk in stream:
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = {
                        "prompt_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(chunk.usage, "total_tokens", 0) or 0,
                    }
                continue

            delta = chunk.choices[0].delta

            if hasattr(delta, "content") and delta.content:
                content_parts.append(delta.content)

            # reasoning_content (Kimi, Fireworks) or reasoning (Together)
            rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if rc:
                reasoning_parts.append(rc)

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index if hasattr(tc_delta, "index") and tc_delta.index is not None else 0
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {"id": "", "name": "", "arguments": ""}
                    entry = tool_calls_by_index[idx]
                    if hasattr(tc_delta, "id") and tc_delta.id:
                        entry["id"] = tc_delta.id
                    if hasattr(tc_delta, "function") and tc_delta.function:
                        if hasattr(tc_delta.function, "name") and tc_delta.function.name:
                            entry["name"] = tc_delta.function.name
                        if hasattr(tc_delta.function, "arguments") and tc_delta.function.arguments:
                            entry["arguments"] += tc_delta.function.arguments

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            if hasattr(chunk, "usage") and chunk.usage:
                usage = {
                    "prompt_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(chunk.usage, "total_tokens", 0) or 0,
                }

        # Assemble tool calls
        tool_calls: list[ToolCallRequest] = []
        for idx in sorted(tool_calls_by_index):
            entry = tool_calls_by_index[idx]
            args = entry["arguments"]
            if isinstance(args, str) and args:
                args = json_repair.loads(args)
            elif not args:
                args = {}
            tool_calls.append(ToolCallRequest(
                id=_short_tool_id(),
                name=entry["name"],
                arguments=args,
            ))

        content = "".join(content_parts) or None
        reasoning = "".join(reasoning_parts) or None

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content=reasoning,
        )

    def get_default_model(self) -> str:
        return self.default_model
