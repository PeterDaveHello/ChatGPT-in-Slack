"""Adapt Responses function calling to the existing Slack text stream."""

import json
from functools import lru_cache
from typing import Any, Generator

import tiktoken
from openai.types.chat import ChatCompletionChunk

from app.openai_constants import GPT_6_ASTRA_MODEL, MAX_TOKENS, MODEL_CONTEXT_LENGTHS


@lru_cache(maxsize=1)
def _token_encoding() -> tiktoken.Encoding:
    # Match the existing fallback for models unknown to the installed tokenizer.
    try:
        return tiktoken.encoding_for_model(GPT_6_ASTRA_MODEL)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _estimated_tokens(value: Any) -> int:
    if isinstance(value, dict):
        value = {
            key: item
            for key, item in value.items()
            if key not in ("image_url", "encrypted_content")
        }
        return sum(_estimated_tokens(item) for item in value.values()) + 4
    if isinstance(value, list):
        return sum(_estimated_tokens(item) for item in value)
    return len(_token_encoding().encode(str(value)))


def _trim_groups(groups: list, tools: list, prompt_group: list) -> list:
    budget = MODEL_CONTEXT_LENGTHS[GPT_6_ASTRA_MODEL] - MAX_TOKENS - 1
    while sum(cost for _, cost in groups) + _estimated_tokens(tools) > budget:
        for index, (items, _) in enumerate(groups[:-1]):
            if items is prompt_group:
                raise ValueError(
                    "GPT-6 Astra tool conversation exceeds the context budget"
                )
            if not any(item.get("role") in ("system", "developer") for item in items):
                del groups[index]
                break
        else:
            raise ValueError("GPT-6 Astra tool conversation exceeds the context budget")
    return [item for items, _ in groups for item in items]


def response_tools(functions: list) -> list:
    # Responses defaults to strict schemas; preserve legacy optional arguments.
    return [{**function, "type": "function", "strict": False} for function in functions]


def response_input(messages: list) -> list:
    items = []
    pending_call_id = None
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "function":
            if pending_call_id is None:
                continue  # A context-window trim may have removed the call.
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": pending_call_id,
                    "output": message["content"],
                }
            )
            pending_call_id = None
            continue
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append({"type": "input_text", "text": part["text"]})
                elif part["type"] == "image_url":
                    image = part["image_url"]
                    parts.append(
                        {
                            "type": "input_image",
                            "image_url": image["url"],
                            "detail": image.get("detail", "auto"),
                        }
                    )
            content = parts
        if content:
            items.append({"role": role, "content": content})
        if message.get("function_call"):
            call = message["function_call"]
            pending_call_id = f"legacy_call_{index}"
            items.append({"type": "function_call", "call_id": pending_call_id, **call})
    return items


def _chunk(model: str, content=None, finish_reason=None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="responses-adapter",
        created=0,
        model=model,
        object="chat.completion.chunk",
        choices=[
            {
                "index": 0,
                "delta": {"content": content} if content is not None else {},
                "finish_reason": finish_reason,
            }
        ],
    )


def stream_function_responses(
    *, client: Any, model: str, messages: list, user: str, module: Any
) -> Generator[ChatCompletionChunk, None, None]:
    inputs = response_input(messages)
    tools = response_tools(module.functions)
    # Keep each historical user turn and each tool round intact during trimming.
    groups = []
    for item in inputs:
        if not groups or item.get("role") in ("user", "system", "developer"):
            groups.append(([item], _estimated_tokens(item)))
        else:
            items, cost = groups[-1]
            items.append(item)
            groups[-1] = (items, cost + _estimated_tokens(item))
    prompt_group = groups[-1][0] if groups else []
    allowed_names = {tool["name"] for tool in tools}
    while True:
        inputs = _trim_groups(groups, tools, prompt_group)
        stream = client.responses.create(
            model=model,
            input=inputs,
            tools=tools,
            user=user,
            max_output_tokens=MAX_TOKENS,
            stream=True,
            store=False,
            include=["reasoning.encrypted_content"],
        )
        response = None
        try:
            for event in stream:
                # Also give the caller a chance to enforce its deadline while reasoning.
                yield _chunk(model)
                if event.type in (
                    "response.output_text.delta",
                    "response.refusal.delta",
                ):
                    yield _chunk(model, content=event.delta)
                elif event.type == "response.completed":
                    response = event.response
                    break
                elif event.type == "response.incomplete":
                    reason = event.response.incomplete_details.reason
                    yield _chunk(
                        model,
                        finish_reason=(
                            "length"
                            if reason == "max_output_tokens"
                            else "content_filter"
                        ),
                    )
                    return
                elif event.type in ("response.failed", "error"):
                    error = getattr(getattr(event, "response", None), "error", None)
                    detail = getattr(event, "message", None) or getattr(
                        error, "message", None
                    )
                    message = "OpenAI Responses request failed"
                    if detail:
                        message += f": {detail}"
                    raise RuntimeError(message)
        finally:
            stream.close()
        if response is None:
            raise RuntimeError("OpenAI Responses stream ended before completion")
        # Replay all output, including encrypted reasoning, without server-side storage.
        round_items = [item.model_dump(exclude_none=True) for item in response.output]
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            yield _chunk(model, finish_reason="stop")
            return
        for call in calls:
            yield _chunk(model)
            if call.name not in allowed_names:
                raise ValueError(f"OpenAI requested an unknown function: {call.name}")
            result = getattr(module, call.name)(**json.loads(call.arguments))
            round_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result if isinstance(result, str) else json.dumps(result),
                }
            )
        usage = getattr(response, "usage", None)
        # Opaque reasoning cannot be tokenized locally; reserve the reported output
        # usage in addition to the visible round as a conservative estimate.
        reasoning_reserve = getattr(usage, "output_tokens", 0) or 0
        groups.append((round_items, _estimated_tokens(round_items) + reasoning_reserve))
