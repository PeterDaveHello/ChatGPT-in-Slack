import copy
import json
from types import SimpleNamespace

import pytest
import httpx
from openai import OpenAI

from app import openai_ops as ops
from app.openai_responses_ops import response_input, stream_function_responses
from app import openai_responses_ops as responses_ops


class Item(SimpleNamespace):
    def model_dump(self, **kwargs):
        return vars(self).copy()


class FakeStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


def completed(*items):
    return SimpleNamespace(
        type="response.completed", response=SimpleNamespace(output=list(items))
    )


def make_client(*batches):
    requests = []
    streams = [FakeStream(events) for events in batches]

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return streams[len(requests) - 1]

    return SimpleNamespace(responses=SimpleNamespace(create=create)), requests, streams


def make_module():
    return SimpleNamespace(
        functions=[{"name": "add", "parameters": {"type": "object", "properties": {}}}],
        add=lambda x: str(x + 1),
    )


@pytest.mark.parametrize("api_type", ["openai", "azure"])
def test_astra_tools_route_and_replay_reasoning(monkeypatch, api_type):
    reasoning = Item(
        type="reasoning", id="rs_1", summary=[], encrypted_content="encrypted"
    )
    call = Item(
        type="function_call", name="add", arguments='{"x": 2}', call_id="call_1"
    )
    client, requests, streams = make_client(
        [completed(reasoning, call)],
        [SimpleNamespace(type="response.output_text.delta", delta="3"), completed()],
    )
    monkeypatch.setattr(ops, "build_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(ops, "import_module", lambda name: make_module())
    stream = ops.start_receiving_openai_response(
        openai_api_key="k",
        model="gpt-6-astra",
        temperature=0.5,
        messages=[{"role": "user", "content": "add one to two"}],
        user="U1",
        openai_api_type=api_type,
        openai_api_base="https://example.invalid",
        openai_deployment_id="astra-deployment",
        openai_organization_id=None,
        function_call_module_name="fake_functions",
    )
    chunks = list(stream)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "3"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert len(requests) == 2
    assert requests[0]["model"] == (
        "astra-deployment" if api_type == "azure" else "gpt-6-astra"
    )
    assert requests[0]["tools"][0]["strict"] is False
    assert requests[0]["store"] is False
    assert requests[0]["include"] == ["reasoning.encrypted_content"]
    assert (
        not {"temperature", "top_p", "functions", "max_completion_tokens"}
        & requests[0].keys()
    )
    assert requests[1]["input"][-3:] == [
        reasoning.model_dump(),
        call.model_dump(),
        {"type": "function_call_output", "call_id": "call_1", "output": "3"},
    ]
    assert all(stream.closed for stream in streams)


def test_response_input_preserves_text_and_image():
    messages = [
        {"role": "system", "content": "Be helpful"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA", "detail": "low"},
                },
            ],
        },
    ]
    assert response_input(messages) == [
        messages[0],
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AA",
                    "detail": "low",
                },
            ],
        },
    ]


@pytest.mark.parametrize("event_type", ["response.failed", "error", "truncated"])
def test_response_errors_are_not_success(event_type):
    client, _, streams = make_client([SimpleNamespace(type=event_type)])
    with pytest.raises(RuntimeError):
        list(
            stream_function_responses(
                client=client,
                model="gpt-6-astra",
                messages=[],
                user="U1",
                module=make_module(),
            )
        )
    assert streams[0].closed


@pytest.mark.parametrize("event_type", ["response.failed", "error"])
def test_response_errors_preserve_diagnostic_message(event_type):
    detail = "Upstream model is unavailable"
    event = SimpleNamespace(
        type=event_type,
        **(
            {"message": detail}
            if event_type == "error"
            else {"response": SimpleNamespace(error=SimpleNamespace(message=detail))}
        ),
    )
    client, _, streams = make_client([event])
    with pytest.raises(RuntimeError, match=detail):
        list(
            stream_function_responses(
                client=client,
                model="gpt-6-astra",
                messages=[],
                user="U1",
                module=make_module(),
            )
        )
    assert streams[0].closed


def test_incomplete_response_does_not_execute_partial_function():
    event = SimpleNamespace(
        type="response.incomplete",
        response=SimpleNamespace(
            incomplete_details=SimpleNamespace(reason="max_output_tokens")
        ),
    )
    client, requests, streams = make_client([event])
    chunks = list(
        stream_function_responses(
            client=client,
            model="gpt-6-astra",
            messages=[],
            user="U1",
            module=make_module(),
        )
    )
    assert chunks[-1].choices[0].finish_reason == "length"
    assert len(requests) == 1
    assert streams[0].closed


def test_closing_adapter_closes_underlying_stream():
    client, _, streams = make_client([SimpleNamespace(type="response.created")])
    stream = stream_function_responses(
        client=client, model="gpt-6-astra", messages=[], user="U1", module=make_module()
    )
    next(stream)
    stream.close()
    assert streams[0].closed


def test_unknown_function_is_not_executed():
    call = Item(type="function_call", name="unlisted", arguments="{}", call_id="call_1")
    client, _, _ = make_client([completed(call)])
    module = make_module()
    module.unlisted = lambda: pytest.fail("Unlisted function must not execute")
    with pytest.raises(ValueError, match="unknown function"):
        list(
            stream_function_responses(
                client=client,
                model="gpt-6-astra",
                messages=[],
                user="U1",
                module=module,
            )
        )


def test_astra_function_token_probe_uses_responses(monkeypatch):
    requests = []
    monkeypatch.setattr(ops, "OPENAI_TIMEOUT_SECONDS", 7)

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=30 if kwargs["tools"] else 10)
        )

    monkeypatch.setattr(ops, "_prompt_tokens_used_by_function_call_cache", None)
    monkeypatch.setattr(
        ops,
        "create_openai_client",
        lambda context: SimpleNamespace(responses=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(ops, "import_module", lambda name: make_module())
    context = {
        "OPENAI_MODEL": "gpt-6-astra",
        "OPENAI_FUNCTION_CALL_MODULE_NAME": "fake",
    }
    assert ops.calculate_tokens_necessary_for_function_call(context) == 20
    assert len(requests) == 2
    assert requests[0]["tools"][0]["type"] == "function"
    assert requests[1]["tools"] == []
    for request in requests:
        assert request["reasoning"] == {"effort": "low"}
        assert request["timeout"] == 7
    assert ops.calculate_tokens_necessary_for_function_call(context) == 20
    assert len(requests) == 2


def test_real_sdk_serializes_tool_round_trip():
    requests = []

    def handle(request):
        assert request.url.path == "/v1/responses"
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            output = [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "name": "add",
                    "arguments": '{"x": 2}',
                    "call_id": "call_1",
                    "status": "completed",
                }
            ]
            events = []
        else:
            assert body["input"][-1] == {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "3",
            }
            output = []
            events = [
                {
                    "type": "response.output_text.delta",
                    "delta": "3",
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": 1,
                }
            ]
        events.append(
            {
                "type": "response.completed",
                "sequence_number": 2,
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "model": "gpt-6-astra",
                    "output": output,
                },
            }
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join("data: " + json.dumps(event) + "\n\n" for event in events),
        )

    with OpenAI(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handle))
    ) as client:
        chunks = list(
            stream_function_responses(
                client=client,
                model="gpt-6-astra",
                messages=[{"role": "user", "content": "add one to two"}],
                user="U1",
                module=make_module(),
            )
        )
    assert len(requests) == 2
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "3"


def test_tool_round_trimming_preserves_current_prompt_and_call_pairs(monkeypatch):
    monkeypatch.setitem(responses_ops.MODEL_CONTEXT_LENGTHS, "gpt-6-astra", 1100)
    system = [{"role": "system", "content": "system"}]
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old reply"},
    ]
    prompt = [{"role": "user", "content": "current"}]
    round_items = [
        {"type": "reasoning"},
        {"type": "function_call", "call_id": "c"},
        {"type": "function_call_output", "call_id": "c", "output": "result"},
    ]
    groups = [(system, 10), (history, 50), (prompt, 10), (round_items, 30)]
    assert (
        responses_ops._trim_groups(groups, [], prompt) == system + prompt + round_items
    )


def test_oversized_tool_result_stops_before_followup(monkeypatch):
    monkeypatch.setitem(responses_ops.MODEL_CONTEXT_LENGTHS, "gpt-6-astra", 1200)
    call = Item(
        type="function_call", name="add", arguments='{"x": 2}', call_id="call_1"
    )
    client, requests, streams = make_client([completed(call)])
    module = make_module()
    module.add = lambda x: "large result " * 1000
    with pytest.raises(ValueError, match="context budget"):
        list(
            stream_function_responses(
                client=client,
                model="gpt-6-astra",
                messages=[{"role": "user", "content": "add"}],
                user="U1",
                module=module,
            )
        )
    assert len(requests) == 1
    assert streams[0].closed
