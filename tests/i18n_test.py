from unittest.mock import MagicMock

import pytest

import app.i18n as i18n
from app.openai_constants import GPT_4O_MODEL, GPT_5_6_LUNA_MODEL, GPT_6_ASTRA_MODEL


@pytest.fixture(autouse=True)
def clear_translation_cache():
    i18n._translation_result_cache.clear()


def make_context(
    *,
    locale="zh-TW",
    api_type=None,
    api_base="https://api.example/v1",
    model=GPT_4O_MODEL,
    deployment_id="dep-1"
):
    context = MagicMock()

    def _get(key, default=None):
        values = {
            "locale": locale,
            "OPENAI_API_TYPE": api_type,
            "OPENAI_API_BASE": api_base,
            "OPENAI_DEPLOYMENT_ID": deployment_id,
            "OPENAI_MODEL": model,
        }
        return values.get(key, default)

    context.get.side_effect = _get
    return context


def test_translate_returns_original_text_without_api_key():
    text = "Translate me"

    assert (
        i18n.translate(openai_api_key=None, context=make_context(), text=text) == text
    )


def test_translate_returns_original_text_for_english_locale():
    text = "Translate me"

    assert (
        i18n.translate(
            openai_api_key="sk-test",
            context=make_context(locale="en-US"),
            text=text,
        )
        == text
    )


@pytest.mark.parametrize(
    "api_type,expected_model",
    [(None, GPT_5_6_LUNA_MODEL), ("azure", "dep-1")],
)
def test_translate_uses_configured_translation_model_and_caches(
    monkeypatch, api_type, expected_model
):
    captured = {"calls": 0, "kwargs": None}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["calls"] += 1
            captured["kwargs"] = kwargs
            return MagicMock(
                model_dump=lambda: {"choices": [{"message": {"content": "翻譯結果"}}]}
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    monkeypatch.setattr(
        i18n,
        "build_openai_client",
        lambda **kwargs: FakeClient(),
    )

    context = make_context(api_type=api_type)
    first = i18n.translate(openai_api_key="sk-test", context=context, text="Hello")
    second = i18n.translate(openai_api_key="sk-test", context=context, text="Hello")

    assert first == "翻譯結果"
    assert second == "翻譯結果"
    assert captured["calls"] == 1
    assert captured["kwargs"]["model"] == expected_model
    assert captured["kwargs"]["n"] == 1
    assert captured["kwargs"]["user"] == "system"
    assert captured["kwargs"]["max_completion_tokens"] == 1024
    assert "max_tokens" not in captured["kwargs"]
    assert captured["kwargs"]["reasoning_effort"] == "none"
    assert "temperature" not in captured["kwargs"]
    assert "top_p" not in captured["kwargs"]


@pytest.mark.parametrize(
    "api_type,deployment_id,expected_model,expected_effort",
    [
        (None, "astra-deployment", GPT_5_6_LUNA_MODEL, "none"),
        ("azure", "astra-deployment", "astra-deployment", "low"),
        ("azure", None, GPT_5_6_LUNA_MODEL, "none"),
    ],
)
def test_astra_translation_uses_compatible_deployment_parameters(
    monkeypatch, api_type, deployment_id, expected_model, expected_effort
):
    client = MagicMock()
    client.chat.completions.create.return_value.model_dump.return_value = {
        "choices": [{"message": {"content": "翻譯結果"}}]
    }
    monkeypatch.setattr(i18n, "build_openai_client", lambda **kwargs: client)
    result = i18n.translate(
        openai_api_key="test",
        context=make_context(
            api_type=api_type, model=GPT_6_ASTRA_MODEL, deployment_id=deployment_id
        ),
        text="Hello",
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert result == "翻譯結果"
    assert kwargs["model"] == expected_model
    assert kwargs["reasoning_effort"] == expected_effort
    assert kwargs["max_completion_tokens"] == 1024
    assert not {"max_tokens", "temperature", "top_p"} & kwargs.keys()


@pytest.mark.parametrize("empty_content", [None, "", "   "])
def test_empty_astra_translation_falls_back_without_caching(monkeypatch, empty_content):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(
            model_dump=lambda: {
                "choices": [
                    {"finish_reason": "length", "message": {"content": empty_content}}
                ]
            }
        ),
        MagicMock(
            model_dump=lambda: {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "翻譯結果"}}
                ]
            }
        ),
    ]
    monkeypatch.setattr(i18n, "build_openai_client", lambda **kwargs: client)
    context = make_context(api_type="azure", model=GPT_6_ASTRA_MODEL)
    assert (
        i18n.translate(openai_api_key="test", context=context, text="Hello") == "Hello"
    )
    assert not i18n._translation_result_cache
    assert (
        i18n.translate(openai_api_key="test", context=context, text="Hello")
        == "翻譯結果"
    )
    assert (
        i18n.translate(openai_api_key="test", context=context, text="Hello")
        == "翻譯結果"
    )
    assert client.chat.completions.create.call_count == 2
