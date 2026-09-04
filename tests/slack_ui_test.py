from unittest.mock import MagicMock
import pytest

from app.slack_ui import build_configure_modal
from app.openai_constants import (
    GPT_5_3_CHAT_LATEST_MODEL,
    GPT_5_4_MODEL,
    GPT_5_4_MINI_MODEL,
    GPT_5_4_NANO_MODEL,
    GPT_5_5_MODEL,
    GPT_6_ASTRA_MODEL,
    GPT_5_6_SOL_MODEL,
    GPT_5_6_TERRA_MODEL,
    GPT_5_6_LUNA_MODEL,
    CHAT_LATEST_MODEL,
)


def make_context(*, api_key=None, model=None):
    context = MagicMock()

    def _get(key, default=None):
        values = {
            "OPENAI_API_KEY": api_key,
            "OPENAI_MODEL": model,
        }
        return values.get(key, default)

    context.get.side_effect = _get
    return context


def test_build_configure_modal_includes_new_models():
    modal = build_configure_modal(make_context())

    options = modal["blocks"][1]["element"]["options"]
    values = [option["value"] for option in options]

    assert values[:12] == [
        GPT_6_ASTRA_MODEL,
        GPT_5_6_SOL_MODEL,
        GPT_5_6_TERRA_MODEL,
        GPT_5_6_LUNA_MODEL,
        GPT_5_5_MODEL,
        CHAT_LATEST_MODEL,
        GPT_5_4_MODEL,
        GPT_5_4_MINI_MODEL,
        GPT_5_4_NANO_MODEL,
        GPT_5_3_CHAT_LATEST_MODEL,
        "gpt-5.2-chat-latest",
        "gpt-5.2",
    ]


def test_build_configure_modal_keeps_saved_model_selected(monkeypatch):
    monkeypatch.setattr(
        "app.slack_ui.translate",
        lambda *, text, **kwargs: text,
    )
    modal = build_configure_modal(
        make_context(api_key="sk-test", model=GPT_5_3_CHAT_LATEST_MODEL)
    )

    initial_option = modal["blocks"][1]["element"]["initial_option"]

    assert initial_option["value"] == GPT_5_3_CHAT_LATEST_MODEL


@pytest.mark.parametrize(
    "model,expected",
    [
        (GPT_5_6_LUNA_MODEL, GPT_5_6_LUNA_MODEL),
        (GPT_6_ASTRA_MODEL, GPT_6_ASTRA_MODEL),
        ("gpt-5.6", GPT_5_6_SOL_MODEL),
    ],
)
@pytest.mark.parametrize("api_key", [None, "sk-test"])
def test_configure_modal_preserves_configured_model(
    monkeypatch, model, expected, api_key
):
    monkeypatch.setattr("app.slack_ui.translate", lambda *, text, **kwargs: text)
    modal = build_configure_modal(make_context(api_key=api_key, model=model))
    assert modal["blocks"][1]["element"]["initial_option"]["value"] == expected


def test_configure_modal_uses_environment_model_without_saved_selection(monkeypatch):
    monkeypatch.setattr("app.slack_ui.OPENAI_MODEL", GPT_5_6_LUNA_MODEL)
    modal = build_configure_modal(make_context())
    assert (
        modal["blocks"][1]["element"]["initial_option"]["value"] == GPT_5_6_LUNA_MODEL
    )
