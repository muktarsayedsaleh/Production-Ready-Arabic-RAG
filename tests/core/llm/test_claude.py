import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaMessage

from core.llm import ClaudeClient, Message, TextPart, TransientLLMError
from core.llm.claude import FALLBACK_BETA


class FakeMessages:
    def __init__(self, response: BetaMessage | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> BetaMessage:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeAnthropic:
    def __init__(self, response: BetaMessage | Exception) -> None:
        self.messages = FakeMessages(response)
        self.beta = SimpleNamespace(messages=self.messages)


def make_message(
    *,
    text: str = "answer",
    model: str = "claude-opus-5-5",
    stop_reason: str = "end_turn",
    usage: dict[str, Any] | None = None,
    stop_details: dict[str, Any] | None = None,
) -> BetaMessage:
    payload: dict[str, Any] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 1000, "output_tokens": 500},
    }
    if stop_details is not None:
        payload["stop_details"] = stop_details
    return BetaMessage.model_validate(payload)


def make_client(
    response: BetaMessage | Exception, **kwargs: Any
) -> tuple[ClaudeClient, FakeMessages]:
    fake = FakeAnthropic(response)
    client = ClaudeClient(client=fake, **kwargs)  # type: ignore[arg-type]
    return client, fake.messages


def status_error(status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError(
        f"HTTP {status}", response=httpx2.Response(status, request=request), body=None
    )


def ask(client: ClaudeClient, **kwargs: Any) -> Any:
    return asyncio.run(client.complete([Message(role="user", content="Question?")], **kwargs))


def test_parses_text_usage_and_cost() -> None:
    client, _ = make_client(make_message(text="الجواب"))

    response = ask(client)

    assert response.text == "الجواب"
    assert response.stop_reason == "end"
    assert response.model == "claude-opus-5-5"
    assert response.usage.input_tokens == 1000
    assert response.usage.output_tokens == 500
    # 1000 * $4/M + 500 * $20/M
    assert response.cost_usd == pytest.approx(0.004 + 0.010)
    assert not response.served_by_fallback
    assert response.unpriced_models == ()


def test_sends_effort_and_default_refusal_fallback() -> None:
    client, calls = make_client(make_message(), effort="high")

    ask(client)

    call = calls.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "high"}
    assert call["betas"] == [FALLBACK_BETA]
    assert call["fallbacks"] == "default"
    assert "thinking" not in call


def test_refusal_fallback_can_be_turned_off() -> None:
    client, calls = make_client(make_message(), refusal_fallback=False)

    ask(client)

    assert "betas" not in calls.calls[0]
    assert "fallbacks" not in calls.calls[0]


def test_cache_marked_parts_get_cache_control_with_configured_ttl() -> None:
    client, calls = make_client(make_message(), cache_ttl="1h")
    system = [TextPart("Stable instructions", cache=True)]
    messages = [
        Message(
            role="user",
            content=[TextPart("Fixed document set", cache=True), TextPart("Question?")],
        )
    ]

    asyncio.run(client.complete(messages, system=system))

    call = calls.calls[0]
    assert call["system"] == [
        {
            "type": "text",
            "text": "Stable instructions",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    assert call["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "Fixed document set",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
        {"type": "text", "text": "Question?"},
    ]


def test_plain_string_content_is_never_cached() -> None:
    client, calls = make_client(make_message())

    ask(client, system="You answer from context.")

    call = calls.calls[0]
    assert call["system"] == [{"type": "text", "text": "You answer from context."}]
    assert call["messages"][0]["content"] == [{"type": "text", "text": "Question?"}]


def test_more_than_four_cache_breakpoints_is_rejected_before_calling() -> None:
    client, calls = make_client(make_message())
    parts = [TextPart(f"part {i}", cache=True) for i in range(5)]

    with pytest.raises(ValueError, match="at most 4"):
        asyncio.run(client.complete([Message(role="user", content=parts)]))
    assert calls.calls == []


def test_cache_reads_and_writes_are_priced_by_ttl() -> None:
    usage = {
        "input_tokens": 100,
        "output_tokens": 200,
        "cache_read_input_tokens": 10_000,
        "cache_creation_input_tokens": 3_000,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 2_000,
            "ephemeral_1h_input_tokens": 1_000,
        },
    }
    client, _ = make_client(make_message(usage=usage))

    response = ask(client)

    assert response.usage.cache_read_tokens == 10_000
    assert response.usage.cache_write_tokens == 3_000
    assert response.usage.total_input_tokens == 13_100
    expected = (
        100 * 4.00  # uncached input
        + 200 * 20.00  # output
        + 10_000 * 0.20  # cache reads, Opus 5.5 rate
        + 2_000 * 4.00 * 1.25  # 5-minute writes
        + 1_000 * 4.00 * 2.0  # 1-hour writes
    ) / 1_000_000
    assert response.cost_usd == pytest.approx(expected)


def test_cache_writes_without_breakdown_use_the_configured_ttl() -> None:
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 1_000_000}
    client, _ = make_client(make_message(usage=usage), cache_ttl="1h")

    response = ask(client)

    assert response.cost_usd == pytest.approx(4.00 * 2.0)


def test_fallback_attempts_are_summed_and_priced_per_model() -> None:
    usage = {
        "input_tokens": 1000,
        "output_tokens": 300,
        "iterations": [
            # Declined by Opus 5.5 after some output, billed at its rates.
            {
                "type": "message",
                "model": "claude-opus-5-5",
                "input_tokens": 1000,
                "output_tokens": 50,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            # Served by the fallback model, billed at its rates.
            {
                "type": "fallback_message",
                "model": "claude-opus-4-8",
                "input_tokens": 1000,
                "output_tokens": 300,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        ],
    }
    client, _ = make_client(make_message(model="claude-opus-4-8", usage=usage))

    response = ask(client)

    assert response.served_by_fallback
    assert response.model == "claude-opus-4-8"
    assert response.stop_reason == "end"
    assert response.usage.input_tokens == 2000
    assert response.usage.output_tokens == 350
    opus_55 = (1000 * 4.00 + 50 * 20.00) / 1_000_000
    opus_48 = (1000 * 5.00 + 300 * 25.00) / 1_000_000
    assert response.cost_usd == pytest.approx(opus_55 + opus_48)


def test_unpriced_fallback_model_is_reported_not_hidden(caplog: pytest.LogCaptureFixture) -> None:
    usage = {
        "input_tokens": 10,
        "output_tokens": 10,
        "iterations": [
            {
                "type": "fallback_message",
                "model": "claude-some-future-model",
                "input_tokens": 10,
                "output_tokens": 10,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        ],
    }
    client, _ = make_client(make_message(model="claude-some-future-model", usage=usage))

    with caplog.at_level(logging.WARNING):
        response = ask(client)

    assert response.unpriced_models == ("claude-some-future-model",)
    assert response.cost_usd == 0.0
    assert "lower bound" in caplog.text


def test_refusal_is_reported_with_category() -> None:
    message = make_message(
        text="",
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": None},
    )
    client, _ = make_client(message)

    response = ask(client)

    assert response.stop_reason == "refusal"
    assert response.refusal_category == "cyber"
    assert response.text == ""


@pytest.mark.parametrize(
    ("api_reason", "expected"),
    [("end_turn", "end"), ("stop_sequence", "end"), ("max_tokens", "max_tokens")],
)
def test_stop_reasons_are_normalised(api_reason: str, expected: str) -> None:
    client, _ = make_client(make_message(stop_reason=api_reason))

    assert ask(client).stop_reason == expected


@pytest.mark.parametrize("status", [408, 409, 429, 500, 529])
def test_retryable_status_codes_raise_transient_error(status: int) -> None:
    client, _ = make_client(status_error(status))

    with pytest.raises(TransientLLMError):
        ask(client)


def test_connection_errors_raise_transient_error() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client, _ = make_client(anthropic.APIConnectionError(request=request))

    with pytest.raises(TransientLLMError):
        ask(client)


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_not_retried(status: int) -> None:
    client, _ = make_client(status_error(status))

    with pytest.raises(anthropic.APIStatusError):
        ask(client)


def test_unpriced_model_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="No pricing configured"):
        ClaudeClient(model="claude-unknown", client=FakeAnthropic(make_message()))  # type: ignore[arg-type]
