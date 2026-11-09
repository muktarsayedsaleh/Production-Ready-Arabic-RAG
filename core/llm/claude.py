"""Claude implementation of `LLMClient`, with prompt caching and refusal fallbacks.

TODO: backport prompt caching (cache_control on marked parts) and cache-aware cost to
Production-Ready-Claude-Agent, whose ClaudeClient sends no cache markers and prices only
plain input and output tokens.
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal, cast

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessage, BetaMessageParam, BetaTextBlockParam

from core.llm.base import (
    Content,
    LLMClient,
    LLMResponse,
    Message,
    StopReason,
    TextPart,
    TokenUsage,
    TransientLLMError,
)
from core.llm.pricing import PRICING, CacheTTL, pricing_for

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# The API accepts at most four cache breakpoints per request.
MAX_CACHE_BREAKPOINTS = 4

# With `fallbacks="default"`, a request declined by a safety classifier is re-run
# server-side on the model Anthropic recommends for that refusal category.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_TRANSIENT_STATUS_CODES = {408, 409, 429}

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end",
    "stop_sequence": "end",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


class ClaudeClient(LLMClient):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        # Set explicitly: Claude Opus 5.5 defaults to "medium", other models to "high".
        effort: Effort = "medium",
        cache_ttl: CacheTTL = "5m",
        refusal_fallback: bool = True,
        max_retries: int = 2,
        client: AsyncAnthropic | None = None,
    ) -> None:
        # Resolve pricing up front so an unpriced model fails at startup, not mid-request.
        pricing_for(model)
        self._model = model
        self._effort = effort
        self._cache_ttl: CacheTTL = cache_ttl
        self._refusal_fallback = refusal_fallback
        # The SDK retries 408/409/429/5xx and connection errors with backoff. What is still
        # failing after that surfaces as TransientLLMError.
        self._client = client or AsyncAnthropic(max_retries=max_retries)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: Content | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        system_blocks = self._to_blocks(system) if system is not None else []
        message_params = [self._to_message_param(message) for message in messages]
        _check_breakpoints(system, messages)

        kwargs: dict[str, Any] = {}
        if system_blocks:
            kwargs["system"] = system_blocks
        if self._refusal_fallback:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        try:
            response = await self._client.beta.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=message_params,
                output_config={"effort": self._effort},
                **kwargs,
            )
        except anthropic.APIConnectionError as exc:
            raise TransientLLMError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code in _TRANSIENT_STATUS_CODES or exc.status_code >= 500:
                raise TransientLLMError(str(exc)) from exc
            raise

        return self._to_response(response)

    def _to_blocks(self, content: Content) -> list[BetaTextBlockParam]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        blocks: list[BetaTextBlockParam] = []
        for part in content:
            block: BetaTextBlockParam = {"type": "text", "text": part.text}
            if part.cache:
                block["cache_control"] = {"type": "ephemeral", "ttl": self._cache_ttl}
            blocks.append(block)
        return blocks

    def _to_message_param(self, message: Message) -> BetaMessageParam:
        # A plain dict: the SDK's message TypedDicts are split per role and mypy can't pick one.
        return cast(
            BetaMessageParam, {"role": message.role, "content": self._to_blocks(message.content)}
        )

    def _to_response(self, response: BetaMessage) -> LLMResponse:
        text = "".join(block.text for block in response.content if block.type == "text")
        stop_reason = _STOP_REASONS.get(response.stop_reason or "", "other")
        refusal_category = None
        if stop_reason == "refusal" and response.stop_details is not None:
            refusal_category = response.stop_details.category

        usage, cost, unpriced, served_by_fallback = self._account(response)
        if unpriced:
            logger.warning("No pricing for %s; cost_usd is a lower bound", ", ".join(unpriced))

        return LLMResponse(
            text=text,
            stop_reason=stop_reason,
            usage=usage,
            cost_usd=cost,
            model=response.model,
            served_by_fallback=served_by_fallback,
            refusal_category=refusal_category,
            unpriced_models=unpriced,
        )

    def _account(self, response: BetaMessage) -> tuple[TokenUsage, float, tuple[str, ...], bool]:
        """Sum tokens and cost over every billed attempt.

        With fallbacks, top-level `usage` covers only the attempt that produced the returned
        message; `usage.iterations` lists every attempt (the declined one included), each
        billed at the rates of the model that ran it.
        """
        iterations = [
            item
            for item in (response.usage.iterations or [])
            if item.type in ("message", "fallback_message")
        ]
        attempts: list[tuple[str, Any]] = (
            [(item.model or self._model, item) for item in iterations]
            if iterations
            else [(response.model, response.usage)]
        )
        served_by_fallback = any(item.type == "fallback_message" for item in iterations)

        totals = {"input": 0, "output": 0, "write": 0, "read": 0}
        cost = 0.0
        unpriced: list[str] = []
        for model, usage in attempts:
            write_5m, write_1h = self._split_cache_writes(usage)
            totals["input"] += usage.input_tokens
            totals["output"] += usage.output_tokens
            totals["write"] += write_5m + write_1h
            totals["read"] += usage.cache_read_input_tokens or 0
            pricing = PRICING.get(model)
            if pricing is None:
                if model not in unpriced:
                    unpriced.append(model)
                continue
            cost += pricing.cost(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_input_tokens or 0,
                cache_write_5m_tokens=write_5m,
                cache_write_1h_tokens=write_1h,
            )

        usage_total = TokenUsage(
            input_tokens=totals["input"],
            output_tokens=totals["output"],
            cache_write_tokens=totals["write"],
            cache_read_tokens=totals["read"],
        )
        return usage_total, cost, tuple(unpriced), served_by_fallback

    def _split_cache_writes(self, usage: Any) -> tuple[int, int]:
        """Cache-write tokens by lifetime, which are priced differently."""
        breakdown = getattr(usage, "cache_creation", None)
        if breakdown is not None:
            return breakdown.ephemeral_5m_input_tokens, breakdown.ephemeral_1h_input_tokens
        written = usage.cache_creation_input_tokens or 0
        return (written, 0) if self._cache_ttl == "5m" else (0, written)


def _check_breakpoints(system: Content | None, messages: Sequence[Message]) -> None:
    contents = [system, *(message.content for message in messages)]
    marked = sum(
        1
        for content in contents
        if content is not None and not isinstance(content, str)
        for part in content
        if isinstance(part, TextPart) and part.cache
    )
    if marked > MAX_CACHE_BREAKPOINTS:
        raise ValueError(
            f"{marked} parts are marked for caching; Claude allows at most "
            f"{MAX_CACHE_BREAKPOINTS} cache breakpoints per request"
        )
