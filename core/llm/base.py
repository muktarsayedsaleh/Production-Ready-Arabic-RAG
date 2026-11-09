"""Provider-agnostic interface for one model call.

RAG generation is a single request and response: a system prompt, the retrieved context
and the question go in, an answer comes out. There is no tool use here, so the types are
smaller than in Production-Ready-Claude-Agent.

Prompt caching is expressed as a hint (`TextPart.cache`) so callers don't depend on one
provider. Providers that cache explicitly (Claude) place a cache breakpoint after a
marked part; providers that cache automatically or not at all ignore the hint.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["user", "assistant"]

# Why the model stopped. "refusal" means a safety classifier declined the request (after
# any fallback the provider ran); callers must not treat its text as an answer.
StopReason = Literal["end", "max_tokens", "refusal", "other"]


@dataclass(frozen=True)
class TextPart:
    text: str
    # Ask the provider to cache the prompt up to and including this part. Only mark parts
    # that are identical across requests (system prompt, instructions, a fixed document
    # set); marking per-request content pays the cache-write premium for nothing.
    cache: bool = False


Content = str | Sequence[TextPart]


@dataclass(frozen=True)
class Message:
    role: Role
    content: Content


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one call, summed over every attempt the provider billed."""

    input_tokens: int = 0  # uncached input only
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens


@dataclass(frozen=True)
class LLMResponse:
    text: str
    stop_reason: StopReason
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    # The model that produced `text`. Differs from the requested model when a provider-side
    # fallback served the request.
    model: str = ""
    served_by_fallback: bool = False
    # Provider category for a refusal (e.g. "cyber"), when one is given.
    refusal_category: str | None = None
    # Models that were billed for this call but have no entry in the pricing table. When
    # non-empty, `cost_usd` is a lower bound, not the full cost.
    unpriced_models: tuple[str, ...] = ()


class TransientLLMError(Exception):
    """A failure worth retrying later: rate limits, overload, timeouts, connection errors."""


class LLMClient(ABC):
    """One model call. Implementations must be safe to share across concurrent requests."""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: Content | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse: ...
