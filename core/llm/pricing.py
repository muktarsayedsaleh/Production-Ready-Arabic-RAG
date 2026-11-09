"""Per-model token prices and cost calculation, including prompt-cache reads and writes.

TODO: backport cache-aware pricing to Production-Ready-Claude-Agent (core/llm/pricing.py
there only prices plain input and output tokens).
"""

from dataclasses import dataclass
from typing import Literal

CacheTTL = Literal["5m", "1h"]

# Cache writes cost a multiple of the base input price, set by the cache lifetime.
CACHE_WRITE_MULTIPLIER: dict[CacheTTL, float] = {"5m": 1.25, "1h": 2.0}


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    # Cache reads are priced per model (not a fixed fraction of input on every model).
    cache_read_per_mtok: float

    def cost(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_5m_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
    ) -> float:
        write_5m = self.input_per_mtok * CACHE_WRITE_MULTIPLIER["5m"]
        write_1h = self.input_per_mtok * CACHE_WRITE_MULTIPLIER["1h"]
        return (
            input_tokens * self.input_per_mtok
            + output_tokens * self.output_per_mtok
            + cache_read_tokens * self.cache_read_per_mtok
            + cache_write_5m_tokens * write_5m
            + cache_write_1h_tokens * write_1h
        ) / 1_000_000


# Anthropic first-party API rates. Keep in sync with https://www.anthropic.com/pricing
# The Opus 4.8, Opus 5 and Sonnet 5 entries are here because server-side refusal
# fallbacks can route a request to them, and their tokens are billed at their own rates.
# Claude Haiku 5.5 is left out on purpose: its prices are tiered by prompt length, which
# this table can't express yet.
PRICING: dict[str, ModelPricing] = {
    "claude-fable-5-1": ModelPricing(10.00, 50.00, cache_read_per_mtok=0.25),
    "claude-opus-5-5": ModelPricing(4.00, 20.00, cache_read_per_mtok=0.20),
    "claude-opus-5": ModelPricing(5.00, 25.00, cache_read_per_mtok=0.50),
    "claude-opus-4-8": ModelPricing(5.00, 25.00, cache_read_per_mtok=0.50),
    "claude-sonnet-5-5": ModelPricing(2.00, 10.00, cache_read_per_mtok=0.20),
    "claude-sonnet-5": ModelPricing(2.00, 10.00, cache_read_per_mtok=0.20),
}


def pricing_for(model: str) -> ModelPricing:
    try:
        return PRICING[model]
    except KeyError:
        raise ValueError(f"No pricing configured for model {model!r}") from None
