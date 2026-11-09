import pytest

from core.llm.pricing import ModelPricing, pricing_for


def test_cost_adds_every_token_kind_at_its_rate() -> None:
    pricing = ModelPricing(input_per_mtok=4.00, output_per_mtok=20.00, cache_read_per_mtok=0.20)

    cost = pricing.cost(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_5m_tokens=1_000_000,
        cache_write_1h_tokens=1_000_000,
    )

    assert cost == pytest.approx(4.00 + 20.00 + 0.20 + 5.00 + 8.00)


def test_cost_of_nothing_is_zero() -> None:
    assert pricing_for("claude-opus-5-5").cost() == 0.0


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="claude-nope"):
        pricing_for("claude-nope")
