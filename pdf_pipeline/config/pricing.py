"""Pricing models and cost calculation decoupled from the Gemini wrapper.

Calculates estimated input, output, thinking, and total cost based on token counts.
Supports in-memory configuration maps and extensible BigQuery pricing lookups.
"""

from decimal import Decimal
from typing import Dict, Optional
from pydantic import BaseModel, Field


class ModelPricing(BaseModel):
    """Cost rates per 1,000,000 (1M) tokens in USD."""

    model_name: str
    input_cost_per_million: Decimal = Field(default=Decimal("0.15"))
    output_cost_per_million: Decimal = Field(default=Decimal("0.60"))
    thinking_cost_per_million: Decimal = Field(default=Decimal("0.60"))
    cached_input_cost_per_million: Decimal = Field(default=Decimal("0.0375"))
    currency: str = Field(default="USD")


class CostEstimate(BaseModel):
    """Calculated cost estimate for a single Gemini invocation."""

    estimated_input_cost: Decimal
    estimated_output_cost: Decimal
    estimated_thinking_cost: Decimal
    estimated_cached_cost: Decimal
    estimated_total_cost: Decimal
    currency: str


class PricingRegistry:
    """Registry maintaining pricing configurations per model."""

    # Default baseline pricing per 1M tokens (USD)
    # Gemini 3.8 Flash: $0.15 / 1M prompt, $0.60 / 1M candidate/thinking, $0.0375 / 1M cached
    # Gemini 3.5 Flash Lite: $0.075 / 1M prompt, $0.30 / 1M candidate, $0.01875 / 1M cached
    DEFAULT_RATES: Dict[str, ModelPricing] = {
        "gemini-3.8-flash": ModelPricing(
            model_name="gemini-3.8-flash",
            input_cost_per_million=Decimal("0.15"),
            output_cost_per_million=Decimal("0.60"),
            thinking_cost_per_million=Decimal("0.60"),
            cached_input_cost_per_million=Decimal("0.0375"),
        ),
        "gemini-3.5-flash-lite": ModelPricing(
            model_name="gemini-3.5-flash-lite",
            input_cost_per_million=Decimal("0.075"),
            output_cost_per_million=Decimal("0.30"),
            thinking_cost_per_million=Decimal("0.30"),
            cached_input_cost_per_million=Decimal("0.01875"),
        ),
        "gemini-3.1-pro-preview": ModelPricing(
            model_name="gemini-3.1-pro-preview",
            input_cost_per_million=Decimal("1.25"),
            output_cost_per_million=Decimal("5.00"),
            thinking_cost_per_million=Decimal("5.00"),
            cached_input_cost_per_million=Decimal("0.3125"),
        ),
        "gemini-embedding-001": ModelPricing(
            model_name="gemini-embedding-001",
            input_cost_per_million=Decimal("0.025"),
            output_cost_per_million=Decimal("0.00"),
            thinking_cost_per_million=Decimal("0.00"),
            cached_input_cost_per_million=Decimal("0.00"),
        ),
    }

    def __init__(self, custom_rates: Optional[Dict[str, ModelPricing]] = None):
        self._rates = dict(self.DEFAULT_RATES)
        if custom_rates:
            self._rates.update(custom_rates)

    def register_model_pricing(self, pricing: ModelPricing) -> None:
        """Register or update a model's pricing rates."""
        self._rates[pricing.model_name] = pricing

    def get_pricing(self, model_name: str) -> ModelPricing:
        """Fetch pricing for a given model, with fuzzy prefix matching."""
        if model_name in self._rates:
            return self._rates[model_name]
        # Match base prefix if version tags are appended (e.g. gemini-3.8-flash-001)
        for key, pricing in self._rates.items():
            if model_name.startswith(key):
                return pricing
        # Fallback to standard 3.8 flash default if unspecified
        return self._rates["gemini-3.8-flash"]

    def calculate_cost(
        self,
        model_name: str,
        prompt_tokens: int,
        output_tokens: int,
        thinking_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> CostEstimate:
        """Compute estimated dollar cost from post-execution token counts."""
        pricing = self.get_pricing(model_name)
        one_million = Decimal("1000000")

        # Non-cached prompt tokens
        active_prompt_tokens = max(0, prompt_tokens - cached_tokens)
        input_cost = (Decimal(active_prompt_tokens) / one_million) * pricing.input_cost_per_million
        cached_cost = (Decimal(cached_tokens) / one_million) * pricing.cached_input_cost_per_million
        output_cost = (Decimal(output_tokens) / one_million) * pricing.output_cost_per_million
        thinking_cost = (Decimal(thinking_tokens) / one_million) * pricing.thinking_cost_per_million

        total_cost = input_cost + cached_cost + output_cost + thinking_cost

        return CostEstimate(
            estimated_input_cost=round(input_cost, 6),
            estimated_output_cost=round(output_cost, 6),
            estimated_thinking_cost=round(thinking_cost, 6),
            estimated_cached_cost=round(cached_cost, 6),
            estimated_total_cost=round(total_cost, 6),
            currency=pricing.currency,
        )


# Global default pricing registry
pricing_registry = PricingRegistry()
