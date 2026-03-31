"""
Pricing configuration loader and cost calculator.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Path to pricing config
PRICING_CONFIG_PATH = Path(__file__).parent / "pricing.yaml"


@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a single model (USD per token)."""
    input_per_token: float
    output_per_token: float
    cached_input_per_token: float = 0.0


@dataclass(frozen=True)
class UsageCost:
    """Cost breakdown for a single API call."""
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    cached_input_tokens: int = 0
    cached_input_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd + self.cached_input_cost_usd


class PricingConfig:
    """Loads and provides access to model pricing."""

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or PRICING_CONFIG_PATH
        self._models: Dict[str, ModelPricing] = {}
        self._load()

    def _load(self):
        """Load pricing from YAML file."""
        try:
            with open(self._config_path) as f:
                data = yaml.safe_load(f)

            for model_name, prices in data.get("models", {}).items():
                self._models[model_name] = ModelPricing(
                    input_per_token=prices.get("input", 0.0),
                    output_per_token=prices.get("output", 0.0),
                    cached_input_per_token=prices.get("cached_input", 0.0),
                )

            logger.info(f"Loaded pricing for {len(self._models)} models")

        except Exception as e:
            logger.error(f"Failed to load pricing config: {e}")
            raise

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
    ) -> UsageCost:
        """
        Calculate cost for an API call.

        Args:
            model: Model name (e.g., "gpt-4o")
            input_tokens: Number of input tokens (non-cached portion)
            output_tokens: Number of output tokens
            cached_input_tokens: Number of cached input tokens

        Returns:
            UsageCost with breakdown

        Raises:
            ValueError if model pricing not found
        """
        pricing = self._models.get(model)
        if not pricing:
            raise ValueError(f"No pricing found for model: {model}")

        input_cost = input_tokens * pricing.input_per_token
        output_cost = output_tokens * pricing.output_per_token
        cached_input_cost = cached_input_tokens * pricing.cached_input_per_token

        return UsageCost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            cached_input_tokens=cached_input_tokens,
            cached_input_cost_usd=cached_input_cost,
        )


# Global singleton
_pricing_config: Optional[PricingConfig] = None


def get_pricing_config() -> PricingConfig:
    """Get the global pricing config singleton."""
    global _pricing_config
    if _pricing_config is None:
        _pricing_config = PricingConfig()
    return _pricing_config