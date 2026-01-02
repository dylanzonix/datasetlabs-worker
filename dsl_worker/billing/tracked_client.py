"""
Tracked OpenAI client that calculates costs for all API calls.

Wraps the AsyncOpenAI client and returns cost information alongside responses.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Any, Dict

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse

from dsl_worker.billing.pricing import get_pricing_config, UsageCost

logger = logging.getLogger(__name__)


@dataclass
class TrackedChatCompletion:
    """Chat completion response with cost tracking."""
    response: ChatCompletion
    cost: UsageCost


@dataclass
class TrackedEmbeddingResponse:
    """Embedding response with cost tracking."""
    response: CreateEmbeddingResponse
    cost: UsageCost


class TrackedOpenAIClient:
    """
    Wrapper around AsyncOpenAI that tracks costs for all API calls.

    Usage:
        client = TrackedOpenAIClient(AsyncOpenAI())

        # Chat completion
        result = await client.chat_completion(model="gpt-4o", messages=[...])
        print(result.cost.total_cost_usd)

        # Embeddings
        result = await client.create_embeddings(model="text-embedding-3-large", input=[...])
        print(result.cost.total_cost_usd)
    """

    def __init__(self, client: AsyncOpenAI):
        self._client = client
        self._pricing = get_pricing_config()

    @property
    def raw_client(self) -> AsyncOpenAI:
        """Access the underlying OpenAI client for unsupported operations."""
        return self._client

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> TrackedChatCompletion:
        """
        Create a chat completion and track the cost.

        Returns:
            TrackedChatCompletion with response and cost
        """
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        # Extract token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        cost = self._pricing.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        logger.debug(
            f"Chat completion: model={model}, "
            f"input={input_tokens}, output={output_tokens}, "
            f"cost=${cost.total_cost_usd:.6f}"
        )

        return TrackedChatCompletion(response=response, cost=cost)

    async def create_embeddings(
        self,
        model: str,
        input: List[str],
        **kwargs,
    ) -> TrackedEmbeddingResponse:
        """
        Create embeddings and track the cost.

        Returns:
            TrackedEmbeddingResponse with response and cost
        """
        response = await self._client.embeddings.create(
            model=model,
            input=input,
            encoding_format="float",
            **kwargs,
        )

        # Extract token usage
        usage = response.usage
        input_tokens = usage.total_tokens if usage else 0

        cost = self._pricing.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=0,  # Embeddings have no output tokens
        )

        logger.debug(
            f"Embeddings: model={model}, "
            f"input={input_tokens}, "
            f"cost=${cost.total_cost_usd:.6f}"
        )

        return TrackedEmbeddingResponse(response=response, cost=cost)

    async def responses_create(
        self,
        prompt: Dict[str, Any],
        input: List[Any],
        model: str = "o1",
        **kwargs,
    ) -> Any:
        """
        Create a response using the responses API (for prompt management).

        Note: This is a special API that may have different cost tracking.
        For now, we estimate based on the model.

        Returns:
            Tuple of (response, cost)
        """
        response = await self._client.responses.create(
            prompt=prompt,
            input=input,
            model=model,
            **kwargs,
        )

        # The responses API doesn't always return token counts directly
        # We need to extract from response.usage if available
        input_tokens = 0
        output_tokens = 0

        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)

        cost = self._pricing.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        logger.debug(
            f"Responses API: model={model}, "
            f"input={input_tokens}, output={output_tokens}, "
            f"cost=${cost.total_cost_usd:.6f}"
        )

        return response, cost