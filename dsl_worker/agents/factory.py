"""
Agent conversation factory — picks the right conversation class for the
tracked client passed in. Callers stop caring which provider they're on.

Usage:
    agent = make_conversation(
        tracked_client,   # TrackedOpenAIClient or TrackedAnthropicClient
        model=...,
        system_prompt=...,
        tools=...,
        ...
    )
"""

from __future__ import annotations

from typing import Any, Union

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.anthropic_base import AnthropicAgentConversation
from dsl_worker.billing.tracked_anthropic_client import TrackedAnthropicClient
from dsl_worker.billing.tracked_client import TrackedOpenAIClient


Conversation = Union[AgentConversation, AnthropicAgentConversation]


def make_conversation(client: Any, **kwargs) -> Conversation:
    """Return the conversation class matching the tracked client type.

    All kwargs match AgentConversation.__init__ — including `openai_client`,
    which is positional-first in most callers. Pass it via this factory:
        make_conversation(tracked_client, openai_client=tracked_client, ...)
    or drop the duplicate by relying on positional binding.
    """
    if isinstance(client, TrackedAnthropicClient):
        # AnthropicAgentConversation accepts the same `openai_client` kwarg
        # name for drop-in compatibility.
        kwargs.setdefault("openai_client", client)
        return AnthropicAgentConversation(**kwargs)
    if isinstance(client, TrackedOpenAIClient):
        kwargs.setdefault("openai_client", client)
        return AgentConversation(**kwargs)
    raise TypeError(
        f"Unsupported tracked client type: {type(client).__name__}. "
        f"Expected TrackedOpenAIClient or TrackedAnthropicClient."
    )


__all__ = [
    "make_conversation",
    "Conversation",
    "AgentResult",
]
