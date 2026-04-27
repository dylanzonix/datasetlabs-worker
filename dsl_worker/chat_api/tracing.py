"""Optional Langfuse tracing for the chat worker API.

If LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY are set in the environment
(loaded by app.py via load_dotenv), the SDK auto-configures and traces
get sent. Otherwise every helper here is a no-op and chat continues to
work exactly the same.

Usage in streaming.py:

    with start_trace("chat_send_message", user_id, project_id, user_content):
        with start_generation("openai.responses", model=...) as gen:
            ... stream ...
            update_generation(gen, output=text, usage=..., cost_usd=...)
        with start_span("tool", name=tool_name, input=args) as span:
            ... run tool ...
            update_span(span, output=result, cost_usd=cost)
"""
from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Dict, Iterator, Optional

log = logging.getLogger(__name__)

try:
    from langfuse import get_client as _get_lf_client  # type: ignore

    def _get_lf():
        try:
            return _get_lf_client()
        except Exception:
            return None
except ImportError:  # pragma: no cover
    def _get_lf():
        return None


def is_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_SECRET_KEY")) and _get_lf() is not None


@contextlib.contextmanager
def start_trace(
    name: str,
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    input_text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Optional[Any]]:
    lf = _get_lf()
    if lf is None:
        yield None
        return
    md = dict(metadata or {})
    if project_id:
        md["project_id"] = project_id
    # Set up the langfuse span. If setup fails, degrade to no-op tracing.
    # See start_generation for the rationale (yielding twice is illegal).
    try:
        cm = lf.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_text,
            metadata=md,
        )
    except Exception:
        log.exception("langfuse start_trace setup failed; continuing without tracing")
        yield None
        return
    with cm as span:
        try:
            trace_kwargs: Dict[str, Any] = {
                "name": name,
                "input": input_text,
                "metadata": md,
            }
            if user_id:
                trace_kwargs["user_id"] = str(user_id)
            if project_id:
                trace_kwargs["session_id"] = str(project_id)
            span.update_trace(**trace_kwargs)
        except Exception:
            pass
        yield span


@contextlib.contextmanager
def start_generation(
    name: str,
    model: Optional[str] = None,
    input_payload: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Optional[Any]]:
    lf = _get_lf()
    if lf is None:
        yield None
        return
    # Set up the langfuse observation. If the SETUP fails (langfuse misconfig,
    # transient API issue), degrade to no-op tracing — don't fail the user's
    # request. If the user's `with` body raises, propagate it normally;
    # contextmanager forbids yielding twice, so the prior version's
    # `except: yield None` actually masked real errors with a misleading
    # "generator didn't stop after throw()".
    try:
        cm = lf.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input_payload,
            metadata=metadata or {},
        )
    except Exception:
        log.exception("langfuse start_generation setup failed; continuing without tracing")
        yield None
        return
    with cm as gen:
        yield gen


def update_generation(
    gen: Optional[Any],
    *,
    output: Any = None,
    usage: Optional[Dict[str, Any]] = None,
    cost_usd: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if gen is None:
        return
    try:
        kwargs: Dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if usage is not None:
            kwargs["usage_details"] = usage
        if cost_usd is not None:
            kwargs["cost_details"] = {"total": cost_usd}
        if metadata is not None:
            kwargs["metadata"] = metadata
        if kwargs:
            gen.update(**kwargs)
    except Exception:
        log.exception("langfuse update_generation failed")


@contextlib.contextmanager
def start_span(
    name: str,
    input_payload: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Optional[Any]]:
    lf = _get_lf()
    if lf is None:
        yield None
        return
    try:
        cm = lf.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_payload,
            metadata=metadata or {},
        )
    except Exception:
        log.exception("langfuse start_span setup failed; continuing without tracing")
        yield None
        return
    with cm as span:
        yield span


def update_span(
    span: Optional[Any],
    *,
    output: Any = None,
    cost_usd: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if span is None:
        return
    try:
        kwargs: Dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if cost_usd is not None:
            kwargs["cost_details"] = {"total": cost_usd}
        if metadata is not None:
            kwargs["metadata"] = metadata
        if kwargs:
            span.update(**kwargs)
    except Exception:
        log.exception("langfuse update_span failed")


def flush() -> None:
    """Best-effort flush so traces aren't lost when a request ends."""
    lf = _get_lf()
    if lf is None:
        return
    try:
        lf.flush()
    except Exception:
        pass
