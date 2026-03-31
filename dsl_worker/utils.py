"""Lightweight utilities for the worker."""

from __future__ import annotations

import tiktoken

_encoding: tiktoken.Encoding | None = None


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding. Lazy-loads the encoder."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return len(_encoding.encode(text))


