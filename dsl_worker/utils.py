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


def truncate_to_tokens(text: str, max_tokens: int, suffix: str = "\n[truncated]") -> str:
    """Truncate text to fit within max_tokens. Appends suffix if truncated."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    tokens = _encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    suffix_tokens = _encoding.encode(suffix)
    kept = max_tokens - len(suffix_tokens)
    if kept <= 0:
        return suffix
    return _encoding.decode(tokens[:kept]) + suffix
