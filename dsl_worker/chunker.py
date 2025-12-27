"""
File chunking strategies for different file types.
"""
import json
import logging
from typing import List
import tiktoken

logger = logging.getLogger(__name__)


def chunk_csv(content: bytes, chunk_size: int = 50) -> List[str]:
    """
    Chunk CSV by rows, including header in each chunk.

    Args:
        content: Raw CSV file bytes
        chunk_size: Number of rows per chunk

    Returns:
        List of CSV chunks (each with header)
    """
    text = content.decode('utf-8', errors='ignore')
    lines = text.split('\n')

    if not lines:
        return []

    header = lines[0]
    chunks = []

    for i in range(1, len(lines), chunk_size):
        chunk_lines = [header] + lines[i:i + chunk_size]
        chunk_text = '\n'.join(chunk_lines)
        if chunk_text.strip():
            chunks.append(chunk_text)

    return chunks


def chunk_jsonl(content: bytes) -> List[str]:
    """
    Chunk JSONL by lines (each line is one record).

    Args:
        content: Raw JSONL file bytes

    Returns:
        List of JSON strings (one per line)
    """
    text = content.decode('utf-8', errors='ignore')
    chunks = []

    for line in text.split('\n'):
        line = line.strip()
        if line:
            try:
                obj = json.loads(line)
                chunks.append(json.dumps(obj, indent=2))
            except json.JSONDecodeError:
                # Not valid JSON, keep as text
                chunks.append(line)

    return chunks


def chunk_json_array(content: bytes) -> List[str]:
    """
    Chunk JSON array by items.

    Args:
        content: Raw JSON file bytes

    Returns:
        List of JSON strings (one per array item)
    """
    text = content.decode('utf-8', errors='ignore')

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [json.dumps(item, indent=2) for item in data]
        else:
            # Single object
            return [json.dumps(data, indent=2)]
    except json.JSONDecodeError:
        return []


def chunk_text_by_tokens(
    content: bytes,
    chunk_size: int = 512,
    overlap: int = 50,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Chunk text by tokens with overlap.

    Args:
        content: Raw file bytes
        chunk_size: Target tokens per chunk
        overlap: Number of overlapping tokens between chunks
        encoding_name: tiktoken encoding to use

    Returns:
        List of text chunks
    """
    text = content.decode('utf-8', errors='ignore')

    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to load encoding {encoding_name}, using cl100k_base: {e}")
        encoding = tiktoken.get_encoding("cl100k_base")

    # Encode entire text
    tokens = encoding.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move forward by (chunk_size - overlap)
        start += chunk_size - overlap

    return chunks