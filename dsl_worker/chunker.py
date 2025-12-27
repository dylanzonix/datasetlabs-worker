"""
File chunking strategies for different file types.

Philosophy:
- Structured data (CSV, JSONL, JSON): Each record is one chunk
- If a single record exceeds token limit, split it by tokens
- Text files: Token-based chunking with overlap
"""
import json
import logging
import csv
from io import StringIO
from typing import List
import tiktoken

logger = logging.getLogger(__name__)


def chunk_csv(
    content: bytes,
    max_tokens: int = 7000,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Chunk CSV by rows (properly parsed).

    Strategy:
    - Use csv.reader to properly parse rows (handles quotes, newlines in fields, etc.)
    - Each row becomes one chunk (as CSV with header)
    - If a single row exceeds max_tokens, split its content by tokens

    Args:
        content: Raw CSV file bytes
        max_tokens: Max tokens per chunk (default: 7000, leaves room for header)
        encoding_name: tiktoken encoding to use

    Returns:
        List of CSV chunks (each row with header, or token-split if row is huge)
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to load encoding {encoding_name}, using cl100k_base: {e}")
        encoding = tiktoken.get_encoding("cl100k_base")

    text = content.decode('utf-8', errors='ignore')

    # Properly parse CSV
    csv_reader = csv.reader(StringIO(text))
    rows = list(csv_reader)

    if not rows:
        return []

    # First row is header
    header = rows[0]
    header_csv = _row_to_csv(header)
    header_tokens = len(encoding.encode(header_csv))

    chunks = []

    # Process each data row
    for row in rows[1:]:
        # Skip completely empty rows
        if not any(field.strip() for field in row):
            continue

        row_csv = _row_to_csv(row)
        row_tokens = len(encoding.encode(row_csv))

        # Check if this single row + header fits
        total_tokens = header_tokens + 1 + row_tokens  # +1 for newline

        if total_tokens <= max_tokens:
            # Normal case: row fits
            chunk = f"{header_csv}\n{row_csv}"
            chunks.append(chunk)
        else:
            # Edge case: single row is too big, split it by tokens
            logger.warning(
                f"CSV row exceeds {max_tokens} tokens ({row_tokens} tokens), "
                f"splitting by tokens"
            )
            # Convert row to text and chunk by tokens
            row_text = row_csv
            text_chunks = chunk_text_by_tokens(
                row_text.encode('utf-8'),
                chunk_size=max_tokens,
                overlap=200
            )
            for text_chunk in text_chunks:
                chunks.append(f"{header_csv}\n{text_chunk}")

    logger.info(f"Chunked CSV into {len(chunks)} chunks (one per row)")
    return chunks


def _row_to_csv(row: List[str]) -> str:
    """Convert a list of fields to a CSV line."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(row)
    return output.getvalue().rstrip('\r\n')


def chunk_jsonl(
    content: bytes,
    max_tokens: int = 8000,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Chunk JSONL by records (one record per chunk).

    Strategy:
    - Each line is one JSON record
    - Parse and pretty-print each record
    - If a single record exceeds max_tokens, split by tokens

    Args:
        content: Raw JSONL file bytes
        max_tokens: Max tokens per chunk
        encoding_name: tiktoken encoding to use

    Returns:
        List of JSON strings (one per record, or token-split if record is huge)
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to load encoding {encoding_name}, using cl100k_base: {e}")
        encoding = tiktoken.get_encoding("cl100k_base")

    text = content.decode('utf-8', errors='ignore')
    chunks = []

    for line_num, line in enumerate(text.split('\n'), 1):
        line = line.strip()
        if not line:
            continue

        try:
            # Parse JSON
            obj = json.loads(line)
            # Pretty-print for better readability
            json_text = json.dumps(obj, indent=2)

            # Check token count
            tokens = len(encoding.encode(json_text))

            if tokens <= max_tokens:
                # Normal case: record fits
                chunks.append(json_text)
            else:
                # Edge case: single record is too big, split by tokens
                logger.warning(
                    f"JSONL record at line {line_num} exceeds {max_tokens} tokens "
                    f"({tokens} tokens), splitting by tokens"
                )
                text_chunks = chunk_text_by_tokens(
                    json_text.encode('utf-8'),
                    chunk_size=max_tokens,
                    overlap=200
                )
                chunks.extend(text_chunks)

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON at line {line_num}: {e}, keeping as text")
            # Not valid JSON, keep as text
            line_tokens = len(encoding.encode(line))
            if line_tokens <= max_tokens:
                chunks.append(line)
            else:
                # Split by tokens
                text_chunks = chunk_text_by_tokens(
                    line.encode('utf-8'),
                    chunk_size=max_tokens,
                    overlap=200
                )
                chunks.extend(text_chunks)

    logger.info(f"Chunked JSONL into {len(chunks)} chunks (one per record)")
    return chunks


def chunk_json_array(
    content: bytes,
    max_tokens: int = 8000,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Chunk JSON array by items (one item per chunk).

    Strategy:
    - Parse JSON array
    - Each array item becomes one chunk
    - If a single item exceeds max_tokens, split by tokens

    Args:
        content: Raw JSON file bytes
        max_tokens: Max tokens per chunk
        encoding_name: tiktoken encoding to use

    Returns:
        List of JSON strings (one per array item, or token-split if item is huge)
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to load encoding {encoding_name}, using cl100k_base: {e}")
        encoding = tiktoken.get_encoding("cl100k_base")

    text = content.decode('utf-8', errors='ignore')
    chunks = []

    try:
        data = json.loads(text)

        if isinstance(data, list):
            # Array of items
            for idx, item in enumerate(data):
                item_text = json.dumps(item, indent=2)
                item_tokens = len(encoding.encode(item_text))

                if item_tokens <= max_tokens:
                    # Normal case: item fits
                    chunks.append(item_text)
                else:
                    # Edge case: single item is too big, split by tokens
                    logger.warning(
                        f"JSON array item {idx} exceeds {max_tokens} tokens "
                        f"({item_tokens} tokens), splitting by tokens"
                    )
                    text_chunks = chunk_text_by_tokens(
                        item_text.encode('utf-8'),
                        chunk_size=max_tokens,
                        overlap=200
                    )
                    chunks.extend(text_chunks)
        else:
            # Single object (not an array)
            obj_text = json.dumps(data, indent=2)
            obj_tokens = len(encoding.encode(obj_text))

            if obj_tokens <= max_tokens:
                chunks.append(obj_text)
            else:
                # Single object is too big, split by tokens
                logger.warning(
                    f"JSON object exceeds {max_tokens} tokens "
                    f"({obj_tokens} tokens), splitting by tokens"
                )
                text_chunks = chunk_text_by_tokens(
                    obj_text.encode('utf-8'),
                    chunk_size=max_tokens,
                    overlap=200
                )
                chunks.extend(text_chunks)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return []

    logger.info(f"Chunked JSON into {len(chunks)} chunks")
    return chunks


def chunk_text_by_tokens(
    content: bytes,
    chunk_size: int = 4096,
    overlap: int = 512,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Chunk text by tokens with overlap.

    Used for:
    - Plain text files
    - Fallback when structured data items are too large

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