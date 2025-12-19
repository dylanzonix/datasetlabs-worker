import os
import json
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional
import tiktoken
import pandas as pd
from unstructured.partition.auto import partition


class SourceDataEngine:
    """
    Main engine for processing source data and retrieving chunks by topic.

    Manages:
    - Processing source files into topic-classified chunks
    - Indexing chunks by topic
    - Round-robin retrieval of chunks for a given topic
    """

    def __init__(self, topic_tree: dict, openai_client, use_web: bool = False):
        """
        Initialize the source data engine.

        Args:
            topic_tree: Topic tree dictionary with 'title' and 'children' keys
            openai_client: Configured OpenAI client for LLM judge
            use_web: Whether to use web search in LLM calls (future feature)
        """
        self.topic_tree = topic_tree
        self.openai_client = openai_client
        self.use_web = use_web

        # State management
        self.chunk_results = {}  # chunk_id -> topic_path
        self.topic_index = {}  # topic_key -> [chunk_ids]
        self.topic_counters = {}  # topic_key -> current_index
        self.output_location = None  # Where chunks are stored

    def process_seed_data(self, seed_location: str, output_location: str, chunk_size: int = 512):
        """
        Process all source files and classify chunks by topic.

        Args:
            seed_location: Directory containing source files
            output_location: Directory to save processed chunks
            chunk_size: Maximum tokens per chunk (default: 512)
        """
        from worker.source_engine.chunk_handler import LLMJudgeChunkHandler

        # Store output location for later retrieval
        self.output_location = output_location

        # Create LLM judge handler
        handler = LLMJudgeChunkHandler(self.openai_client, self.topic_tree, self.use_web)

        # Process files
        results = _process_source_files(seed_location, output_location, chunk_size, handler)

        # Index chunks by topic
        for chunk_id, topic_path in results.items():
            if topic_path is not None:  # Skip irrelevant chunks
                topic_key = self._topic_path_to_key(topic_path)

                # Initialize topic index if needed
                if topic_key not in self.topic_index:
                    self.topic_index[topic_key] = []
                    self.topic_counters[topic_key] = 0

                # Add chunk to topic index
                self.topic_index[topic_key].append(chunk_id)
                self.chunk_results[chunk_id] = topic_path

        print(f"\n=== Indexing Summary ===")
        print(f"Total chunks processed: {len(results)}")
        print(f"Relevant chunks indexed: {len(self.chunk_results)}")
        print(f"Unique topics found: {len(self.topic_index)}")
        print(f"\nTopic distribution:")
        for topic_key, chunk_ids in sorted(self.topic_index.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {topic_key}: {len(chunk_ids)} chunks")

    def get_source(self, topic_path: List[Dict[str, str]], instruction: Optional[str] = None) -> Optional[str]:
        """
        Get a chunk for the specified topic with round-robin iteration.

        Args:
            topic_path: Path from root to leaf topic (list of dicts with 'title' key)
            instruction: Optional instruction for future processing (not used yet)

        Returns:
            Chunk text, or None if no chunks available for this topic
        """
        topic_key = self._topic_path_to_key(topic_path)

        # Check if topic exists in index
        if topic_key not in self.topic_index or not self.topic_index[topic_key]:
            return None

        # Get chunk IDs for this topic
        chunk_ids = self.topic_index[topic_key]

        # Get current index (round-robin)
        current_idx = self.topic_counters[topic_key]
        chunk_id = chunk_ids[current_idx]

        # Update counter (loop back to 0 when reaching end)
        self.topic_counters[topic_key] = (current_idx + 1) % len(chunk_ids)

        # Read chunk from disk
        chunk_text = self._read_chunk_from_disk(chunk_id)

        return chunk_text

    def _topic_path_to_key(self, topic_path: List[Dict[str, str]]) -> str:
        """Convert topic path to string key for indexing."""
        return " > ".join([node["title"] for node in topic_path])

    def _read_chunk_from_disk(self, chunk_id: int) -> Optional[str]:
        """Read a chunk file from disk."""
        if self.output_location is None:
            return None

        chunk_file = Path(self.output_location) / f"chunk_{chunk_id}.txt"

        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading chunk {chunk_id}: {e}")
            return None

    def get_topic_stats(self) -> Dict[str, int]:
        """Get statistics on chunks per topic."""
        return {topic: len(chunks) for topic, chunks in self.topic_index.items()}


def _process_source_files(
        seed_location: str,
        output_location: str,
        chunk_size: int,
        chunk_handler: Any
) -> Dict[int, Any]:
    """
    Internal function to process all files in seed_location, chunk them, and process with handler.

    Args:
        seed_location: Directory containing source files
        output_location: Directory to save processed chunks
        chunk_size: Maximum tokens per chunk for text, or rows for tables
        chunk_handler: Object with process_with_llm(chunk: str) -> Any method

    Returns:
        Dictionary mapping chunk_id to handler results
    """
    # Initialize
    os.makedirs(output_location, exist_ok=True)

    # Try to get tiktoken encoding, fallback to simple chunking if unavailable
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        print("Using tiktoken for tokenization")
    except Exception as e:
        print(f"Warning: Could not load tiktoken encoding ({e})")
        print("Falling back to simple word-based chunking")
        encoding = None

    results_map = {}
    chunk_id = 0

    # Get all files in seed_location
    seed_path = Path(seed_location)
    if not seed_path.exists():
        raise ValueError(f"Seed location does not exist: {seed_location}")

    files = [f for f in seed_path.iterdir() if f.is_file()]

    print(f"Found {len(files)} files to process")

    for file_path in files:
        print(f"Processing: {file_path.name}")

        # Get chunks from file (generator for memory efficiency)
        chunks = _get_chunks_from_file(file_path, chunk_size, encoding)

        # Process each chunk
        for chunk_text in chunks:
            # Get LLM result
            llm_result = chunk_handler.process_with_llm(chunk_text)

            # Store result
            results_map[chunk_id] = llm_result

            # Save chunk to disk
            chunk_file = Path(output_location) / f"chunk_{chunk_id}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk_text)

            print(f"  Processed chunk {chunk_id}")
            chunk_id += 1

    print(f"\nProcessing complete. Total chunks: {chunk_id}")
    return results_map


def _get_chunks_from_file(file_path: Path, chunk_size: int, encoding) -> Generator[str, None, None]:
    """
    Get chunks from a file based on its type.
    Returns a generator to avoid loading everything in memory.
    """
    extension = file_path.suffix.lower()

    # Handle structured data formats
    if extension == '.csv':
        yield from _process_csv(file_path, chunk_size, encoding)
    elif extension == '.jsonl':
        yield from _process_jsonl(file_path, chunk_size, encoding)
    elif extension == '.json':
        # Try to parse as array first
        is_array, chunks = _process_json(file_path, chunk_size, encoding)
        if is_array:
            yield from chunks
        else:
            # Treat as regular text file
            yield from _process_text_file(file_path, chunk_size, encoding)
    else:
        # All other files treated as text
        yield from _process_text_file(file_path, chunk_size, encoding)


def _count_tokens(text: str, encoding) -> int:
    """Count tokens in text, with fallback to word count if encoding unavailable."""
    if encoding is not None:
        return len(encoding.encode(text))
    else:
        # Fallback: approximate 1 token ≈ 0.75 words (rough estimate)
        return int(len(text.split()) * 1.33)


def _chunk_by_tokens(text: str, chunk_size: int, encoding) -> List[str]:
    """Chunk text by tokens, with fallback to word-based chunking."""
    chunks = []

    if encoding is not None:
        # Use proper token-based chunking
        tokens = encoding.encode(text)
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
    else:
        # Fallback: word-based chunking
        words = text.split()
        # Approximate chunk size in words (assuming 1 token ≈ 0.75 words)
        words_per_chunk = int(chunk_size * 0.75)

        for i in range(0, len(words), words_per_chunk):
            chunk_words = words[i:i + words_per_chunk]
            chunks.append(' '.join(chunk_words))

    return chunks


def _process_row(row_text: str, chunk_size: int, encoding) -> Generator[str, None, None]:
    """Process a single row, chunking if it exceeds chunk_size tokens."""
    token_count = _count_tokens(row_text, encoding)

    if token_count <= chunk_size:
        # Row fits in one chunk
        yield row_text
    else:
        # Row too large, chunk it by tokens
        for chunk in _chunk_by_tokens(row_text, chunk_size, encoding):
            yield chunk


def _process_csv(file_path: Path, chunk_size: int, encoding) -> Generator[str, None, None]:
    """Process CSV file row-by-row."""
    # Read CSV in chunks to avoid memory issues
    for chunk_df in pd.read_csv(file_path, chunksize=1000):
        for _, row in chunk_df.iterrows():
            row_text = row.to_json()
            yield from _process_row(row_text, chunk_size, encoding)


def _process_jsonl(file_path: Path, chunk_size: int, encoding) -> Generator[str, None, None]:
    """Process JSONL file line-by-line using pandas for memory efficiency."""
    # Read JSONL in chunks to avoid memory issues with large files
    for chunk_df in pd.read_json(file_path, lines=True, chunksize=1000):
        for _, row in chunk_df.iterrows():
            row_text = row.to_json()
            yield from _process_row(row_text, chunk_size, encoding)


def _process_json(file_path: Path, chunk_size: int, encoding) -> tuple[bool, Generator[str, None, None]]:
    """
    Process JSON file - if array, row-by-row; otherwise return False.
    Returns (is_array, chunks_generator)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if it's an array
        if isinstance(data, list):
            def json_array_chunks():
                for item in data:
                    item_text = json.dumps(item)
                    yield from _process_row(item_text, chunk_size, encoding)

            return True, json_array_chunks()
        else:
            # Not an array, treat as text
            return False, None
    except:
        # If parsing fails, treat as text
        return False, None


def _process_text_file(file_path: Path, chunk_size: int, encoding) -> Generator[str, None, None]:
    """Process text file using unstructured and chunk by tokens."""
    try:
        # Try to extract text using unstructured
        elements = partition(filename=str(file_path))
        full_text = '\n'.join([str(el) for el in elements])
    except Exception as e:
        # Fallback: just read the file directly
        print(f"  Warning: unstructured failed ({type(e).__name__}), reading file directly")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()

    # Chunk by tokens using helper function
    chunks = _chunk_by_tokens(full_text, chunk_size, encoding)

    for chunk in chunks:
        yield chunk