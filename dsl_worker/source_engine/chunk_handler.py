"""
LLM-as-judge chunk handler for topic classification.
"""

from typing import Optional, List, Dict
from openai import OpenAI


class LLMJudgeChunkHandler:
    """
    LLM-based judge that classifies chunks into topic tree leaves.

    Traverses the topic tree in a BFS-like manner, asking the LLM to select
    the most relevant topic at each level until reaching a leaf node.
    """

    def __init__(self, openai_client: OpenAI, topic_tree: dict, use_web: bool = False):
        """
        Initialize the LLM judge.

        Args:
            openai_client: Configured OpenAI client
            topic_tree: Topic tree dictionary with 'title' and 'children' keys
            use_web: Whether to use web search in LLM calls
        """
        self.client = openai_client
        self.topic_tree = topic_tree
        self.use_web = use_web
        self.prompt_id = "pmpt_6934f50f0ad48194a20e9a348c753119045275275dc8352b"
        self.prompt_version = "3"

    def process_with_llm(self, chunk: str) -> Optional[List[Dict[str, str]]]:
        """
        Classify a chunk into the most relevant topic path.

        Args:
            chunk: Text chunk to classify

        Returns:
            List of topic nodes from root to leaf, or None if irrelevant.
            Each node is a dict with 'title' key.
        """
        path = []
        current_node = self.topic_tree

        # Check root-level relevance first (single topic case)
        selected_idx = self._select_topic([current_node["title"]], chunk)
        if selected_idx == -1:
            return None

        # Add root to path
        path.append({"title": current_node["title"]})

        # Traverse down the tree level by level
        while self._has_children(current_node):
            children = current_node["children"]
            child_titles = [child["title"] for child in children]

            # Ask LLM to pick the most relevant child topic
            selected_idx = self._select_topic(child_titles, chunk)

            if selected_idx == -1:
                # Chunk is not relevant at this level
                return None

            # Get the selected child node by index
            if selected_idx < 0 or selected_idx >= len(children):
                # Should not happen due to validation in _select_topic
                print(f"Warning: Invalid index {selected_idx} for {len(children)} children")
                return None

            current_node = children[selected_idx]

            # Add to path and continue
            path.append({"title": current_node["title"]})

        # Reached a leaf node
        return path

    def _has_children(self, node: dict) -> bool:
        """Check if a node has children."""
        return bool(node.get("children") and len(node["children"]) > 0)

    def _find_child(self, children: List[dict], title: str) -> Optional[dict]:
        """Find a child node by title."""
        for child in children:
            if child["title"] == title:
                return child
        return None

    def _select_topic(self, topic_titles: List[str], chunk: str) -> int:
        """
        Ask the LLM to choose the most relevant topic index for this chunk.

        Returns:
            int: index into `topic_titles` (0..len-1), or -1 if irrelevant.
        """
        try:
            classes_lines = [f"{i}: {title}" for i, title in enumerate(topic_titles)]
            classes_block = "\n".join(classes_lines)

            user_input = (
                "<text>\n"
                f"{chunk[:2000]}\n"
                "</text>\n"
                "<classes>\n"
                f"{classes_block}\n"
                "</classes>"
            )

            response = self.client.responses.create(
                prompt={
                    "id": self.prompt_id,
                    "version": self.prompt_version,
                },
                input=user_input,
                reasoning={"summary": "auto"},
                store=True,
                include=[
                    "reasoning.encrypted_content",
                    "web_search_call.action.sources",
                ],
            )

            # Prefer modern helper if available
            raw_answer = getattr(response, "output_text", None)

            # Fallback to low-level fields if needed
            if raw_answer is None and getattr(response, "output", None):
                first_item = response.output[0]
                content = getattr(first_item, "content", None)
                if content:
                    first_content = content[0]
                    raw_answer = getattr(first_content, "text", None)

            if not raw_answer:
                print("Warning: could not extract text from LLM response")
                return -1

            answer = raw_answer.strip()

            # Extract a single integer (tolerate minor extra text)
            m = re.search(r"-?\d+", answer)
            if not m:
                print(f"Warning: LLM returned non-numeric answer '{answer}'")
                return -1

            idx = int(m.group(0))

            # -1 is explicitly “no class”
            if idx == -1:
                return -1

            # Valid index?
            if 0 <= idx < len(topic_titles):
                return idx

            print(
                f"Warning: LLM returned out-of-range index {idx} "
                f"for {len(topic_titles)} topics"
            )
            return -1

        except Exception as e:
            print(f"Error in topic selection: {e}")
            return -1