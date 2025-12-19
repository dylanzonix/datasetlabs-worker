import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import yaml  # pip install pyyaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the YAML configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Optional sanity check: make sure required keys exist
    required_keys = ["topic_tree"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")

    return config


def collect_leaf_paths(
    node: Dict[str, Any],
    path: List[str] | None = None,
    results: List[str] | None = None,
) -> List[str]:
    """
    Recursively collect paths to leaf nodes (nodes with no children).
    Each path is formatted as 'Parent > Child > Leaf'.
    """
    if path is None:
        path = []
    if results is None:
        results = []

    current_path = path + [node.get("title", "")]
    children = node.get("children") or []

    # Leaf node: no children
    if not children:
        results.append(" > ".join(current_path))
    else:
        for child in children:
            collect_leaf_paths(child, current_path, results)

    return results


def main(config_file: str) -> None:
    # 1. Load YAML config
    config = load_config(config_file)

    instructions = config.get("instructions")
    sources_dir = config.get("sources_dir")
    topic_tree_path = Path(config["topic_tree"])
    samples = config.get("samples")

    # (You can use instructions / sources_dir / samples later as needed.)
    # For now we just show they're available:
    print(f"Instructions: {instructions}")
    print(f"Sources dir: {sources_dir}")
    print(f"Topic tree file: {topic_tree_path}")
    print(f"Samples: {samples}")
    print()

    # 2. Load topic tree JSON using path from config
    if not topic_tree_path.is_file():
        print(f"Error: topic_tree file does not exist: {topic_tree_path}")
        sys.exit(1)

    with topic_tree_path.open("r", encoding="utf-8") as f:
        topic_tree = json.load(f)

    # 3. Collect leaf topic paths
    leaf_paths = collect_leaf_paths(topic_tree)

    # 4. Print them (or return / write to file, etc.)
    print("Leaf topics:")
    for p in leaf_paths:
        print(p)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        script_name = Path(sys.argv[0]).name
        print(f"Usage: python {script_name} path/to/config.yaml")
        sys.exit(1)

    main(sys.argv[1])
