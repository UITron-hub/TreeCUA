# /path/to/project/openrlhf/agent/exploration/node.py
import json
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class Node:
    """A dataclass representing a single state in the exploration tree."""
    # Core identifiers & structure
    node_id: str
    parent_node_id: Optional[str]
    depth: int
    status: str  # UNEXPLORED, SUCCESS, FAILURE

    # --- Information about the action REQUIRED TO REACH this node ---
    # This is the central piece of information for execution.
    action_command: Optional[str] = None
    step_goal: Optional[str] = None
    final_goal: Optional[str] = None
    step_reason: Optional[str] = None
    expected_observation: Optional[str] = None
    step_action: Optional[List[Dict[str, Any]]] = field(default_factory=list)

    # --- Information captured AFTER REACHING this node ---
    screenshot: Optional[str] = None
    accessibility_tree: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = field(default_factory=dict)

    # Agent's plan starting FROM this node.
    next_nodes_candidates: Optional[List[Dict[str, Any]]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        # Handle legacy or missing fields gracefully
        fields = cls.__annotations__
        kwargs = {k: data.get(k) for k in fields}
        return cls(**kwargs)

    def save(self, nodes_dir: str):
        file_path = os.path.join(nodes_dir, f"{self.node_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, nodes_dir: str, node_id: str) -> Optional["Node"]:
        file_path = os.path.join(nodes_dir, f"{node_id}.json")
        if not os.path.exists(file_path): return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, TypeError) as e:
            logging.error(f"Failed to load or parse node file: {file_path}", exc_info=True)
            return None