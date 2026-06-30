#!/usr/bin/env python3
"""
Replay TreeCUA action sequences in OSWorld to obtain full trajectories with screenshots.

The TreeCUA dataset provides environment configurations (config.json) and action
sequences (nodes.jsonl). This script replays them using the official OSWorld
DesktopEnv to reconstruct the complete trajectories including visual observations.

The output mirrors the exploration pipeline's session directory layout, so
post-processing scripts (scripts/post_process/run_post_processing.py) can be
run on the replayed output directly.

Usage:
    python scripts/replay.py \\
        --dataset_dir ./data_cache/TreeCUA_Datasets \\
        --output_dir ./replay_output \\
        --provider_name docker \\
        --path_to_vm /path/to/Ubuntu.qcow2

    # Replay a specific app or tree
    python scripts/replay.py \\
        --dataset_dir ./data_cache/TreeCUA_Datasets \\
        --output_dir ./replay_output \\
        --app chrome \\
        --tree_id root_001_4bb91e98
"""

import argparse
import json
import os
import sys
import uuid
import logging
from collections import defaultdict
from typing import Dict, List, Optional

# Ensure the TreeCUA root is on sys.path so that OSWorld and exploration imports work.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from OSWorld.desktop_env import DesktopEnv
from exploration.exploration_agent import ExplorationAgent
from exploration.node import Node

logger = logging.getLogger("replay")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Action conversion: reuses ExplorationAgent.parse_actions_from_tool_call
# ---------------------------------------------------------------------------

# Lightweight ExplorationAgent instance for its action parser only.
# All __init__ params have defaults — no API key needed for parsing.
_agent = ExplorationAgent()

# Coordinate scaling ratios. Actions in the TreeCUA dataset use 1024×768
# coordinates. If your VM runs at a different resolution, set these environment
# variables to scale the coordinates accordingly:
#   TREE_CUA_REPLAY_RATIO_X = original_width  / 1024
#   TREE_CUA_REPLAY_RATIO_Y = original_height / 768
_X_RATIO = float(os.environ.get("TREE_CUA_REPLAY_RATIO_X", "1.0"))
_Y_RATIO = float(os.environ.get("TREE_CUA_REPLAY_RATIO_Y", "1.0"))


def step_actions_to_pyautogui(step_actions: List[Dict]) -> str:
    """Convert step_action list to pyautogui command string.

    Uses the exact same wrapping pattern as ExplorationAgent.enrich_candidates
    (exploration_agent.py lines 186-196): each action dict is wrapped in a
    mock tool_call and passed to parse_actions_from_tool_call.
    """
    command_list = []
    for action_dict in step_actions:
        mock_tool_call = {
            "name": "computer",
            "input": action_dict,
            "id": "mock_tool_call",
        }
        command = _agent.parse_actions_from_tool_call(
            mock_tool_call, _X_RATIO, _Y_RATIO,
        )
        command_list.append(command.strip())
    return "\n".join(command_list)


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------

def load_nodes(nodes_path: str) -> Dict[str, dict]:
    """Load nodes.jsonl and return a dict keyed by node_id."""
    nodes: Dict[str, dict] = {}
    with open(nodes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            node = json.loads(line)
            nodes[node["node_id"]] = node
    return nodes


def build_tree(nodes: Dict[str, dict]) -> Dict[str, List[str]]:
    """Build a parent_id -> [child_id, ...] mapping."""
    children: Dict[str, List[str]] = defaultdict(list)
    for node_id, node in nodes.items():
        parent = node.get("parent_node_id")
        if parent:
            children[parent].append(node_id)
    return children


def collect_leaf_paths(
    nodes: Dict[str, dict],
    children: Dict[str, List[str]],
) -> List[List[str]]:
    """Collect all root -> leaf paths."""
    roots = [nid for nid, nd in nodes.items()
             if nd.get("parent_node_id") is None or nd.get("depth") == 0]

    paths: List[List[str]] = []

    def dfs(node_id: str, path: List[str]):
        path = path + [node_id]
        if node_id not in children or not children[node_id]:
            paths.append(path)
        else:
            for child_id in children[node_id]:
                dfs(child_id, path)

    for root_id in roots:
        dfs(root_id, [])
    return paths


# ---------------------------------------------------------------------------
# Replay logic
# ---------------------------------------------------------------------------

class ReplayRunner:
    """Replays TreeCUA trees and saves output in the exploration session format.

    Output layout (compatible with post-processing pipeline):
        output_dir/
        ├── nodes/            # Node JSONs (Node.to_dict() format)
        ├── screenshots/       # {node_id}.png
        └── trajectories/      # {traj_id}/{traj_id}.json manifests
    """

    def __init__(
        self,
        output_dir: str,
        provider_name: str = "docker",
        path_to_vm: Optional[str] = None,
        headless: bool = True,
        sleep_after_execution: float = 2.0,
    ):
        self.output_dir = output_dir
        self.provider_name = provider_name
        self.path_to_vm = path_to_vm
        self.headless = headless
        self.sleep_after_execution = sleep_after_execution

        self.nodes_dir = os.path.join(output_dir, "nodes")
        self.screenshots_dir = os.path.join(output_dir, "screenshots")
        self.trajectories_dir = os.path.join(output_dir, "trajectories")
        for d in [self.nodes_dir, self.screenshots_dir, self.trajectories_dir]:
            os.makedirs(d, exist_ok=True)

    def _create_env(self) -> DesktopEnv:
        kwargs = {"provider_name": self.provider_name, "headless": self.headless}
        if self.path_to_vm:
            kwargs["path_to_vm"] = self.path_to_vm
        return DesktopEnv(**kwargs)

    def replay_tree(
        self,
        config_path: str,
        nodes_path: str,
        app: str,
        category: str,
        tree_id: str,
    ) -> int:
        """Replay all leaf paths in a tree. Returns the number of replayed paths."""
        with open(config_path, "r", encoding="utf-8") as f:
            outer_config = json.load(f)

        inner_config = outer_config.get("config", outer_config)
        nodes = load_nodes(nodes_path)
        children = build_tree(nodes)
        leaf_paths = collect_leaf_paths(nodes, children)

        logger.info(
            "Tree %s [%s/%s]: %d nodes, %d leaf paths",
            tree_id, app, category, len(nodes), len(leaf_paths),
        )

        path_count = 0
        for path in leaf_paths:
            try:
                self._replay_path(
                    inner_config, nodes, path, tree_id,
                )
                path_count += 1
            except Exception:
                logger.exception(
                    "Failed to replay path %s in tree %s", path, tree_id,
                )
                continue

        return path_count

    def _replay_path(
        self,
        task_config: dict,
        nodes: Dict[str, dict],
        path: List[str],
        tree_id: str,
    ):
        """Replay a single root -> leaf path and save all artifacts."""
        env = self._create_env()
        traj_id = f"traj_{uuid.uuid4().hex[:12]}"
        traj_dir = os.path.join(self.trajectories_dir, traj_id)
        os.makedirs(traj_dir, exist_ok=True)

        try:
            env.reset(task_config)

            traj_detail: List[dict] = []

            for i, node_id in enumerate(path):
                node_data = nodes[node_id]

                # Take screenshot at current state
                obs = env._get_obs()
                screenshot_bytes = obs.get("screenshot", b"")
                screenshot_filename = f"{node_id}.png"
                screenshot_path = os.path.join(
                    self.screenshots_dir, screenshot_filename,
                )
                if screenshot_bytes:
                    with open(screenshot_path, "wb") as f:
                        f.write(screenshot_bytes)

                # Convert action to pyautogui command string
                step_actions = node_data.get("step_action") or []
                action_command = step_actions_to_pyautogui(step_actions) if step_actions else None

                # Save node as JSON (matching Node.to_dict() format)
                node_obj = Node(
                    node_id=node_id,
                    parent_node_id=node_data.get("parent_node_id"),
                    depth=node_data.get("depth", i),
                    status=node_data.get("verification_result", "UNKNOWN"),
                    action_command=action_command,
                    step_goal=node_data.get("step_goal"),
                    step_action=node_data.get("step_action") or [],
                    screenshot=screenshot_filename,
                )
                node_obj.save(self.nodes_dir)

                # Build traj_detail entry (matching create_trajectory_manifest format)
                traj_detail.append({
                    "step": i,
                    "goal": node_data.get("step_goal"),
                    "action": node_data.get("step_action"),
                    "reason": None,
                    "future_impact": "N/A",
                    "verification_result": {
                        "result_type": node_data.get("verification_result"),
                        "feedback": None,
                    },
                    "final_goal_at_step": "N/A",
                })

                # Execute action to advance to the next node (root has null action)
                if i < len(path) - 1:
                    next_node = nodes[path[i + 1]]
                    next_actions = next_node.get("step_action")
                    if next_actions:
                        command = step_actions_to_pyautogui(next_actions)
                        env.step(command, pause=self.sleep_after_execution)

            # Save trajectory manifest (matching create_trajectory_manifest format)
            manifest = {
                "trajectory_id": traj_id,
                "session_id": os.path.basename(self.output_dir),
                "termination_reason": "replay_complete",
                "length": len(path) - 1,
                "node_path": path,
                "final_task": {},
                "traj_detail": traj_detail,
            }

            manifest_path = os.path.join(traj_dir, f"{traj_id}.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

        finally:
            try:
                env.close()
            except Exception:
                logger.warning("Failed to close environment", exc_info=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def iter_trees(
    dataset_dir: str,
    app_filter: Optional[str] = None,
    tree_id_filter: Optional[str] = None,
):
    """Yield (config_path, nodes_path, app, category, tree_id) tuples."""
    trees_dir = os.path.join(dataset_dir, "trees")
    if not os.path.isdir(trees_dir):
        raise FileNotFoundError(f"trees/ directory not found under {dataset_dir}")

    for app in sorted(os.listdir(trees_dir)):
        if app_filter and app != app_filter:
            continue
        app_dir = os.path.join(trees_dir, app)
        if not os.path.isdir(app_dir):
            continue
        for category in sorted(os.listdir(app_dir)):
            cat_dir = os.path.join(app_dir, category)
            if not os.path.isdir(cat_dir):
                continue
            for tree_id in sorted(os.listdir(cat_dir)):
                if tree_id_filter and tree_id != tree_id_filter:
                    continue
                tree_dir = os.path.join(cat_dir, tree_id)
                config_path = os.path.join(tree_dir, "config.json")
                nodes_path = os.path.join(tree_dir, "nodes.jsonl")
                if os.path.isfile(config_path) and os.path.isfile(nodes_path):
                    yield config_path, nodes_path, app, category, tree_id


def main():
    parser = argparse.ArgumentParser(
        description="Replay TreeCUA action sequences in OSWorld",
    )
    parser.add_argument(
        "--dataset_dir", required=True,
        help="Path to the TreeCUA_Datasets directory (contains trees/ and data_resource/).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write replayed session data (nodes/, screenshots/, trajectories/).",
    )
    parser.add_argument(
        "--provider_name", default="docker",
        choices=["docker", "vmware", "virtualbox", "aws", "azure", "gcp",
                 "aliyun", "volcengine", "fastvm"],
        help="OSWorld provider name (default: docker).",
    )
    parser.add_argument(
        "--path_to_vm",
        help="Path to the VM image (required for vmware/virtualbox providers).",
    )
    parser.add_argument(
        "--headless", action="store_true", default=True,
        help="Run the environment in headless mode.",
    )
    parser.add_argument(
        "--no-headless", dest="headless", action="store_false",
        help="Show the VM GUI (not headless).",
    )
    parser.add_argument(
        "--app", help="Only replay trees for this app (e.g. chrome, gimp).",
    )
    parser.add_argument(
        "--tree_id", help="Only replay a specific tree by its tree_id.",
    )
    parser.add_argument(
        "--sleep_after_execution", type=float, default=2.0,
        help="Seconds to sleep after each action execution (default: 2.0).",
    )

    args = parser.parse_args()

    # Set RESOURCE_DIR so OSWorld can resolve data_resource/ paths in config steps.
    data_resource_dir = os.path.join(args.dataset_dir, "data_resource")
    if os.path.isdir(data_resource_dir) and "RESOURCE_DIR" not in os.environ:
        os.environ["RESOURCE_DIR"] = args.dataset_dir

    runner = ReplayRunner(
        output_dir=args.output_dir,
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        headless=args.headless,
        sleep_after_execution=args.sleep_after_execution,
    )

    total_trees = 0
    total_paths = 0

    for config_path, nodes_path, app, category, tree_id in iter_trees(
        args.dataset_dir, args.app, args.tree_id,
    ):
        total_trees += 1
        n_paths = runner.replay_tree(config_path, nodes_path, app, category, tree_id)
        total_paths += n_paths
        logger.info("  -> %d paths replayed", n_paths)

    logger.info(
        "Replay complete: %d trees, %d leaf paths -> %s",
        total_trees, total_paths, args.output_dir,
    )


if __name__ == "__main__":
    main()
