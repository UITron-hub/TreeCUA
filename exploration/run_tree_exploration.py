#!/usr/bin/env python3
"""
Entry point for TreeCUA exploration.

Usage:
    python exploration/run_tree_exploration.py \
        --domain libreoffice_calc \
        --api_key "sk-xxx" \
        --api_base_url "https://api.claudecode.uk/v1/messages" \
        --path_to_vm "/path/to/Ubuntu.qcow2" \
        --provider_name docker
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
import signal
from multiprocessing import Process, Manager

# Ensure the TreeCUA root is on the Python path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from exploration.tree_manager import ExplorationTreeManager
from exploration.worker import worker_process

logger = logging.getLogger(__name__)


def get_config():
    parser = argparse.ArgumentParser(
        description="Run collaborative, tree-based GUI exploration (TreeCUA).")

    # Session & Worker
    parser.add_argument("--session_name", type=str,
                        default=f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--num_envs", type=int, default=2,
                        help="Number of parallel worker processes.")
    parser.add_argument("--results_base_dir", type=str, default="./exploration_results")

    # Trajectory Control
    parser.add_argument("--max_trajectories", type=int, default=5000,
                        help="Total number of trajectories to collect.")
    parser.add_argument("--max_trajectories_per_tree", type=int, default=1000,
                        help="Start a new tree after this many trajectories.")
    parser.add_argument("--max_steps_exploration", type=int, default=20,
                        help="Maximum depth/steps for a single trajectory.")
    parser.add_argument("--exploration_strategy", type=str,
                        choices=['random', 'dfs', 'bfs', 'hybrid'],
                        default='hybrid')
    parser.add_argument("--continue_existing", type=bool, default=True,
                        help="Prioritize existing trees before starting new ones.")

    # VM & Env
    parser.add_argument("--path_to_vm", type=str, required=True,
                        help="Path to VM image (qcow2 for docker, vmx for vmware, etc.).")
    parser.add_argument("--headless", action="store_true",
                        help="Run VM in headless mode.")
    parser.add_argument("--action_space", type=str, default="claude_computer_use")
    parser.add_argument("--provider_name", type=str, default="docker",
                        choices=["docker", "vmware", "virtualbox", "aws", "azure", "gcp", "aliyun", "volcengine", "fastvm"])
    parser.add_argument("--screen_width", type=int, default=1024)
    parser.add_argument("--screen_height", type=int, default=768)
    parser.add_argument("--sleep_after_execution", type=float, default=5.0)

    # Agent (Anthropic-native API)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5")
    parser.add_argument("--api_base_url", type=str, required=True,
                        help="Anthropic Messages API endpoint.")
    parser.add_argument("--api_key", type=str, required=True,
                        help="Anthropic API key (x-api-key header).")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--domain", type=str, default="libreoffice_writer",
                        help="Application domain (e.g., chrome, libreoffice_calc, gimp, vlc, vs_code).")
    parser.add_argument("--sub_category", type=str, default=None,
                        help="Sub-category within the domain.")

    # Verifier
    parser.add_argument("--use_verifier", action="store_true")
    parser.add_argument("--verifier_model", type=str, default=None)
    parser.add_argument("--verifier_api_base_url", type=str, default=None)
    parser.add_argument("--verifier_api_key", type=str, default=None)
    parser.add_argument("--verifier_max_tokens", type=int, default=512)

    # Post-processing
    parser.add_argument("--enable_inline_scoring", action="store_true",
                        help="Run scoring inline during exploration.")
    parser.add_argument("--enable_inline_summary", action="store_true",
                        help="Run summary inline during exploration.")

    # Logging
    parser.add_argument("--log_level", type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    return parser.parse_args()


def setup_master_logging(log_level, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "master_orchestrator.log")
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.handlers = []
    formatter = logging.Formatter(
        f"%(asctime)s [%(levelname)s] [Master] %(message)s")
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root_logger.addHandler(sh)
    logger.info(f"Master logger configured. Log file: {log_file}")


def signal_handler(sig, frame):
    logger.info("Master process received shutdown signal. Exiting...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = get_config()

    session_dir = os.path.join(args.results_base_dir, args.session_name)
    os.makedirs(session_dir, exist_ok=True)

    setup_master_logging(args.log_level, session_dir)

    logger.info(f"Starting TreeCUA Exploration Session: {args.session_name}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Provider: {args.provider_name}")
    logger.info(f"Session dir: {session_dir}")

    with open(os.path.join(session_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    with Manager() as manager:
        tree_manager = ExplorationTreeManager(
            session_dir=session_dir,
            domain=args.domain,
            max_traj_per_tree=args.max_trajectories_per_tree,
            sub_category=args.sub_category,
            continue_existing_trees=args.continue_existing,
            selection_strategy=args.exploration_strategy,
            max_depth=args.max_steps_exploration,
        )

        completion_queue = manager.Queue()
        processes = []

        for i in range(args.num_envs):
            p = Process(
                target=worker_process,
                args=(i + 1, tree_manager, completion_queue, args, session_dir),
                name=f"Worker-{i+1}",
            )
            processes.append(p)
            p.start()
            logger.info(f"Started worker process {p.name} with PID {p.pid}")
            time.sleep(5)

        completed_count = 0
        try:
            while completed_count < args.max_trajectories:
                try:
                    completion_queue.get(timeout=600)
                    completed_count += 1
                    if completed_count % 10 == 0:
                        logger.info(
                            f"Progress: {completed_count}/{args.max_trajectories}")
                except Exception:
                    if not any(p.is_alive() for p in processes):
                        logger.error("All workers have died. Shutting down.")
                        break

            logger.info(f"Goal reached! Total: {completed_count}")

        except Exception as e:
            logger.error(f"Master error: {e}", exc_info=True)

        finally:
            logger.info("Terminating worker processes...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
            start_wait = time.time()
            while any(p.is_alive() for p in processes):
                if time.time() - start_wait > 30:
                    for p in processes:
                        if p.is_alive():
                            p.kill()
                    break
                time.sleep(1)
            logger.info("Exploration session finished.")
