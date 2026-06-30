#!/usr/bin/env python3
"""
Standalone CLI for running post-processing on exploration results.

Usage:
    python scripts/post_process/run_post_processing.py --session_dir /path/to/session

Environment variables:
    LLM_API_URL     - LLM API base URL (required)
    LLM_API_KEY     - LLM API key (required)
    LLM_MODEL       - Model name for post-processing agents
"""

import argparse
import logging
import os
import sys

# Ensure the TreeCUA root is on the Python path.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from exploration.config import get_api_config
from exploration.post_processing import (
    PostProcessingPipeline,
    PostProcessConfig,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run post-processing on completed exploration sessions."
    )
    parser.add_argument(
        "--session_dir", type=str, required=True,
        help="Path to the exploration session directory."
    )
    parser.add_argument(
        "--screenshot_root", type=str, default=None,
        help="Root directory for screenshots (defaults to session_dir)."
    )
    parser.add_argument(
        "--api_url", type=str, default=None,
        help="LLM API base URL (default: $LLM_API_URL)."
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="LLM API key (default: $LLM_API_KEY)."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (default: $LLM_MODEL)."
    )

    parser.add_argument(
        "--step", type=str, nargs="*",
        choices=["scoring", "summary_overall", "summary_stages", "reason_synthesis", "all"],
        default=["all"],
        help="Which post-processing steps to run (default: all)."
    )

    parser.add_argument("--force_rescore", action="store_true")
    parser.add_argument("--force_re_summary", action="store_true")
    parser.add_argument("--force_re_stages", action="store_true")
    parser.add_argument("--force_reason_synthesis", action="store_true")

    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not os.path.isdir(args.session_dir):
        logger.error(f"Session directory not found: {args.session_dir}")
        sys.exit(1)

    api_cfg = get_api_config(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
    )
    if not api_cfg.api_url or not api_cfg.api_key:
        logger.error(
            "API URL and API key are required. "
            "Set LLM_API_URL and LLM_API_KEY environment variables."
        )
        sys.exit(1)

    cfg = PostProcessConfig(
        api_config=api_cfg,
        screenshot_root=args.screenshot_root or args.session_dir,
        force_rescore=args.force_rescore,
        force_re_summary=args.force_re_summary,
        force_re_stages=args.force_re_stages,
        force_reason_synthesis=args.force_reason_synthesis,
    )

    pipeline = PostProcessingPipeline(cfg)

    steps = set(args.step)
    run_all = "all" in steps

    if run_all or "scoring" in steps:
        logger.info("=== Step 1: Trajectory Scoring ===")
        pipeline.run_scoring(args.session_dir)

    if run_all or "summary_overall" in steps:
        logger.info("=== Step 2: Overall Task Summary ===")
        pipeline.run_summary_overall(args.session_dir)

    if run_all or "summary_stages" in steps:
        logger.info("=== Step 3: Stage Breakdown ===")
        pipeline.run_summary_stages(args.session_dir)

    if run_all or "reason_synthesis" in steps:
        logger.info("=== Step 4: Reason Synthesis ===")
        pipeline.run_reason_synthesis(args.session_dir)

    logger.info("Post-processing complete.")


if __name__ == "__main__":
    main()
