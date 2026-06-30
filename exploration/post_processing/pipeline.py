# openrlhf/agent/exploration/post_processing/pipeline.py
"""
Orchestrator for running post-processing steps on exploration results.

Usage:
    from exploration.post_processing import PostProcessingPipeline, PostProcessConfig
    from exploration.config import get_api_config

    cfg = PostProcessConfig(
        api_config=get_api_config(),
        screenshot_root="/path/to/session",
    )
    pipeline = PostProcessingPipeline(cfg)
    pipeline.run_all("/path/to/session")
"""

import os
import json
import logging
import glob
from dataclasses import dataclass, field
from typing import Optional, List

from exploration.config import ApiConfig
from .summary_overall import SummaryOverallAgent
from .summary_stages import SummaryStagesAgent
from .trajectory_scoring import TrajectoryScoringAgent
from .reason_synthesis import ReasonSynthesisAgent

logger = logging.getLogger(__name__)


@dataclass
class PostProcessConfig:
    """Configuration for the post-processing pipeline."""
    api_config: ApiConfig
    screenshot_root: str = ""
    max_workers: int = 10
    force_rescore: bool = False
    force_re_summary: bool = False
    force_re_stages: bool = False
    force_reason_synthesis: bool = False


class PostProcessingPipeline:
    """
    Runs post-processing steps on exploration session data.

    Steps:
    1. summary_overall - Generate final_task_summary for each trajectory
    2. summary_stages - Break trajectories into logical stages
    3. trajectory_scoring - Score trajectory quality
    4. reason_synthesis - Generate step-level thinking for each step
    """

    def __init__(self, config: PostProcessConfig):
        self.config = config
        self._summary_overall: Optional[SummaryOverallAgent] = None
        self._summary_stages: Optional[SummaryStagesAgent] = None
        self._scorer: Optional[TrajectoryScoringAgent] = None
        self._reason_synthesis: Optional[ReasonSynthesisAgent] = None

    @property
    def summary_overall(self) -> SummaryOverallAgent:
        if self._summary_overall is None:
            self._summary_overall = SummaryOverallAgent(
                model=self.config.api_config.model,
                api_url=self.config.api_config.api_url,
                api_key=self.config.api_config.api_key,
            )
        return self._summary_overall

    @property
    def summary_stages(self) -> SummaryStagesAgent:
        if self._summary_stages is None:
            self._summary_stages = SummaryStagesAgent(
                model=self.config.api_config.model,
                api_url=self.config.api_config.api_url,
                api_key=self.config.api_config.api_key,
            )
        return self._summary_stages

    @property
    def scorer(self) -> TrajectoryScoringAgent:
        if self._scorer is None:
            self._scorer = TrajectoryScoringAgent(
                model=self.config.api_config.model,
                api_url=self.config.api_config.api_url,
                api_key=self.config.api_config.api_key,
            )
        return self._scorer

    @property
    def reason_synthesis(self) -> ReasonSynthesisAgent:
        if self._reason_synthesis is None:
            self._reason_synthesis = ReasonSynthesisAgent(
                model=self.config.api_config.model,
                api_url=self.config.api_config.api_url,
                api_key=self.config.api_config.api_key,
            )
        return self._reason_synthesis

    def _find_trajectory_files(self, session_dir: str) -> List[str]:
        """Find all trajectory manifest JSON files in a session directory."""
        pattern = os.path.join(session_dir, "trajectories", "*", "*.json")
        files = []
        for f in glob.iglob(pattern):
            # Skip already-processed files
            if "_thinking.json" in f or "_stage_thinking.json" in f:
                continue
            # Match traj_xxx.json pattern
            if "traj_" in os.path.basename(f):
                files.append(f)
        return sorted(files)

    def run_summary_overall(self, session_dir: str) -> int:
        """Generate overall task summaries. Returns count of processed files."""
        files = self._find_trajectory_files(session_dir)
        count = 0
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                final_task = data.get("final_task", {})
                if final_task.get("final_task_summary") and not self.config.force_re_summary:
                    continue
                result = self.summary_overall.process(data)
                if result:
                    if "final_task_summary" in result:
                        final_task["final_task_summary"] = result["final_task_summary"]
                    data["final_task"] = final_task
                    with open(f, 'w', encoding='utf-8') as fh:
                        json.dump(data, fh, indent=2, ensure_ascii=False)
                    count += 1
            except Exception as e:
                logger.error(f"Error processing {f}: {e}")
        logger.info(f"Summary overall: {count} files processed.")
        return count

    def run_summary_stages(self, session_dir: str) -> int:
        """Generate stage breakdowns. Returns count of processed files."""
        files = self._find_trajectory_files(session_dir)
        count = 0
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                final_task = data.get("final_task", {})
                if final_task.get("stages") and not self.config.force_re_stages:
                    continue
                result = self.summary_stages.process(data)
                if result:
                    if "stages" in result:
                        final_task["stages"] = result["stages"]
                    if "final_task_summary_effective" in result:
                        final_task["final_task_summary_effective"] = result["final_task_summary_effective"]
                    data["final_task"] = final_task
                    with open(f, 'w', encoding='utf-8') as fh:
                        json.dump(data, fh, indent=2, ensure_ascii=False)
                    count += 1
            except Exception as e:
                logger.error(f"Error processing {f}: {e}")
        logger.info(f"Summary stages: {count} files processed.")
        return count

    def run_scoring(self, session_dir: str) -> int:
        """Score trajectory quality. Returns count of processed files."""
        files = self._find_trajectory_files(session_dir)
        count = 0
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                if 'quality_assessment' in data and data['quality_assessment'] and not self.config.force_rescore:
                    continue
                result = self.scorer.process(data)
                if result and "error" not in result:
                    data['quality_assessment'] = result
                    with open(f, 'w', encoding='utf-8') as fh:
                        json.dump(data, fh, indent=2, ensure_ascii=False)
                    count += 1
            except Exception as e:
                logger.error(f"Error scoring {f}: {e}")
        logger.info(f"Scoring: {count} files processed.")
        return count

    def run_reason_synthesis(self, session_dir: str) -> int:
        """Generate step-level thinking. Returns count of processed files."""
        count = self.reason_synthesis.process_directory(
            root_dir=os.path.join(session_dir, "trajectories"),
            screenshot_root=self.config.screenshot_root or session_dir,
        )
        logger.info(f"Reason synthesis: {count} files processed.")
        return count

    def run_all(self, session_dir: str):
        """Run all post-processing steps."""
        logger.info(f"Starting post-processing pipeline for: {session_dir}")
        n1 = self.run_scoring(session_dir)
        n2 = self.run_summary_overall(session_dir)
        n3 = self.run_summary_stages(session_dir)
        n4 = self.run_reason_synthesis(session_dir)
        logger.info(f"Pipeline complete: scoring={n1}, summary_overall={n2}, "
                     f"summary_stages={n3}, reason_synthesis={n4}")
