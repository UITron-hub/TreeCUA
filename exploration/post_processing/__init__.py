# openrlhf/agent/exploration/post_processing/__init__.py
"""
Post-processing pipeline for exploration trajectories.

Provides tools for:
- Overall task summary generation
- Stage-level breakdown and classification
- Trajectory quality scoring
- Step-level reason/thinking synthesis

All modules accept config via constructor and can be used both inline (during
exploration) and standalone (offline batch processing).
"""

from .pipeline import PostProcessingPipeline, PostProcessConfig
from .summary_overall import SummaryOverallAgent
from .summary_stages import SummaryStagesAgent
from .trajectory_scoring import TrajectoryScoringAgent
from .reason_synthesis import ReasonSynthesisAgent

__all__ = [
    "PostProcessingPipeline",
    "PostProcessConfig",
    "SummaryOverallAgent",
    "SummaryStagesAgent",
    "TrajectoryScoringAgent",
    "ReasonSynthesisAgent",
]
