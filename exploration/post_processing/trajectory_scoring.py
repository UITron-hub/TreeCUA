# openrlhf/agent/exploration/post_processing/trajectory_scoring.py
"""
Scores trajectory quality on 4 metrics: task_utility, step_efficiency,
task_consistency, action_coherence (0-3 scale each).

Can be used standalone or imported into the exploration pipeline.
"""

import json
import logging
import time
from typing import Dict, List, Optional

import requests

from exploration.prompts import POSTPROCESS_SCORING_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class TrajectoryScoringAgent:
    """Scores trajectory quality using an LLM evaluator."""

    def __init__(self, model: str, api_url: str, api_key: str, max_tokens: int = 5120):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.max_tokens = max_tokens

    def _extract_storyline(self, traj_detail: List[Dict]) -> List[str]:
        """Extract a compact storyline from trajectory detail steps."""
        storyline = []
        for i, step in enumerate(traj_detail):
            goal = step.get('goal', 'No goal')
            verification = step.get('verification_result', {})
            feedback = verification.get('feedback', 'No feedback')
            action_list = step.get('action', [])
            act_str = "Unknown"
            if action_list:
                act_data = action_list[0]
                act_type = act_data.get('action', '')
                if act_type == 'type':
                    act_str = f"Type '{act_data.get('text', '')}'"
                elif act_type == 'scroll':
                    act_str = "Scroll"
                elif act_type == 'key':
                    act_str = f"Press Key '{act_data.get('text', '')}'"
                elif act_type == 'left_click':
                    target_text = act_data.get('text', '')
                    act_str = f"Click '{target_text}'" if target_text else "Click Position"
                else:
                    act_str = act_type
            line = f"Step {i+1}: Intent=[{goal}] -> Action=[{act_str}] -> Feedback=[{feedback}]"
            storyline.append(line)
        return storyline

    def _call_llm(self, messages: List[Dict]) -> Optional[Dict]:
        """Call the LLM API and return parsed JSON."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        for attempt in range(5):
            try:
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                if resp.status_code != 200:
                    logger.error(f"API Error {resp.status_code}")
                    if attempt < 4:
                        time.sleep(1)
                    continue
                content = resp.json()["choices"][0]["message"]["content"]
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                return json.loads(content)
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {e}")
                if attempt < 4:
                    time.sleep(1)
        return None

    def process(self, data: Dict) -> Optional[Dict]:
        """
        Score a trajectory.

        Args:
            data: Trajectory manifest dict (must contain 'final_task' and 'traj_detail').
        Returns:
            Dict with 'reason', 'task_utility', 'step_efficiency', 'task_consistency',
            'action_coherence' keys, or None on failure.
        """
        final_task = data.get('final_task', {}).get('final_task_summary', 'No summary')
        traj_detail = data.get('traj_detail', [])

        storyline_list = self._extract_storyline(traj_detail)
        if len(storyline_list) < 2:
            return {
                "reason": "Trajectory too short (<2 steps).",
                "task_utility": 0,
                "step_efficiency": 0,
                "task_consistency": 0,
                "action_coherence": 0,
            }

        story_text = "\n".join(storyline_list)
        user_content = f"""Target Task: "{final_task}"

Execution Storyline:
{story_text}

Evaluate this trajectory using the 0-3 scale for the 4 metrics."""

        messages = [
            {"role": "system", "content": POSTPROCESS_SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self._call_llm(messages)
