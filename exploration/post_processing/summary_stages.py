# openrlhf/agent/exploration/post_processing/summary_stages.py
"""
Breaks a trajectory into logical stages (EFFECTIVE, NAVIGATION, NOISE) and
generates per-stage summaries plus an effective-only final task summary.

Can be used standalone or imported into the exploration pipeline.
"""

import json
import logging
from typing import Dict, List, Optional

import requests

from exploration.prompts import POSTPROCESS_SUMMARY_STAGES_PROMPT

logger = logging.getLogger(__name__)


class SummaryStagesAgent:
    """Breaks a trajectory into classified stages and generates summaries."""

    def __init__(self, model: str, api_url: str, api_key: str, max_tokens: int = 4096):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.max_tokens = max_tokens

    def _format_history(self, traj_detail: List[Dict]) -> str:
        """Format trajectory steps into a prompt-friendly history string."""
        formatted = []
        for item in traj_detail:
            step_num = item.get('step')
            goal = item.get('goal', 'N/A')
            actions = item.get('action', [])
            action_strs = []
            if isinstance(actions, list):
                for act in actions:
                    if isinstance(act, dict):
                        act_type = act.get('action', 'UNKNOWN')
                        if 'coordinate' in act:
                            action_strs.append(f"{act_type}{act['coordinate']}")
                        elif 'text' in act:
                            action_strs.append(f"{act_type}('{act['text']}')")
                        else:
                            action_strs.append(str(act_type))
                    else:
                        action_strs.append(str(act))
            action_summary = ", ".join(action_strs)

            verification = item.get('verification_result', {})
            result_type = verification.get('result_type', 'UNKNOWN')
            feedback = verification.get('feedback', '')

            formatted.append(
                f"Step {step_num}:\n"
                f"  - Goal: {goal}\n"
                f"  - Action: {action_summary}\n"
                f"  - Result: {result_type}\n"
                f"  - Feedback: {feedback}"
            )
        return "\n".join(formatted)

    def _call_llm(self, messages: List[Dict]) -> Optional[Dict]:
        """Call the LLM API and return parsed JSON."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
            "top_p": 1.0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            content = content.strip()
            if "```json" in content:
                content = content.split("```json", 1)[1]
            if "```" in content:
                content = content.split("```", 1)[0]
            return json.loads(content)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def process(self, data: Dict) -> Optional[Dict]:
        """
        Generate stage breakdown for a trajectory.

        Args:
            data: Trajectory manifest dict (must contain 'traj_detail').
        Returns:
            Dict with 'stages' and optionally 'final_task_summary_effective', or None.
        """
        traj_detail = data.get("traj_detail", [])
        if not traj_detail:
            logger.warning("No traj_detail found.")
            return None

        history_str = self._format_history(traj_detail)
        prompt = POSTPROCESS_SUMMARY_STAGES_PROMPT.format(history=history_str)
        messages = [{"role": "user", "content": prompt}]

        return self._call_llm(messages)
