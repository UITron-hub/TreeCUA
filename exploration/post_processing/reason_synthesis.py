# openrlhf/agent/exploration/post_processing/reason_synthesis.py
"""
Generates step-level thinking (observation, progress, plan, impact) for each
step in a trajectory by reverse-reasoning from the final task.

Can be used standalone or imported into the exploration pipeline.
"""

import base64
import io
import json
import logging
import os
import threading
from typing import Dict, List, Optional, Callable
import re

import requests
from PIL import Image

from exploration.prompts import POSTPROCESS_REASON_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)


def _detect_language(text: str, threshold: float = 0.05) -> str:
    """Detect if text is Chinese or English based on character ratio."""
    if not text:
        return "English"
    clean = "".join(text.split())
    if not clean:
        return "English"
    chinese_count = len(re.findall(r'[一-龥]', clean))
    return "Chinese" if chinese_count / len(clean) > threshold else "English"


class ReasonSynthesisAgent:
    """Generates step-level thinking for trajectory steps."""

    def __init__(self, model: str, api_url: str, api_key: str, max_tokens: int = 4096):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.max_tokens = max_tokens

    def _encode_image(self, image_path: str, target_size=(1024, 768)) -> Optional[str]:
        """Load and base64-encode an image for the LLM."""
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception:
            return None

    def _call_llm(self, messages: List[Dict], temperature: float = 0.1) -> Optional[Dict]:
        """Call the LLM API and return parsed JSON."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=90)
            if resp.status_code != 200:
                raise ValueError(f"API Error {resp.status_code}")
            content = resp.json()["choices"][0]["message"]["content"]
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def process_trajectory(
        self,
        data: Dict,
        screenshot_root: str,
        save_callback: Optional[Callable] = None,
    ) -> Optional[Dict]:
        """
        Generate step-level thinking for a single trajectory.

        Args:
            data: Trajectory manifest dict.
            screenshot_root: Root directory containing 'screenshots/' folder.
            save_callback: Optional callback(data) called after each step to save progress.
        Returns:
            The updated data dict with 'thinking' fields, or None on failure.
        """
        final_task = data.get("final_task", "")
        if isinstance(final_task, dict):
            final_task = final_task.get("final_task_summary", "")
        language = _detect_language(str(final_task))

        node_path = data.get("node_path", [])
        traj_detail = data.get("traj_detail", [])
        if not traj_detail:
            return None

        history_accumulator = []
        total_steps = len(traj_detail)

        for i, step_data in enumerate(traj_detail):
            step_id = step_data.get("step", i + 1)
            goal = step_data.get("goal", "")
            if goal == "Confirm task completion and finish.":
                step_data["goal"] = "Finish"
                goal = "Finish"
            verification = step_data.get("verification_result", "N/A")

            # Skip if already has thinking
            existing = step_data.get("thinking")
            if existing and _detect_language(str(existing)) == language:
                history_accumulator.append({
                    "step": len(history_accumulator) + 1,
                    "goal": goal,
                    "action": step_data.get("action"),
                    "verification_result": verification,
                    "thinking": existing,
                })
                continue

            # Build future trajectory
            future_list = []
            for future_step in traj_detail[i + 1:]:
                future_list.append({
                    "step": future_step.get("step", "N/A"),
                    "goal": future_step.get("goal"),
                    "verification_result": future_step.get("verification_result", "N/A"),
                })
            future_str = json.dumps(future_list, ensure_ascii=False, indent=1) if future_list else "None (Last step)"
            history_str = json.dumps(history_accumulator, ensure_ascii=False, indent=1) if history_accumulator else "None (First step)"

            # Get screenshot
            img_path = None
            node_idx = min(i, len(node_path) - 1)
            if node_path:
                img_path = os.path.join(screenshot_root, "screenshots", f"{node_path[node_idx]}.png")
            base64_image = self._encode_image(img_path)

            # Build prompt
            prompt = POSTPROCESS_REASON_SYNTHESIS_PROMPT.format(
                final_task=final_task,
                history_trajectory=history_str,
                step_id=step_id,
                goal=goal,
                verification_result=json.dumps(verification, ensure_ascii=False),
                future_trajectory=future_str,
                language=language,
            )

            content_payload = [{"type": "text", "text": prompt}]
            if base64_image:
                content_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"},
                })

            result = self._call_llm([{"role": "user", "content": content_payload}])
            if result:
                synthesized = result.get("synthesized_thoughts", {})
                step_data["thinking"] = synthesized
                if save_callback:
                    save_callback(data)

            history_accumulator.append({
                "step": len(history_accumulator) + 1,
                "goal": goal,
                "action": step_data.get("action"),
                "verification_result": verification,
                "thinking": step_data.get("thinking"),
            })

        return data

    def process_directory(
        self,
        root_dir: str,
        screenshot_root: str,
        data_mode: str = "trajectory",
    ) -> int:
        """
        Process all trajectory files in a directory tree.

        Args:
            root_dir: Root directory to scan for trajectory JSON files.
            screenshot_root: Root directory containing 'screenshots/' folder.
            data_mode: 'trajectory' or 'stage'.
        Returns:
            Number of successfully processed files.
        """
        target_files = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if not f.endswith(".json") or "_thinking.json" in f:
                    continue
                if data_mode == "trajectory" and "traj_" in f:
                    target_files.append(os.path.join(root, f))
                elif data_mode == "stage" and "stg_" in f:
                    target_files.append(os.path.join(root, f))

        count = 0
        for file_path in target_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)

                suffix = "_thinking.json" if data_mode == "trajectory" else "_stage_thinking.json"
                output_path = file_path.replace(".json", suffix)

                # Resume from existing output if available
                if os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)

                def save_cb(d):
                    with open(output_path, 'w', encoding='utf-8') as fh:
                        json.dump(d, fh, indent=2, ensure_ascii=False)

                result = self.process_trajectory(data, screenshot_root, save_cb)
                if result:
                    with open(output_path, 'w', encoding='utf-8') as fh:
                        json.dump(result, fh, indent=2, ensure_ascii=False)
                    count += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        logger.info(f"Reason synthesis complete: {count}/{len(target_files)} files.")
        return count
