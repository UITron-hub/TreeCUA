
# openrlhf/agent/exploration/verification_agent.py
import base64
import json
import logging
import requests
import time
from typing import Dict, Tuple, List

from .prompts import VERIFICATION_PROMPT_TEMPLATE

logger = logging.getLogger("desktopenv.agent.verification")

class VerificationAgent:
    """
    An agent that verifies GUI actions and classifies the outcome into one of
    four categories: SUCCESS, NO_CHANGE, UNEXPECTED_CHANGE, NEEDS_MORE_TIME.
    """
    def __init__(self, model: str, api_url: str, api_key: str, max_tokens: int = 8192):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.verification_prompt=VERIFICATION_PROMPT_TEMPLATE
        logger.info(f"VerificationAgent initialized with model: {self.model}")

    def _call_verifier_llm(self, messages: List[Dict]) -> requests.Response:
        """Calls the LLM and returns the full response object."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        if 'gemini' in self.model:
            payload["extra_body"] = {
                "google": {
                    "thinking_config": {
                        "include_thoughts": False,
                        "thinking_budget": 0
                    },
                    "thought_tag_marker": "think"
                }
            }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        return requests.post(self.api_url, headers=headers, json=payload, timeout=90)

    def verify(
        self,
        prev_obs_bytes: bytes,
        action_command: str,
        expected_outcome: str,
        current_obs_bytes: bytes
    ) -> Tuple[str, str]:
        """
        Performs categorized verification with a retry mechanism.

        Returns:
            A tuple of (result_type: str, feedback: str).
            result_type can be "SUCCESS", "NO_CHANGE", "UNEXPECTED_CHANGE", "NEEDS_MORE_TIME", or "VERIFIER_FAILURE".
        """
        logger.info("Performing categorized verification...")
        
        prompt =  self.verification_prompt.format(
            action=action_command,
            expected_outcome=expected_outcome
        )
        prev_img_base64 = base64.b64encode(prev_obs_bytes).decode('utf-8')
        curr_img_base64 = base64.b64encode(current_obs_bytes).decode('utf-8')
        messages = [{
            "role": "user", "content": [
                {"type": "text", "text": "This is the **Previous Screenshot**:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prev_img_base64}"}},
                {"type": "text", "text": "This is the **Current Screenshot**:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{curr_img_base64}"}},
                {"type": "text", "text": prompt}
            ]}]

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                response = self._call_verifier_llm(messages)
                if response.status_code != 200:
                    logger.error(f"Verifier API error (Status {response.status_code}): {response.text}")
                    if attempt == MAX_RETRIES - 1:
                        raise requests.exceptions.HTTPError(f"API failed with status {response.status_code}")
                    time.sleep(2 ** attempt)
                    continue

                response_json = response.json()
                content = response_json["choices"][0]["message"]["content"]
                
                json_str = content.strip()
                if json_str.startswith("```json"):
                    json_str = json_str.split("```json\n", 1)[1].rsplit("\n```", 1)[0]

                result = json.loads(json_str)
                result_type = result.get("result_type", "None")
                feedback = result.get("feedback", "No feedback from verifier.")
                future_impact = result.get("future_impact", "")
                
                logger.info(f"Verification Result: Type='{result_type}',Future Impact={future_impact}, Feedback='{feedback}'")
                return result_type, future_impact,feedback

            except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError, KeyError) as e:
                logger.error(f"Verification failed on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                if response is not None:
                    logger.error(f"Response content: {response.text}")
                else:
                    logger.error("No response received from verifier.")
                if attempt == MAX_RETRIES - 1:
                    logger.error("All verification attempts failed.")
                    return "Success", "", "The verifier failed to produce a valid response."
                time.sleep(2 ** attempt)

        return "Success", "", "VERIFIER_FAILURE"