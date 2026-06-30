# openrlhf/agent/exploration/exploration_agent.py
"""
Exploration agent for GUI task exploration.

NOTE: The action space used in this module (defined by the prompt templates and
parse_actions_from_tool_call) follows the Anthropic Claude Computer Use action format
(actions like left_click, type, scroll, key, etc. with coordinates and text params).
This action space is designed to work with Claude models via the native Anthropic
Messages API. The parse_actions_from_tool_call method converts Claude-format tool
calls into pyautogui commands for desktop execution.
"""

import base64
import json
import logging
import requests
import io
from PIL import Image
from typing import Dict, List, Any, Tuple
import time
from .prompts import *
from .prompts import EXPLORATION_PROMPT_TEMPLATE_EN
import random
import os
from exploration.node import Node

logger = logging.getLogger("desktopenv.agent.exploration")



def get_max_candidates(current_step: int) -> int:
    """
    Gets the maximum number of candidates based on the current step and total steps.
    """
    if current_step <= 2:
        max_candidates = 10
    elif current_step <= 10:
        max_candidates = 6
    else:
        max_candidates = 3
    return max_candidates

def resize_image_to_1024_768(image_bytes: bytes) -> Tuple[bytes, float, float]:
    """
    Resizes an image to 1024x768 for the model and calculates scaling ratios.
    This is a utility function placed here for clarity.
    """
    img = Image.open(io.BytesIO(image_bytes))
    original_width, original_height = img.size
    new_width, new_height = 1024, 768
    
    if (original_width, original_height) == (new_width, new_height):
        # No resize needed
        return image_bytes, 1.0, 1.0

    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img_resized.save(buffer, format='PNG')
    resized_bytes = buffer.getvalue()
    
    x_ratio = original_width / new_width
    y_ratio = original_height / new_height
    
    return resized_bytes, x_ratio, y_ratio


class ExplorationAgent:
    """
    A self-contained agent for GUI exploration.

    Handles image preprocessing, LLM interaction, and post-processing of
    candidate actions. Uses the Anthropic Claude Computer Use action format
    -- the parse_actions_from_tool_call method converts Claude-format tool
    calls into pyautogui commands, so this agent is designed to work with
    Claude-family models via an OpenAI-compatible API.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_url: str = "",
        api_key: str = "",
        max_tokens: int = 4096,
        action_space: str = "claude_computer_use",
        screen_size: tuple = (1024, 768),
        **kwargs,
    ):
        self.model_name = model
        self.api_url = api_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.action_space = action_space
        self.screen_size = screen_size

        logger.info(f"ExplorationAgent initialized for model: {self.model_name}.")
        self.history: List[Dict[str, Any]] = []
        self.final_task_summary: str = ""
        self.stages: List[Dict[str, Any]] = []
        self.explore_prompt_template = EXPLORATION_PROMPT_TEMPLATE_EN

    def reset(self, *args, **kwargs):
        """Reset the agent's internal state and history."""
        self.history = []
        logger.info("ExplorationAgent state and history have been reset.")

    def update_history(self, chosen_candidate: Dict[str, Any], verification_result: Tuple[str, str]):
        """
        Adds the executed action's summary AND the categorized verification result to the history.
        """
        result_type, future_impact, feedback = verification_result
        history_entry = {
            "step_goal": chosen_candidate.get("step_goal", "N/A"),
            "step_action": chosen_candidate.get("step_action", []),
            "step_reason": chosen_candidate.get("step_reason", ""),
            "final_goal_at_step": chosen_candidate.get("final_goal", "N/A"),
            'expected_observation': chosen_candidate.get("expected_observation", "N/A"),
            "future_impact": future_impact,
            "verification_result_type": result_type,
            "verification_feedback": feedback
        }
        self.history.append(history_entry)

    def _format_history_for_prompt(self, add_final_goal=True) -> str:
        """
        Formats the history for the prompt, including the categorized verification feedback.
        """
        # This check is added to prevent an error on the very first step when history is empty.
        if not self.history:
            return "No actions have been taken yet.\n- last_final_goal: None, please infer a starting goal."

        last_final_goal = self.history[-1].get("final_goal_at_step", "N/A")
        formatted_actions = []
        for i, entry in enumerate(self.history):
            action_str = json.dumps(entry['step_action'])
            result_type = entry['verification_result_type']
            feedback = entry['verification_feedback']
            
            formatted_actions.append(
                f"Step {i+1}:\n"
                f"  - Goal: {entry['step_goal']}\n"
                f"  - Action: {action_str}\n"
                f"  - Result: {result_type}\n"
                f"  - Feedback: {feedback}"
            )
        
        actions_string = "\n".join(formatted_actions)
        if add_final_goal:
            actions_string += f"\n- last_final_goal: '{last_final_goal}'"
        return actions_string


    def _call_exploration_llm(self, messages: List[Dict], payload: Dict = None) -> Dict:
        """
        Internal method to call the Claude API via the native Anthropic Messages endpoint.
        """
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        logger.info(f"Calling Claude API at: {self.api_url} with model: {self.model_name}")
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Claude API: {e}\n Response: {response.text if 'response' in locals() else 'No response'}", exc_info=True)
            return {'error': str(e)}
        

    def _enrich_candidates(self, candidates: list, x_ratio: float, y_ratio: float) -> list:
        """
        Internal method to process raw candidates from the LLM.
        It parses action strings and adds executable commands with scaled coordinates.
        """
        enriched_candidates = []
        for candidate in candidates:
            try:
                action_str = candidate.get("step_action", "[]")
                if isinstance(action_str,str):
                    parsed_actions = json.loads(action_str)
                else:
                    parsed_actions = action_str
                candidate["step_action"] = parsed_actions

                command_list = []
                for sub_action_input in parsed_actions:
                    mock_tool_call = {
                        "name": "computer",
                        "input": sub_action_input,
                        "id": "mock_tool_call" # ID is not used by parser
                    }
                    command = self.parse_actions_from_tool_call(mock_tool_call, x_ratio, y_ratio)
                    command_list.append(command.strip())
                
                candidate["action_command"] = "\n".join(command_list)
                enriched_candidates.append(candidate)
            except Exception as e:
                logger.error(f"Could not process candidate: {e}. Candidate: {candidate}")
                candidate["_processing_error"] = str(e)
                enriched_candidates.append(candidate)
        return enriched_candidates

    def predict(self, obs: Dict, domain=None, extra_guidance: str = None) -> List[Dict[str, Any]]:
        """
        Takes a raw observation, processes it, calls the LLM, and returns
        a list of fully enriched, ready-to-use candidate actions.
        """

        # 1. Pre-process the observation
        if "screenshot" not in obs:
            logger.error("No screenshot in observation.")
            return []
        
        # Resize image for the LLM and get scaling ratios for post-processing
        resized_bytes, x_ratio, y_ratio = resize_image_to_1024_768(obs["screenshot"])
        logger.info(f"Image resized. Original: {obs['screenshot'].__sizeof__()}B, Resized: {resized_bytes.__sizeof__()}B. Ratios: x={x_ratio:.2f}, y={y_ratio:.2f}")

        # 2. Prepare prompt and message for LLM
        history_str = self._format_history_for_prompt()
        prompt_text = self.explore_prompt_template.format(history=history_str, max_candidates=get_max_candidates(len(self.history)),extra_guidance=extra_guidance)
        screenshot_base64 = base64.b64encode(resized_bytes).decode('utf-8')

        
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_base64}},
        ]}]

        payload = {
            "model": self.model_name, "messages": messages, "max_tokens": self.max_tokens,
            "temperature": 0.2, "top_p": 1.0,
        }

        # Include computer_use tools for Claude models. For other models,
        # the prompt asks for JSON output directly in the content field.
        if "claude" in self.model_name.lower():
            payload["tools"] = [{'name': 'computer', 'type': 'computer_20250124',
                                 'display_width_px': 1024, 'display_height_px': 768, 'display_number': 1}]
            payload["anthropic_beta"] = ["computer-use-2025-01-24"]

        MAX_RETRIES = 5
        for retry in range(MAX_RETRIES):
        # 3. Call the LLM
            try:
                response_json = self._call_exploration_llm(messages, payload)

            # 4. Parse the raw candidates from the Anthropic response
                # Anthropic native response: content is a list of blocks
                content_blocks = response_json.get("content", [])
                response_text = ""
                for block in content_blocks:
                    if block.get("type") == "text":
                        response_text += block.get("text", "")
                    elif block.get("type") == "tool_use":
                        # If Claude returns a tool_use directly, extract the input
                        tool_input = block.get("input", {})
                        response_text = json.dumps(tool_input)
                logger.info(f"LLM Response: {response_text}")

                stop_reason = response_json.get("stop_reason", "")
                if stop_reason == "tool_use":
                    payload["messages"][0]["content"].append({"type": "text", "text": "Once again, please do not only return a single action of the 'computer' tool. Instead, return the JSON-format file as shown in the example, which includes several candidate actions."})
                    logger.info("Claude response was a tool_use. Retrying with additional guidance.")
                    continue
                

                json_str = response_text.strip()
                json_str = json_str.split("```json\n", 1)[1].rsplit("\n```", 1)[0]

                data = json.loads(json_str)
                raw_candidates = data.get("candidates", [])
                if not isinstance(raw_candidates, list) or not raw_candidates:
                    logger.error(f"Invalid or empty candidates in LLM response. Candidates: {raw_candidates}")
                    continue
                
                logger.info(f"Successfully generated {len(raw_candidates)} raw candidates.")
                # 5. Post-process and enrich the candidates with scaled commands
                enriched_candidates = self._enrich_candidates(raw_candidates, x_ratio, y_ratio)
                return enriched_candidates
            
            except Exception as e:
                logger.error(f"Failed to process LLM response: {e}. Response: {response_json}")
                #log text messages for debugging
                # logger.error(f"Messages sent: {messages}")
                if retry == MAX_RETRIES - 1:
                    return []
                logger.info(f"Retrying... ({retry + 1}/{MAX_RETRIES})")
                time.sleep(random.randint(1, 40) * retry)

        return []
    
    def parse_actions_from_tool_call(self, tool_call: Dict, screenshot_resize_ratio_x=1.0, screenshot_resize_ratio_y=1.0) -> str:
        result = ""
        function_args = tool_call.get("input", {})
        
        action = function_args.get("action")
        if not action:
            action = tool_call.get("function", {}).get("name")

        action_conversion = {
            "left click": "left_click",
            "right click": "right_click"
        }
        action = action_conversion.get(action, action)

        # 提取所有可能的参数
        text = function_args.get("text")
        coordinate = function_args.get("coordinate")
        start_coordinate = function_args.get("start_coordinate")
        scroll_direction = function_args.get("scroll_direction")
        scroll_amount = function_args.get("scroll_amount")
        duration = function_args.get("duration")

        # 统一处理坐标缩放
        if coordinate and (screenshot_resize_ratio_x != 1.0 or screenshot_resize_ratio_y != 1.0):
            coordinate = (
                int(coordinate[0] * screenshot_resize_ratio_x),
                int(coordinate[1] * screenshot_resize_ratio_y)
            )
            logger.info(f"Resized coordinate={coordinate}")

        if start_coordinate and (screenshot_resize_ratio_x != 1.0 or screenshot_resize_ratio_y != 1.0):
            start_coordinate = (
                int(start_coordinate[0] * screenshot_resize_ratio_x),
                int(start_coordinate[1] * screenshot_resize_ratio_y)
            )
            logger.info(f"Resized start_coordinate={start_coordinate}")
            
        # --- 辅助函数：处理组合键 ---
        def hold_keys_while_action(action_code):
            if not text or not isinstance(text, str):
                return action_code
            
            keys_to_hold = [k.strip().lower() for k in text.split('+')]
            
            # 按下所有组合键
            press_code = ""
            for key in keys_to_hold:
                press_code += f"pyautogui.keyDown('{key}')\n"
                
            # 释放所有组合键
            release_code = ""
            for key in reversed(keys_to_hold):
                release_code += f"pyautogui.keyUp('{key}')\n"
                
            return f"{press_code}{action_code}{release_code}"

        if action == "screenshot":
            result += "pyautogui.sleep(0.1)\n"
        
        elif action == "key":
            if not isinstance(text, str):
                raise ValueError(f"'text' must be a string for key action")
            
            key_conversion = { "page_down": "pagedown", "page_up": "pageup", "super_l": "win", "super": "command", "escape": "esc", "return": "enter" }
            keys = [key_conversion.get(k.strip().lower(), k.strip().lower()) for k in text.split('+')]
            
            # 使用 pyautogui.hotkey 来处理组合键更安全
            formatted_keys = ", ".join([f"'{k}'" for k in keys])
            result += f"pyautogui.hotkey({formatted_keys})\n"

        elif action == "hold_key":
            if not isinstance(text, str) or duration is None:
                raise ValueError("'text' and 'duration' are required for hold_key action")
            
            keys = [k.strip().lower() for k in text.split('+')]
            for key in keys:
                result += f"pyautogui.keyDown('{key}')\n"
            result += f"time.sleep({duration})\n"
            for key in reversed(keys):
                result += f"pyautogui.keyUp('{key}')\n"

        elif action == "type":
            if not isinstance(text, str):
                raise ValueError(f"'text' must be a string for type action")
            result += f"pyautogui.typewrite({repr(text)}, interval=0.01)\n"

        elif action == "mouse_move":
            if coordinate is None: raise ValueError("'coordinate' is required")
            x, y = coordinate
            result += f"pyautogui.moveTo({x}, {y}, duration={duration or 0.5})\n"

        elif action == "left_click_drag":
            if start_coordinate is None or coordinate is None:
                raise ValueError("'start_coordinate' and 'coordinate' are required for left_click_drag")
            start_x, start_y = start_coordinate
            end_x, end_y = coordinate
            result += f"pyautogui.moveTo({start_x}, {start_y}, duration=0.2)\n"
            result += f"pyautogui.dragTo({end_x}, {end_y}, duration={duration or 0.5})\n"

        elif action in ("left_click", "right_click", "middle_click", "double_click", "triple_click"):
            click_map = {
                "left_click": "click", "right_click": "rightClick", "middle_click": "middleClick",
                "double_click": "doubleClick", "triple_click": "tripleClick"
            }
            func_name = click_map[action]
            
            if coordinate:
                x, y = coordinate
                action_code = f"pyautogui.{func_name}({x}, {y})\n"
            else:
                action_code = f"pyautogui.{func_name}()\n"
                
            result += hold_keys_while_action(action_code)

        elif action == "left_mouse_down":
            result += "pyautogui.mouseDown(button='left')\n"
        
        elif action == "left_mouse_up":
            result += "pyautogui.mouseUp(button='left')\n"

        elif action == "scroll":
            if scroll_direction is None or scroll_amount is None:
                raise ValueError("'scroll_direction' and 'scroll_amount' are required for scroll")

            scroll_val = scroll_amount if scroll_direction in ('up', 'right') else -scroll_amount
            
            if scroll_direction in ("up", "down"):
                scroll_func = "scroll"
            else: # left, right
                scroll_func = "hscroll"

            if coordinate:
                x, y = coordinate
                action_code = f"pyautogui.{scroll_func}({scroll_val}, x={x}, y={y})\n"
            else:
                action_code = f"pyautogui.{scroll_func}({scroll_val})\n"
                
            result += hold_keys_while_action(action_code)

        elif action == "wait":
            if duration is None: 
                result += "time.sleep(1)\n"
            else:
                result += f"time.sleep({duration})\n"

        # 其他您自定义的动作
        elif action == "done":
            result += "DONE"
        elif action == "fail":
            result += "FAIL"

        # 未在代码中处理的动作
        elif action == "cursor_position":
            # 这个动作是获取信息，而不是执行动作，所以在这个函数中通常不生成代码
            # 可以考虑在另一个流程中处理它，或者直接返回一个注释
            result += "# Action 'cursor_position' is not an executable command in this context.\n"
        else:
            raise ValueError(f"Invalid or unsupported action: {action}")
        
        return result

    def summarize(self) -> str:
        """
        Generates a final, high-quality task summary based on the agent's
        internal history of the completed trajectory.

        Returns:
            A string containing the final task summary command.
        """
        if not self.history:
            return "No history available to summarize."

        logger.info("Generating final task summary...")
        
        # 1. Format the history for the summary prompt
        history_str = self._format_history_for_prompt(add_final_goal=False)

        # 2. Prepare prompt and message for the summary LLM call
        prompt = SUMMARY_PROMPT_TEMPLATE.format(history=history_str)
        messages = [{"role": "user", "content": prompt}]


        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.2, "top_p": 1.0,
        }


        # 3. Call the LLM for summary
        MAX_RETRIES = 3
        for retry in range(MAX_RETRIES):
        # 3. Call the LLM
            try:
                response_json = self._call_exploration_llm(messages,payload)

            # 4. Parse the raw candidates from the response
                # Parse Anthropic native response
                content_blocks = response_json.get("content", [])
                response_text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
                logger.info(f"LLM Response of summary: {response_text}")


                json_str = response_text.strip()
                json_str = json_str.split("```json\n", 1)[1].rsplit("\n```", 1)[0]

                summary  = json.loads(json_str)

                self.final_task_summary = summary.get("final_task_summary", "")
                # self.stages = summary.get("stages", [])
                
                # 5. Post-process and enrich the candidates with scaled commands
                return summary 
            
            except Exception as e:
                logger.error(f"Failed to generate or parse summary: {e}")
                if retry == MAX_RETRIES - 1:
                    return []
                logger.info(f"Retrying... ({retry + 1}/{MAX_RETRIES})")
                time.sleep(random.randint(1, 20))

    def _populate_history_from_path(self, node_path: List[str], nodes_dir: str) -> bool:
        """
        Private helper to populate self.history from a validated node_path.
        
        It iterates through the node IDs (skipping the root), reads each
        corresponding node file, and constructs the history entries.

        Args:
            node_path: An ordered list of node IDs, from root to the final node.
            nodes_dir: The path to the directory containing the node JSON files.

        Returns:
            True if the process completes (even with warnings), False on critical failure.
        """
        self.reset()
        new_history = []

        if len(node_path) < 2:
            logger.info("Node path has less than two nodes. History will be empty.")
            return True

        # Skip the root node (index 0) as it represents the initial state
        for node_id in node_path[1:]:
            node_file_path = os.path.join(nodes_dir, f"{node_id}.json")
            try:
                with open(node_file_path, 'r', encoding='utf-8') as f:
                    node_data = json.load(f)
                
                # Map data from the node file to the history entry structure
                history_entry = {
                    "step_goal": node_data.get("step_goal", "N/A"),
                    "step_action": node_data.get("step_action", []),
                    "step_reason": node_data.get("step_reason", ""),
                    "expected_observation": node_data.get("expected_observation", "N/A"),
                    "final_goal_at_step": node_data.get("final_goal", "N/A"),
                    "verification_result_type": node_data.get("verification_result", {}).get("result_type", "UNKNOWN"),
                    "future_impact":node_data.get("verification_result", {}).get("future_impact", ""),
                    "verification_feedback": node_data.get("verification_result", {}).get("feedback", "")
                }
                new_history.append(history_entry)

            except FileNotFoundError:
                logger.warning(f"Node file not found while loading history: {node_file_path}. Skipping this step.")
                continue
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from node file {node_file_path}. Error: {e}. Skipping this step.")
                continue

        self.history = new_history
        logger.info(f"History successfully populated with {len(self.history)} steps from the provided path.")
        return True

    def load_history_from_trajectory(self, trajectory_manifest_path: str) -> bool:
        """
        Loads the agent's history from a trajectory manifest file.

        This method reads the sequence of nodes defined in the provided manifest
        to reconstruct the agent's internal history.

        Args:
            trajectory_manifest_path: The path to the trajectory manifest JSON file.

        Returns:
            True if the history was loaded successfully, False otherwise.
        """
        logger.info(f"Attempting to load history from trajectory manifest: {trajectory_manifest_path}")
        if not os.path.isfile(trajectory_manifest_path):
            logger.error(f"Trajectory manifest file not found at: {trajectory_manifest_path}")
            return False

        try:
            with open(trajectory_manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            
            node_path = manifest_data.get("node_path")
            if not isinstance(node_path, list):
                logger.error("Manifest file does not contain a valid 'node_path' list.")
                return False
            
            # Infer the session and nodes directory from the manifest path
            # manifest path: .../<session_dir>/trajectories/<traj_dir>/<file>.json
            session_dir = os.path.dirname(os.path.dirname(os.path.dirname(trajectory_manifest_path)))
            nodes_dir = os.path.join(session_dir, "nodes")

            return self._populate_history_from_path(node_path, nodes_dir)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from manifest file {trajectory_manifest_path}. Error: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading from trajectory: {e}", exc_info=True)
            return False

    def load_history_from_node(self, end_node_path: str) -> bool:
        """
        Loads the agent's history by traversing up from a given end node.

        This reconstructs the execution path by following 'parent_node_id' links
        from the specified end node up to the root, then populates the history.

        Args:
            end_node_path: The path to a node JSON file, typically the last in a sequence.

        Returns:
            True if the history was loaded successfully, False otherwise.
        """
        logger.info(f"Attempting to load history by traversing up from node: {end_node_path}")
        if not os.path.isfile(end_node_path):
            logger.error(f"End node file not found at: {end_node_path}")
            return False
        
        nodes_dir = os.path.dirname(end_node_path)
        reconstructed_path = []
        current_node_id = os.path.splitext(os.path.basename(end_node_path))[0]
        
        MAX_TRAVERSAL = 200 # Safety break to prevent infinite loops
        for i in range(MAX_TRAVERSAL):
            if not current_node_id:
                break # Reached the root's parent (None), which is the correct termination
            
            reconstructed_path.append(current_node_id)
            current_file_path = os.path.join(nodes_dir, f"{current_node_id}.json")

            try:
                with open(current_file_path, 'r', encoding='utf-8') as f:
                    node_data = json.load(f)
                current_node_id = node_data.get("parent_node_id")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(f"Failed to read or parse parent node file {current_file_path}. Aborting traversal. Error: {e}")
                return False
        else:
            logger.warning(f"Node traversal stopped after reaching the limit of {MAX_TRAVERSAL} steps. The reconstructed history might be incomplete.")

        # The path was built from end to start, so we reverse it
        reconstructed_path.reverse()
        logger.info(f"Reconstructed a path of length {len(reconstructed_path)}.")
        
        return self._populate_history_from_path(reconstructed_path, nodes_dir)


    def _generate_thinking_by_rule(self, context: Dict) -> str:
        """Generates a thinking process using a fixed rule-based template."""
        future_steps_str = "\n".join(
            f"- {goal}" for goal in context['future_step_goals']
        ) if context['future_step_goals'] else "This is the final planned action for this stage."

        thinking = f"""#### Observation
Based on the current task, I am observing the state prior to executing the goal: '{context['current_step_goal']}'.

#### Reasoning
My overall objective is to '{context['global_task_summary']}'. To achieve this, the immediate goal is '{context['current_step_goal']}'. This action is necessary because it directly enables the following planned steps:
{future_steps_str}

#### Anticipation
After executing the command `{context['current_action_command']}`, I expect the screen to change in a way that confirms the completion of the current step goal.

#### Self-Correction/Confirmation
If the expected outcome is observed, I will proceed with the next step. Otherwise, I would need to re-evaluate the state and re-plan.
"""
        return thinking

    def _generate_thinking_by_llm(self, context: Dict) -> str:
        """Generates a thinking process by calling a large language model."""
        
        future_steps_list = [f"- Next Step's Goal: {g}" for g in context['future_step_goals']]
        if not future_steps_list:
            future_steps_formatted = "- This is the final step in the current stage."
        else:
            future_steps_formatted = "\n".join(future_steps_list)

        prompt = THINKING_PROMPT_TEMPLATE.format(
            global_task_summary=context['global_task_summary'],
            current_step_goal=context['current_step_goal'],
            current_action_command=context['current_action_command'],
            future_steps_formatted=future_steps_formatted
        )
        
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model_name, 
            "messages": messages, 
            "max_tokens": 2048, # Allocate enough tokens for a detailed thought process
            "temperature": 0.3, 
            "top_p": 1.0,
            # No tools needed for this generation task
        }

        MAX_RETRIES = 3
        for i in range(MAX_RETRIES):
            try:
                response_json = self._call_exploration_llm(messages, payload)
                if not response_json:
                    raise ValueError("API call returned None.")
                
                # Parse Anthropic native response
                content_blocks = response_json.get("content", [])
                response_text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")

                # Robustly find and parse the JSON block
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if not json_match:
                    raise json.JSONDecodeError("No JSON object found in the response.", response_text, 0)
                
                thinking_data = json.loads(json_match.group())
                
                if "thinking" in thinking_data and isinstance(thinking_data["thinking"], str):
                    return thinking_data["thinking"]
                else:
                    raise ValueError("JSON response does not contain a valid 'thinking' string.")
            
            except Exception as e:
                logger.error(f"Attempt {i+1}/{MAX_RETRIES}: Failed to generate or parse thinking from LLM. Error: {e}")
                if i == MAX_RETRIES - 1:
                    return f"Error: Failed to generate thinking after {MAX_RETRIES} attempts."
                time.sleep(2) # Wait before retrying
        
        return "Error: Exited retry loop unexpectedly."


    def generate_thinking_for_trajectory(
        self, 
        trajectory_manifest_path: str, 
        mode: str = 'llm', 
        future_k: int = 3
    ):
        """
        Main method to generate and save 'step_thinking' for an entire trajectory.

        Args:
            trajectory_manifest_path: Path to the trajectory manifest JSON file.
            mode: The generation mode, either 'rule' or 'llm'.
            future_k: The number of future steps to consider for context (planning horizon).
        """
        logger.info(f"Starting reverse thinking generation for: {trajectory_manifest_path}")
        logger.info(f"Mode: {mode}, Future K: {future_k}")

        if not os.path.isfile(trajectory_manifest_path):
            logger.error(f"Manifest file not found: {trajectory_manifest_path}")
            return
        
        try:
            with open(trajectory_manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Could not parse manifest file: {e}")
            return

        node_path = manifest_data.get("node_path", [])
        final_task_stages = manifest_data.get("final_task", [])

        if len(node_path) < 2:
            logger.warning("Trajectory has less than 2 nodes, skipping.")
            return

        session_dir = os.path.dirname(os.path.dirname(os.path.dirname(trajectory_manifest_path)))
        nodes_dir = os.path.join(session_dir, "nodes")

        # Create a quick lookup map for step number to global task
        step_to_global_task_map = {}
        if isinstance(final_task_stages, list):
            for stage in final_task_stages:
                if isinstance(stage, dict) and 'start_step' in stage and 'end_step' in stage:
                    for step_num in range(stage['start_step'], stage['end_step'] + 1):
                        step_to_global_task_map[step_num] = stage.get('task_summary', 'N/A')

        # Pre-load all node data for efficiency
        all_node_data = {}
        for node_id in node_path:
            node_file = os.path.join(nodes_dir, f"{node_id}.json")
            try:
                with open(node_file, 'r', encoding='utf-8') as f:
                    all_node_data[node_id] = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logger.error(f"Could not load or parse node file: {node_file}")
                return

        # Iterate through each step (node) in the trajectory to generate thinking
        # Skip the root node (index 0)
        for i in range(1, len(node_path)):
            current_node_id = node_path[i]
            current_node_data = all_node_data[current_node_id]
            step_num = current_node_data.get('step_number', i)

            logger.info(f"--- Processing Step {step_num} (Node: {current_node_id}) ---")

            # 1. Gather context
            global_task = step_to_global_task_map.get(step_num, "Overall goal not defined for this step.")
            
            future_step_goals = []
            for k in range(1, future_k + 1):
                if (i + k) < len(node_path):
                    future_node_id = node_path[i + k]
                    future_node_data = all_node_data.get(future_node_id, {})
                    future_goal = future_node_data.get('step_goal', None)
                    if future_goal:
                        future_step_goals.append(future_goal)

            context = {
                "global_task_summary": global_task,
                "current_step_goal": current_node_data.get('step_goal', 'N/A'),
                "current_action_command": current_node_data.get('action_command', 'N/A'),
                "future_step_goals": future_step_goals
            }

            # 2. Generate thinking based on mode
            if mode == 'rule':
                thinking_text = self._generate_thinking_by_rule(context)
            elif mode == 'llm':
                thinking_text = self._generate_thinking_by_llm(context)
            else:
                logger.error(f"Invalid mode '{mode}'. Choose 'rule' or 'llm'.")
                return

            logger.info(f"Generated Thinking: {thinking_text[:100]}...")

            # 3. Save the generated thinking back to the node file
            current_node_data['step_thinking'] = thinking_text
            node_file_path = os.path.join(nodes_dir, f"{current_node_id}.json")
            try:
                with open(node_file_path, 'w', encoding='utf-8') as f:
                    json.dump(current_node_data, f, indent=2, ensure_ascii=False)
            except IOError as e:
                logger.error(f"Failed to write updated thinking to node file {node_file_path}: {e}")
        
        logger.info(f"\nSuccessfully generated and saved thinking for all {len(node_path)-1} steps in the trajectory.")


    def update_history_with_node(self, executed_node: Node):
        """
        Updates the agent's history based on the node that was just executed.
        This version correctly accesses the top-level attributes of the Node object.
        """
        result = executed_node.execution_result or {}
        result_type = result.get("result_type", "UNKNOWN")
        feedback = result.get("feedback", "")
        future_impact = result.get("future_impact", "")

        # Directly access attributes from the Node object
        history_entry = {
            "step_goal": executed_node.step_goal or "N/A",
            "step_action": executed_node.step_action or [],
            "step_reason": executed_node.step_reason or "",
            "final_goal_at_step": executed_node.final_goal or "N/A",
            'expected_observation': executed_node.expected_observation or "N/A",
            "future_impact":future_impact,
            "verification_result_type": result_type,
            "verification_feedback": feedback
        }
        self.history.append(history_entry)

    def load_history_from_path(self, node_path: List[str], nodes_dir: str) -> bool:
        """
        Loads the agent's history by reading the sequence of nodes in a given path.
        """
        self.reset()
        if len(node_path) < 2: return True # Nothing to load for a single node

        for i in range(len(node_path) - 1):
            child_node_id = node_path[i+1]
            child_node = Node.load(nodes_dir, child_node_id)
            if not child_node:
                self.logger.warning(f"Could not load node {child_node_id}. History may be incomplete.")
                continue
            self.update_history_with_node(child_node)

        self.logger.info(f"History loaded with {len(self.history)} steps for path ending at {node_path[-1]}.")
        return True