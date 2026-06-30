# openrlhf/agent/exploration/worker.py
import os
import sys
import logging
import time
import random
import datetime
import signal
import json
import glob
from argparse import Namespace
from multiprocessing import Queue
from typing import Dict, Any, Tuple, List, Optional
import uuid

# Ensure the TreeCUA root is on the Python path so that imports resolve correctly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from OSWorld.desktop_env import DesktopEnv
from exploration.exploration_agent import ExplorationAgent
from exploration.verification_agent import VerificationAgent
from exploration.traj_score_agent import TrajectoryScoringAgent
from exploration.tree_manager import ExplorationTreeManager
from exploration.node import Node
from exploration.trajectory_utils import create_trajectory_manifest, is_screenshot_consistent,create_annotated_trajectory_montage
from exploration.prompts import AVOID_REPETITION_PROMPT

class ExplorationWorker:
    """Encapsulates the entire logic for a single exploration worker."""
    def __init__(self, worker_id: int, tree_manager: ExplorationTreeManager, completion_queue: Queue, args: Namespace, session_dir: str):
        self.worker_id = worker_id
        self.tree_manager = tree_manager
        self.completion_queue = completion_queue
        self.args = args
        self.session_dir = session_dir
        self.nodes_dir = os.path.join(session_dir, "nodes")
        self.screenshots_dir = os.path.join(session_dir, "screenshots")
        self.logger = logging.getLogger(f"worker.{self.worker_id}")
        
        # Agents
        self.agent: Optional[ExplorationAgent] = None
        self.verifier: Optional[VerificationAgent] = None
        self.scorer: Optional[TrajectoryScoringAgent] = None 
        self.env: Optional[DesktopEnv] = None
        
        self._shutdown_requested = False
        self.logger.info(f"Worker initialized. Session Dir: {self.session_dir}")

        self.config = []
        
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
    
    def _handle_shutdown_signal(self, signum, frame):
        signal_name = signal.Signals(signum).name
        self.logger.info(f"Received {signal_name} signal. Initiating graceful shutdown...")
        self._shutdown_requested = True

    def run(self):
        self.logger.info(">>> Worker Loop Started. Setting up components...")
        self._setup_components()
        
        try:
            consecutive_no_task_count = 0
            max_consecutive_no_task = 600
            
            while not self._shutdown_requested:
                self.logger.info("--- Polling for a new task ---")
                
                # Update unpack to handle sub_task_info
                task_data = self.tree_manager.get_task()

                if task_data:
                    consecutive_no_task_count = 0
                    start_node_id, full_node_path, initial_config, sub_task_info = task_data

                    
                    self.config = initial_config
                    self.logger.info(f"Task acquired! Start Node: {start_node_id}, Path Length: {len(full_node_path)}")
                    


                    try:
                        if self._shutdown_requested: break
                        if self.env is None:
                            if not self._setup_env(): break
                            
                        t0 = time.time()
                        # Pass sub_task_info to the runner
                        self._run_trajectory_from_start_node(start_node_id, full_node_path, sub_task_info)
                        self.logger.info(f"Task {start_node_id} finished. Duration: {time.time() - t0:.2f}s")
                        
                    except Exception as e:
                        self.logger.error(f"!!! Unexpected error during task: {e}", exc_info=True)
                        if self.env:
                            try: self.env.close()
                            except: pass
                            self.env = None
                else:
                    consecutive_no_task_count += 1
                    if consecutive_no_task_count >= max_consecutive_no_task:
                        break
                    for _ in range(300):
                        if self._shutdown_requested: break
                        time.sleep(1)
            
        except Exception as e:
            self.logger.critical(f"Critical error in worker main loop: {e}", exc_info=True)
        finally:
            if self.env: 
                try: self.env.close()
                except: pass

    def _run_trajectory_from_start_node(self, start_node_id: str, full_node_path: List[str], sub_task_info: Optional[Dict]):
        self.logger.info(f"=== Starting Trajectory. Target Start Node: {start_node_id} ===")
        traj_id = f"traj_{datetime.datetime.now().strftime('%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"
        
        try:
            obs, current_node = self._replay_and_verify_path(full_node_path)
            if not obs:
                return # Replay failed
        except Exception as e:
            self.logger.error(f"Critical error during replay logic: {e}", exc_info=True)
            return

        consecutive_failures = 0
        termination_reason = "UNKNOWN"
        step_count = 0
        
        self.logger.info("Replay complete. Starting Exploration Loop.")

        while len(full_node_path) < self.args.max_steps_exploration:
            step_count += 1
            self.logger.info(f"--- Step {step_count} (Node Depth: {current_node.depth}) ---")
            
            # Pass sub_task_info to exploration step
            obs, current_node, step_result = self._perform_exploration_step(current_node, obs, sub_task_info)

            if step_result == "PREDICT_FAILURE":
                termination_reason = "PREDICT_FAILURE"
                break
            
            full_node_path.append(current_node.node_id)

            if step_result == "SUCCESS_DONE":
                termination_reason = "SUCCESS_DONE"
                break
            
            if step_result == "STEP_FAILURE":
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    termination_reason = "MAX_FAILURES"
                    break
            else:
                consecutive_failures = 0
        
        if termination_reason == "UNKNOWN":
            termination_reason = "MAX_STEP"
        
        self._finalize_trajectory(full_node_path, current_node, termination_reason,traj_id)

    def _replay_and_verify_path(self, full_node_path: List[str]) -> Tuple[Optional[Dict], Optional[Node]]:
        start_node = Node.load(self.nodes_dir, full_node_path[-1])

        try:
            self.agent.reset()
            self.env.reset(self.config)
            time.sleep(10)
        except Exception as e:
            self.logger.error(f"Reset Faild when loading {full_node_path[-1]}", exc_info=True)
            start_node.status = "UNEXPLORED"
            self.tree_manager.update_node(start_node)
            time.sleep(10)
            return None, None


        if not start_node: return None, None

        if start_node.depth == 0:
            obs = self.env._get_obs()
            start_node.screenshot = f"{start_node.node_id}.png"
            with open(os.path.join(self.screenshots_dir, start_node.screenshot), "wb") as f: 
                f.write(obs['screenshot'])
            start_node.status = "EXPLORING"
            self.tree_manager.update_node(start_node)
            return obs, start_node
       

        parent_path = full_node_path[:-1]
        self.agent.load_history_from_path(parent_path, self.nodes_dir)
        
        replay_action_commands = []
        for node_id in parent_path[1:]:
            node = Node.load(self.nodes_dir, node_id)
            if node:
                self.env.step(node.action_command)
                # replay_action_commands.append(node.action_command)
                # replay_action_commands.append("pyautogui.sleep(8)")
            else:
                self.logger.error(f"Could not load ancestor node {node_id} during replay.")
                return None, None
        # replay_action_commands = "\n".join(replay_action_commands)
        # self.env.step(replay_action_commands)
        
        parent_obs = self.env._get_obs() 

        # === 校验:当前的父节点是否与历史轨迹的父节点状态一致===
        parent_node_id = parent_path[-1]
        parent_screenshot_path = os.path.join(self.screenshots_dir, f"{parent_node_id}.png")
        current_parent_screenshot_path = os.path.join(self.screenshots_dir, f"{parent_node_id}_replay_{full_node_path[-1]}.png")
        with open(current_parent_screenshot_path, "wb") as f: f.write(parent_obs['screenshot'])
        
        try:
            with open(parent_screenshot_path, "rb") as f:
                historical_parent_screenshot = f.read()

            verify_result, _, feedback = self._verify_step(
                prev_obs=historical_parent_screenshot,
                command="Check for state consistency.", # 这是一个校验动作
                done=False,
                # 在预期观察中明确指出允许的差异
                expected_observation=(
                    "The screen should be identical to the reference image. "
                    "Ignore differences in time/clock, rotating ads, dynamic news feeds, "
                    "blinking cursors, or minor scroll shifts. "
                    "Focus on whether the main window, open files, and available tools are the same."
                ),
                current_obs=parent_obs
            )

            if verify_result != "SUCCESS":
                self.logger.error(f"Consistency check failed for node {start_node.node_id}. Parent state does not match replay state. Feedback: {feedback}")
                
                start_node.status = "LOAD_FAILED"
                start_node.execution_result = {
                    "result_type": "REPLAY_DIVERGENCE", 
                    "feedback": f"Environment state inconsistent with parent node after replay. VLM Feedback: {feedback}"
                }
                self.tree_manager.update_node(start_node)
                return None, None
            else:
                self.logger.info(f"Successful load parent({parent_node_id}) for {start_node.node_id}")

        except FileNotFoundError:
            self.logger.error(f"Historical screenshot missing for node {parent_node_id}, skipping consistency check.")
        except Exception as e:
            self.logger.error(f"Error during consistency check node {parent_node_id}: {e}", exc_info=True)



        done = "DONE" in (start_node.action_command or "")
        expected_observation = (start_node.expected_observation or "") 
        
        obs = self.env.step(start_node.action_command, self.args.sleep_after_execution)[0] if not done else parent_obs
        result_type, future_impact, feedback = self._verify_step(parent_obs['screenshot'], start_node.action_command, done, expected_observation, obs)
        
        start_node.screenshot = f"{start_node.node_id}.png"
        with open(os.path.join(self.screenshots_dir, start_node.screenshot), "wb") as f: f.write(obs['screenshot'])
        start_node.execution_result = {"result_type":result_type, "future_impact":future_impact, "feedback": feedback}

        if result_type != "SUCCESS":
            start_node.status = "FAILURE"
            self.tree_manager.update_node(start_node)
            return None, None

        else:
            self.agent.update_history_with_node(start_node)
            start_node.status = "LOAD_SUCCESS"
            self.tree_manager.update_node(start_node)

        # === 3. 记录 Step 1 + Step 2 逻辑 ===
        if start_node.depth == 2 and result_type in ["SUCCESS", "SUCCESS_DONE"]:
            try:
                parent_node = Node.load(self.nodes_dir, start_node.parent_node_id)
                if parent_node:
                    step1_goal = parent_node.step_goal or "N/A"
                    step2_goal = start_node.step_goal or "N/A"
                    self.tree_manager.record_prefix_history(step1_goal, step2_goal)
            except Exception as e:
                self.logger.error(f"Failed to record prefix history: {e}")
                
        return obs, start_node

    def _perform_exploration_step(self, current_node: Node, obs: Dict, sub_task_info: Optional[Dict]) -> Tuple[Dict, Node, str]:
        """Runs prediction, execution, and verification for one step."""
        
        
        # === 1. 构建额外 Prompt 逻辑 (Global History) ===
        #Get global prefix history to avoid repetition
        extra_guidance = ""
        avoid_history = None
        if current_node.depth < 4:
            prefix_history = self.tree_manager.get_global_prefix_history()
            if prefix_history:
                sampled_history = random.sample(prefix_history, min(30, len(prefix_history)))
                history_text = "\n".join([f"- {h}" for h in sampled_history])
                avoid_history = AVOID_REPETITION_PROMPT.format(forbidden_paths=history_text)
                self.logger.info(f"Injecting global prefix avoidance prompt.")
        if avoid_history:
            extra_guidance += f"\n{avoid_history}\n"

        if current_node.depth <= 3:
            extra_guidance += "If you believe that the current interface (such as a file, website, etc.) is not suitable for completing the current type of task (for example, the file is not editable, the file lacks the elements necessary to complete this type of task, the website fails to open, or the website lacks interactive elements, etc.), please directly generate a unique candidate that includes the 'done' action."
            
        #task-specific guidance      
        task_guidance = None
        if sub_task_info:
            category_name = sub_task_info.get("category_name", "Specified Task")
            description = sub_task_info.get("description", "")
            all_prompts = sub_task_info.get("prompts", [])
            
            # 随机采样 3 个例子
            selected_examples = []
            if all_prompts:
                selected_examples = random.sample(all_prompts, min(3, len(all_prompts)))
            
            examples_text = "\n".join([f"- {ex}" for ex in selected_examples])
            
            task_guidance = (
                f"### Current Task Focus: {category_name}\n"
                f"**Description**: {description}\n\n"
                f"**Examples of goals in this category**:\n{examples_text}\n"
                f"\nPlease align your exploration goals with the nature of '{category_name}' described above.",
                f"The task types described in the 'Description' and 'Examples' sections are merely for reference. We encourage designing more diverse tasks that are different from the examples based on the current observation, but most of the tasks should conform to the {category_name} category."
            )
        if task_guidance:
            extra_guidance += f"\n{task_guidance}\n"

        # config additional instructions
        if 'instruction' in self.config and self.config['instruction'].strip():
            extra_guidance += f"\n##Additional information regarding the current initial state: {self.config['instruction']}\n"
        
        # ===============================================

        candidates = self.agent.predict(
            obs, 
            domain=self.args.domain, 
            extra_guidance=extra_guidance, 
        )

        if not candidates:
            current_node.status = "PREDICT_FAILURE"
            self.tree_manager.update_node(current_node)
            return obs, current_node, "PREDICT_FAILURE"

        current_node.status = "SUCCESS" 
        self.tree_manager.update_node(current_node)

        new_nodes = self.tree_manager.report_new_candidates(current_node, candidates)
        chosen_node = random.choice(new_nodes)
        chosen_node.status = "EXPLORING"
        self.tree_manager.update_node(chosen_node)

        prev_obs_bytes = obs['screenshot']
        command = chosen_node.action_command
        done = "DONE" in (command or "")
        
        next_obs = self.env.step(command, self.args.sleep_after_execution)[0] if not done else obs
        expected_observation = (chosen_node.expected_observation or "")

        result_type, future_impact, feedback = self._verify_step(prev_obs_bytes, command, done, expected_observation, next_obs)
        
        chosen_node.screenshot = f"{chosen_node.node_id}.png"
        with open(os.path.join(self.screenshots_dir, chosen_node.screenshot), "wb") as f: f.write(next_obs['screenshot'])
        
        chosen_node.execution_result = {"result_type": result_type, "future_impact":future_impact,"feedback": feedback}
        self.agent.update_history_with_node(chosen_node)

        step_result = "SUCCESS"
        if done: step_result = "SUCCESS_DONE"
        if result_type != "SUCCESS": step_result = "STEP_FAILURE"

        # === 3. 记录 Step 1 + Step 2 逻辑 ===
        # if chosen_node.depth == 2 and step_result in ["SUCCESS", "SUCCESS_DONE"]:
        #     try:
        #         parent_node = Node.load(self.nodes_dir, chosen_node.parent_node_id)
        #         if parent_node:
        #             step1_goal = parent_node.step_goal or "N/A"
        #             step2_goal = chosen_node.step_goal or "N/A"
        #             self.tree_manager.record_prefix_history(step1_goal, step2_goal)
        #     except Exception as e:
        #         self.logger.error(f"Failed to record prefix history: {e}")
        # ===================================

        return next_obs, chosen_node, step_result

    def _finalize_trajectory(self, node_path: List[str], final_node: Node, reason: str,traj_id:str):
        traj_id = f"{traj_id}_{reason.lower()}"
        self.logger.info(f"Finalizing trajectory. Reason: {reason}. Path len: {len(node_path)} Path: {(node_path)}")
        
        final_status = "SUCCESS" if "SUCCESS" in reason or reason == "MAX_STEP" else "FAILURE"
        if final_node.status == "EXPLORING":
            final_node.status = final_status
            self.tree_manager.update_node(final_node)
        
        if reason == "PREDICT_FAILURE":
            self.logger.warning("Trajectory ended due to prediction failure.")
            return
    
        
        traj_dir = os.path.join(self.session_dir, 'trajectories', traj_id)
        # 1. 报告轨迹完成计数 (传入 Root Node ID)
        if node_path and final_status == "SUCCESS":
            self.tree_manager.report_trajectory_completion(node_path[0])
        create_trajectory_manifest(
            self.session_dir, node_path, reason, self.agent, traj_id,
            generate_summary=getattr(self.args, 'enable_inline_summary', False)
        )

        try:
            create_annotated_trajectory_montage(self.session_dir, traj_dir, node_path)
        except Exception as e:
            self.logger.error(f"Failed to create trajectory montage: {e}", exc_info=True)


        # === 3. 运行打分 Agent (Traj Score) ===
        if self.scorer:
            try:
                json_path = os.path.join(traj_dir, f"{traj_id}.json")
                if os.path.exists(json_path):
                    self.logger.info(f"Scoring trajectory: {json_path}")
                    with open(json_path, 'r', encoding='utf-8') as f:
                        traj_data = json.load(f)
                    
                    score_result = self.scorer.score_trajectory(traj_data)
                    
                    traj_data['quality_assessment'] = score_result
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(traj_data, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"Score saved: {score_result}")
            except Exception as e:
                self.logger.error(f"Error during trajectory scoring: {e}", exc_info=True)
        # =====================================

        self.completion_queue.put(1)

    def _setup_components(self):
        self.logger.info(f"Setting up Agent (Model: {self.args.model})...")
        self.agent = ExplorationAgent(
            model=self.args.model, api_url=self.args.api_base_url, api_key=self.args.api_key,
            max_tokens=self.args.max_tokens, action_space=self.args.action_space,
            screen_size=(self.args.screen_width, self.args.screen_height)
        )
        
        if self.args.use_verifier:
            self.verifier = VerificationAgent(
                model=self.args.verifier_model or self.args.model,
                api_url=self.args.verifier_api_base_url or self.args.api_base_url,
                api_key=self.args.verifier_api_key or self.args.api_key,
                max_tokens=self.args.verifier_max_tokens
            )
        
        # 初始化打分 Agent (only if inline scoring is enabled)
        if getattr(self.args, 'enable_inline_scoring', False):
            self.scorer = TrajectoryScoringAgent(
                model=self.args.model,
                api_url=self.args.api_base_url,
                api_key=self.args.api_key,
                max_tokens=1024
            )
            self.logger.info("Scoring Agent initialized (inline mode).")
        else:
            self.scorer = None
            self.logger.info("Inline scoring disabled. Use post-processing pipeline offline.")

    def _setup_env(self) -> bool:
        if self.env is None:
            try:
                self.env = DesktopEnv(
                    path_to_vm=self.args.path_to_vm, action_space=self.args.action_space,
                    provider_name=self.args.provider_name,
                    screen_size=(self.args.screen_width, self.args.screen_height),
                    headless=self.args.headless
                )
                return True
            except Exception as e:
                self.logger.critical(f"Failed to create DesktopEnv: {e}", exc_info=True)
                return False
        return True

    def _verify_step(self, prev_obs: bytes, command: str, done: bool, expected_observation:str, current_obs: Dict) -> Tuple[str, str, str]:
        if self.verifier:
            if done: return "SUCCESS", "Successfully completed","Terminated."
            return self.verifier.verify(prev_obs, command, expected_observation, current_obs['screenshot'])
        return "SUCCESS", "Successfully completed", "Verification disabled."

def worker_process(worker_id: int, tree_manager: ExplorationTreeManager, completion_queue: Queue, args: Namespace, session_dir: str):
    log_dir = os.path.join(session_dir, "worker_logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"worker_{worker_id}.log"),
        level=getattr(logging, args.log_level.upper()),
        format=f"%(asctime)s [%(levelname)s] [W-{worker_id}] [%(name)s] %(message)s"
    )
    worker = ExplorationWorker(worker_id, tree_manager, completion_queue, args, session_dir)
    worker.run()