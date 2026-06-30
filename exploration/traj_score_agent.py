import os
import json
import logging
import requests
import time
import concurrent.futures
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import threading

from exploration.prompts import POSTPROCESS_SCORING_SYSTEM_PROMPT

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("scoring_agent")


class RateLimiter:
    """简单的速率限制器"""
    def __init__(self, calls_per_second: float = 2):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """如果需要，等待适当的时间"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_call_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_call_time = time.time()


class TrajectoryScoringAgent:
    def __init__(self, model: str, api_url: str, api_key: str, max_tokens: int = 5120):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.rate_limiter = RateLimiter(calls_per_second=2)

    def _call_llm(self, messages: List[Dict]) -> requests.Response:
        """调用LLM API"""
        # 应用速率限制
        self.rate_limiter.wait_if_needed()
        
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
                        "include_thoughts": True,
                        "thinking_budget": 2048
                    },
                    "thought_tag_marker": "think"
                }
            }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        
        try:
            return requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    def _extract_storyline(self, traj_detail: List[Dict]) -> List[str]:
        """
        提取精简的故事线，包含 Goal, Action 类型, 和 Feedback。
        """
        storyline = []
        for i, step in enumerate(traj_detail):
            # 1. 获取 Goal
            goal = step.get('goal', 'No goal')
            
            # 2. 获取 Feedback
            verification = step.get('verification_result', {})
            feedback = verification.get('feedback', 'No feedback')
            
            # 3. 获取 Action 类型 (用于判断 Purity)
            action_list = step.get('action', [])
            act_str = "Unknown"
            if action_list:
                act_data = action_list[0]
                act_type = act_data.get('action', '')
                
                # 简化 Action 描述
                if act_type == 'type':
                    act_str = f"Type '{act_data.get('text', '')}'"
                elif act_type == 'scroll':
                    act_str = "Scroll"
                elif act_type == 'key':
                    act_str = f"Press Key '{act_data.get('text', '')}'"
                elif act_type == 'left_click':
                    # 可选：如果是点击，且有 text 字段（OCR结果），可以带上
                    target_text = act_data.get('text', '')
                    if target_text:
                        act_str = f"Click '{target_text}'"
                    else:
                        act_str = "Click Position"
                else:
                    act_str = act_type
            
            # 组合单行描述
            line = f"Step {i+1}: Intent=[{goal}] -> Action=[{act_str}] -> Feedback=[{feedback}]"
            storyline.append(line)
            
        return storyline

    def score_trajectory(self, json_data: Dict) -> Dict:
        """保持原有接口不变"""
        traj_id = json_data.get('trajectory_id', 'unknown')
        final_task = json_data.get('final_task', {}).get('final_task_summary', 'No summary provided')
        traj_detail = json_data.get('traj_detail', [])

        # 1. 提取故事线
        storyline_list = self._extract_storyline(traj_detail)
        
        # 简单规则：如果步骤过少，返回全0
        if len(storyline_list) < 2:
            return {
                "reason": "Trajectory too short (<2 steps).",
                "semantic_coherence": 0,
                "goal_achievement": 0,
                "step_purity": 0,
                "task_alignment": 0
            }

        story_text = "\n".join(storyline_list)

        # 2. 构造 Prompt
        user_content = f"""
        Target Task: "{final_task}"

        Execution Storyline:
        {story_text}

        Evaluate this trajectory using the 0-3 scale for the 4 metrics.
        """

        messages = [
            {"role": "system", "content": POSTPROCESS_SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        # 3. 调用 LLM (带重试)
        MAX_RETRIES = 5
        for attempt in range(MAX_RETRIES):
            try:
                response = self._call_llm(messages)
                if response.status_code != 200:
                    logger.error(f"API Error {response.status_code}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(1)
                    continue
                
                content = response.json()["choices"][0]["message"]["content"]
                # 清洗可能的 Markdown
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                
                # 计算一个简单的加权总分供参考 (可选)
                logger.info(f"Scored {traj_id}: {result}")
                return result

            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed for {traj_id}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)

        return {"error": "Scoring failed"}


# ==========================================
# 新增：并发处理函数
# ==========================================

def process_single_file(file_path: str, scorer: TrajectoryScoringAgent, force_rescore: bool = False) -> tuple:
    """处理单个文件的辅助函数（用于并发）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果已经打过分且不强制重打，跳过
        if not force_rescore and 'quality_assessment' in data:
            required_fields = ["reason", "task_utility", "step_efficiency", "task_consistency", "action_coherence"]
            missing_fields = [field for field in required_fields if field not in data['quality_assessment']]
            if not missing_fields:
                return file_path, True, "Already scored"
        
        logger.info(f"Processing: {file_path}")
        
        # 执行打分
        assessment = scorer.score_trajectory(data)
        
        # 将结果写回 quality_assessment 字段
        data['quality_assessment'] = assessment
        
        # 覆写文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return file_path, True, "Success"
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        return file_path, False, f"JSON decode error: {e}"
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return file_path, False, f"Error: {e}"


def find_json_files(root_path: str) -> List[str]:
    """查找所有JSON文件"""
    json_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".json") and not file.endswith("_thinking.json"):
                file_path = os.path.join(root, file)
                json_files.append(file_path)
    return json_files


def process_all_trajectories_concurrent(root_path: str, scorer: TrajectoryScoringAgent, 
                                        max_workers: int = 4, force_rescore: bool = False):
    """
    并发版本的处理函数
    
    Args:
        root_path: 根目录路径
        scorer: TrajectoryScoringAgent实例
        max_workers: 最大并发数
        force_rescore: 是否强制重新评分
    """
    if not os.path.exists(root_path):
        logger.error(f"Path does not exist: {root_path}")
        return
    
    # 查找所有文件
    json_files = find_json_files(root_path)
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    if not json_files:
        logger.warning("No JSON files found to process")
        return
    
    successful = 0
    failed = 0
    
    # 使用线程池并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_file, file_path, scorer, force_rescore): file_path
            for file_path in json_files
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result(timeout=180)  # 增加超时时间
                file_path, success, message = result
                if success:
                    successful += 1
                else:
                    failed += 1
                    logger.error(f"Failed: {file_path} - {message}")
            except concurrent.futures.TimeoutError:
                failed += 1
                logger.error(f"Timeout processing file: {file_path}")
            except Exception as e:
                failed += 1
                logger.error(f"Unexpected error processing {file_path}: {e}")
    
    logger.info(f"Finished. Successfully processed: {successful}, Failed: {failed}")


# ==========================================
# 原有的遍历文件夹逻辑（保持兼容）
# ==========================================

def process_all_trajectories(root_path: str, scorer: TrajectoryScoringAgent):
    """
    原有的单线程处理函数（保持向后兼容）
    """
    count = 0
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".json") and not file.endswith("_scored.json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 如果已经打过分，跳过
                    if 'quality_assessment' in data:
                        continue

                    logger.info(f"Processing: {file}")
                    
                    # 执行打分
                    assessment = scorer.score_trajectory(data)
                    
                    # 将结果写回 quality_assessment 字段
                    data['quality_assessment'] = assessment
                    
                    # 覆写文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    count += 1
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Finished. Scored {count} files.")


# ==========================================
# 主函数示例
# ==========================================

if __name__ == "__main__":
    import argparse
    from exploration.config import get_postprocess_config

    parser = argparse.ArgumentParser(description="Run trajectory scoring on exploration results.")
    parser.add_argument("--target_dir", type=str, required=True,
                        help="Path to the trajectories directory to score.")
    parser.add_argument("--api_url", type=str, default=None,
                        help="API base URL (default: $POSTPROCESS_API_URL).")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (default: $POSTPROCESS_API_KEY).")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: $POSTPROCESS_MODEL).")
    parser.add_argument("--max_workers", type=int, default=20,
                        help="Number of concurrent workers.")
    parser.add_argument("--force_rescore", action="store_true",
                        help="Re-score trajectories that already have scores.")
    args = parser.parse_args()

    cfg = get_postprocess_config(api_url=args.api_url, api_key=args.api_key, model=args.model)
    if not cfg.api_url or not cfg.api_key:
        print("Error: API URL and API key are required. Set them via environment variables or CLI args.")
        exit(1)

    if not os.path.exists(args.target_dir):
        print(f"Path does not exist: {args.target_dir}")
        exit(1)

    scorer = TrajectoryScoringAgent(cfg.model, cfg.api_url, cfg.api_key)
    process_all_trajectories_concurrent(
        root_path=args.target_dir,
        scorer=scorer,
        max_workers=args.max_workers,
        force_rescore=args.force_rescore
    )