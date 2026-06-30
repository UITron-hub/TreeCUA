# openrlhf/agent/exploration/tree_manager.py
import os
import json
import uuid
import logging
import filelock
import random
import re  # 新增：用于处理文件名中的特殊字符
from typing import Dict, List, Any, Tuple, Optional

from exploration.node import Node
from exploration.initial_state_generator import InitialStateGenerator

logger = logging.getLogger("desktopenv.tree_manager")

WORLD_KNOWLEDGE_PATH = os.getenv(
    "WORLD_KNOWLEDGE_PATH",
    "data_resource/world_knowledge.json"
)

class ExplorationTreeManager:
    """
    Manages the exploration forest (multiple trees), supports resuming sessions,
    and handles global history for diversity.
    """
    def __init__(self, session_dir: str, domain: str, max_traj_per_tree: int = 1000, 
                 sub_category: str = None, continue_existing_trees: bool = False, 
                 selection_strategy: str = "random", max_depth: int = 20):        
        """
        Args:
            session_dir: 会话目录
            domain: 领域名称 (e.g. libreoffice_calc)
            max_traj_per_tree: 每棵树最大轨迹数
            sub_category: [新功能] 如果指定，该Manager生成的所有树都将固定属于该子类。
                                如果为None，则根据策略自动选择（均衡模式或无限制模式）。
        """
        self.session_dir = session_dir
        self.domain = domain
        self.max_traj_per_tree = max_traj_per_tree
        self.sub_category = sub_category  # Mode C: Fixed Category
        self.continue_existing_trees = continue_existing_trees
        self.selection_strategy = selection_strategy
        self.max_depth = max_depth
        
        self.nodes_dir = os.path.join(session_dir, "nodes")
        self.screenshots_dir = os.path.join(session_dir, "screenshots")
        self.manifest_path = os.path.join(session_dir, "tree_manifest.json")
        # Global history path 不再是单一固定的，而是动态生成的，这里只保留目录概念
        self.history_dir = os.path.join(session_dir, "global_histories")
        
        self.lock = filelock.FileLock(self.manifest_path + ".lock", timeout=60)
        self.state_generator = InitialStateGenerator()

        # --- Load World Knowledge ---
        self.world_knowledge = {}
        if os.path.exists(WORLD_KNOWLEDGE_PATH):
            try:
                with open(WORLD_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
                    self.world_knowledge = json.load(f)
                logger.info(f"Loaded world knowledge from {WORLD_KNOWLEDGE_PATH}")
            except Exception as e:
                logger.error(f"Failed to load world knowledge: {e}")
        else:
            logger.warning(f"World knowledge file not found at {WORLD_KNOWLEDGE_PATH}")
        # ----------------------------
        # if sub_task_info is None and self.tree_manager.sub_category is not None:
        # sub_task_info = self.world_knowledge[self.domain].get(sub_task_category)
        if self.domain not in self.world_knowledge:
            logger.error(f"Domain '{self.domain}' not found in world knowledge. Available domains: {list(self.world_knowledge.keys())}")
            raise ValueError(f"Domain '{self.domain}' not found in world knowledge.")
        
        domain_knowledge = self.world_knowledge[self.domain]
        
        # --- 如果指定了子类，检查子类是否合法 ---
        if self.sub_category is not None:
            if not isinstance(domain_knowledge, dict):
                logger.error(f"Domain '{self.domain}' knowledge structure is invalid or missing 'sub_categories' field.")
                raise ValueError(f"Domain '{self.domain}' knowledge structure is invalid.")
            
            if self.sub_category not in domain_knowledge and self.sub_category != "general":
                available_categories = list(domain_knowledge.keys())
                logger.error(f"Sub category '{self.sub_category}' not found in domain '{self.domain}'. Available sub-categories: {available_categories}") 
                raise ValueError(f"Sub category '{self.sub_category}' not found in domain '{self.domain}'.")
            
        logger.info(f"Using fixed sub-category: {self.sub_category}")
            

            

        os.makedirs(self.nodes_dir, exist_ok=True)
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True) # 确保存放历史的目录存在
        os.makedirs(os.path.join(self.session_dir, "trajectories"), exist_ok=True)
        
        self._initialize_or_resume_session()

    def _initialize_or_resume_session(self):
        with self.lock:
            # 1. 初始化或加载 Manifest
            if not os.path.exists(self.manifest_path):
                logger.info(f"Initializing new session at {self.manifest_path}")
                manifest = {
                    "session_id": os.path.basename(self.session_dir),
                    "current_tree_index": 0,    
                    "current_root_id": None,    
                    "tree_traj_counts": {},     
                    "category_counts": {}, # 记录各类任务已完成的轨迹总数
                    "nodes": {},                
                    "metadata": {"total_nodes": 0},
                    "config_mode": "fixed" if self.sub_category else "balanced/auto"
                }
                # 立即开启第一棵树
                self._start_new_tree(manifest)
            else:
                logger.info(f"Resuming existing exploration session from {self.manifest_path}")
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    
                    reset = False
                    if "category_counts" not in manifest:
                        manifest["category_counts"] = {}
                    
                    if self.continue_existing_trees:
                        logger.info("Searching for existing trees to resume...")
                        found_root = self._find_available_existing_tree(manifest)
                        if found_root:
                            manifest["current_root_id"] = found_root
                            logger.info(f"Resuming existing tree: {found_root}")
                            next_root = found_root
                        reset = True

                    # Crash Recovery: 重置意外中断的 EXPLORING 节点
                    for node_id, data in manifest['nodes'].items():
                        if data.get('status') == 'EXPLORING':
                            data['status'] = 'UNEXPLORED'
                            reset = True
                    if reset:
                        logger.info(f"Reset 'EXPLORING' nodes to 'UNEXPLORED'.")
                        self._save_manifest(manifest)

    def _save_manifest(self, manifest):
        """Helper to save manifest safely inside a lock."""
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def _get_history_file_path(self, category: Optional[str]) -> str:
        """
        Helper: 根据 category 获取对应的 history 文件路径。
        将特殊字符替换为下划线，以保证文件名合法。
        """
        if not category:
            filename = "global_history_generic.json"
        else:
            # 替换空格、&、/ 等非字母数字字符为下划线
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', category)
            # 避免文件名过长
            safe_name = safe_name[:50]
            filename = f"global_history_{safe_name}.json"
        
        return os.path.join(self.history_dir, filename)

    def _start_new_tree(self, manifest: Dict):
        """
        Internal: Starts a new tree with a fresh configuration.
        Must be called while holding self.lock.
        """
        tree_idx = manifest.get("current_tree_index", 0) + 1
        manifest["current_tree_index"] = tree_idx
        
        selected_category = None
        domain_tasks = self.world_knowledge.get(self.domain, {})

        # --- 策略选择逻辑 ---
        
        # 1. Mode C: 外部指定 (Fixed)
        if self.sub_category and self.sub_category != "general":
            selected_category = self.sub_category
            logger.info(f"Tree #{tree_idx} using FIXED sub-category: '{selected_category}'")
        
        #  2. Mode A: 无子类 (Generic)
        elif self.sub_category == "general":
            selected_category = "general"
            logger.info(f"No specific sub-tasks or constraints. Tree #{tree_idx} using GENERIC generation.")

        # 3. Mode B: 自动均衡 (Balanced)
        elif domain_tasks:
            if "category_counts" not in manifest:
                manifest["category_counts"] = {}
            
            # 确保计数器初始化
            all_categories = list(domain_tasks.keys())
            for cat in all_categories:
                if cat not in manifest["category_counts"]:
                    manifest["category_counts"][cat] = 0
            
            current_counts = manifest["category_counts"]
            
            # 找出最小完成数
            relevant_counts = [(cat, current_counts.get(cat, 0)) for cat in all_categories]
            min_count = min(count for _, count in relevant_counts)
            
            # 筛选候选并随机选择
            candidates = [cat for cat, count in relevant_counts if count == min_count]
            selected_category = random.choice(candidates)
            
            logger.info(f"Tree #{tree_idx} auto-assigned category: '{selected_category}' (Balanced Strategy, Completed: {min_count})")
        

        # --- 生成配置 (传入 sub_category) ---
        logger.info(f"Generating new Initial Config for Tree #{tree_idx} (Domain: {self.domain}, Category: {selected_category})...")
        
        # [修改点] 将 selected_category 传递给 state_generator
        new_config = self.state_generator.generate_task(self.domain, sub_category=selected_category)
        
        config_filename = f"config_tree_{tree_idx}_{uuid.uuid4().hex[:6]}.json"
        config_path = os.path.join(self.session_dir, config_filename)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False)

        # -------------------------------------------------------
            
        # 4. 创建新 Root Node
        root_id = f"root_{tree_idx:03d}_{uuid.uuid4().hex[:8]}"
        manifest["current_root_id"] = root_id
        manifest["tree_traj_counts"][root_id] = 0
        
        manifest["nodes"][root_id] = {
            "parent_id": None, 
            "depth": 0, 
            "status": "UNEXPLORED",
            "tree_root_id": root_id,
            "config_path": config_path,
            "sub_task_category": selected_category # 记录该树归属的 Category
        }
        manifest['metadata']['total_nodes'] += 1
        
        root_node = Node(node_id=root_id, parent_node_id=None, depth=0, status="UNEXPLORED")
        root_node.save(self.nodes_dir)
        
        logger.info(f"Started NEW TREE #{tree_idx}. Root: {root_id}. Config: {config_filename}")
        self._save_manifest(manifest)

    def get_full_node_path(self, target_node_id: str) -> List[str]:
        with open(self.manifest_path, 'r', encoding='utf-8') as f: 
            manifest = json.load(f)
        path_ids = []
        current_id = target_node_id
        while current_id:
            path_ids.append(current_id)
            node_entry = manifest['nodes'].get(current_id)
            if not node_entry: break
            current_id = node_entry.get('parent_id')
        path_ids.reverse()
        return path_ids

    def _select_node_by_priority(self, unexplored_nodes: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """
        根据策略选择节点。
        unexplored_nodes: List of (node_id, node_data)
        """
        if not unexplored_nodes:
            return []

        # 1. 纯随机 (Random)
        if self.selection_strategy == "random":
            random.shuffle(unexplored_nodes)
            return unexplored_nodes

        # 2. 深度优先 (DFS) - 优先选深度大的
        elif self.selection_strategy == "dfs":
            # depth越大越靠前
            return sorted(unexplored_nodes, key=lambda x: x[1].get('depth', 0), reverse=True)

        # 3. 浅层优先 (BFS) - 优先选深度小的
        elif self.selection_strategy == "bfs":
            # depth越小越靠前
            return sorted(unexplored_nodes, key=lambda x: x[1].get('depth', 0), reverse=False)

        # 4. 混合方案 (Hybrid)
        # 优先选择: (深度 <= 2) 且 (最后一步深度 < 最大长度 - 3)
        # 这里的“最后一步深度”即当前节点的深度，因为从它出发就是下一步。
        elif self.selection_strategy == "hybrid":
            priority_candidates = []
            other_candidates = []

            for nid, data in unexplored_nodes:
                depth = data.get('depth', 0)
                # 混合策略条件：浅层探索(<=3) 且 剩余步数充足(避免死胡同)
                if depth <= 2 and depth < (self.max_depth - 3):
                    priority_candidates.append((nid, data))
                else:
                    other_candidates.append((nid, data))
            
            # 组内随机
            random.shuffle(priority_candidates)
            random.shuffle(other_candidates)
            
            return priority_candidates + other_candidates

        # 默认回退到随机
        random.shuffle(unexplored_nodes)
        return unexplored_nodes

    def _find_available_existing_tree(self, manifest: Dict) -> Optional[str]:
        """
        在 continue_existing_trees 模式下，寻找一个既没满又有待探索节点的旧树。
        返回 root_id 或 None。
        """
        # 获取所有树的 root_id (通过遍历 nodes 里的 tree_root_id 集合，或者 metadata，这里假设可以通过 tree_traj_counts 的 keys 获取所有已知的 root)
        all_roots = list(manifest.get("tree_traj_counts", {}).keys())

        for root_id in all_roots:
            # 1. 检查是否已满
            count = manifest["tree_traj_counts"].get(root_id, 0)
            if count >= self.max_traj_per_tree:
                continue

            # 2. 检查是否有 UNEXPLORED 节点
            has_unexplored = False
            for nid, data in manifest['nodes'].items():
                if data.get('tree_root_id') == root_id and data.get('status') == 'UNEXPLORED':
                    has_unexplored = True
                    break
            
            if has_unexplored:
                return root_id
        
        return None

    def get_task(self) -> Optional[Tuple[str, List[str], Dict, Optional[Dict]]]:
        """
        Gets a task. 
        Returns: (target_node_id, full_node_path, initial_config, sub_task_info)
        """
        with self.lock:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            current_root = manifest.get("current_root_id")
            
            # Helper: 获取某棵树的可用节点
            def get_tree_nodes(root_id):
                unexplored = []
                active = []
                for nid, data in manifest['nodes'].items():
                    if data.get('tree_root_id') == root_id:
                        if data['status'] == 'UNEXPLORED':
                            unexplored.append((nid, data))
                        elif data['status'] == 'EXPLORING':
                            active.append(nid)
                return unexplored, active

            # --- 状态检查与树切换逻辑 ---
            need_switch = False
            
            # 1. 如果还没有树，或者当前树满了
            if not current_root:
                need_switch = True
            else:
                current_count = manifest["tree_traj_counts"].get(current_root, 0)
                if current_count >= self.max_traj_per_tree:
                    logger.info(f"Tree {current_root} reached limit ({current_count}/{self.max_traj_per_tree}).")
                    need_switch = True
                else:
                    # 检查当前树是否死掉 (无 UNEXPLORED 且 无 ACTIVE)
                    unexplored, active = get_tree_nodes(current_root)
                    if not unexplored and not active:
                        logger.info(f"Tree {current_root} is EXHAUSTED/DEAD.")
                        need_switch = True
            
            # --- 执行切换 (如果需要) ---
            if need_switch:
                next_root = None
                
                # 策略 A: 尝试在旧树上继续 (Continue Existing)
                if self.continue_existing_trees:
                    logger.info("Searching for existing trees to resume...")
                    found_root = self._find_available_existing_tree(manifest)
                    if found_root:
                        logger.info(f"Resuming existing tree: {found_root}")
                        next_root = found_root
                    else:
                        logger.info("No available existing trees found.")

                # 策略 B: 如果没找到旧树，或者没开启该模式，则开新树
                if not next_root:
                    self._start_new_tree(manifest)
                    next_root = manifest["current_root_id"]
                
                # 更新当前 Root
                manifest["current_root_id"] = next_root
                current_root = next_root
                # 这里不需要 save_manifest，因为下面选节点时会一起存，或者 start_new_tree 已经存了
                # 但为了保险起见，如果是切换回旧树，状态发生了变更，需要保存 current_root_id
                if self.continue_existing_trees and next_root != manifest.get("current_root_id"):
                     self._save_manifest(manifest)

            # --- 再次获取当前树的节点 (因为可能切换了树) ---
            current_tree_unexplored, current_tree_active = get_tree_nodes(current_root)

            # 双重检查：如果新开的树或者切过来的树还是空的（理论上start_new_tree会创建root节点，不会空）
            if not current_tree_unexplored:
                # 唯一的边缘情况：正在跑的 Worker 把这棵“复活”的树的最后节点做完了，或者刚好都在 EXPLORING
                if current_tree_active:
                    return None # 等待其他 Worker
                else:
                    # 极度异常：刚切过来就死掉了？递归再找一次或者报错，这里简单起见再次强制开新树
                    logger.warning(f"Unexpected state: Tree {current_root} has no nodes immediately after selection. Forcing new tree.")
                    self._start_new_tree(manifest)
                    current_root = manifest["current_root_id"]
                    current_tree_unexplored, _ = get_tree_nodes(current_root)

            # --- 任务选择 (Select Task) ---
            sorted_nodes = self._select_node_by_priority(current_tree_unexplored)
            
            # 如果排序后为空（防御性编程）
            if not sorted_nodes:
                return None

            target_node_id, target_data = sorted_nodes[0]
            
            manifest['nodes'][target_node_id]['status'] = 'EXPLORING'
            
            # 确保 current_root_id 持久化正确
            manifest["current_root_id"] = current_root
            self._save_manifest(manifest)

            # --- 后续加载 Config 逻辑保持不变 ---
            root_id = target_data.get('tree_root_id')
            root_entry = manifest['nodes'].get(root_id)
            config_path = root_entry.get('config_path')
            



            with open(config_path, 'r', encoding='utf-8') as f:
                initial_config = json.load(f)

            # --- 6. 获取子任务信息 ---
            sub_task_category = root_entry.get('sub_task_category')
            sub_task_info = None
            if sub_task_category and self.domain in self.world_knowledge:
                sub_task_info = self.world_knowledge[self.domain].get(sub_task_category)
                if sub_task_info:
                    sub_task_info['category_name'] = sub_task_category
            elif sub_task_category == 'general':
                sub_task_info["category_name"] = "general"
                sub_task_info['description'] = f"You need to explore various possible tasks within the app {self.domain}"
                #self.world_knowledge[self.domain] all prompts
                sub_task_info['prompts'] = []
                # [category['prompts'] for category in self.world_knowledge[self.domain]]
                for category, details in self.world_knowledge[self.domain].items():
                    sub_task_info['prompts'].extend(details['prompts'])

            logger.info(f"Assigning task: {target_node_id} (Tree: {root_id}, Category: {sub_task_category})")
            # sub_task_info.get("description", "")
            logger.info(f"Description: {sub_task_info.get('description', 'N/A') if sub_task_info else 'N/A'}")
            full_node_path = self.get_full_node_path(target_node_id)
            
            return target_node_id, full_node_path, initial_config, sub_task_info

    def report_new_candidates(self, parent_node: Node, candidates: List[Dict]) -> List[Node]:
        new_nodes = []
        with self.lock:
            with open(self.manifest_path, 'r', encoding='utf-8') as f_manifest:
                manifest = json.load(f_manifest)
                
                parent_entry = manifest['nodes'].get(parent_node.node_id)
                if not parent_entry:
                    tree_root_id = manifest["current_root_id"]
                else:
                    tree_root_id = parent_entry.get('tree_root_id')

                for cand in candidates:
                    child_depth = parent_node.depth + 1
                    child_id = f"node_{child_depth:02d}_{uuid.uuid4().hex[:8]}"
                    
                    manifest['nodes'][child_id] = {
                        "parent_id": parent_node.node_id, 
                        "depth": child_depth, 
                        "status": "UNEXPLORED",
                        "tree_root_id": tree_root_id 
                    }
                    manifest['metadata']['total_nodes'] += 1

                    child_node = Node(
                        node_id=child_id, parent_node_id=parent_node.node_id,
                        depth=child_depth, status="UNEXPLORED",
                        action_command=cand.get("action_command"),
                        step_goal=cand.get("step_goal"),
                        final_goal=cand.get("final_goal"),
                        step_reason=cand.get("step_reason"),
                        expected_observation=cand.get("expected_observation"),
                        step_action=cand.get("step_action"),
                    )
                    child_node.save(self.nodes_dir)
                    new_nodes.append(child_node)
                
                self._save_manifest(manifest)

            parent_node.next_nodes_candidates = [
                {"step_goal": cand.get("step_goal"), "child_node_id": node.node_id}
                for cand, node in zip(candidates, new_nodes)
            ]
            parent_node.save(self.nodes_dir)

        logger.info(f"Added {len(candidates)} new candidates from parent {parent_node.node_id}")
        return new_nodes

    def update_node(self, node: Node):
        with self.lock:
            with open(self.manifest_path, 'r+', encoding='utf-8') as f:
                manifest = json.load(f)
                if node.node_id in manifest['nodes']:
                    manifest['nodes'][node.node_id]['status'] = node.status
                    f.seek(0)
                    json.dump(manifest, f, indent=2, ensure_ascii=False)
                    f.truncate()
            node.save(self.nodes_dir)
            logger.info(f"Node {node.node_id} updated with status {node.status}.")

    def report_trajectory_completion(self, root_id: str):
        """
        Reports that a trajectory starting from root_id has finished.
        Updates counts for both the tree and the task category.
        """
        with self.lock:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            # 1. 更新该树的轨迹计数
            if root_id in manifest["tree_traj_counts"]:
                manifest["tree_traj_counts"][root_id] += 1
            else:
                manifest["tree_traj_counts"][root_id] = 1
            
            # 2. 更新该任务类别的轨迹计数
            root_node_entry = manifest['nodes'].get(root_id)
            if root_node_entry:
                category = root_node_entry.get('sub_task_category')
                if category:
                    if "category_counts" not in manifest:
                        manifest["category_counts"] = {}
                    
                    if category not in manifest["category_counts"]:
                        manifest["category_counts"][category] = 0
                    
                    manifest["category_counts"][category] += 1
                    logger.info(f"Trajectory completed for category '{category}'. Total: {manifest['category_counts'][category]}")

            self._save_manifest(manifest)

    def get_global_prefix_history(self, category: Optional[str] = None) -> List[str]:
        """
        [修改] 根据 category 读取对应的 global history 文件。
        """
        file_path = self._get_history_file_path(category)
        if os.path.exists(file_path):
            try:
                with self.lock:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                logger.error(f"Error reading global history from {file_path}: {e}")
                return []
        return []

    def record_prefix_history(self, step1_goal: str, step2_goal: str, category: Optional[str] = None):
        """
        [修改] 根据 category 将 history 写入对应的文件。
        """
        combo = f"Step 1 Goal: {step1_goal} -> Step 2 Goal: {step2_goal}"
        file_path = self._get_history_file_path(category)
        
        with self.lock:
            history = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                except:
                    history = []
            
            if combo not in history:
                history.append(combo)
                if len(history) > 500:
                    history = history[-500:]
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                logger.info(f"Recorded new global prefix for category '{category}' in {os.path.basename(file_path)}: {combo}")