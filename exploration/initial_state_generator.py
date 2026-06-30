# openrlhf/cli/initial_state_generator.py
import random
import json
import logging
import uuid
import os
from typing import Optional
import zipfile

# --- Logger setup for this module ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [StateGenerator] %(message)s")
logger = logging.getLogger(__name__)

class InitialStateGenerator:
    """
    A class to generate random initial environment configurations ('configs')
    for different application domains, supporting specific sub-category initialization.
    """
    def __init__(self, resource_dir: str = None):
        self.base_resource_dir = resource_dir or os.getenv(
            "RESOURCE_DIR",
            "data_resource/useful"
        )
        
        # --- 1. LibreOffice Impress (PowerPoint) ---
        self.impress_asset_dir = os.path.join(self.base_resource_dir, "ppts")
        self.impress_files = self._load_files(self.impress_asset_dir, ['.pptx', '.odp'])

        # --- 2. LibreOffice Writer (Word) ---
        self.writer_asset_dir = os.path.join(self.base_resource_dir, "docs") 
        self.writer_files = self._load_files(self.writer_asset_dir, ['.docx', '.doc', '.odt'])

        # --- 3. LibreOffice Calc (Excel) ---
        self.calc_asset_dir = os.path.join(self.base_resource_dir, "excels")
        self.calc_files = self._load_files(self.calc_asset_dir, ['.xlsx', '.xls', '.ods', '.csv'])

        # --- 4. VLC (Media) ---
        self.media_asset_dir = os.path.join(self.base_resource_dir, "videos")
        self.media_files = self._load_files(self.media_asset_dir, ['.mp4', '.mkv', '.avi'])

        # --- 5. VS Code (Code Projects) ---
        self.code_asset_dir = os.path.join(self.base_resource_dir, "codes/projects")
        self.code_files = self._load_files(self.code_asset_dir, ['.py', '.js', '.json', '.cpp', '.txt'])
        self.zip_asset_dir = os.path.join(self.base_resource_dir, "code_project_zips")
        self.vscode_projects = {
            "web": self._load_files(self.zip_asset_dir, prefix="web_"),     # 前端项目
            "algo": self._load_files(self.zip_asset_dir, prefix="algo_") + \
                    self._load_files(self.zip_asset_dir, prefix="py_algo_"), # 算法/Python项目
            "data": self._load_files(self.zip_asset_dir, prefix="data_"),   # 数据/ML项目
            "doc": self._load_files(self.zip_asset_dir, prefix="doc_"),     # 文档/Markdown项目
            "latex": self._load_files(self.zip_asset_dir, prefix="latex_")  # LaTeX项目
        }
        

        # --- 6. GIMP (Images) ---
        self.xcf_asset_dir = os.path.join(self.base_resource_dir, "xcfs")
        self.xcf_files = self._load_files(self.xcf_asset_dir, ['.xcf'])
        self.image_asset_dir = os.path.join(self.base_resource_dir, "images")
        self.image_files = self._load_files(self.image_asset_dir, ['.jpg', '.jpeg', '.png'])

        # --- 7. Thunderbird (Email Profiles) ---
        self.tb_asset_dir = os.path.join(self.base_resource_dir, "thunderbird_files")
        # self.profile_rich_name = "thunderbird-profile.tar.gz"
        # self.profile_blank_name = "thunderbird-profile-blank.tar.gz"
        self.profile_rich_file = os.path.join(self.tb_asset_dir, "thunderbird-profile.tar.gz")
        self.profile_blank_file = os.path.join(self.tb_asset_dir, "thunderbird-profile-blank.tar.gz")

        # --- Chrome URLs (verified reachable, 2026-04) ---
        self.chrome_urls = {
            "shopping_and_commerce": [
                "https://www.ebay.com", "https://www.bestbuy.com",
                "https://www.target.com", "https://www.walmart.com",
            ],
            "travel_and_transport": [
                "https://www.expedia.com", "https://www.booking.com",
                "https://www.tripadvisor.com", "https://www.kayak.com",
            ],
            "information_and_reference": [
                "https://www.wikipedia.org", "https://www.imdb.com",
                "https://www.accuweather.com", "https://www.yelp.com",
                "https://www.dictionary.com",
            ],
            "health_gov_and_law": [
                "https://www.drugs.com", "https://www.webmd.com",
                "https://www.cdc.gov", "https://www.usa.gov",
                "https://www.medlineplus.gov", "https://www.recreation.gov",
            ],
            "news_and_finance": [
                "https://www.cnn.com", "https://www.bbc.com",
                "https://www.bloomberg.com", "https://www.forbes.com",
            ],
            "tools_and_education": [
                "https://www.google.com/maps", "https://translate.google.com",
                "https://www.coursera.org", "https://www.khanacademy.org",
                "https://github.com",
            ],
        }

        
        # --- Chrome History Pool ---
        self.chrome_history_pool = [
            {"url": "https://stackoverflow.com/questions/4114095/how-to-revert-a-git-repository-to-a-previous-commit", "title": "How to revert a Git repository - Stack Overflow"},
            {"url": "https://docs.python.org/3/library/asyncio.html", "title": "asyncio — Asynchronous I/O — Python 3 documentation"},
            {"url": "https://github.com/facebookresearch/llama", "title": "GitHub - facebookresearch/llama"},
            {"url": "https://huggingface.co/docs/transformers/index", "title": "Welcome to Transformers - Hugging Face"},
            {"url": "https://www.allrecipes.com/recipe/23600/worlds-best-lasagna/", "title": "World's Best Lasagna Recipe"},
            {"url": "https://www.instructables.com/How-to-Build-a-Bookshelf/", "title": "How to Build a Bookshelf (with Pictures)"},
            {"url": "https://letterboxd.com/film/parasite-2019/", "title": "Parasite (2019) - Letterboxd"},
            {"url": "https://www.thingiverse.com/thing:340333", "title": "Articulated Butterfly by 8ran"},
            {"url": "https://www.etsy.com/search?q=custom+leather+journal", "title": "Custom Leather Journal - Etsy"},
            {"url": "https://www.rei.com/product/893894/rei-co-op-half-dome-2-plus-tent", "title": "REI Co-op Half Dome 2 Plus Tent"},
            {"url": "https://www.goodreads.com/book/show/136251.The_Name_of_the_Wind", "title": "The Name of the Wind by Patrick Rothfuss"},
            {"url": "https://en.wikipedia.org/wiki/History_of_the_Internet", "title": "History of the Internet - Wikipedia"},
            {"url": "https://arstechnica.com/science/2023/10/the-curious-case-of-the-exploding-ants/", "title": "The curious case of the exploding ants - Ars Technica"},
        ]

    def _load_files(self, directory: str, extensions: list = None, prefix: str = None) -> list:
        """Helper to load files with specific extensions OR prefix."""
        files = []
        if os.path.isdir(directory):
            files = [f for f in os.listdir(directory)]
            if extensions:
                files = [f for f in files if any(f.lower().endswith(ext) for ext in extensions)]
            if prefix:
                files = [f for f in files if f.startswith(prefix)]
        return files
    
    def _get_base_task_template(self, domain: str, sub_category: str = "general") -> dict:
        """ 
        Returns a base dictionary with all the required top-level fields for a task.
        """
        return {
            "id": str(uuid.uuid4()),
            "snapshot": domain, # e.g., "chrome"
            "sub_category": sub_category if sub_category else "general",
            "instruction": f"The environment is pre-configured. Your task is to explore {domain}, infer a reasonable user goal, and carry it out.",
            "source": "synthetic_exploration_starter",
            "config": [], 
            "trajectory": "trajectories/", 
            "related_apps": [domain],
            "evaluator": {
                "func": "infeasible" 
            }
        }

    def _get_base_chrome_config(self) -> list:
        """Returns the essential config to just launch Chrome."""
        return [
            {
                "type": "launch",
                "parameters": {
                    "command": ["google-chrome", "--remote-debugging-port=1337"]
                }
            },
            {
                "type": "launch",
                "parameters": {
                    "command": ["socat", "tcp-listen:9222,fork", "tcp:localhost:1337"]
                }
            }
        ]

    def generate_task(self, domain: str, sub_category: Optional[str] = "general") -> dict:
        """
        Main method to generate a complete task JSON for a given domain.
        
        Args:
            domain: The main application domain (e.g., 'libreoffice_calc').
            sub_category: (Optional) The specific sub-task category (e.g., 'Data Calculation & Statistics').
                          This allows for more targeted initialization.
        """
        if "libreoffice_impress" in domain:
            return self._generate_libreoffice_impress_task(sub_category)
        elif "libreoffice_writer" in domain:
            return self._generate_libreoffice_writer_task(sub_category)
        elif "libreoffice_calc" in domain:
            return self._generate_libreoffice_calc_task(sub_category)
        elif "chrome" in domain:
            return self._generate_chrome_task(sub_category)
        elif "vlc" in domain:
            return self._generate_vlc_task(sub_category)
        elif "vs_code" in domain:
            return self._generate_vs_code_task(sub_category)
        elif "thunderbird" in domain:
            return self._generate_thunderbird_task(sub_category)
        elif "gimp" in domain:
            return self._generate_gimp_task(sub_category)
        elif "os" in domain:
            return self._generate_os_task(sub_category)
        else:
            logger.warning(f"No specific task generator for domain '{domain}'. Returning a blank task.")
            return self._get_base_task_template(domain)

    # =========================================================================
    # Domain-Specific Generators
    # =========================================================================

    def _generate_libreoffice_impress_task(self, sub_category: Optional[str] = None) -> dict:
        """Generates an initial task for LibreOffice Impress (PowerPoint)."""
        domain_name = "libreoffice_impress"
        task = self._get_base_task_template(domain_name)
        
        # --- [INTERFACE] Sub-Category Specific Initialization ---
        if sub_category:
            # TODO: Add logic here to return specific configs based on sub_category
            # Example:
            # if sub_category == "Slide Layout Design":
            #     return self._setup_impress_layout_task(task)
            pass 
        # --------------------------------------------------------

        if not self.impress_files:
            logger.error("No PowerPoint files available. Launching blank Impress.")
            task["config"] = [{"type": "launch", "parameters": {"command": ["libreoffice", "--impress"]}}]
            task["instruction"] += " You start with a blank presentation."
            return task

        chosen_filename = random.choice(self.impress_files)
        absolute_local_path = os.path.join(self.impress_asset_dir, chosen_filename)
        vm_path = f"/home/user/Desktop/{chosen_filename}"
        
        logger.info(f"Generating '{domain_name}' state with '{chosen_filename}'.")

        config = [
            {"type": "upload_file", "parameters": {"files": [{"local_path": absolute_local_path, "path": vm_path}]}},
            {"type": "open", "parameters": {"path": vm_path}}
        ]
        
        task["config"] = config
        task["instruction"] += f" You start with the presentation '{chosen_filename}' already open."
        return task

    def _generate_libreoffice_writer_task(self, sub_category: Optional[str] = None) -> dict:
        """Generates an initial task for LibreOffice Writer (Word)."""
        domain_name = "libreoffice_writer"
        task = self._get_base_task_template(domain_name)

        # --- [INTERFACE] Sub-Category Specific Initialization ---
        if sub_category:
            # TODO: Add logic here to return specific configs based on sub_category
            pass 
        # --------------------------------------------------------

        if not self.writer_files:
            logger.error("No Word files available. Launching blank Writer.")
            task["config"] = [{"type": "launch", "parameters": {"command": ["libreoffice", "--writer"]}}]
            task["instruction"] += " You start with a blank document."
            return task

        chosen_filename = random.choice(self.writer_files)
        absolute_local_path = os.path.join(self.writer_asset_dir, chosen_filename)
        vm_path = f"/home/user/Desktop/{chosen_filename}"
        
        logger.info(f"Generating '{domain_name}' state with '{chosen_filename}'.")

        config = [
            {"type": "upload_file", "parameters": {"files": [{"local_path": absolute_local_path, "path": vm_path}]}},
            {"type": "open", "parameters": {"path": vm_path}}
        ]
        
        task["config"] = config
        task["instruction"] += f" You start with the document '{chosen_filename}' already open."
        return task

    def _generate_libreoffice_calc_task(self, sub_category: Optional[str] = None) -> dict:
        """Generates an initial task for LibreOffice Calc (Excel)."""
        domain_name = "libreoffice_calc"
        task = self._get_base_task_template(domain_name)

        # --- [INTERFACE] Sub-Category Specific Initialization ---
        if sub_category:
            logger.info(f"Attempting specific initialization for Calc sub-category: {sub_category}")
            # TODO: Implement specific logic.
            # E.g. If "Data Cleaning", maybe open a specific CSV with messy data.
            # if sub_category == "Data Calculation & Statistics":
            #     # return specific_config
            #     pass
            pass
        # --------------------------------------------------------

        if not self.calc_files:
            logger.error("No Excel files available. Launching blank Calc.")
            task["config"] = [{"type": "launch", "parameters": {"command": ["libreoffice", "--calc"]}}]
            task["instruction"] += " You start with a blank spreadsheet."
            return task

        chosen_filename = random.choice(self.calc_files)
        absolute_local_path = os.path.join(self.calc_asset_dir, chosen_filename)
        vm_path = f"/home/user/Desktop/{chosen_filename}"
        
        logger.info(f"Generating '{domain_name}' state with '{chosen_filename}'.")

        config = [
            {"type": "upload_file", "parameters": {"files": [{"local_path": absolute_local_path, "path": vm_path}]}},
            {"type": "open", "parameters": {"path": vm_path}}
        ]
        
        task["config"] = config
        task["instruction"] += f" You start with the spreadsheet '{chosen_filename}' already open."
        return task

    def _generate_chrome_task(self, sub_category: Optional[str] = None) -> dict:
        """
        Generates a context-aware initial state for Chrome based on specific sub-categories.
        The goal is to create a 'generalized' environment suitable for any task within that category.
        """
        domain_name = "chrome"
        task = self._get_base_task_template(domain_name, sub_category)
        
        # 1. 基础启动配置 (Base Config)
        # 始终包含 Chrome 调试端口和 socat 转发，这是所有 Chrome 任务的基础
        config = self._get_base_chrome_config()
        
        # 2. 辅助资源池 (Helper Pools)
        all_urls = [url for category in self.chrome_urls.values() for url in category]
        
        # 默认指令后缀
        instruction_suffix = ""

        # =====================================================================
        # Category 1: Browser Configuration & Personalization
        # 场景：用户刚打开浏览器，想要修改设置（语言、主页、搜索引擎、主题等）。
        # 策略：保持简单。打开一个通用页面（如 Google），模拟日常状态。
        #       偶尔模拟“主页被篡改”的场景（参考 reference id: 3299584d...）
        # =====================================================================
        if sub_category == "Browser Configuration & Personalization":
            # 20% 概率：模拟主页被设置为奇怪的网站 (Troubleshooting 场景)
            if random.random() < 0.2:
                weird_site = "http://www.example.com"
                # 使用 jq 修改 Chrome Preferences (模拟 reference 中的逻辑)
                # 注意：这里简化处理，直接打开这个网页模拟“一启动就是这个”
                config.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": [weird_site]}})
                instruction_suffix = " Chrome has started, but the startup page seems incorrect."
            else:
                # 80% 概率：标准启动，打开 Google 或 New Tab
                start_url = "https://www.google.com"
                config.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": [start_url]}})
                instruction_suffix = " Chrome is open. You can access settings to personalize the browser."

        # =====================================================================
        # Category 2: Privacy, Security & Data Management
        # 场景：用户需要清理历史、Cookies 或管理隐私设置。
        # 关键：必须有一个"丰富"且"特定"的历史记录池，以便 Agent 有东西可删。
        # =====================================================================
        elif sub_category == "Privacy, Security & Data Management":
            # 构建一个包含高频网站（YouTube, Amazon）的混合历史记录
            # 这样无论是"删除 YouTube 记录"还是"删除 Amazon 记录"的任务都能满足
            
            rich_history = []
            current_time_offset = 60 # start from 1 min ago
            
            # 1. 注入 YouTube 记录 (模拟大量观看历史)
            youtube_titles = ["Music Video", "News Clip", "Tutorial", "Vlog", "Review"]
            for i in range(15): # 15 entries
                rich_history.append({
                    "url": f"https://www.youtube.com/watch?v=video_id_{i}",
                    "title": f"YouTube - {random.choice(youtube_titles)} {i}",
                    "visit_time_from_now_in_seconds": current_time_offset
                })
                current_time_offset += random.randint(300, 3600)

            # 2. 注入 Shopping 记录 (模拟 Amazon/Etsy)
            shopping_items = ["Laptop", "Shoes", "Book", "Coffee Maker"]
            for item in shopping_items:
                rich_history.append({
                    "url": f"https://www.amazon.com/s?k={item}",
                    "title": f"Amazon.com : {item}",
                    "visit_time_from_now_in_seconds": current_time_offset
                })
                current_time_offset += random.randint(600, 7200)

            # 3. 注入普通浏览记录
            random_pool = random.sample(self.chrome_history_pool, min(5, len(self.chrome_history_pool)))
            for item in random_pool:
                item_copy = item.copy()
                item_copy["visit_time_from_now_in_seconds"] = current_time_offset
                rich_history.append(item_copy)
                current_time_offset += random.randint(1000, 5000)

            # 动作：注入历史
            # Insert at index 0 (or before launch) typically, but config list execution order matters.
            # Reference usually puts `update_browse_history` first or standalone.
            # We insert it at the beginning of our custom actions.
            config.insert(0, {"type": "update_browse_history", "parameters": {"history": rich_history}}) 
            
            # 动作：打开一个新标签页作为开始
            config.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://www.google.com"]}})
            
            instruction_suffix = " Chrome is open. The browser history contains extensive activity from YouTube, Amazon, and other sites."

        # =====================================================================
        # Category 3: Search, Navigation & Information Retrieval
        # 场景：搜索信息、比价、查机票。
        # 策略：随机化起始点。有时从通用搜索开始，有时从垂类网站（旅行/购物）开始。
        # =====================================================================
        elif sub_category == "Search, Navigation & Information Retrieval":
            scenario = random.choice(["general_search", "shopping", "travel"])
            
            if scenario == "general_search":
                # 从 Google/Bing 开始
                url = random.choice(["https://www.google.com", "https://www.bing.com"])
                config.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": [url]}})
                instruction_suffix = f" Chrome is open at {url}. Ready for queries."
            
            elif scenario == "shopping":
                # 从 Amazon/Ebay/Shopping 开始
                url = random.choice(["https://www.amazon.com", "https://www.ebay.com", "https://shopping.google.com"])
                config.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": [url]}})
                instruction_suffix = " Chrome is open on a shopping platform."
                
            elif scenario == "travel":
                # 从 Expedia/Booking/FlightAware 开始
                url = random.choice(["https://www.expedia.com", "https://www.booking.com", "https://www.tripadvisor.com"])
                config.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": [url]}})
                instruction_suffix = " Chrome is open on a travel website."

        # =====================================================================
        # Category 4: Bookmarks, Tabs & Session Management
        # 场景：恢复标签、整理书签。
        # 关键：必须构造“Session”上下文，即有打开的标签，也有刚关闭的标签。
        # =====================================================================
        elif sub_category == "Bookmarks, Tabs & Session Management":
            # 1. 构造"最近关闭"的标签 (Open -> Close)
            # 选取 2-3 个 URL
            urls_to_close = random.sample(all_urls, 4)
            config.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": urls_to_close}})
            config.append({"type": "chrome_close_tabs", "parameters": {"urls_to_close": urls_to_close[:-2]}})
            
            instruction_suffix = " Chrome has several tabs open, and some were recently closed. The session is active."

        # =====================================================================
        # Category 5: File Management, Conversion & Downloads
        # 场景：下载文件、格式转换、扩展管理。
        # 策略：
        #   A (Download): 打开有下载链接的页面 (GitHub/Arxiv)。
        #   B (Upload/Convert): 桌面放置文件 + 打开转换工具网站。
        # =====================================================================
        elif sub_category == "File Management, Conversion & Downloads":
            task_type = random.choice(["download_source", "local_process"])
            
            if task_type == "download_source":
                # 打开一个典型的资源网站
                # 例如 Arxiv (PDF), GitHub (Code/Zip), Unsplash (Image)
                source_url = random.choice([
                    "https://arxiv.org/list/cs/recent", 
                    "https://github.com/torvalds/linux",
                    "https://unsplash.com/t/wallpapers"
                ])
                config.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": [source_url]}})
                instruction_suffix = " Chrome is open on a page with downloadable resources."
                
            else: # local_process (Upload/Convert/Install Extension)
                # 1. 准备本地文件 (从资源池中选一个 Writer 或 Image 文件)
                if self.writer_files:
                    f_name = random.choice(self.writer_files)
                    local_path = os.path.join(self.writer_asset_dir, f_name)
                    vm_path = f"/home/user/Desktop/{f_name}"
                    config.append({"type": "upload_file", "parameters": {"files": [{"local_path": local_path, "path": vm_path}]}})
                    file_msg = f"File '{f_name}' is on the Desktop."
                else:
                    file_msg = "Files are ready."

                # 2. 打开工具网站 (PDF转换, 扩展管理页)
                tool_url = random.choice([
                    "https://www.ilovepdf.com",
                    "https://convertio.co",
                    "chrome://extensions"
                ])
                config.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": [tool_url]}})
                
                instruction_suffix = f" Chrome is open at {tool_url}. {file_msg}"

        # =====================================================================
        # Category 6: Cross-Application Integration & Complex Workflows
        # 场景：Chrome 与本地应用 (Excel, Word, Terminal, Nautilus) 交互。
        # 策略：双开应用。
        # =====================================================================
        elif sub_category == "Cross-Application Integration & Complex Workflows":
            # 随机选择一个协作应用
            partner = random.choice(["calc", "writer", "terminal", "file_manager"])
            
            # Chrome 始终打开一个“云端”或“资源”页面
            web_url = random.choice(["https://drive.google.com", "https://github.com", "https://gmail.com"])
            config.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": [web_url]}})
            
            if partner == "calc" and self.calc_files:
                # 场景：从网页提取数据到 Excel，或从 Excel 上传到网页
                f_name = random.choice(self.calc_files)
                local_path = os.path.join(self.calc_asset_dir, f_name)
                vm_path = f"/home/user/Desktop/{f_name}"
                
                # Upload & Launch Calc
                config.append({"type": "upload_file", "parameters": {"files": [{"local_path": local_path, "path": vm_path}]}})
                config.append({"type": "launch", "parameters": {"command": ["libreoffice", "--calc", vm_path]}})
                instruction_suffix = f" Chrome and LibreOffice Calc (with '{f_name}') are both open."
                
            elif partner == "writer" and self.writer_files:
                # 场景：文档转换上传
                f_name = random.choice(self.writer_files)
                local_path = os.path.join(self.writer_asset_dir, f_name)
                vm_path = f"/home/user/Desktop/{f_name}"
                
                # Upload & Launch Writer
                config.append({"type": "upload_file", "parameters": {"files": [{"local_path": local_path, "path": vm_path}]}})
                config.append({"type": "launch", "parameters": {"command": ["libreoffice", "--writer", vm_path]}})
                instruction_suffix = f" Chrome and LibreOffice Writer (with '{f_name}') are both open."
                
            elif partner == "terminal":
                # 场景：Git clone, wget, 系统配置
                config.append({"type": "launch", "parameters": {"command": ["gnome-terminal"]}})
                instruction_suffix = " Chrome and Terminal are open. Ready for development tasks."
                
            else: # file_manager (nautilus)
                # 场景：整理下载文件，上传文件
                # 确保 Downloads 或 Desktop 有文件夹
                config.append({"type": "launch", "parameters": {"command": ["nautilus", "/home/user/Desktop"]}})
                instruction_suffix = " Chrome and File Manager are open."

        # =====================================================================
        # Fallback (Default)
        # =====================================================================
        else:
            config.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://www.google.com"]}})
            instruction_suffix = " Chrome is open."

        task['config'] = config
        task['instruction'] += instruction_suffix
        return task
        
    def _generate_vlc_task(self, sub_category: Optional[str] = None) -> dict:
        """
        Generates a VLC task with context-aware initialization based on 4 major categories.
        """
        domain_name = "vlc"
        task = self._get_base_task_template(domain_name, sub_category)
        
        # 默认配置容器
        files_to_upload = []  # List of (local_path, vm_path)
        launch_configs = []   # List of launch config dicts
        instruction_suffix = ""

        # ---------------------------------------------------------------------
        # 1. 基础播放与导航 (Playback_Control_and_Navigation)
        # ---------------------------------------------------------------------
        if sub_category == "Playback_Control_and_Navigation" or sub_category=="general":
            # 必须有一个媒体文件
            if self.media_files:
                f_name = random.choice(self.media_files)
                f_local = os.path.join(self.media_asset_dir, f_name)
                f_vm = f"/home/user/Desktop/{f_name}"
                
                files_to_upload.append((f_local, f_vm))
                # 直接播放该文件
                launch_configs.append({
                    "type": "launch",
                    "parameters": {"command": ["vlc", "--repeat",f_vm]}
                })
                instruction_suffix = f" VLC is playing '{f_name}'."
            else:
                # Fallback: 只启动 VLC
                logger.warning("No media files available for VLC playback. Launching blank VLC.")
                launch_configs.append({"type": "launch", "parameters": {"command": ["vlc"]}})
                instruction_suffix = " VLC is open."

        # ---------------------------------------------------------------------
        # 2. 转码、转换与提取 (Transcoding_Conversion_and_Extraction)
        # ---------------------------------------------------------------------
        elif sub_category == "Transcoding_Conversion_and_Extraction":
            # 需要素材文件在桌面
            if self.media_files:
                f_name = random.choice(self.media_files)
                f_local = os.path.join(self.media_asset_dir, f_name)
                f_vm = f"/home/user/Desktop/{f_name}"
                files_to_upload.append((f_local, f_vm))
                
                # 随机策略：
                # 方式 A: VLC 打开着这个文件 (从当前播放中提取)
                # 方式 B: VLC 空载启动 (通过菜单选择文件进行转换) - 这种更常见于转码任务
                start_with_file = random.choice([True, False])
                
                if start_with_file:
                    launch_configs.append({"type": "launch", "parameters": {"command": ["vlc","--repeat", f_vm]}})
                    instruction_suffix = f" VLC is open with '{f_name}'. You may need to convert or extract from it."
                else:
                    launch_configs.append({"type": "launch", "parameters": {"command": ["vlc"]}})
                    instruction_suffix = f" VLC is open. The source file '{f_name}' is on the Desktop."

        # ---------------------------------------------------------------------
        # 3. 跨应用协作 (Cross_Application_Integration)
        # ---------------------------------------------------------------------
        elif sub_category == "Cross_Application_Integration":
            # 必须有媒体文件
            video_vm_path = None
            if self.media_files:
                v_name = random.choice(self.media_files)
                v_local = os.path.join(self.media_asset_dir, v_name)
                video_vm_path = f"/home/user/Desktop/{v_name}"
                files_to_upload.append((v_local, video_vm_path))
            
            # 随机选择一个“伙伴应用”场景
            # 场景包括: Office (PPT/Word), GIMP (GIF制作), VSCode (教程)
            partner_app = random.choice(["impress", "writer", "gimp", "vscode"])
            
            # --- Scenario A: LibreOffice Impress (PPT) ---
            if partner_app == "impress" and self.impress_files:
                ppt_name = random.choice(self.impress_files)
                ppt_local = os.path.join(self.impress_asset_dir, ppt_name)
                ppt_vm = f"/home/user/Desktop/{ppt_name}"
                files_to_upload.append((ppt_local, ppt_vm))
                
                # 启动 PPT 和 VLC
                launch_configs.append({"type": "launch", "parameters": {"command": ["libreoffice", "--impress", ppt_vm]}})
                if video_vm_path:
                    launch_configs.append({"type": "launch", "parameters": {"command": ["vlc", video_vm_path]}})
                instruction_suffix = f" You are working on slides '{ppt_name}' with video reference '{v_name}'."

            # --- Scenario B: LibreOffice Writer (Word) ---
            elif partner_app == "writer" and self.writer_files:
                doc_name = random.choice(self.writer_files)
                doc_local = os.path.join(self.writer_asset_dir, doc_name)
                doc_vm = f"/home/user/Desktop/{doc_name}"
                files_to_upload.append((doc_local, doc_vm))
                
                launch_configs.append({"type": "launch", "parameters": {"command": ["libreoffice", "--writer", doc_vm]}})
                if video_vm_path:
                    launch_configs.append({"type": "launch", "parameters": {"command": ["vlc", video_vm_path]}})
                instruction_suffix = f" You are editing '{doc_name}' while watching '{v_name}'."

            # --- Scenario C: GIMP (Image/GIF) ---
            elif partner_app == "gimp":
                # GIMP 启动，VLC 启动
                launch_configs.append({"type": "launch", "parameters": {"command": ["gimp"]}})
                if video_vm_path:
                    launch_configs.append({"type": "launch", "parameters": {"command": ["vlc","--repeat", video_vm_path]}})
                instruction_suffix = f" GIMP and VLC are both open. Source video '{v_name}' is ready."

            # --- Scenario D: VS Code (Coding Tutorial) ---
            else: # vscode
                launch_configs.append({"type": "launch", "parameters": {"command": ["code", "/home/user/Desktop"]}})
                if video_vm_path:
                    launch_configs.append({"type": "launch", "parameters": {"command": ["vlc","--repeat", video_vm_path]}})
                instruction_suffix = " VS Code and VLC are open for a coding session."

        # ---------------------------------------------------------------------
        # 4. 系统配置与网络 (System_Configuration_and_Network)
        # ---------------------------------------------------------------------
        elif sub_category == "System_Configuration_and_Network":
            # 随机决定是“网络流”任务还是“系统设置”任务
            is_network_task = random.choice([True, False])
            
            if is_network_task:
                # 网络任务：只需启动 VLC，不需要本地文件
                launch_configs.append({"type": "launch", "parameters": {"command": ["vlc"]}})
                instruction_suffix = " VLC is open. Ready to stream content."
            else:
                # 系统设置任务 (如设置默认应用)：
                # 最好从 Terminal 或 Desktop 开始，而不是 VLC 内部
                launch_configs.append({"type": "launch", "parameters": {"command": ["gnome-terminal"]}})
                instruction_suffix = " You are starting from the Terminal to configure system settings."

        # =====================================================================
        # 组装最终 Config
        # =====================================================================
        
        # 1. Uploads
        for local, remote in files_to_upload:
            task["config"].append({
                "type": "upload_file",
                "parameters": {"files": [{"local_path": local, "path": remote}]}
            })
            
        # 2. Launches (顺序很重要，通常最后启动的在最上面)
        # 为了避免遮挡，如果有多应用，Launch 顺序可以随机，或者固定顺序
        for l_conf in launch_configs:
            task["config"].append(l_conf)

        task["instruction"] += instruction_suffix
        return task
    
    def _generate_vs_code_task(self, sub_category: Optional[str] = None) -> dict:
        """
        Generates a context-rich initial state for VS Code tasks.
        Automatically detects interesting files inside the project zip to open them on launch.
        """
        domain_name = "vs_code"
        task = self._get_base_task_template(domain_name, sub_category)
        config = []
        env_description = []

        # ---------------------------------------------------------------------
        # Helper 1: 项目选择、上传与文件探测
        # ---------------------------------------------------------------------
        def setup_project(project_type: str, dest_folder_name: str = "current_project"):
            """
            1. 从指定类型中选 ZIP
            2. 探测 ZIP 内的 '有趣' 文件 (py, md, html, js)
            3. 生成 Upload 和 Unzip 的 config
            返回: (project_name, remote_root_path, remote_file_path_to_open)
            """
            candidates = self.vscode_projects.get(project_type, [])
            
            # Fallback: 如果没有对应类型的 zip，创建一个空文件夹
            if not candidates:
                logger.warning(f"No zip found for type {project_type}, creating empty folder.")
                config.append({"type": "execute", "parameters": {"command": f"mkdir -p /home/user/Desktop/{dest_folder_name}"}})
                return "Empty Project", f"/home/user/Desktop/{dest_folder_name}", None
            
            # 1. 选文件
            chosen_zip = random.choice(candidates)
            local_zip_path = os.path.join(self.zip_asset_dir, chosen_zip)
            
            # 处理项目名 (去掉前缀和后缀)
            raw_name = chosen_zip.replace(".zip", "")
            clean_name = raw_name.split("_", 1)[1] if "_" in raw_name else raw_name
            
            remote_zip_path = f"/home/user/Desktop/{chosen_zip}"
            
            # 2. 探测内部文件 (Peek inside the local zip)
            target_file_relative = None
            try:
                with zipfile.ZipFile(local_zip_path, 'r') as zf:
                    file_list = zf.namelist()
                    # 过滤掉文件夹和隐藏文件
                    valid_files = [f for f in file_list if not f.endswith('/') and '/.' not in f and not f.startswith('.')]
                    
                    # 定义优先级后缀
                    priority_exts = ['.py', '.js', '.html', '.md', '.tex', '.java', '.cpp', '.json', '.ipynb']
                    
                    # 策略 A: 找 'main', 'index', 'readme' 且符合后缀的文件
                    for f in valid_files:
                        f_lower = f.lower()
                        if any(k in f_lower for k in ['main', 'index', 'readme']) and any(f_lower.endswith(ext) for ext in priority_exts):
                            target_file_relative = f
                            break
                    
                    # 策略 B: 如果没找到，找任意一个符合后缀的文件
                    if not target_file_relative:
                        for f in valid_files:
                            if any(f.lower().endswith(ext) for ext in priority_exts):
                                target_file_relative = f
                                break
                    
                    # 策略 C: 实在不行，选第一个文件
                    if not target_file_relative and valid_files:
                        target_file_relative = valid_files[0]

            except Exception as e:
                logger.error(f"Failed to inspect zip {local_zip_path}: {e}")

            # 3. 构造远程路径
            # 假设 zip 解压后会保留顶层文件夹结构 (通常 zip -r 会这样)
            # 我们通过 target_file_relative 的第一部分来确定解压后的根目录名
            if target_file_relative:
                top_folder = target_file_relative.split('/')[0]
                remote_project_root = f"/home/user/Desktop/{top_folder}"
                remote_file_path = f"/home/user/Desktop/{target_file_relative}"
            else:
                # 如果没找到文件，或者 zip 结构很奇怪，就假设解压到了 Desktop 下的 clean_name
                remote_project_root = f"/home/user/Desktop/{clean_name}"
                remote_file_path = None

            # 4. 生成 Config: 上传 & 解压
            config.append({
                "type": "upload_file", 
                "parameters": {"files": [{"local_path": local_zip_path, "path": remote_zip_path}]}
            })
            
            # 解压到 Desktop，并删除 zip 包
            config.append({
                "type": "execute",
                "parameters": {
                    "command": f"unzip -q -o {remote_zip_path} -d /home/user/Desktop/ && rm {remote_zip_path}",
                    "shell": True
                }
            })
            
            return clean_name, remote_project_root, remote_file_path

        # ---------------------------------------------------------------------
        # Helper 2: 辅助文件上传逻辑 (跨应用)
        # ---------------------------------------------------------------------
        def setup_aux_file(file_type="doc"):
            """上传一个 Word 或 Excel 文件作为参考资料"""
            source_list = self.writer_files if file_type == "doc" else self.calc_files
            source_dir = self.writer_asset_dir if file_type == "doc" else self.calc_asset_dir
            
            if source_list:
                f_name = random.choice(source_list)
                local_p = os.path.join(source_dir, f_name)
                remote_p = f"/home/user/Desktop/{f_name}"
                config.append({
                    "type": "upload_file", 
                    "parameters": {"files": [{"local_path": local_p, "path": remote_p}]}
                })
                return f_name
            return None

        # ---------------------------------------------------------------------
        # Category Logic
        # ---------------------------------------------------------------------
        
        # 1. Core Editor Settings & Customization
        # 场景：Web 项目通常文件结构丰富，适合用来测试界面设置（如隐藏文件、侧边栏、主题）
        if sub_category == "Core_Editor_Settings_and_Customization":
            proj_name, proj_root, target_file = setup_project("web")
            env_description.append(f"A web project '{proj_name}' is currently open.")
            
            launch_cmd = ["code", proj_root]
            if target_file:
                launch_cmd.append(target_file) # code <folder> <file>
                env_description.append(f"The file '{os.path.basename(target_file)}' is active in the editor.")
            
            config.append({"type": "launch", "parameters": {"command": launch_cmd}})

        # 2. Extension Management & External Tools
        # 场景：需要特定语言环境（Python 或 LaTeX），迫使用户检查或安装插件
        elif sub_category == "Extension_Management_and_External_Tools":
            # 随机选择 Python 算法项目或 LaTeX 项目
            p_type = "algo" if random.random() < 0.5 else "latex"
            proj_name, proj_root, target_file = setup_project(p_type)
            env_description.append(f"Project '{proj_name}' is loaded.")
            
            launch_cmd = ["code", proj_root]
            if target_file:
                launch_cmd.append(target_file)
            
            config.append({"type": "launch", "parameters": {"command": launch_cmd}})
            env_description.append("You may need to check or install extensions compatible with this project type.")

        # 3. Project, Workspace & File Operations
        # 场景：文件操作。有时从 Terminal 启动，有时打开空窗口。
        elif sub_category == "Project_Workspace_and_File_Operations":
            proj_name, proj_root, _ = setup_project("web") # 不需要特定打开某个文件
            env_description.append(f"The project folder '{proj_name}' is located on the Desktop.")
            
            if random.random() < 0.4:
                # 模拟从终端启动的场景
                config.append({"type": "launch", "parameters": {"command": ["gnome-terminal"]}})
                env_description.append("You are currently in the Terminal.")
            else:
                # 模拟打开空 VS Code 的场景
                config.append({"type": "launch", "parameters": {"command": ["code"]}})
                env_description.append("VS Code is open (empty window).")

        # 4. Code Editing, Refactoring & Navigation
        # 场景：核心代码编辑。提供算法项目 + 辅助文档（模拟需求文档）。
        elif sub_category == "Code_Editing_Refactoring_and_Navigation" or sub_category=="general":
            proj_name, proj_root, target_file = setup_project("algo")
            env_description.append(f"You are working on the code project '{proj_name}'.")
            
            # 注入辅助文档 (Word)
            aux = setup_aux_file("doc")
            if aux: 
                env_description.append(f"A reference document '{aux}' is available on the Desktop.")
            
            launch_cmd = ["code", proj_root]
            if target_file:
                launch_cmd.append(target_file)
                env_description.append(f"You are editing '{os.path.basename(target_file)}'.")
            
            config.append({"type": "launch", "parameters": {"command": launch_cmd}})

        # 5. Debugging, Execution & Language Support
        # 场景：调试运行。提供数据项目 + 辅助数据表（模拟原始数据）。
        elif sub_category == "Debugging_Execution_and_Language_Support":
            proj_name, proj_root, target_file = setup_project("data")
            env_description.append(f"You are debugging the data analysis project '{proj_name}'.")
            
            # 注入辅助数据 (Excel)
            aux = setup_aux_file("calc")
            if aux: 
                env_description.append(f"Raw data file '{aux}' is on the Desktop for verification.")
            
            launch_cmd = ["code", proj_root]
            if target_file:
                launch_cmd.append(target_file)
                env_description.append(f"The script '{os.path.basename(target_file)}' is open.")
            
            config.append({"type": "launch", "parameters": {"command": launch_cmd}})

        elif sub_category is not None and sub_category.lower() != "general":
            # 未知子类别处理
            logger.error(f"Unknown VS Code sub-category: {sub_category}")
            raise ValueError(f"Unknown VS Code sub-category: {sub_category}")
        # Fallback
        else:
            config.append({"type": "launch", "parameters": {"command": ["code"]}})
            env_description.append("VS Code is open.")

        # ---------------------------------------------------------------------
        # Finalize
        # ---------------------------------------------------------------------
        task['config'] = config
        
        # 将动态生成的环境描述追加到 instruction 中
        if env_description:
            info_str = " ".join(env_description)
            task['instruction'] += f" [Environment Context]: {info_str}"
            
        return task
    
    def _generate_thunderbird_task(self, sub_category: Optional[str] = None) -> dict:
        """
        Generates a Thunderbird task using pre-defined profile paths from __init__.
        """
        domain_name = "thunderbird"
        task = self._get_base_task_template(domain_name, sub_category)
        
        # Define VM paths (Destination paths in the environment)
        # We derive the filename from the local path to ensure consistency
        vm_p_rich = f"/home/user/{os.path.basename(self.profile_rich_file)}"
        vm_p_blank = f"/home/user/{os.path.basename(self.profile_blank_file)}"
        
        # Initialize config list
        task["config"] = []

        # =====================================================================
        # Category 1: Account Lifecycle Management (Clean Slate)
        # =====================================================================
        if sub_category == "Account_Lifecycle_Management":
            # 1. Upload Blank Profile (Use self.profile_blank_file)
            task["config"].append({
                "type": "upload_file",
                "parameters": {"files": [{"local_path": self.profile_blank_file, "path": vm_p_blank}]}
            })

            # 2. Extract Profile
            task["config"].append({
                "type": "execute",
                "parameters": {
                    "command": ["tar", "-xzv", "--recursive-unlink", "-f", vm_p_blank, "-C", "/home/user/"]
                }
            })

            # 3. Launch Thunderbird
            task["config"].append({
                "type": "launch",
                "parameters": {"command": ["/usr/bin/thunderbird"]}
            })
            
            task["instruction"] += " Thunderbird is open with a clean profile. You can set up or manage accounts."

        # =====================================================================
        # Category 2: Email Organization & Automation (Rich Data)
        # =====================================================================
        elif sub_category == "Email_Organization_and_Automation" or sub_category=="general":
            # 1. Upload Rich Profile (Use self.profile_rich_file)
            task["config"].append({
                "type": "upload_file",
                "parameters": {"files": [{"local_path": self.profile_rich_file, "path": vm_p_rich}]}
            })

            # 2. Extract Profile
            task["config"].append({
                "type": "execute",
                "parameters": {
                    "command": ["tar", "-xzv", "--recursive-unlink", "-f", vm_p_rich, "-C", "/home/user/"]
                }
            })

            # 3. Create Dummy Folders
            task["config"].append({
                "type": "execute",
                "parameters": {
                    "command": ["mkdir", "-p", "/home/user/Documents/Work", "/home/user/Documents/Personal"]
                }
            })

            # 4. Launch Thunderbird
            task["config"].append({
                "type": "launch",
                "parameters": {"command": ["/usr/bin/thunderbird"]}
            })

            task["instruction"] += " Thunderbird is open with your email history. Ready for organization."

        # =====================================================================
        # Category 3: Interface Personalization (Rich Data)
        # =====================================================================
        elif sub_category == "Interface_Personalization_and_Settings":
            # 1. Upload Rich Profile
            task["config"].append({
                "type": "upload_file",
                "parameters": {"files": [{"local_path": self.profile_rich_file, "path": vm_p_rich}]}
            })

            # 2. Extract Profile
            task["config"].append({
                "type": "execute",
                "parameters": {
                    "command": ["tar", "-xzv", "--recursive-unlink", "-f", vm_p_rich, "-C", "/home/user/"]
                }
            })

            # 3. Launch Thunderbird
            task["config"].append({
                "type": "launch",
                "parameters": {"command": ["/usr/bin/thunderbird"]}
            })

            task["instruction"] += " Thunderbird is open. Please customize the interface."

        # =====================================================================
        # Category 4: Composition & Communication (Complex Logic)
        # =====================================================================
        elif sub_category == "Composition_and_Communication_Flow":
            # Randomly decide: CLI Automation OR Standard GUI Attachment
            is_cli_task = random.choice([True, False])

            # --- Logic A: CLI Automation (Start with Write Window & Template) ---
            if is_cli_task:
                # 1. Prepare Template (Found in self.tb_asset_dir)
                template_name = random.choice(["New-month AWS Bill.html", "Payment Reminder.html"])
                local_tmpl = os.path.join(self.tb_asset_dir, template_name)
                vm_tmpl = f"/home/user/.{template_name.replace(' ', '_').lower()}" # Hidden file
                
                # 2. Upload Template
                task["config"].append({
                    "type": "upload_file",
                    "parameters": {"files": [{"local_path": local_tmpl, "path": vm_tmpl}]}
                })
                
                # 3. Upload & Extract Rich Profile
                task["config"].append({
                    "type": "upload_file",
                    "parameters": {"files": [{"local_path": self.profile_rich_file, "path": vm_p_rich}]}
                })
                task["config"].append({
                    "type": "execute",
                    "parameters": {"command": ["tar", "-xzv", "--recursive-unlink", "-f", vm_p_rich, "-C", "/home/user/"]}
                })

                # 4. Launch with CLI Compose Command
                subject = "Invoice" if "Bill" in template_name else "Payment Reminder"
                cmd_str = f"/usr/bin/thunderbird -compose \"subject='{subject}',body='$(cat {vm_tmpl})'\""
                
                task["config"].append({
                    "type": "launch",
                    "parameters": {"command": cmd_str, "shell": True}
                })
                task["instruction"] += " A composition window is opened via command line using a template."

            # --- Logic B: Standard GUI (Start with Attachment on Desktop) ---
            else:
                # 1. Upload Attachment File (From Writer/Word assets if available)
                if hasattr(self, 'writer_files') and self.writer_files:
                    att_name = random.choice(self.writer_files)
                    att_local = os.path.join(self.writer_asset_dir, att_name)
                    
                    task["config"].append({
                        "type": "upload_file",
                        "parameters": {"files": [{"local_path": att_local, "path": f"/home/user/Desktop/{att_name}"}]}
                    })
                    file_info = f"with file '{att_name}' on Desktop"
                else:
                    file_info = ""

                # 2. Upload & Extract Rich Profile
                task["config"].append({
                    "type": "upload_file",
                    "parameters": {"files": [{"local_path": self.profile_rich_file, "path": vm_p_rich}]}
                })
                task["config"].append({
                    "type": "execute",
                    "parameters": {"command": ["tar", "-xzv", "--recursive-unlink", "-f", vm_p_rich, "-C", "/home/user/"]}
                })

                # 3. Launch Thunderbird Normally
                task["config"].append({
                    "type": "launch",
                    "parameters": {"command": ["/usr/bin/thunderbird"]}
                })
                task["instruction"] += f" Thunderbird is open {file_info}. Ready to compose."

        # =====================================================================
        # Category 5: Data Extraction & Cross App (Multi-App)
        # =====================================================================
        elif sub_category == "Data_Extraction_and_Cross_App_Workflow":
            # 1. Upload Rich Profile
            task["config"].append({
                "type": "upload_file",
                "parameters": {"files": [{"local_path": self.profile_rich_file, "path": vm_p_rich}]}
            })

            # 2. Extract Profile
            task["config"].append({
                "type": "execute",
                "parameters": {
                    "command": ["tar", "-xzv", "--recursive-unlink", "-f", vm_p_rich, "-C", "/home/user/"]}
            })

            # 3. Prepare Target Directories
            task["config"].append({
                "type": "execute",
                "parameters": {
                    "command": ["mkdir", "-p", "/home/user/Documents/Finance/receipts", "/home/user/Desktop/Exports"]
                }
            })

            # 4. Launch Thunderbird
            task["config"].append({
                "type": "launch",
                "parameters": {"command": ["/usr/bin/thunderbird"]}
            })

            # 5. Launch Partner App
            if random.choice([True, False]):
                task["config"].append({
                    "type": "launch",
                    "parameters": {"command": ["nautilus", "/home/user/Documents"]}
                })
                task["instruction"] += " Thunderbird and File Manager are open."
            else:
                task["instruction"] += " Thunderbird is open. Please export data to Documents."

        return task

    def _generate_gimp_task(self, sub_category: Optional[str] = None) -> dict:
        """
        Optimized generator based on State-Centric Classification.
        """
        domain_name = "gimp"
        task = self._get_base_task_template(domain_name, sub_category)
        
        # 默认变量
        launch_cmd = ["gimp"]
        files_to_upload = [] # List of (local_path, vm_path) tuples
        instruction_suffix = ""

        
        # 1. 普通图片编辑 (Standard_Image_Editing)
        # 状态: GIMP 打开 + JPG/PNG
        if sub_category in ["Color_Light_Mode","Spatial_Transform_Filters", "File_IO_Conversion", "general"]:
            if self.image_files:
                f_name = random.choice(self.image_files)
                f_local = os.path.join(self.image_asset_dir, f_name)
                f_vm = f"/home/user/Desktop/{f_name}"
                
                files_to_upload.append((f_local, f_vm))
                launch_cmd.append(f_vm)
                instruction_suffix = f" GIMP is open with '{f_name}'."
            else:
                # Fallback
                instruction_suffix = " GIMP is open."

        # 2. 图层/工程编辑 (Project_Layer_Manipulation)
        # 状态: GIMP 打开 + XCF
        elif sub_category == "Project_Layer_Manipulation":
            # 优先找 XCF，没有则回退到 PNG
            target_files = self.xcf_files if self.xcf_files else self.image_files
            target_dir = self.xcf_asset_dir if self.xcf_files else self.image_asset_dir
            
            if target_files:
                f_name = random.choice(target_files)
                f_local = os.path.join(target_dir, f_name)
                f_vm = f"/home/user/Desktop/{f_name}"
                
                files_to_upload.append((f_local, f_vm))
                launch_cmd.append(f_vm)
                instruction_suffix = f" GIMP is open with project '{f_name}'."

        # 3. 环境配置 (Environment_Configuration)
        elif sub_category == "Environment_Configuration":
            # 不需要上传文件，直接启动
            instruction_suffix = " GIMP is open and ready for configuration."

        # 4. 命令行与脚本 (CLI_Scripting_External)
        elif sub_category == "CLI_Scripting_External":
            # 1. 准备素材 (图片/脚本文件)
            # 不仅要有图片，有时候也可以放一个空的 python 脚本方便用户开始写代码
            files_to_upload = []
            
            # 必须上传图片素材
            if self.image_files:
                f_name = random.choice(self.image_files)
                f_local = os.path.join(self.image_asset_dir, f_name)
                f_vm = f"/home/user/Desktop/{f_name}"
                files_to_upload.append((f_local, f_vm))
            
            # 2. 决定启动环境：终端 (Terminal) 还是 编辑器 (VS Code)
            # 既然这个分类包含 Scripting 和 VS Code 配置，我们需要让 Agent 熟悉这两种环境
            start_env = random.choice(["terminal", "vscode"])
            
            if start_env == "terminal":
                launch_cmd = ["gnome-terminal"]
                instruction_suffix = " You are in the Terminal. Assets are on the Desktop."
            
            else: # vscode
                # 启动 VS Code 并直接打开 Desktop 文件夹，方便看到素材
                launch_cmd = ["code", "/home/user/Desktop"]
                instruction_suffix = " VS Code is open on the Desktop directory. You can write scripts or configure the editor. Assets are on the Desktop"


        # 5. 复杂流/特殊格式 (Complex_Workflow_Format_Specific)
        # 状态: 视具体 Prompt 关键词而定 (Video / Raw / Office)
        elif sub_category == "Complex_Workflow_Format_Specific":
            # 简单起见，我们随机一种场景
            scene = random.choice(["video", "office", "raw"])
            
            if scene == "video" and self.media_files:
                f_name = random.choice(self.media_files)
                f_local = os.path.join(self.media_asset_dir, f_name)
                f_vm = f"/home/user/Desktop/{f_name}"
                files_to_upload.append((f_local, f_vm))
                # GIMP 也可以尝试打开视频，或者不开
                launch_cmd = ["gimp", f_vm] 
                instruction_suffix = f" GIMP is set up for video editing with '{f_name}'."
            
            elif scene == "office":
                # 启动 GIMP 空白 + Writer
                launch_cmd = ["gimp"]
                # 注意：Config 里需要额外添加一个 LibreOffice 的启动项，下面处理
                task['config'].append({"type": "launch", "parameters": {"command": ["libreoffice", "--writer"]}})
                instruction_suffix = " GIMP and LibreOffice Writer are both open."
            
            else: # RAW / General
                if self.image_files:
                    f_name = random.choice(self.image_files)
                    files_to_upload.append((os.path.join(self.image_asset_dir, f_name), f"/home/user/Desktop/{f_name}"))
                    launch_cmd.append(f"/home/user/Desktop/{f_name}")

        # ==========================================================
        # 统一构建 Config
        # ==========================================================
        
        # 1. 生成 Upload Config
        for local, remote in files_to_upload:
            task["config"].append({
                "type": "upload_file",
                "parameters": {"files": [{"local_path": local, "path": remote}]}
            })
        
        # 2. 生成 Main Launch Config
        task["config"].append({
            "type": "launch",
            "parameters": {"command": launch_cmd}
        })
        
        task["instruction"] += instruction_suffix
        return task

    def _generate_os_task(self, sub_category: Optional[str] = None) -> dict:
        """
        Generates OS tasks by HEAVILY REUSING existing assets (zips, docs, images).
        This creates realistic file systems without manual generation.
        """
        domain_name = "os"
        task = self._get_base_task_template(domain_name, sub_category)
        config = []
        env_description = []

        # =====================================================================
        # Helper: 利用现有 ZIP 包快速构建复杂文件树
        # =====================================================================
        def setup_rich_file_tree(dest_folder="workspace"):
            """
            从现有的 VS Code 项目包中随机选一个解压，作为文件操作的'靶子'。
            比 touch 空文件真实得多。
            """
            # 优先选 Web 或 Algo 项目，因为它们文件多、层级深
            candidates = self.vscode_projects.get("web", []) + self.vscode_projects.get("algo", [])
            if not candidates:
                # Fallback: 如果没包，创建一个简单的目录
                config.append({"type": "execute", "parameters": {"command": f"mkdir -p /home/user/Desktop/{dest_folder}/data"}})
                return f"/home/user/Desktop/{dest_folder}"

            chosen_zip = random.choice(candidates)
            local_zip = os.path.join(self.zip_asset_dir, chosen_zip)
            remote_zip = f"/home/user/Desktop/{chosen_zip}"
            remote_dest = f"/home/user/Desktop/{dest_folder}"

            # 1. Upload
            config.append({"type": "upload_file", "parameters": {"files": [{"local_path": local_zip, "path": remote_zip}]}})
            
            # 2. Unzip into destination
            # -d 指定解压目录
            config.append({"type": "execute", "parameters": {
                "command": f"unzip -q -o {remote_zip} -d {remote_dest} && rm {remote_zip}", 
                "shell": True
            }})
            
            return remote_dest

        # =====================================================================
        # Helper: 注入一些独立文件 (Word/Excel/Image) 到指定目录
        # =====================================================================
        def inject_loose_files(target_dir):
            """在目录里撒一些 Word/Excel/图片，增加杂乱度"""
            # 选 1-2 个 Word
            for _ in range(random.randint(1, 2)):
                if self.writer_files:
                    f = random.choice(self.writer_files)
                    config.append({"type": "upload_file", "parameters": {
                        "files": [{"local_path": os.path.join(self.writer_asset_dir, f), 
                                   "path": f"{target_dir}/{f}"}]
                    }})
            
            # 选 1 个 Excel
            if self.calc_files:
                f = random.choice(self.calc_files)
                config.append({"type": "upload_file", "parameters": {
                    "files": [{"local_path": os.path.join(self.calc_asset_dir, f), 
                               "path": f"{target_dir}/{f}"}]
                }})

        # =====================================================================
        # Category 1: GUI_File_Manager_Operations
        # 复用策略：解压一个 Web 项目 -> 得到几百个文件 -> 用 Nautilus 打开
        # =====================================================================
        if sub_category == "GUI_File_Manager_Operations":
            # 在 Desktop/project 下生成文件树
            root = setup_rich_file_tree("project_files")
            
            # 再撒点 Word/Excel 进去，方便练习"移动不同类型文件"
            inject_loose_files(root)
            
            # 启动 Nautilus
            config.append({"type": "launch", "parameters": {"command": ["nautilus", root]}})
            
            env_description.append(f"File Manager is open at '{root}'.")
            env_description.append("The folder is populated with project files and documents.")

        # =====================================================================
        # Category 2: System_Settings_Configuration
        # 复用策略：上传图片资源用于改壁纸
        # =====================================================================
        elif sub_category == "System_Settings_Configuration":
            config.append({"type": "launch", "parameters": {"command": ["gnome-control-center"]}})
            
            # 确保 Pictures 文件夹里有图
            if self.image_files:
                # 随机选 2 张图上传
                for _ in range(2):
                    img = random.choice(self.image_files)
                    local_p = os.path.join(self.image_asset_dir, img)
                    remote_p = f"/home/user/Pictures/{img}"
                    config.append({"type": "upload_file", "parameters": {"files": [{"local_path": local_p, "path": remote_p}]}})
                env_description.append("Some images are available in the Pictures folder.")
            
            env_description.append("Settings panel is open.")

        # =====================================================================
        # Category 3: CLI_Software_Installation_and_Setup
        # 复用策略：这里主要靠 Chrome 和 Terminal，不太需要文件资源
        # =====================================================================
        elif sub_category == "CLI_Software_Installation_and_Setup":
            # Chrome + Terminal 组合
            base_chrome = self._get_base_chrome_config()
            config.extend(base_chrome)
            config.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://www.google.com"]}})
            config.append({"type": "launch", "parameters": {"command": ["gnome-terminal"]}})
            
            env_description.append("Terminal and Browser are ready.")

        # =====================================================================
        # Category 4: CLI_Advanced_Workflow_and_Scripting
        # 复用策略：
        # 1. Git: 解压一个 Algo 项目 -> 变成 Git Repo
        # 2. Grep: 解压一个 Doc/Web 项目 -> 搜索文本
        # =====================================================================
        elif sub_category == "CLI_Advanced_Workflow_and_Scripting":
            # 随机决定是 Git 任务还是 文本处理任务
            if random.random() < 0.5:
                # --- Git 场景 ---
                # 解压一个算法项目 (通常代码比较纯粹)
                root = setup_rich_file_tree("git_repo")
                
                # 初始化为 Git 仓库
                git_cmds = f"cd {root} && git init && git config user.email 'you@example.com' && git config user.name 'User' && git add . && git commit -m 'initial commit'"
                config.append({"type": "execute", "parameters": {"command": git_cmds, "shell": True}})
                
                # 启动 Terminal 进入该目录
                config.append({"type": "launch", "parameters": {"command": ["gnome-terminal", f"--working-directory={root}"]}})
                env_description.append(f"Terminal is open in a Git repository at '{root}'.")
                
            else:
                # --- Grep/Find 场景 ---
                # 解压一个文档或Web项目 (文本内容多)
                root = setup_rich_file_tree("data_analysis")
                
                # 启动 Terminal
                config.append({"type": "launch", "parameters": {"command": ["gnome-terminal", f"--working-directory={root}"]}})
                env_description.append(f"Terminal is open at '{root}' containing various source files.")

        # Fallback
        else:
            config.append({"type": "launch", "parameters": {"command": ["gnome-terminal"]}})

        task['config'] = config
        if env_description:
            task['instruction'] += f" [Environment Context]: {' '.join(env_description)}"
            
        return task