
# /path/to/project/openrlhf/agent/exploration/trajectory_utils.py
import datetime
import json
import logging
import os
import re
import math
import io
from PIL import Image, ImageDraw, ImageFont, ImageChops

# Import for type hinting
from exploration.exploration_agent import ExplorationAgent

logger = logging.getLogger("desktopenv.trajectory_utils")

def create_annotated_trajectory_montage(session_dir: str, traj_dir: str, node_path: list):
    """
    Creates an annotated montage from a specific trajectory path within a session.
    This version is confirmed to be compatible with the new Node structure.
    """
    logger.info("Attempting to create annotated trajectory montage from node path...")
    nodes_dir = os.path.join(session_dir, 'nodes')
    screenshots_dir = os.path.join(session_dir, 'screenshots')
    output_path = os.path.join(traj_dir, f'summary.png')

    if not node_path:
        logger.warning("Node path is empty. Cannot create montage.")
        return

    image_data = []
    # Load all node data once to avoid repeated file I/O
    node_data_map = {}
    for node_id in node_path:
        node_file_path = os.path.join(nodes_dir, f"{node_id}.json")
        if os.path.exists(node_file_path):
            with open(node_file_path, 'r', encoding='utf-8') as f:
                node_data_map[node_id] = json.load(f)
        else:
            logger.warning(f"Node file not found for montage: {node_file_path}")

    for i, node_id in enumerate(node_path):
        node_data = node_data_map.get(node_id)
        if not node_data:
            continue
            
        screenshot_filename = node_data.get('screenshot')
        if not screenshot_filename or not os.path.exists(os.path.join(screenshots_dir, screenshot_filename)):
            logger.warning(f"Screenshot file not found for node {node_id}")
            continue

        action_text = "Initial State"
        if i > 0:
            action_text = node_data.get('action_command', 'Action N/A')

        image_data.append({'file': screenshot_filename, 'action': action_text})

    if not image_data:
        logger.warning("No valid screenshot data found for the node path.")
        return

    TARGET_WIDTH, FONT_SIZE, PADDING = 768, 14, 5
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", FONT_SIZE)
    except IOError:
        font = ImageFont.load_default()

    first_img_path = os.path.join(screenshots_dir, image_data[0]['file'])
    with Image.open(first_img_path) as img:
        aspect_ratio = img.height / img.width
        target_height = int(TARGET_WIDTH * aspect_ratio)

    cols = 5
    rows = math.ceil(len(image_data) / cols)
    montage = Image.new('RGB', (cols * TARGET_WIDTH, rows * target_height), (240, 240, 240))

    for i, data in enumerate(image_data):
        row, col = divmod(i, cols)
        paste_x, paste_y = col * TARGET_WIDTH, row * target_height
        
        img_path = os.path.join(screenshots_dir, data['file'])
        with Image.open(img_path) as img:
            resized_img = img.resize((TARGET_WIDTH, target_height), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(resized_img)
            action_text = str(data['action'])
            
            try:
                text_bbox = draw.multiline_textbbox((PADDING, PADDING), action_text, font=font)
                bg_y_start = target_height - (text_bbox[3] - text_bbox[1] + 2 * PADDING)
                draw.rectangle([(0, bg_y_start), (TARGET_WIDTH, target_height)], fill=(0, 0, 0, 180))
            except AttributeError:
                text_width, text_height = draw.textsize(action_text, font=font)
                bg_y_start = target_height - (text_height + 2 * PADDING)
                draw.rectangle([(0, bg_y_start), (TARGET_WIDTH, target_height)], fill="black")

            draw.multiline_text((PADDING, bg_y_start + PADDING), action_text, fill="white", font=font)
            montage.paste(resized_img, (paste_x, paste_y))

    montage.save(output_path)
    logger.info(f"Successfully created annotated trajectory summary at: {output_path}")


def create_trajectory_manifest(session_dir: str, node_path: list, termination_reason: str,
                               agent: ExplorationAgent, traj_id: str, generate_summary: bool = True):
    """
    Generates a manifest and summary for a completed trajectory.
    Uses agent.history as the single source of truth for the trajectory's details.

    Args:
        generate_summary: If False, skips the LLM summary call (for offline post-processing).
    """
    logger.info(f"Generating manifest for a path of length {len(node_path)} from agent history...")

    if not agent.history:
        logger.warning("Agent history is empty. Manifest will have no detailed steps.")
        step_by_step_summary = []
    else:
        step_by_step_summary = []
        for i, history_entry in enumerate(agent.history):
            step_by_step_summary.append({
                "step": i + 1,
                "goal": history_entry.get("step_goal"),
                "action": history_entry.get("step_action", {}),
                "reason": history_entry.get("step_reason"),
                "future_impact": history_entry.get("future_impact", "N/A"),
                "verification_result": {
                    "result_type": history_entry.get("verification_result_type"),
                    "feedback": history_entry.get("verification_feedback")
                },
                'final_goal_at_step': history_entry.get("final_goal_at_step", "N/A")
            })

    # Generate summary via LLM only if requested (can be done offline)
    if generate_summary:
        final_task = agent.summarize()
    else:
        final_task = {}

    traj_dir = os.path.join(session_dir, 'trajectories', traj_id)
    os.makedirs(traj_dir, exist_ok=True)

    manifest = {
        "trajectory_id": traj_id,
        "session_id": os.path.basename(session_dir),
        "termination_reason": termination_reason,
        "length": len(node_path) - 1,
        "node_path": node_path,
        "final_task": final_task,
        "traj_detail": step_by_step_summary,
    }

    manifest_path = os.path.join(traj_dir, f"{traj_id}.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Trajectory manifest saved to: {manifest_path}")





def is_screenshot_consistent(current_img_bytes: bytes, saved_img_path: str, threshold: float = 10.0) -> bool:
    """
    Compares current screenshot bytes with a saved image file using RMS difference.
    
    Args:
        current_img_bytes: The raw bytes of the current screenshot.
        saved_img_path: The file path to the saved screenshot (reference).
        threshold: The RMS difference threshold. Lower is stricter. 
                   5.0 allows for minor clock changes or cursor blinking.
                   
    Returns:
        True if the images are consistent (difference <= threshold), False otherwise.
    """
    if not os.path.exists(saved_img_path):
        logger.warning(f"Saved screenshot not found at {saved_img_path}, skipping consistency check.")
        return True

    try:
        # Load images
        img1 = Image.open(io.BytesIO(current_img_bytes)).convert('RGB')
        img2 = Image.open(saved_img_path).convert('RGB')

        # Check dimensions
        if img1.size != img2.size:
            logger.warning(f"Image size mismatch: Current {img1.size} vs Saved {img2.size}")
            return False

        # Calculate RMS difference
        h1 = img1.histogram()
        h2 = img2.histogram()
        
        # Fast fail check (optional, but good for performance if totally different)
        # if h1 == h2: return True

        diff = ImageChops.difference(img1, img2)
        histogram = diff.histogram()

        def rms_diff(histogram):
            sq = (value * ((idx % 256) ** 2) for idx, value in enumerate(histogram))
            sum_of_squares = sum(sq)
            rms = math.sqrt(sum_of_squares / float(img1.size[0] * img1.size[1]))
            return rms

        error = rms_diff(histogram)
        
        is_consistent = error <= threshold
        
        if not is_consistent:
            logger.warning(f"Replay Consistency Check Failed. RMS Error: {error:.2f} (Threshold: {threshold})")
        else:
            logger.info(f"Replay Consistency Check Passed. RMS Error: {error:.2f}")
            
        return is_consistent

    except Exception as e:
        logger.error(f"Error during screenshot comparison: {e}", exc_info=True)
        return True