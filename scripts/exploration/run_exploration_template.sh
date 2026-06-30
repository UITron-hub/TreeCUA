#!/bin/bash
# ============================================================
# TreeCUA Exploration Launch Script (Template)
# ============================================================
#
# Required environment variables:
#   LLM_API_URL                - Anthropic Messages API endpoint
#   LLM_API_KEY                - Anthropic API key (x-api-key)
#   LLM_MODEL                  - Model name (e.g. claude-sonnet-4-5)
#
# Usage:
#   source config/env.template.sh
#   ./run_exploration_template.sh
# ============================================================

set -e

# --- Conda Environment ---
CONDA_BASE_PATH="${CONDA_BASE_PATH:-/path/to/anaconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-osworld}"
echo "Activating Conda environment: ${CONDA_ENV_NAME}..."
source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# --- Project Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# --- Domain Configuration ---
DOMAIN="${DOMAIN:-gimp}"
NUM_ENVS="${NUM_ENVS:-10}"
MAX_STEPS_PER_TRAJECTORY="${MAX_STEPS_PER_TRAJECTORY:-20}"

# --- VM Configuration ---
VM_PATH="${VM_PATH:-/path/to/Ubuntu.qcow2}"
PROVIDER="${PROVIDER:-docker}"
ACTION_SPACE="${ACTION_SPACE:-claude_computer_use}"
SCREEN_WIDTH="${SCREEN_WIDTH:-1024}"
SCREEN_HEIGHT="${SCREEN_HEIGHT:-768}"

# --- LLM Configuration (from environment) ---
MODEL="${LLM_MODEL}"
API_BASE_URL="${LLM_API_URL}"
API_KEY="${LLM_API_KEY}"
MAX_TOKENS="${MAX_TOKENS:-16384}"

# --- Results ---
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-$PROJECT_ROOT/results}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ENABLE_INLINE_SCORING="${ENABLE_INLINE_SCORING:-false}"
ENABLE_INLINE_SUMMARY="${ENABLE_INLINE_SUMMARY:-false}"

# --- Validate required variables ---
if [ -z "$API_KEY" ] || [ -z "$API_BASE_URL" ]; then
    echo "ERROR: LLM_API_KEY and LLM_API_URL must be set."
    exit 1
fi

# --- Workloads: "SubCategoryName:MaxTrajectories:MaxTrajectoriesPerTree" ---
WORKLOADS=(
    "Category_1:500:250"
    # Add more categories here...
)

echo "================================================"
echo "Starting TreeCUA Batch Exploration"
echo "  Domain:       $DOMAIN"
echo "  Model:        $MODEL"
echo "  Num Envs:     $NUM_ENVS"
echo "  Categories:   ${#WORKLOADS[@]}"
echo "================================================"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for config in "${WORKLOADS[@]}"; do
    IFS=":" read -r CURRENT_SUB_CAT CURRENT_MAX_TRAJ CURRENT_MAX_PER_TREE <<< "$config"
    SESSION_NAME="tree_explore_${DOMAIN}_${TIMESTAMP}/${CURRENT_SUB_CAT}"

    echo ">>> Processing: $CURRENT_SUB_CAT (max_traj=$CURRENT_MAX_TRAJ)"

    python "$PROJECT_ROOT/exploration/run_tree_exploration.py" \
        --session_name "$SESSION_NAME" \
        --num_envs "$NUM_ENVS" \
        --results_base_dir "$RESULTS_BASE_DIR" \
        --max_trajectories "$CURRENT_MAX_TRAJ" \
        --max_trajectories_per_tree "$CURRENT_MAX_PER_TREE" \
        --max_steps_exploration "$MAX_STEPS_PER_TRAJECTORY" \
        --path_to_vm "$VM_PATH" \
        --provider_name "$PROVIDER" \
        --action_space "$ACTION_SPACE" \
        --screen_width "$SCREEN_WIDTH" \
        --screen_height "$SCREEN_HEIGHT" \
        --model "$MODEL" \
        --api_base_url "$API_BASE_URL" \
        --api_key "$API_KEY" \
        --max_tokens "$MAX_TOKENS" \
        --log_level "$LOG_LEVEL" \
        --domain "$DOMAIN" \
        --sub_category "$CURRENT_SUB_CAT" \
        $( [ "$ENABLE_INLINE_SCORING" = "true" ] && echo "--enable_inline_scoring" ) \
        $( [ "$ENABLE_INLINE_SUMMARY" = "true" ] && echo "--enable_inline_summary" )

    echo ">>> Finished: $CURRENT_SUB_CAT"
    echo "------------------------------------------------"
    sleep 5
done

echo "--- All Categories Finished ---"
