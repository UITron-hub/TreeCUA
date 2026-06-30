#!/bin/bash
# ============================================================
# Post-Processing Pipeline for TreeCUA Exploration Results
# ============================================================
#
# Runs all post-processing steps on a completed exploration session:
#   1. Trajectory scoring (4 metrics, 0-3 scale)
#   2. Overall task summary generation
#   3. Stage-level breakdown (EFFECTIVE/NAVIGATION/NOISE)
#   4. Step-level reason synthesis
#
# Required environment variables:
#   LLM_API_URL   - LLM API base URL
#   LLM_API_KEY   - LLM API key
#   LLM_MODEL     - Model name (optional, will use server default)
#
# Usage:
#   ./run_post_processing.sh /path/to/session_dir [--step all]
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_post_processing.py"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <session_dir> [additional args...]"
    echo ""
    echo "Arguments:"
    echo "  session_dir    Path to exploration session directory (required)"
    echo ""
    echo "Optional:"
    echo "  --step <steps>        Comma-separated steps"
    echo "  --api_url <url>       LLM API base URL"
    echo "  --api_key <key>       LLM API key"
    echo "  --model <name>        Model name"
    echo "  --force_rescore       Re-score existing assessments"
    echo "  --force_re_summary    Re-generate existing summaries"
    echo "  --force_re_stages     Re-generate existing stage breakdowns"
    echo "  --force_reason_synthesis  Re-generate existing step thinking"
    echo ""
    echo "Environment variables:"
    echo "  LLM_API_URL, LLM_API_KEY, LLM_MODEL"
    exit 1
fi

PYTHON_BIN="${CONDA_PREFIX:-}/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

echo "=== TreeCUA Post-Processing Pipeline ==="
echo "Session dir: $1"
echo "Python: ${PYTHON_BIN}"
echo ""

exec "${PYTHON_BIN}" "${PYTHON_SCRIPT}" --session_dir "$@"
