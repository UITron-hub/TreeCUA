#!/bin/bash
# ============================================================
# TreeCUA Environment Configuration Template
# ============================================================
# Copy this file to env.local.sh and fill in your values.
#   cp config/env.template.sh config/env.local.sh
#   source config/env.local.sh
# ============================================================

# ============================================================
# Exploration API — Anthropic Messages format (required)
#   Protocol: Native Anthropic Messages API
#   Headers:  x-api-key, anthropic-version: 2023-06-01
#   Endpoint: /v1/messages
#   Used by:  ExplorationAgent (candidate generation, summarization)
# ============================================================
export LLM_API_URL="https://your-api-endpoint/v1/messages"
export LLM_API_KEY="your-api-key-here"
export LLM_MODEL="claude-sonnet-4-5"
export LLM_MAX_TOKENS="16384"

# ============================================================
# Verifier API — OpenAI Chat Completions format
#   Protocol: OpenAI-compatible Chat Completions
#   Headers:  Authorization: Bearer <key>
#   Endpoint: /v1/chat/completions
#   Used by:  VerificationAgent (step-level verification)
#   NOTE: Must use a different API/endpoint than exploration!
#         Typically gpt-4o-mini or similar fast model.
# ============================================================
export VERIFIER_API_URL="https://your-openai-endpoint/v1/chat/completions"
export VERIFIER_API_KEY="your-verifier-api-key"
export VERIFIER_MODEL="gpt-4o-mini"
export VERIFIER_MAX_TOKENS="5120"

# ============================================================
# Post-processing API — OpenAI Chat Completions format
#   Protocol: OpenAI-compatible Chat Completions
#   Headers:  Authorization: Bearer <key>
#   Endpoint: /v1/chat/completions
#   Used by:  TrajectoryScoringAgent, summary_overall,
#            summary_stages, reason_synthesis
#   NOTE: Can share the same endpoint as VERIFIER if using the
#         same API provider. Typically gpt-4o-mini.
# ============================================================
export POSTPROCESS_API_URL="https://your-openai-endpoint/v1/chat/completions"
export POSTPROCESS_API_KEY="your-postprocess-api-key"
export POSTPROCESS_MODEL="gpt-4o-mini"

# ============================================================
# Project paths
# ============================================================
# Automatically set to the current directory. Source env.local.sh from
# the project root after `cd TreeCUA`, or change this to an absolute path.
export TREE_CUA_ROOT="$PWD"
export PYTHONPATH="$TREE_CUA_ROOT:$PYTHONPATH"

# ============================================================
# Resource paths
# ============================================================
# A simplified reference version of world_knowledge.json is provided
# at data_resource/world_knowledge.json in this repository.
export WORLD_KNOWLEDGE_PATH="$TREE_CUA_ROOT/data_resource/world_knowledge.json"
# Point to the downloaded TreeCUA-Datasets (Hugging Face).
export RESOURCE_DIR="$TREE_CUA_ROOT/data_cache/TreeCUA_Datasets"
