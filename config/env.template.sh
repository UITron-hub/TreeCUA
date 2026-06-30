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
export LLM_MAX_TOKENS="4096"

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
export VERIFIER_MAX_TOKENS="512"

# ============================================================
# Scoring & Post-processing API — OpenAI Chat Completions format
#   Protocol: OpenAI-compatible Chat Completions
#   Headers:  Authorization: Bearer <key>
#   Endpoint: /v1/chat/completions
#   Used by:  TrajectoryScoringAgent, post_processing/*
#   NOTE: Can share the same endpoint as VERIFIER if using the
#         same API provider. Typically gpt-4o-mini.
# ============================================================
export SCORING_API_URL="https://your-openai-endpoint/v1/chat/completions"
export SCORING_API_KEY="your-scoring-api-key"
export SCORING_MODEL="gpt-4o-mini"

# ============================================================
# Resource paths
# ============================================================
export WORLD_KNOWLEDGE_PATH="/path/to/world_knowledge.json"
export RESOURCE_DIR="/path/to/test/resources"
