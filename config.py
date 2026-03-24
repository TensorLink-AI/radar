"""
Configuration for RADAR.

All settings in one place. Override via environment variables prefixed with RADAR_.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # read .env file from project root


class Config:
    """Central configuration — edit here or set RADAR_* env vars."""

    # ── Runner Backend ────────────────────────────────────────────────
    # Where to execute training: "local" | "affinetes"
    RUNNER_BACKEND: str = os.getenv("RADAR_RUNNER_BACKEND", "local")

    # ── LLM Configuration ──────────────────────────────────────────────
    LLM_MODEL_STRONG: str = os.getenv("RADAR_LLM_MODEL_STRONG", "o3")  # for planning

    # ── Desearch Proxy ────────────────────────────────────────────────
    # SN22 Desearch endpoint URL
    DESEARCH_SN22_URL: str = os.getenv("RADAR_DESEARCH_SN22_URL", "https://desearch.ai/api/v1")
    # Max queries per miner per tempo
    DESEARCH_MAX_QUERIES: int = int(os.getenv("RADAR_DESEARCH_MAX_QUERIES", "20"))
    # Enable desearch proxy (set to "true" to enable)
    DESEARCH_ENABLED: bool = os.getenv("RADAR_DESEARCH_ENABLED", "false").lower() == "true"

    # ── Official Training Image ──────────────────────────────────────
    OFFICIAL_TRAINING_IMAGE: str = os.getenv("OFFICIAL_TRAINING_IMAGE", "ghcr.io/tensorlink-ai/radar/ts-runner:latest")
    OFFICIAL_TRAINING_IMAGE_DIGEST: str = os.getenv("OFFICIAL_TRAINING_IMAGE_DIGEST", "")

    # ── R2 Audit Log ────────────────────────────────────────────────
    R2_ACCOUNT_ID: str = os.getenv("R2_ACCOUNT_ID", "")
    R2_ACCESS_KEY_ID: str = os.getenv("R2_ACCESS_KEY_ID", "")
    R2_SECRET_ACCESS_KEY: str = os.getenv("R2_SECRET_ACCESS_KEY", "")
    R2_BUCKET: str = os.getenv("R2_BUCKET", "")
    R2_PRESIGNED_TTL: int = int(os.getenv("RADAR_PRESIGNED_TTL", "5400"))

    # ── Round Timing (blocks, ~12s/block) ────────────────────────
    ROUND_INTERVAL_BLOCKS: int = int(os.getenv("RADAR_ROUND_INTERVAL", "275"))
    SUBMISSION_WINDOW_BLOCKS: int = int(os.getenv("RADAR_SUBMISSION_WINDOW", "50"))
    TRAINING_WINDOW_BLOCKS: int = int(os.getenv("RADAR_TRAINING_WINDOW", "150"))
    EVAL_WINDOW_BLOCKS: int = int(os.getenv("RADAR_EVAL_WINDOW", "25"))
    SKIP_TRAINING_WAIT: bool = os.getenv("RADAR_SKIP_TRAINING_WAIT", "false").lower() == "true"
    SIZE_GATE_TOLERANCE: float = float(os.getenv("RADAR_SIZE_GATE_TOLERANCE", "0.10"))
    FALLBACK_ENABLED: bool = os.getenv("RADAR_FALLBACK_ENABLED", "false").lower() == "true"
    FALLBACK_WINDOW_BLOCKS: int = int(os.getenv("RADAR_FALLBACK_WINDOW", "50"))

    # ── Agent (Phase A) ──────────────────────────────────────────
    AGENT_TIMEOUT: int = int(os.getenv("RADAR_AGENT_TIMEOUT", "600"))

    # ── Training Dispatch ────────────────────────────────────────
    TRAINING_TIMEOUT: int = int(os.getenv("RADAR_TRAINING_TIMEOUT", "1800"))

    # ── Phase C Eval ─────────────────────────────────────────────
    EVAL_DEVICE: str = os.getenv("RADAR_EVAL_DEVICE", "cpu")

    # ── Scoring ──────────────────────────────────────────────────
    EMA_ALPHA: float = float(os.getenv("RADAR_EMA_ALPHA", "0.3"))
    SOFTMAX_TEMPERATURE: float = float(os.getenv("RADAR_SOFTMAX_TEMP", "0.1"))

    # ── Agent Scratchpad ────────────────────────────────────────
    SCRATCHPAD_ENABLED: bool = os.getenv("RADAR_SCRATCHPAD_ENABLED", "true").lower() == "true"
    SCRATCHPAD_MAX_MB: int = int(os.getenv("RADAR_SCRATCHPAD_MAX_MB", "10"))
    SCRATCHPAD_TTL: int = int(os.getenv("RADAR_SCRATCHPAD_TTL", "900"))

    # ── Query API ────────────────────────────────────────────────
    QUERY_RATE_LIMIT: int = int(os.getenv("RADAR_QUERY_RATE_LIMIT", "10"))
