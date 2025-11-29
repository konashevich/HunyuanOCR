"""
MCP Server Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# vLLM Backend
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")

# MCP Server Settings
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "3000"))

# Security (optional)
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "")

# Limits
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))

# Model
MODEL_NAME = "tencent/HunyuanOCR"
