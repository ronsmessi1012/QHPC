from dotenv import load_dotenv
import os

# Load .env if present
load_dotenv()

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Borderline routing threshold (used later)
BORDERLINE_THRESHOLD = 0.15