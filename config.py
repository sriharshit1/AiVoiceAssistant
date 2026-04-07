"""config.py — All settings from .env + prompts.yaml"""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR     = Path(__file__).parent
PROMPTS_FILE = BASE_DIR / "prompts.yaml"

def load_prompts() -> dict:
    if not PROMPTS_FILE.exists():
        raise FileNotFoundError(f"prompts.yaml not found at {PROMPTS_FILE}")
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PROMPTS = load_prompts()

# ── API ─────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set in .env")

# ── Server ──────────────────────────────────────────────────
SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))

# ── Whisper ──────────────────────────────────────────────────
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")

# ── edge-tts ─────────────────────────────────────────────────
_tts = PROMPTS.get("tts", {})
EDGE_TTS_VOICE:  str = os.getenv("EDGE_TTS_VOICE",  _tts.get("voice",  "en-US-JennyNeural"))
EDGE_TTS_RATE:   str = os.getenv("EDGE_TTS_RATE",   _tts.get("rate",   "+0%"))
EDGE_TTS_VOLUME: str = os.getenv("EDGE_TTS_VOLUME", _tts.get("volume", "+0%"))

# ── TTS fallback ─────────────────────────────────────────────
TTS_RATE:        int   = int(os.getenv("TTS_RATE",   "175"))
TTS_VOLUME:      float = float(os.getenv("TTS_VOLUME", "1.0"))
TTS_VOICE_INDEX: int   = int(os.getenv("TTS_VOICE_INDEX", "0"))

# ── Microphone ───────────────────────────────────────────────
SAMPLE_RATE:       int   = int(os.getenv("SAMPLE_RATE",        "16000"))
SILENCE_THRESHOLD: int   = int(os.getenv("SILENCE_THRESHOLD",  "500"))
SILENCE_DURATION:  float = float(os.getenv("SILENCE_DURATION", "1.5"))

# ── Groq LLM ─────────────────────────────────────────────────
_g = PROMPTS.get("groq", {})
GROQ_MODEL:       str   = _g.get("model",       "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_TEMPERATURE: float = _g.get("temperature", 0.7)
GROQ_MAX_TOKENS:  int   = _g.get("max_tokens",  1024)
GROQ_TOP_P:       float = _g.get("top_p",       0.9)

# ── Memory ───────────────────────────────────────────────────
MEMORY_MAX_TURNS:     int = int(os.getenv("MEMORY_MAX_TURNS",     "20"))
MEMORY_SUMMARY_EVERY: int = int(os.getenv("MEMORY_SUMMARY_EVERY", "8"))

# ── User Location & Timezone ─────────────────────────────────
# Override any of these in your .env file:
#   DEFAULT_CITY=Mumbai
#   DEFAULT_COUNTRY=India
#   USER_TIMEZONE=Asia/Kolkata
DEFAULT_CITY:    str = os.getenv("DEFAULT_CITY",    "Lucknow")
DEFAULT_COUNTRY: str = os.getenv("DEFAULT_COUNTRY", "India")
USER_TIMEZONE:   str = os.getenv("USER_TIMEZONE",   "Asia/Kolkata")